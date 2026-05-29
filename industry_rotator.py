"""
行业轮动矩阵：行业评分、筛选、拥挤度检测、行业止损
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from config import StrategyConfig


class IndustryRotator:
    """行业轮动引擎"""

    def __init__(self, data_source):
        self.ds = data_source
        self.config = StrategyConfig()
        self.industries = self.config.SW_INDUSTRY_NAMES

    def calculate_industry_scores(self, date: str) -> pd.DataFrame:
        """
        计算所有行业综合得分
        返回: DataFrame [industry, momentum, trend, flow, congestion, congestion_penalty, total_score]
        """
        end_date = pd.to_datetime(date)
        start_date = end_date - pd.Timedelta(days=200)
        start_str = start_date.strftime('%Y-%m-%d')

        records = []

        # 计算全市场总成交额（用于拥挤度）
        total_amount_5d = 0
        total_amount_60d = 0
        industry_amounts = {}

        for ind in self.industries:
            df = self.ds.load_industry_index(ind, start=start_str, end=date)
            if len(df) < 60:
                continue

            close = df['close']
            amount = df['amount']

            # 动量：20日收益率
            mom_20 = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0

            # 趋势：偏离120日均线
            ma120 = close.rolling(self.config.INDUSTRY_TREND_MA).mean().iloc[-1]
            trend_strength = (close.iloc[-1] / ma120 - 1) if ma120 > 0 else 0

            # 资金：20日成交额占比变化
            amount_20 = amount.iloc[-20:].sum()
            amount_60 = amount.iloc[-60:].sum()
            flow_score = (amount_20 / 20) / (amount_60 / 60) - 1  # 近期成交额相对变化

            # 拥挤度：5日成交额占比历史分位
            amount_5 = amount.iloc[-5:].sum()
            industry_amounts[ind] = amount_5

            records.append({
                'industry': ind,
                'momentum': mom_20,
                'trend': trend_strength,
                'flow': flow_score,
                'amount_5d': amount_5,
                'close': close.iloc[-1],
                'ma20': close.rolling(20).mean().iloc[-1],
                'price_vs_ma20': close.iloc[-1] / close.rolling(20).mean().iloc[-1] - 1
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # 计算全市场成交额
        total_amount_5d = df['amount_5d'].sum()
        df['amount_pct'] = df['amount_5d'] / total_amount_5d if total_amount_5d > 0 else 0

        # 拥挤度历史分位（使用模拟的60日历史）
        # 实际应加载历史数据，这里用当前截面分位近似
        df['congestion_pct'] = df['amount_pct'].rank(pct=True)

        # 拥挤度惩罚（非线性）
        def congestion_penalty(pct):
            if pct > self.config.CONGESTION_EXTREME:
                return 2.0
            elif pct > self.config.CONGESTION_HIGH:
                return 1.5
            elif pct > self.config.CONGESTION_MID:
                return 1.0
            return 0.0

        df['congestion_penalty'] = df['congestion_pct'].apply(congestion_penalty)

        # 综合得分（标准化后加权）
        for col in ['momentum', 'trend', 'flow']:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

        df['total_score'] = (
            0.30 * df['momentum'] + 
            0.30 * df['trend'] + 
            0.20 * df['flow'] - 
            0.20 * df['congestion_penalty']
        )

        return df.sort_values('total_score', ascending=False).reset_index(drop=True)

    def select_industries(self, scores_df: pd.DataFrame, market_state, date: str) -> pd.DataFrame:
        """
        筛选强势行业
        返回: 入选行业DataFrame（含权重）
        """
        if scores_df.empty:
            return pd.DataFrame()

        # 流动性过滤：剔除成交额占比过低的（模拟中跳过）

        # 根据市场状态调整入选数量
        if market_state.value in ['strong_bear', 'weak_bear']:
            top_n = max(5, self.config.INDUSTRY_TOP_N // 2)
        else:
            top_n = self.config.INDUSTRY_TOP_N

        selected = scores_df.head(top_n).copy()

        # 基础等权 + 得分偏离
        base_weight = 1.0 / len(selected)
        # Top 3 加分，末位减分
        weights = []
        for i in range(len(selected)):
            if i < 3:
                w = base_weight * 1.2
            elif i >= len(selected) - 3:
                w = base_weight * 0.8
            else:
                w = base_weight
            weights.append(w)

        weights = np.array(weights)
        weights = weights / weights.sum()  # 归一化

        # 应用行业权重上限
        weights = np.clip(weights, 0, self.config.INDUSTRY_MAX_WEIGHT)
        weights = weights / weights.sum()  # 再次归一化

        selected['target_weight'] = weights

        return selected[['industry', 'total_score', 'congestion_pct', 'target_weight', 'close', 'ma20']]

    def check_industry_stop_loss(self, industry: str, date: str) -> Tuple[bool, str]:
        """
        检查行业是否触发止损
        返回: (是否止损, 原因)
        """
        df = self.ds.load_industry_index(industry, end=date)
        if len(df) < 20:
            return False, ""

        close = df['close']
        ma20 = close.rolling(20).mean().iloc[-1]
        current = close.iloc[-1]

        # 破20日均线
        if current < ma20:
            return True, f"跌破MA20 ({current:.2f} < {ma20:.2f})"

        # 5日暴跌
        if len(close) >= 5:
            ret_5d = (current / close.iloc[-6] - 1)
            if ret_5d < -self.config.INDUSTRY_CRASH_LIMIT:
                return True, f"5日跌幅{ret_5d:.1%}"

        return False, ""
