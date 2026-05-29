"""
多因子选股模型：因子计算、动态权重、MAD去极值、市值中性化、行业内选股
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
from typing import Dict, List, Tuple, Optional
from config import StrategyConfig, StyleRegime


class FactorModel:
    """多因子选股引擎"""

    def __init__(self, data_source):
        self.ds = data_source
        self.config = StrategyConfig()
        self.factor_names = [
            'EP', 'BP', 'DIVIDEND_YIELD',
            'ROE', 'ROE_STABILITY', 'PROFIT_GROWTH', 'OCF_TO_REVENUE',
            'MOMENTUM_20', 'MOMENTUM_60',
            'VOLATILITY_20', 'TURNOVER_STABILITY', 'VOLUME_PRICE_DIVERGENCE'
        ]
        # 因子方向映射（正向/反向）
        self.factor_direction = {
            'EP': 1, 'BP': 1, 'DIVIDEND_YIELD': 1,
            'ROE': 1, 'ROE_STABILITY': 1, 'PROFIT_GROWTH': 1, 'OCF_TO_REVENUE': 1,
            'MOMENTUM_20': 1, 'MOMENTUM_60': 1,
            'VOLATILITY_20': -1, 'TURNOVER_STABILITY': 1, 'VOLUME_PRICE_DIVERGENCE': -1
        }

    def mad_winsorize(self, series: pd.Series, n: int = 5) -> pd.Series:
        """MAD去极值"""
        median = series.median()
        mad = (series - median).abs().median()
        upper = median + n * mad
        lower = median - n * mad
        return series.clip(lower, upper)

    def neutralize_market_cap(self, factor: pd.Series, market_cap: pd.Series) -> pd.Series:
        """市值中性化：对市值做回归取残差"""
        log_cap = np.log(market_cap)
        mask = factor.notna() & log_cap.notna()
        if mask.sum() < 10:
            return factor

        slope, intercept, _, _, _ = linregress(log_cap[mask], factor[mask])
        residual = factor.copy()
        residual[mask] = factor[mask] - (intercept + slope * log_cap[mask])
        return residual

    def zscore_standardize(self, series: pd.Series) -> pd.Series:
        """Z-Score标准化"""
        mean = series.mean()
        std = series.std()
        if std < 1e-8:
            return pd.Series(0, index=series.index)
        return (series - mean) / std

    def process_factor(self, raw_factor: pd.Series, market_cap: pd.Series) -> pd.Series:
        """完整因子处理流程：去极值 -> 市值中性化 -> 标准化"""
        f = self.mad_winsorize(raw_factor)
        f = self.neutralize_market_cap(f, market_cap)
        f = self.zscore_standardize(f)
        return f

    def calculate_factor_ic(self, factor_name: str, date: str, lookback: int = 40) -> float:
        """
        计算因子近lookback日Rank IC
        实际应加载历史因子和收益数据，这里返回模拟值
        """
        # 简化：返回随机IC，实际需用历史数据计算
        return np.random.uniform(0.03, 0.08)

    def get_dynamic_weights(self, style_regime: StyleRegime, date: str) -> Dict[str, float]:
        """
        获取动态因子权重
        基础权重 + IC动态调整
        """
        base_weights = self.config.FACTOR_WEIGHT_MAP[style_regime].copy()

        # 计算各因子近期IC
        ic_scores = {}
        for factor in self.factor_names:
            ic = self.calculate_factor_ic(factor, date, self.config.FACTOR_IC_LOOKBACK)
            if ic < self.config.FACTOR_IC_THRESHOLD:
                ic_scores[factor] = -1.0  # 失效标记
            else:
                ic_scores[factor] = np.tanh(ic * 3)  # 压缩到[-1,1]

        # 动态调整
        adjusted_weights = {}
        for factor, weight in base_weights.items():
            if factor in ic_scores:
                if ic_scores[factor] < -0.5:  # 因子失效
                    adjusted_weights[factor] = 0
                else:
                    adj = ic_scores[factor] * 0.3  # ±30%调整
                    adjusted_weights[factor] = weight * (1 + adj)
            else:
                adjusted_weights[factor] = weight

        # 权重约束检查
        # 价值类
        value_factors = ['EP', 'BP', 'DIVIDEND_YIELD']
        value_sum = sum(adjusted_weights.get(f, 0) for f in value_factors)
        if not (self.config.VALUE_WEIGHT_RANGE[0] <= value_sum <= self.config.VALUE_WEIGHT_RANGE[1]):
            # 缩放到范围内
            target = np.clip(value_sum, *self.config.VALUE_WEIGHT_RANGE)
            if value_sum > 0:
                scale = target / value_sum
                for f in value_factors:
                    adjusted_weights[f] = adjusted_weights.get(f, 0) * scale

        # 动量类
        mom_factors = ['MOMENTUM_20', 'MOMENTUM_60']
        mom_sum = sum(adjusted_weights.get(f, 0) for f in mom_factors)
        if not (self.config.MOMENTUM_WEIGHT_RANGE[0] <= mom_sum <= self.config.MOMENTUM_WEIGHT_RANGE[1]):
            target = np.clip(mom_sum, *self.config.MOMENTUM_WEIGHT_RANGE)
            if mom_sum > 0:
                scale = target / mom_sum
                for f in mom_factors:
                    adjusted_weights[f] = adjusted_weights.get(f, 0) * scale

        # 微观结构（负向）
        micro_factors = ['VOLATILITY_20', 'VOLUME_PRICE_DIVERGENCE']
        micro_sum = sum(adjusted_weights.get(f, 0) for f in micro_factors)
        if not (self.config.MICRO_WEIGHT_RANGE[0] <= micro_sum <= self.config.MICRO_WEIGHT_RANGE[1]):
            target = np.clip(micro_sum, *self.config.MICRO_WEIGHT_RANGE)
            if micro_sum < 0:
                scale = target / micro_sum
                for f in micro_factors:
                    adjusted_weights[f] = adjusted_weights.get(f, 0) * scale

        # 归一化（绝对值之和为1）
        total_abs = sum(abs(v) for v in adjusted_weights.values())
        if total_abs > 0:
            adjusted_weights = {k: v / total_abs for k, v in adjusted_weights.items()}

        return adjusted_weights

    def select_stocks_in_industry(self, 
                                   industry: str, 
                                   industry_weight: float,
                                   factor_weights: Dict[str, float],
                                   date: str,
                                   market_state) -> pd.DataFrame:
        """
        在指定行业内选股
        返回: DataFrame [code, score, target_weight, ...]
        """
        # 加载该行业股票列表
        all_stocks = self.ds.load_factor_data(date)
        if all_stocks.empty:
            return pd.DataFrame()

        # 过滤该行业股票
        stocks = all_stocks[all_stocks['INDUSTRY'] == industry].copy()
        if len(stocks) < 5:
            return pd.DataFrame()

        date_obj = pd.to_datetime(date)

        # 基础过滤
        stocks = stocks[
            (pd.to_datetime(stocks['LIST_DATE']) <= date_obj - pd.Timedelta(days=self.config.STOCK_MIN_LIST_DAYS)) &
            (stocks['MARKET_CAP'] > 0)
        ].copy()

        if len(stocks) < 3:
            return pd.DataFrame()

        # 因子处理与打分
        market_cap = stocks['MARKET_CAP']
        composite_score = pd.Series(0.0, index=stocks.index)

        for factor_name, weight in factor_weights.items():
            if factor_name not in stocks.columns:
                continue

            raw = stocks[factor_name]
            processed = self.process_factor(raw, market_cap)

            # 方向调整（负权重因子反向）
            direction = self.factor_direction.get(factor_name, 1)
            composite_score += processed * weight * direction

        stocks['factor_score'] = composite_score
        stocks = stocks.sort_values('factor_score', ascending=False)

        # 每行业选Top N
        n_select = self.config.STOCKS_PER_INDUSTRY
        selected = stocks.head(n_select).copy()

        # 个股权重：行业权重内等权（后续组合层会做波动率调整）
        selected['base_weight'] = industry_weight / len(selected)

        return selected[['base_weight', 'factor_score', 'MARKET_CAP', 'EP', 'ROE', 'MOMENTUM_20', 'VOLATILITY_20']]
