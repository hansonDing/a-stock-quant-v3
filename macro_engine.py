"""
宏观周期引擎：市场状态判断 + 风格周期判断
输出：市场状态标签、仓位系数、风格偏好
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from config import StrategyConfig, MarketState, StyleRegime


class MacroCycleEngine:
    """宏观周期研判引擎"""

    def __init__(self, data_source):
        self.ds = data_source
        self.config = StrategyConfig()

    def detect_market_state(self, date: str) -> Tuple[MarketState, Dict]:
        """
        判断当前市场状态
        返回: (状态标签, 详细得分字典)
        """
        # 获取万得全A近期数据
        end_date = pd.to_datetime(date)
        start_date = end_date - pd.Timedelta(days=180)
        market_df = self.ds.load_market_index(start=start_date.strftime('%Y-%m-%d'), end=date)

        if len(market_df) < 60:
            return MarketState.CONSOLIDATION, {'error': 'insufficient_data'}

        close = market_df['close']

        # 计算均线
        ma20 = close.rolling(self.config.TREND_MA_SHORT).mean().iloc[-1]
        ma60 = close.rolling(self.config.TREND_MA_LONG).mean().iloc[-1]
        ma120 = close.rolling(self.config.TREND_MA_SUPER).mean().iloc[-1]

        # RSI计算
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.config.RSI_WINDOW).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.config.RSI_WINDOW).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        # 成交额趋势
        amount_ma20 = market_df['amount'].rolling(20).mean().iloc[-1]
        amount_ma60 = market_df['amount'].rolling(60).mean().iloc[-1]

        # 当前价格
        current_price = close.iloc[-1]

        # 判定逻辑
        scores = {
            'price_above_ma60': current_price > ma60,
            'ma60_upward': ma60 > close.rolling(self.config.TREND_MA_LONG).mean().iloc[-5] if len(close) >= 65 else False,
            'price_above_ma120': current_price > ma120,
            'rsi_bullish': rsi > 50,
            'amount_expanding': amount_ma20 > amount_ma60,
        }

        # 状态机
        if scores['price_above_ma60'] and scores['ma60_upward'] and scores['rsi_bullish'] and scores['amount_expanding']:
            state = MarketState.STRONG_BULL
        elif scores['price_above_ma60'] and scores['ma60_upward']:
            state = MarketState.WEAK_BULL
        elif current_price < ma60 and current_price > ma120:
            state = MarketState.CONSOLIDATION
        elif current_price < ma120 and not scores['ma60_upward']:
            state = MarketState.WEAK_BEAR
        elif current_price < ma120 and close.iloc[-1] < ma120 * 0.95:  # 明显跌破
            state = MarketState.STRONG_BEAR
        else:
            state = MarketState.CONSOLIDATION

        detail = {
            'price': current_price,
            'ma20': ma20, 'ma60': ma60, 'ma120': ma120,
            'rsi': rsi,
            'amount_ma20': amount_ma20, 'amount_ma60': amount_ma60,
            'scores': scores,
            'position_limit': self.config.POSITION_LIMIT[state]
        }

        return state, detail

    def detect_style_regime(self, date: str) -> Tuple[StyleRegime, Dict]:
        """
        判断价值/成长风格周期
        使用巨潮价值/成长指数比值（VGRatio）的20日/60日均线
        """
        end_date = pd.to_datetime(date)
        start_date = end_date - pd.Timedelta(days=200)

        # 加载风格指数（实际应接入399371和399372，这里用模拟数据）
        value_df = self.ds.load_market_index(start=start_date.strftime('%Y-%m-%d'), end=date)
        growth_df = self.ds.load_market_index(start=start_date.strftime('%Y-%m-%d'), end=date)

        # 使用模拟数据时，value_df和growth_df可能相同，这里用内部存储的风格指数
        if hasattr(self.ds, 'value_index') and hasattr(self.ds, 'growth_index'):
            value_close = self.ds.value_index.loc[start_date:end_date, 'close']
            growth_close = self.ds.growth_index.loc[start_date:end_date, 'close']
        else:
            # fallback
            return StyleRegime.TRANSITION, {'vg_ratio': 1.0, 'reason': 'no_style_data'}

        if len(value_close) < 60 or len(growth_close) < 60:
            return StyleRegime.TRANSITION, {'vg_ratio': 1.0, 'reason': 'insufficient_data'}

        # 计算VGRatio
        vg_ratio = value_close / growth_close

        ma20 = vg_ratio.rolling(self.config.STYLE_MA_SHORT).mean()
        ma60 = vg_ratio.rolling(self.config.STYLE_MA_LONG).mean()

        # 判断趋势方向（需持续5日确认，避免毛刺）
        ma20_trend = ma20.iloc[-1] > ma20.iloc[-6] if len(ma20) >= 6 else False
        ma60_trend = ma60.iloc[-1] > ma60.iloc[-21] if len(ma60) >= 21 else False

        # 双均线同向确认
        if ma20_trend and ma60_trend:
            regime = StyleRegime.VALUE_DOMINANT
        elif not ma20_trend and not ma60_trend:
            regime = StyleRegime.GROWTH_DOMINANT
        else:
            regime = StyleRegime.TRANSITION

        detail = {
            'vg_ratio_current': vg_ratio.iloc[-1],
            'ma20': ma20.iloc[-1],
            'ma60': ma60.iloc[-1],
            'ma20_trend': ma20_trend,
            'ma60_trend': ma60_trend,
            'regime': regime.value
        }

        return regime, detail

    def get_position_limit(self, market_state: MarketState) -> float:
        """根据市场状态获取仓位上限"""
        return self.config.POSITION_LIMIT.get(market_state, 0.5)
