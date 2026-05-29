"""
组合构建与风控：权重优化、三层熔断、策略失效检测
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from config import StrategyConfig, MarketState


class PortfolioManager:
    """组合管理与风控引擎"""

    def __init__(self, data_source, initial_cash: float = 10_000_000):
        self.ds = data_source
        self.config = StrategyConfig()
        self.initial_cash = initial_cash
        self.cash = initial_cash

        # 持仓状态
        self.positions = {}  # {code: {'shares': int, 'cost': float, 'max_price': float, 'industry': str}}
        self.industry_exposure = {}  # {industry: total_weight}

        # 历史记录
        self.nav_history = []
        self.trade_history = []

    def calculate_volatility_adjusted_weights(self, 
                                               selected_stocks: pd.DataFrame,
                                               date: str) -> pd.Series:
        """
        波动率缩放：风险平价思想
        高波动股票降低权重，低波动股票提高权重
        """
        if selected_stocks.empty:
            return pd.Series()

        base_weights = selected_stocks['base_weight'].copy()

        # 获取个股20日波动率（来自因子数据）
        vols = selected_stocks['VOLATILITY_20'].replace(0, 0.3)  # 防止除零

        # 波动率倒数加权
        avg_vol = vols.mean()
        vol_scalar = (avg_vol / vols) ** 0.5

        adjusted = base_weights * vol_scalar
        adjusted = adjusted / adjusted.sum()  # 归一化

        # 应用个股权重上限
        adjusted = adjusted.clip(upper=self.config.STOCK_MAX_WEIGHT)
        adjusted = adjusted / adjusted.sum()  # 再次归一化

        return adjusted

    def apply_constraints(self, 
                          weights: pd.Series, 
                          industries: pd.Series) -> pd.Series:
        """
        应用组合约束：行业上限、个股权重上限
        """
        weights = weights.copy()

        # 迭代调整行业权重
        for _ in range(10):
            industry_sums = weights.groupby(industries).sum()

            for ind, ind_sum in industry_sums.items():
                if ind_sum > self.config.INDUSTRY_MAX_WEIGHT:
                    ind_stocks = weights[industries == ind].index
                    excess = ind_sum - self.config.INDUSTRY_MAX_WEIGHT
                    weights.loc[ind_stocks] *= (self.config.INDUSTRY_MAX_WEIGHT / ind_sum)

                    # 分配多余权重给其他行业
                    other_stocks = weights[industries != ind].index
                    if len(other_stocks) > 0:
                        weights.loc[other_stocks] += excess / len(other_stocks)

            # 个股权重上限
            weights = weights.clip(upper=self.config.STOCK_MAX_WEIGHT)
            weights = weights / weights.sum()

        return weights

    def build_target_portfolio(self, 
                              stock_selections: Dict[str, pd.DataFrame],
                              date: str,
                              position_limit: float) -> pd.DataFrame:
        """
        构建目标组合
        输入: {industry: selected_stocks_df}
        输出: target_portfolio DataFrame [code, target_weight, industry]
        """
        all_targets = []

        for industry, stocks in stock_selections.items():
            if stocks.empty:
                continue

            # 波动率调整权重
            vol_weights = self.calculate_volatility_adjusted_weights(stocks, date)
            stocks['vol_weight'] = vol_weights

            # 应用约束
            industries = pd.Series(industry, index=stocks.index)
            final_weights = self.apply_constraints(stocks['vol_weight'], industries)

            stocks['target_weight'] = final_weights * position_limit  # 应用总仓位限制
            stocks['industry'] = industry

            all_targets.append(stocks[['target_weight', 'industry', 'factor_score']])

        if not all_targets:
            return pd.DataFrame()

        portfolio = pd.concat(all_targets)

        # 确保总权重不超过position_limit
        if portfolio['target_weight'].sum() > position_limit:
            portfolio['target_weight'] = portfolio['target_weight'] / portfolio['target_weight'].sum() * position_limit

        # 保留最小现金比例
        total_weight = portfolio['target_weight'].sum()
        if total_weight > (1 - self.config.MIN_CASH_RATIO):
            scale = (1 - self.config.MIN_CASH_RATIO) / total_weight
            portfolio['target_weight'] *= scale

        # 重置索引并确保列名为 code
        portfolio = portfolio.reset_index()
        if 'index' in portfolio.columns:
            portfolio = portfolio.rename(columns={'index': 'code'})
        elif 'level_0' in portfolio.columns:
            portfolio = portfolio.rename(columns={'level_0': 'code'})
        else:
            portfolio = portfolio.rename(columns={portfolio.columns[0]: 'code'})

        return portfolio[['code', 'target_weight', 'industry', 'factor_score']]

    def generate_trades(self, target_portfolio: pd.DataFrame, date: str) -> List[Dict]:
        """
        生成交易指令（对比当前持仓与目标持仓）
        返回: [{'code': str, 'action': 'BUY/SELL', 'target_weight': float, 'reason': str}, ...]
        """
        trades = []

        # 当前持仓市值
        current_nav = self.calculate_nav(date)
        current_weights = {}
        for code, pos in self.positions.items():
            price = self._get_price(code, date)
            if price > 0:
                current_weights[code] = (pos['shares'] * price) / current_nav if current_nav > 0 else 0

        # 目标权重
        target_weights = {}
        if not target_portfolio.empty:
            # 兼容不同列名
            code_col = 'code' if 'code' in target_portfolio.columns else target_portfolio.columns[0]
            weight_col = 'target_weight' if 'target_weight' in target_portfolio.columns else 'base_weight'
            target_weights = dict(zip(target_portfolio[code_col], target_portfolio[weight_col]))

        all_codes = set(current_weights.keys()) | set(target_weights.keys())

        for code in all_codes:
            tw = target_weights.get(code, 0)
            cw = current_weights.get(code, 0)

            if tw > cw + 0.005:  # 买入阈值
                trades.append({
                    'code': code,
                    'action': 'BUY',
                    'target_weight': tw,
                    'current_weight': cw,
                    'delta': tw - cw,
                    'reason': 'rebalance'
                })
            elif cw > tw + 0.005 and tw == 0:  # 清仓
                trades.append({
                    'code': code,
                    'action': 'SELL',
                    'target_weight': 0,
                    'current_weight': cw,
                    'delta': cw,
                    'reason': 'rebalance_or_stop'
                })

        return trades

    def execute_trades(self, trades: List[Dict], date: str):
        """执行交易（简化版，不考虑市场冲击）"""
        nav = self.calculate_nav(date)

        for trade in trades:
            code = trade['code']
            action = trade['action']
            price = self._get_price(code, date)

            if price is None or price <= 0:
                continue

            if action == 'BUY':
                target_value = trade['target_weight'] * nav
                shares = int(target_value / price)
                cost = shares * price * (1 + self.config.COMMISSION_RATE)

                if cost <= self.cash and shares > 0:
                    self.cash -= cost
                    if code in self.positions:
                        old = self.positions[code]
                        total_cost = old['cost'] * old['shares'] + cost
                        total_shares = old['shares'] + shares
                        self.positions[code] = {
                            'shares': total_shares,
                            'cost': total_cost / total_shares,
                            'max_price': price,
                            'industry': old['industry']
                        }
                    else:
                        self.positions[code] = {
                            'shares': shares,
                            'cost': price,
                            'max_price': price,
                            'industry': trade.get('industry', 'unknown')
                        }

                    self.trade_history.append({
                        'date': date, 'code': code, 'action': 'BUY',
                        'shares': shares, 'price': price, 'cost': cost
                    })

            elif action == 'SELL':
                if code in self.positions:
                    shares = self.positions[code]['shares']
                    proceeds = shares * price * (1 - self.config.COMMISSION_RATE - self.config.STAMP_TAX_RATE)
                    self.cash += proceeds
                    del self.positions[code]

                    self.trade_history.append({
                        'date': date, 'code': code, 'action': 'SELL',
                        'shares': shares, 'price': price, 'proceeds': proceeds
                    })

    def check_risk(self, date: str, market_state: MarketState) -> List[Dict]:
        """
        三层风控检查
        返回风险信号列表
        """
        signals = []
        nav = self.calculate_nav(date)

        # 第一层：个股止损（最高价回撤10%）
        for code, pos in list(self.positions.items()):
            price = self._get_price(code, date)
            if price > pos['max_price']:
                pos['max_price'] = price

            if pos['max_price'] > 0:
                drawdown = (pos['max_price'] - price) / pos['max_price']
                if drawdown >= self.config.STOCK_STOP_LOSS:
                    signals.append({
                        'level': 'STOCK',
                        'code': code,
                        'action': 'SELL',
                        'reason': f'个股止损: 回撤{drawdown:.1%}'
                    })

        # 第二层：行业止损（由IndustryRotator外部调用，这里仅做个股层面）

        # 第三层：大盘熔断
        if market_state == MarketState.STRONG_BEAR:
            current_position = 1 - (self.cash / nav)
            if current_position > self.config.POSITION_LIMIT[MarketState.STRONG_BEAR]:
                signals.append({
                    'level': 'MACRO',
                    'action': 'REDUCE',
                    'target_position': self.config.POSITION_LIMIT[MarketState.STRONG_BEAR],
                    'reason': '大盘强熊市，强制降仓'
                })

        return signals

    def handle_risk_signals(self, signals: List[Dict], date: str):
        """处理风险信号"""
        for sig in signals:
            if sig['action'] == 'SELL' and 'code' in sig:
                code = sig['code']
                if code in self.positions:
                    price = self._get_price(code, date)
                    shares = self.positions[code]['shares']
                    proceeds = shares * price * (1 - self.config.COMMISSION_RATE - self.config.STAMP_TAX_RATE)
                    self.cash += proceeds
                    del self.positions[code]
                    self.trade_history.append({
                        'date': date, 'code': code, 'action': 'SELL',
                        'shares': shares, 'price': price, 'reason': sig['reason']
                    })

            elif sig['action'] == 'REDUCE':
                # 等比例减仓
                target_pos = sig.get('target_position', 0.3)
                current_nav = self.calculate_nav(date)
                target_value = current_nav * target_pos
                current_value = current_nav - self.cash

                if current_value > target_value:
                    reduce_ratio = (current_value - target_value) / current_value
                    for code in list(self.positions.keys()):
                        pos = self.positions[code]
                        reduce_shares = int(pos['shares'] * reduce_ratio)
                        if reduce_shares > 0:
                            price = self._get_price(code, date)
                            proceeds = reduce_shares * price * (1 - self.config.COMMISSION_RATE - self.config.STAMP_TAX_RATE)
                            self.cash += proceeds
                            pos['shares'] -= reduce_shares
                            if pos['shares'] <= 0:
                                del self.positions[code]

    def calculate_nav(self, date: str) -> float:
        """计算当前净值"""
        market_value = 0
        for code, pos in self.positions.items():
            price = self._get_price(code, date)
            market_value += pos['shares'] * price
        return self.cash + market_value

    def _get_price(self, code: str, date: str) -> float:
        """获取某股票某日期收盘价"""
        try:
            df = self.ds.load_stock_daily(code, end=date)
            if df.empty or 'close' not in df.columns:
                return 0.0
            val = df['close'].iloc[-1]
            return float(val) if val is not None and not pd.isna(val) else 0.0
        except Exception:
            return 0.0

    def get_state(self, date: str) -> Dict:
        """获取当前组合状态"""
        nav = self.calculate_nav(date)
        return {
            'date': date,
            'nav': nav,
            'cash': self.cash,
            'cash_ratio': self.cash / nav if nav > 0 else 1,
            'n_positions': len(self.positions),
            'positions': self.positions.copy()
        }
