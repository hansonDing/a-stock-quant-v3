"""
回测引擎：主调度器，串联所有模块，执行向量化回测
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
from config import StrategyConfig, MarketState, StyleRegime
from macro_engine import MacroCycleEngine
from industry_rotator import IndustryRotator
from factor_model import FactorModel
from portfolio_manager import PortfolioManager


class BacktestEngine:
    """策略回测引擎"""

    def __init__(self, data_source, initial_cash: float = 10_000_000):
        self.ds = data_source
        self.config = StrategyConfig()

        # 初始化各模块
        self.macro = MacroCycleEngine(data_source)
        self.industry_rotator = IndustryRotator(data_source)
        self.factor_model = FactorModel(data_source)
        self.portfolio = PortfolioManager(data_source, initial_cash)

        # 记录
        self.records = []
        self.industry_records = []
        self.factor_weight_records = []

    def run(self, start_date: str, end_date: str, rebalance_freq: str = '2W-FRI') -> pd.DataFrame:
        """
        执行回测
        rebalance_freq: '2W-FRI' 表示双周周五
        """
        # 生成调仓日（双周周五）
        all_dates = pd.date_range(start_date, end_date, freq='B')
        # 筛选双周周五
        fridays = all_dates[all_dates.dayofweek == 4]  # 周五
        rebalance_dates = fridays[::2]  # 每两周

        print(f"📅 回测区间: {start_date} ~ {end_date}")
        print(f"🔄 调仓频率: 双周，共 {len(rebalance_dates)} 个调仓日")

        for i, date in enumerate(rebalance_dates):
            date_str = date.strftime('%Y-%m-%d')

            # ===== 步骤1: 宏观周期判断 =====
            market_state, market_detail = self.macro.detect_market_state(date_str)
            style_regime, style_detail = self.macro.detect_style_regime(date_str)
            position_limit = self.macro.get_position_limit(market_state)

            # ===== 步骤2: 行业轮动 =====
            industry_scores = self.industry_rotator.calculate_industry_scores(date_str)
            selected_industries = self.industry_rotator.select_industries(
                industry_scores, market_state, date_str
            )

            # 检查行业止损（从当前持仓中剔除触发止损的行业）
            current_industries = set(p['industry'] for p in self.portfolio.positions.values())
            for ind in current_industries:
                should_stop, reason = self.industry_rotator.check_industry_stop_loss(ind, date_str)
                if should_stop:
                    # 清仓该行业所有股票
                    for code, pos in list(self.portfolio.positions.items()):
                        if pos['industry'] == ind:
                            price = self.portfolio._get_price(code, date_str)
                            shares = pos['shares']
                            proceeds = shares * price * (1 - self.config.COMMISSION_RATE - self.config.STAMP_TAX_RATE)
                            self.portfolio.cash += proceeds
                            del self.portfolio.positions[code]
                            self.portfolio.trade_history.append({
                                'date': date_str, 'code': code, 'action': 'SELL',
                                'shares': shares, 'price': price, 'reason': f'行业止损: {reason}'
                            })

            # ===== 步骤3: 多因子选股 =====
            factor_weights = self.factor_model.get_dynamic_weights(style_regime, date_str)

            stock_selections = {}
            for _, row in selected_industries.iterrows():
                industry = row['industry']
                ind_weight = row['target_weight']

                selected = self.factor_model.select_stocks_in_industry(
                    industry, ind_weight, factor_weights, date_str, market_state
                )
                if not selected.empty:
                    stock_selections[industry] = selected

            # ===== 步骤4: 构建目标组合 =====
            target_portfolio = self.portfolio.build_target_portfolio(
                stock_selections, date_str, position_limit
            )

            # ===== 步骤5: 生成并执行交易 =====
            trades = self.portfolio.generate_trades(target_portfolio, date_str)
            self.portfolio.execute_trades(trades, date_str)

            # ===== 步骤6: 风控检查 =====
            risk_signals = self.portfolio.check_risk(date_str, market_state)
            self.portfolio.handle_risk_signals(risk_signals, date_str)

            # ===== 记录 =====
            nav = self.portfolio.calculate_nav(date_str)
            self.records.append({
                'date': date_str,
                'nav': nav,
                'market_state': market_state.value,
                'style_regime': style_regime.value,
                'position_limit': position_limit,
                'n_industries': len(selected_industries),
                'n_stocks': len(target_portfolio),
                'cash_ratio': self.portfolio.cash / nav if nav > 0 else 1,
                'market_price': market_detail.get('price', 0),
                'market_rsi': market_detail.get('rsi', 0),
                'vg_ratio': style_detail.get('vg_ratio_current', 0)
            })

            self.industry_records.append({
                'date': date_str,
                'selected_industries': selected_industries['industry'].tolist() if not selected_industries.empty else [],
                'industry_weights': selected_industries.set_index('industry')['target_weight'].to_dict() if not selected_industries.empty else {}
            })

            self.factor_weight_records.append({
                'date': date_str,
                'style_regime': style_regime.value,
                'weights': factor_weights
            })

            if (i + 1) % 10 == 0 or i == len(rebalance_dates) - 1:
                print(f"  [{i+1}/{len(rebalance_dates)}] {date_str} | 状态: {market_state.value} | "
                      f"风格: {style_regime.value} | NAV: {nav/1e6:.2f}M | 持仓: {len(self.portfolio.positions)}只")

        return pd.DataFrame(self.records)

    def calculate_performance(self, df: pd.DataFrame) -> Dict:
        """计算绩效指标"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # 收益率
        df['return'] = df['nav'].pct_change()
        df['cum_return'] = (1 + df['return']).cumprod() - 1

        # 年化收益
        total_return = df['nav'].iloc[-1] / df['nav'].iloc[0] - 1
        n_years = (df.index[-1] - df.index[0]).days / 365.25
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # 波动率
        annual_vol = df['return'].std() * np.sqrt(252)

        # 夏普比率（假设无风险利率3%）
        sharpe = (annual_return - 0.03) / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        cummax = df['nav'].cummax()
        drawdown = (df['nav'] - cummax) / cummax
        max_drawdown = drawdown.min()

        # 卡玛比率
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 月度胜率
        monthly = df['return'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
        monthly_win_rate = (monthly > 0).mean()

        # 盈亏比
        wins = df['return'][df['return'] > 0]
        losses = df['return'][df['return'] < 0]
        profit_loss_ratio = abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'monthly_win_rate': monthly_win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'n_rebalances': len(df),
            'final_nav': df['nav'].iloc[-1]
        }
