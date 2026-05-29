#!/usr/bin/env python3
"""
选股推荐脚本：基于当前最新数据，输出下个交易日的推荐组合
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from akshare_data_source import AkShareDataSource
from backtest_engine import BacktestEngine
from macro_engine import MacroCycleEngine
from industry_rotator import IndustryRotator
from factor_model import FactorModel
from portfolio_manager import PortfolioManager
from config import StrategyConfig, MarketState, StyleRegime


def get_next_trading_date(date_str: str) -> str:
    """获取下一个交易日（简单版：如果是周五则周一，否则明天）"""
    dt = pd.to_datetime(date_str)
    weekday = dt.weekday()
    if weekday == 4:  # 周五
        dt += timedelta(days=3)
    elif weekday == 5:  # 周六
        dt += timedelta(days=2)
    elif weekday == 6:  # 周日
        dt += timedelta(days=1)
    else:
        dt += timedelta(days=1)
    return dt.strftime('%Y-%m-%d')


def get_latest_trading_date(date_str: str) -> str:
    """获取最新交易日（如果今天是周末则回退到周五）"""
    dt = pd.to_datetime(date_str)
    weekday = dt.weekday()
    if weekday == 5:  # 周六
        dt -= timedelta(days=1)
    elif weekday == 6:  # 周日
        dt -= timedelta(days=2)
    return dt.strftime('%Y-%m-%d')


def main():
    # 今天日期（周末则回退到最近交易日）
    today = datetime.today().strftime('%Y-%m-%d')
    latest_date = get_latest_trading_date(today)
    next_date = get_next_trading_date(today)

    print("=" * 70)
    print("📊 A股多因子轮动策略 — 选股推荐")
    print("=" * 70)
    print(f"📅 当前日期:   {today}")
    print(f"📅 最新数据:   {latest_date}")
    print(f"📅 推荐目标:   {next_date}")
    print(f"💡 说明：基于 {latest_date} 收盘数据，生成 {next_date} 调仓方案")
    print("=" * 70)

    # 初始化数据源
    print("\n[1/5] 初始化数据源...")
    ds = AkShareDataSource()
    print(f"    ✓ AkShare + 腾讯混合数据源")

    # 测试数据连通
    print(f"\n    测试市场数据...")
    market_df = ds.load_market_index(start='2026-05-01', end=latest_date)
    if market_df.empty:
        print(f"    ✗ 数据获取失败")
        return
    print(f"    ✓ 上证指数: {len(market_df)} 条, 最新收盘 {market_df.iloc[-1]['close']:.2f}")

    print(f"\n    测试财务因子...")
    factors = ds.load_factor_data(latest_date)
    if factors.empty:
        print(f"    ✗ 因子数据获取失败")
        return
    print(f"    ✓ 获取到 {len(factors)} 只股票的因子数据")

    # 初始化各模块
    print("\n[2/5] 初始化策略模块...")
    config = StrategyConfig()
    macro = MacroCycleEngine(ds)
    industry_rotator = IndustryRotator(ds)
    factor_model = FactorModel(ds)
    portfolio = PortfolioManager(ds, initial_cash=1_000_000)
    print(f"    ✓ 宏观周期引擎")
    print(f"    ✓ 行业轮动引擎")
    print(f"    ✓ 多因子选股引擎")
    print(f"    ✓ 组合管理引擎")

    # ===== Step 1: 宏观判断 =====
    print(f"\n[3/5] 宏观周期研判 ({latest_date})...")
    market_state, market_detail = macro.detect_market_state(latest_date)
    style_regime, style_detail = macro.detect_style_regime(latest_date)
    position_limit = macro.get_position_limit(market_state)

    print(f"    ┌────────────────────────────────────┐")
    print(f"    │  市场状态: {market_state.value:<15} │")
    print(f"    │  风格周期: {style_regime.value:<15} │")
    print(f"    │  建议仓位: {position_limit:<15.0%} │")
    print(f"    └────────────────────────────────────┘")
    print(f"    详情: 收盘 {market_detail.get('price', 0):.2f}, MA20 {market_detail.get('ma20', 0):.2f}, MA60 {market_detail.get('ma60', 0):.2f}, RSI {market_detail.get('rsi', 0):.1f}")
    print(f"    风格: VG比值 {style_detail.get('vg_ratio_current', 0):.3f}, MA20趋势 {'↑' if style_detail.get('ma20_trend') else '↓'}")

    # ===== Step 2: 行业轮动 =====
    print(f"\n[4/5] 行业轮动评分 ({latest_date})...")
    industry_scores = industry_rotator.calculate_industry_scores(latest_date)
    selected_industries = industry_rotator.select_industries(industry_scores, market_state, latest_date)

    print(f"\n    🏆 入选行业 ({len(selected_industries)} 个):")
    print(f"    {'排名':<4} {'行业':<10} {'得分':>8} {'拥挤度':>8} {'权重':>8}")
    print(f"    {'-' * 44}")
    for i, (_, row) in enumerate(selected_industries.iterrows(), 1):
        cong = row.get('congestion_pct', 0)
        flag = "⚠️拥挤" if cong > 0.90 else ""
        print(f"    {i:<4} {row['industry']:<10} {row['total_score']:>8.2f} {cong:>7.1%} {row['target_weight']:>7.1%} {flag}")

    # 检查行业止损
    current_industries = set()
    for code, pos in portfolio.positions.items():
        current_industries.add(pos.get('industry', 'unknown'))
    for ind in current_industries:
        should_stop, reason = industry_rotator.check_industry_stop_loss(ind, latest_date)
        if should_stop:
            print(f"    🚨 行业止损: {ind} — {reason}")

    # ===== Step 3: 多因子选股 =====
    print(f"\n[5/5] 多因子选股与组合构建...")
    factor_weights = factor_model.get_dynamic_weights(style_regime, latest_date)

    print(f"\n    📐 当前因子权重 ({style_regime.value}):")
    for factor, weight in sorted(factor_weights.items(), key=lambda x: -abs(x[1])):
        bar = '█' * int(abs(weight) * 30)
        print(f"    {factor:<20} {weight:>+7.2%}  {bar}")

    stock_selections = {}
    for _, row in selected_industries.iterrows():
        industry = row['industry']
        ind_weight = row['target_weight']

        selected = factor_model.select_stocks_in_industry(
            industry, ind_weight, factor_weights, latest_date, market_state
        )
        if not selected.empty:
            stock_selections[industry] = selected

    # 构建目标组合
    target_portfolio = portfolio.build_target_portfolio(stock_selections, latest_date, position_limit)

    if target_portfolio.empty:
        print(f"\n    ⚠️ 未选中任何股票，建议空仓或降低仓位")
        return

    # 输出推荐
    print(f"\n{'=' * 70}")
    print(f"📋 选股推荐 (共 {len(target_portfolio)} 只，目标日期: {next_date})")
    print(f"{'=' * 70}")
    print(f"\n{'#':<4} {'股票代码':<12} {'行业':<10} {'因子分':>8} {'目标权重':>8}")
    print(f"{'-' * 52}")

    for i, (_, row) in enumerate(target_portfolio.iterrows(), 1):
        code = row.get('code', row.index[0] if hasattr(row.index, '__len__') else row.name)
        if isinstance(code, (int, float)):
            code = str(int(code))
        industry = row.get('industry', 'unknown')
        score = row.get('factor_score', 0)
        weight = row.get('target_weight', 0)
        print(f"    {i:<4} {code:<12} {industry:<10} {score:>8.2f} {weight:>7.1%}")

    print(f"\n{'=' * 70}")
    print(f"📊 组合概要")
    print(f"{'=' * 70}")
    total_weight = target_portfolio['target_weight'].sum()
    cash_ratio = 1 - total_weight
    print(f"    总持仓权重: {total_weight:>7.1%}")
    print(f"    现金比例:   {cash_ratio:>7.1%}")
    print(f"    股票数量:   {len(target_portfolio):>7} 只")
    print(f"    涉及行业:   {target_portfolio['industry'].nunique():>7} 个")

    # 行业分布
    print(f"\n    行业分布:")
    ind_dist = target_portfolio.groupby('industry')['target_weight'].sum().sort_values(ascending=False)
    for ind, w in ind_dist.items():
        bar = '█' * int(w / total_weight * 30)
        print(f"    {ind:<10} {w:>7.1%}  {bar}")

    # 保存结果
    output_file = f'stock_picks_{next_date}.csv'
    target_portfolio.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 推荐已保存至: {output_file}")

    # 风险提示
    print(f"\n{'=' * 70}")
    print(f"⚠️ 风险提示")
    print(f"{'=' * 70}")
    print(f"    1. 本推荐基于历史数据和量化模型，不构成投资建议")
    print(f"    2. 市场状态: {market_state.value}，建议仓位上限 {position_limit:.0%}")
    print(f"    3. 请结合自身风险承受能力，独立判断")
    print(f"    4. 调仓后请关注个股止损（回撤10%）和大盘熔断信号")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
