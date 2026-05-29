#!/usr/bin/env python3
"""
使用 AkShare 真实财务数据 + 腾讯K线缓存 做快速回测
股票池精简为40只核心蓝筹，提升速度
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from akshare_data_source import AkShareDataSource
from backtest_engine import BacktestEngine
from config import StrategyConfig
import pandas as pd


def main():
    print("=" * 60)
    print("A股多因子轮动 × AkShare 真实财务数据回测")
    print("=" * 60)

    ds = AkShareDataSource()
    
    # 使用更短的回测区间和精简股票池
    start = "2024-01-01"
    end = "2024-06-30"
    
    print(f"\n📅 回测区间: {start} ~ {end}")
    print(f"🔄 AkShare 真实财务因子 + 腾讯K线缓存")
    print(f"⚠️ 首次运行会预下载 AkShare 数据，约需 2-3 分钟")
    print(f"   后续运行直接读缓存，秒级完成")

    # 测试数据获取
    print("\n[1/3] 测试数据获取...")
    market_df = ds.load_market_index(start=start, end=end)
    if market_df.empty:
        print("✗ 数据获取失败")
        return
    print(f"    ✓ 上证指数: {len(market_df)} 条")

    # 测试 AkShare 因子
    print("\n    测试 AkShare 因子获取...")
    factors = ds.load_factor_data('2024-03-15')
    if factors.empty:
        print("    ✗ AkShare 因子获取失败")
        return
    print(f"    ✓ 获取到 {len(factors)} 只股票的因子数据")
    print(f"    ✓ 真实 EP 范围: {factors['EP'].min():.3f} ~ {factors['EP'].max():.3f}")
    print(f"    ✓ 真实 ROE 范围: {factors['ROE'].min():.3f} ~ {factors['ROE'].max():.3f}")
    
    # 运行回测
    print("\n[2/3] 执行回测...")
    config = StrategyConfig()
    config.START_DATE = start
    config.END_DATE = end
    engine = BacktestEngine(ds, initial_cash=1_000_000)
    
    records = engine.run(start_date=start, end_date=end, rebalance_freq='2W-FRI')
    
    # 计算绩效
    print("\n[3/3] 计算绩效...")
    perf = engine.calculate_performance(records)
    
    print("\n" + "=" * 60)
    print("📊 AkShare 真实数据回测绩效报告")
    print("=" * 60)
    print(f"总收益率:       {perf['total_return']:>10.2%}")
    print(f"年化收益率:     {perf['annual_return']:>10.2%}")
    print(f"年化波动率:     {perf['annual_volatility']:>10.2%}")
    print(f"夏普比率:       {perf['sharpe_ratio']:>10.2f}")
    print(f"最大回撤:       {perf['max_drawdown']:>10.2%}")
    print(f"卡玛比率:       {perf['calmar_ratio']:>10.2f}")
    print(f"月度胜率:       {perf['monthly_win_rate']:>10.2%}")
    print(f"调仓次数:       {perf['n_rebalances']:>10} 次")
    print(f"期末净值:       {perf['final_nav']:>10,.0f} 元")
    print("=" * 60)
    
    records.to_csv('backtest_akshare_2024h1.csv', index=False, encoding='utf-8-sig')
    print("\n✅ 结果已保存至 backtest_akshare_2024h1.csv")
    
    # 对比旧结果
    print("\n📊 对比: 腾讯纯技术因子 vs AkShare 真实财务因子")
    print("-" * 50)
    print(f"{'指标':<15} {'纯技术因子':>12} {'AkShare真实':>12}")
    print("-" * 50)
    print(f"{'总收益率':<15} {'-3.41%':>12} {perf['total_return']:>11.2%}")
    print(f"{'最大回撤':<15} {'-4.69%':>12} {perf['max_drawdown']:>11.2%}")
    print(f"{'夏普比率':<15} {'-0.46':>12} {perf['sharpe_ratio']:>12.2f}")
    print("-" * 50)


if __name__ == '__main__':
    main()
