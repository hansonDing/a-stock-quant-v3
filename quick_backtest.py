#!/usr/bin/env python3
"""
使用腾讯真实数据做快速回测（简化版）
- 缩短区间：2024年1-6月
- 减少股票池：40只核心蓝筹
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tencent_data_source import TencentDataSource
from backtest_engine import BacktestEngine
from config import StrategyConfig
import pandas as pd


def main():
    print("=" * 60)
    print("A股多因子轮动 × 腾讯真实数据快速回测")
    print("=" * 60)

    ds = TencentDataSource()
    
    # 使用更短的回测区间
    start = "2024-01-01"
    end = "2024-06-30"
    
    print(f"\n📅 回测区间: {start} ~ {end}")
    print(f"🔄 数据将通过腾讯免费接口实时获取，请耐心等待...")

    # 测试数据获取
    print("\n[1/3] 测试数据获取...")
    market_df = ds.load_market_index(start=start, end=end)
    if market_df.empty:
        print("✗ 数据获取失败")
        return
    print(f"    ✓ 上证指数: {len(market_df)} 条")
    
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
    print("📊 真实数据回测绩效报告")
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
    
    records.to_csv('backtest_tencent_2024h1.csv', index=False, encoding='utf-8-sig')
    print("\n✅ 结果已保存至 backtest_tencent_2024h1.csv")


if __name__ == '__main__':
    main()
