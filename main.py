"""
A股多因子轮动 × 趋势周期策略 V2.0 - 主运行脚本
使用说明:
1. 默认使用MockDataSource模拟数据演示策略逻辑
2. 接入实盘时，继承DataSource实现真实数据接口
3. 运行: python main.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_layer import MockDataSource
from backtest_engine import BacktestEngine
from config import StrategyConfig


def main():
    print("=" * 60)
    print("A股多因子轮动 × 趋势周期策略 V2.0")
    print("=" * 60)

    # 初始化数据源（模拟数据）
    print("\n[1/4] 初始化数据源...")
    ds = MockDataSource(n_stocks=500, random_seed=42)
    print(f"    ✓ 生成500只股票模拟数据，区间 2020-01-01 ~ 2024-12-31")

    # 初始化回测引擎
    print("\n[2/4] 初始化回测引擎...")
    config = StrategyConfig()
    engine = BacktestEngine(ds, initial_cash=10_000_000)
    print(f"    ✓ 初始资金: 1000万")

    # 执行回测
    print("\n[3/4] 执行回测...")
    records = engine.run(
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        rebalance_freq='2W-FRI'
    )

    # 计算绩效
    print("\n[4/4] 计算绩效指标...")
    perf = engine.calculate_performance(records)

    print("\n" + "=" * 60)
    print("📊 回测绩效报告")
    print("=" * 60)
    print(f"总收益率:       {perf['total_return']:>10.2%}")
    print(f"年化收益率:     {perf['annual_return']:>10.2%}")
    print(f"年化波动率:     {perf['annual_volatility']:>10.2%}")
    print(f"夏普比率:       {perf['sharpe_ratio']:>10.2f}")
    print(f"最大回撤:       {perf['max_drawdown']:>10.2%}")
    print(f"卡玛比率:       {perf['calmar_ratio']:>10.2f}")
    print(f"月度胜率:       {perf['monthly_win_rate']:>10.2%}")
    print(f"盈亏比:         {perf['profit_loss_ratio']:>10.2f}")
    print(f"调仓次数:       {perf['n_rebalances']:>10} 次")
    print(f"期末净值:       {perf['final_nav']:>10,.0f} 元")
    print("=" * 60)

    # 保存结果
    records.to_csv('backtest_records.csv', index=False, encoding='utf-8-sig')
    print("\n✅ 回测完成！结果已保存至 backtest_records.csv")

    return records, perf, engine


if __name__ == '__main__':
    main()
