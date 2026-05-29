#!/usr/bin/env python3
"""
使用腾讯真实A股数据运行多因子轮动策略回测
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tencent_data_source import TencentDataSource
from backtest_engine import BacktestEngine
from config import StrategyConfig


def main():
    print("=" * 60)
    print("A股多因子轮动 × 腾讯真实数据回测")
    print("=" * 60)

    # 初始化腾讯数据源
    print("\n[1/4] 初始化腾讯数据源...")
    ds = TencentDataSource()
    print("    ✓ 腾讯接口已就绪 (免费，无需Token)")

    # 测试获取市场数据
    print("\n    测试获取上证指数数据...")
    market_df = ds.load_market_index(start='2022-01-01', end='2024-12-31')
    if market_df.empty:
        print("    ✗ 获取数据失败，请检查网络连接")
        return
    print(f"    ✓ 获取到 {len(market_df)} 条上证指数数据")
    print(f"    ✓ 数据区间: {market_df.index[0].date()} ~ {market_df.index[-1].date()}")

    # 测试获取个股数据
    print("\n    测试获取个股数据...")
    stock_df = ds.load_stock_daily('600519.SH', start='2022-01-01', end='2024-12-31')
    if not stock_df.empty:
        print(f"    ✓ 获取到贵州茅台 {len(stock_df)} 条数据")

    # 初始化回测引擎
    print("\n[2/4] 初始化回测引擎...")
    config = StrategyConfig()
    # 调整日期为真实数据区间
    config.START_DATE = "2022-01-01"
    config.END_DATE = "2024-12-31"
    engine = BacktestEngine(ds, initial_cash=1_000_000)
    print(f"    ✓ 初始资金: 100万")
    print(f"    ✓ 回测区间: {config.START_DATE} ~ {config.END_DATE}")
    print(f"    ⚠️ 使用纯技术因子替代财务因子（腾讯接口无财务数据）")

    # 执行回测
    print("\n[3/4] 执行回测...")
    print("    注意：真实数据回测耗时较长，请耐心等待...")
    records = engine.run(
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        rebalance_freq='2W-FRI'
    )

    # 计算绩效
    print("\n[4/4] 计算绩效指标...")
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
    print(f"盈亏比:         {perf['profit_loss_ratio']:>10.2f}")
    print(f"调仓次数:       {perf['n_rebalances']:>10} 次")
    print(f"期末净值:       {perf['final_nav']:>10,.0f} 元")
    print("=" * 60)

    # 保存结果
    records.to_csv('backtest_real_data.csv', index=False, encoding='utf-8-sig')
    print("\n✅ 回测完成！结果已保存至 backtest_real_data.csv")

    # 尝试生成可视化
    try:
        from visualization import StrategyVisualizer
        viz = StrategyVisualizer(records)
        viz.plot_all(save_dir='./real_results/')
        print("📊 可视化图表已保存至 ./real_results/")
    except Exception as e:
        print(f"⚠️ 可视化生成失败: {e}")

    return records, perf, engine


if __name__ == '__main__':
    main()
