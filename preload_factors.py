#!/usr/bin/env python3
"""
预计算因子数据，避免回测时重复计算
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tencent_data_source import TencentDataSource
import pandas as pd
import time

def main():
    ds = TencentDataSource(delay=0.05)
    
    start = "2024-01-01"
    end = "2024-06-30"
    
    # 生成调仓日（双周周五）
    all_dates = pd.date_range(start, end, freq='B')
    fridays = all_dates[all_dates.dayofweek == 4]
    rebalance_dates = fridays[::2]
    
    total = len(rebalance_dates)
    print(f"需要预计算 {total} 个调仓日的因子数据")
    print(f"区间: {start} ~ {end}")
    print()
    
    for i, date in enumerate(rebalance_dates):
        date_str = date.strftime('%Y-%m-%d')
        cache_file = os.path.join(ds.cache_dir, f"factors_{date_str}.csv")
        
        if os.path.exists(cache_file):
            print(f"[{i+1}/{total}] {date_str} 已缓存，跳过")
            continue
        
        start_time = time.time()
        try:
            df = ds.load_factor_data(date_str)
            elapsed = time.time() - start_time
            if not df.empty:
                print(f"[{i+1}/{total}] {date_str} ✓ 计算成功 ({len(df)}只, {elapsed:.1f}s)")
            else:
                print(f"[{i+1}/{total}] {date_str} ✗ 无数据 ({elapsed:.1f}s)")
        except Exception as e:
            print(f"[{i+1}/{total}] {date_str} ✗ 失败: {e}")
    
    print("\n因子预计算完成！")

if __name__ == '__main__':
    main()
