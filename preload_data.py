#!/usr/bin/env python3
"""
预下载腾讯数据到本地缓存，避免回测时重复请求接口
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tencent_data_source import TencentDataSource
import pandas as pd
from datetime import datetime
import time


def main():
    cache_dir = './data_cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    start = "2020-01-01"
    end = "2024-12-31"
    
    ds = TencentDataSource(delay=0.3)
    
    # 收集所有需要下载的代码
    codes_to_download = []
    
    # 1. 大盘指数
    codes_to_download.append(('sh000001', 'market_index'))
    
    # 2. 行业代理股
    for ind, code in ds._industry_index_codes.items():
        tc = ds._to_tencent_code(code)
        codes_to_download.append((tc, f'industry_{ind}'))
    
    # 3. 所有候选股票
    stock_list = ds.load_stock_list('2024-01-01')
    all_codes = stock_list['code'].tolist()
    for code in all_codes:
        tc = ds._to_tencent_code(code)
        codes_to_download.append((tc, 'stock'))
    
    # 去重
    unique_codes = []
    seen = set()
    for tc, _ in codes_to_download:
        if tc not in seen:
            seen.add(tc)
            unique_codes.append(tc)
    
    total = len(unique_codes)
    print(f"需要下载 {total} 只证券的K线数据")
    print(f"区间: {start} ~ {end}")
    print(f"缓存目录: {cache_dir}")
    print()
    
    success = 0
    failed = 0
    
    for i, tc in enumerate(unique_codes):
        cache_file = os.path.join(cache_dir, f"{tc}_{start}_{end}.csv")
        
        if os.path.exists(cache_file):
            print(f"[{i+1}/{total}] {tc} 已缓存，跳过")
            success += 1
            continue
        
        try:
            df = ds._fetch_multiple_klines(tc, start, end)
            if not df.empty:
                df.to_csv(cache_file)
                print(f"[{i+1}/{total}] {tc} ✓ 下载成功 ({len(df)}条)")
                success += 1
            else:
                print(f"[{i+1}/{total}] {tc} ✗ 无数据")
                failed += 1
        except Exception as e:
            print(f"[{i+1}/{total}] {tc} ✗ 失败: {e}")
            failed += 1
        
        # 避免请求过快
        time.sleep(0.3)
    
    print()
    print(f"预下载完成: 成功 {success} / 失败 {failed} / 总计 {total}")
    print(f"缓存文件保存在 {cache_dir}/")


if __name__ == '__main__':
    main()
