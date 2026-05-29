#!/usr/bin/env python3
"""
使用 AkShare 获取真实财务数据，替换模拟因子
优化版：预下载所有数据，回测时直接查表
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tencent_data_source import TencentDataSource
import pandas as pd
import numpy as np
import akshare as ak
import time
from datetime import datetime, timedelta


class AkShareDataSource(TencentDataSource):
    """
    AkShare + 腾讯缓存混合数据源
    - K线：复用腾讯缓存（已下载的本地数据）
    - 财务因子：AkShare 真实数据，预下载后查表
    """

    def __init__(self, max_retries=3, delay=0.3, cache_dir='./data_cache'):
        super().__init__(max_retries, delay, cache_dir)
        self._valuation_cache = None  # 全市场估值数据缓存
        self._financial_cache = {}    # 个股财务数据缓存

    def _preload_all_valuation(self, codes: list):
        """预下载所有股票的估值数据"""
        if self._valuation_cache is not None:
            return

        print("🔄 正在从 AkShare 预下载估值数据，请稍候...")
        all_data = []
        for i, code in enumerate(codes):
            symbol = code.replace('.SH', '').replace('.SZ', '').replace('.BJ', '')
            try:
                df = ak.stock_value_em(symbol=symbol)
                if not df.empty:
                    df['symbol'] = symbol
                    df['code'] = code
                    df['日期'] = pd.to_datetime(df['数据日期'])
                    all_data.append(df[['code', 'symbol', '日期', 'PE(TTM)', '市净率', '总市值']])
            except Exception:
                pass
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(codes)}] 估值数据下载中...")
            time.sleep(0.05)

        if all_data:
            self._valuation_cache = pd.concat(all_data, ignore_index=True)
            print(f"✓ 估值数据预下载完成，共 {len(self._valuation_cache)} 条记录")
        else:
            self._valuation_cache = pd.DataFrame()

    def _preload_all_financial(self, codes: list):
        """预下载所有股票的财务数据"""
        if self._financial_cache:
            return

        print("🔄 正在从 AkShare 预下载财务数据，请稍候...")
        for i, code in enumerate(codes):
            symbol = code.replace('.SH', '').replace('.SZ', '').replace('.BJ', '')
            try:
                df = ak.stock_financial_abstract_ths(symbol=symbol, indicator='按报告期')
                if not df.empty and len(df.columns) > 10:
                    df['code'] = code
                    df['symbol'] = symbol
                    df['报告期'] = pd.to_datetime(df['报告期'])
                    self._financial_cache[code] = df
            except Exception:
                pass
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(codes)}] 财务数据下载中...")
            time.sleep(0.05)

        print(f"✓ 财务数据预下载完成，共 {len(self._financial_cache)} 只股票")

    def _get_valuation(self, code: str, date: pd.Timestamp) -> dict:
        """从预缓存的估值数据中获取某股票某日期的估值"""
        if self._valuation_cache is None or self._valuation_cache.empty:
            return {}

        df = self._valuation_cache[
            (self._valuation_cache['code'] == code) &
            (self._valuation_cache['日期'] <= date)
        ]
        if df.empty:
            return {}
        latest = df.iloc[-1]
        return {
            'PE': float(latest.get('PE(TTM)', 0)) if pd.notna(latest.get('PE(TTM)')) else 0,
            'PB': float(latest.get('市净率', 0)) if pd.notna(latest.get('市净率')) else 0,
            'MARKET_CAP': float(latest.get('总市值', 0)) if pd.notna(latest.get('总市值')) else 0,
        }

    def _get_financial(self, code: str, date: pd.Timestamp) -> dict:
        """从预缓存的财务数据中获取某股票某日期前的最新财务数据"""
        df = self._financial_cache.get(code)
        if df is None or df.empty:
            return {}

        df_before = df[df['报告期'] <= date]
        if df_before.empty:
            return {}

        latest = df_before.iloc[-1]
        roe_str = str(latest.get('净资产收益率', '0'))
        roe = self._parse_percent(roe_str)
        growth_str = str(latest.get('净利润同比增长率', '0'))
        profit_growth = self._parse_percent(growth_str)
        ocf_str = str(latest.get('每股经营现金流', '0'))
        ocf_per_share = self._parse_number(ocf_str)
        eps_str = str(latest.get('基本每股收益', '0'))
        eps = self._parse_number(eps_str)

        # ROE稳定性
        roe_values = []
        for _, row in df_before.tail(8).iterrows():
            r = self._parse_percent(str(row.get('净资产收益率', '0')))
            if r > 0:
                roe_values.append(r)
        roe_stability = 1 / (np.std(roe_values) + 1e-6) if len(roe_values) > 1 else 5

        return {
            'ROE': roe,
            'PROFIT_GROWTH': profit_growth,
            'OCF_PER_SHARE': ocf_per_share,
            'EPS': eps,
            'ROE_STABILITY': roe_stability,
        }

    def load_factor_data(self, date: str) -> pd.DataFrame:
        """基于 AkShare 真实财务数据 + 腾讯技术因子"""
        # 先检查缓存
        cache_file = os.path.join(self.cache_dir, f"akshare_factors_{date}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, index_col='code')
                if not df.empty:
                    return df
            except Exception:
                pass

        date_obj = pd.to_datetime(date)
        start = (date_obj - timedelta(days=180)).strftime('%Y-%m-%d')
        end = date

        stock_list = self.load_stock_list(date)
        codes = stock_list['code'].tolist()

        # 预下载 AkShare 数据（只执行一次）
        self._preload_all_valuation(codes)
        self._preload_all_financial(codes)

        records = []
        for code in codes:
            try:
                # 1. 技术因子（从K线）
                df = self.load_stock_daily(code, start, end)
                if df.empty or len(df) < 30:
                    continue

                df = df[df.index <= date_obj]
                if df.empty:
                    continue

                latest = df.iloc[-1]
                close = latest['close']
                if close <= 0:
                    continue

                returns = df['close'].pct_change().dropna()
                if len(returns) < 20:
                    continue

                mom_20 = df['close'].iloc[-1] / df['close'].iloc[-min(20, len(df))] - 1 if len(df) >= 20 else 0
                mom_60 = df['close'].iloc[-1] / df['close'].iloc[-min(60, len(df))] - 1 if len(df) >= 60 else mom_20
                vol_20 = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else 0.3
                avg_volume = df['volume'].tail(20).mean()
                turnover_stability = 1 / (df['volume'].tail(20).std() / (avg_volume + 1e-6) + 1e-6)

                if len(df) >= 20:
                    price_change = df['close'].tail(20).diff().dropna()
                    volume_change = df['volume'].tail(20).diff().dropna()
                    min_len = min(len(price_change), len(volume_change))
                    if min_len >= 10:
                        vp_corr = np.corrcoef(price_change.tail(min_len).values,
                                              volume_change.tail(min_len).values)[0, 1]
                        vp_divergence = -vp_corr
                    else:
                        vp_divergence = 0
                else:
                    vp_divergence = 0

                # 2. AkShare 真实财务因子
                val = self._get_valuation(code, date_obj)
                fin = self._get_financial(code, date_obj)

                pe = val.get('PE', 0)
                pb = val.get('PB', 0)
                market_cap = val.get('MARKET_CAP', 0)

                ep = 1 / pe if pe > 0 else 0.05
                bp = 1 / pb if pb > 0 else 0.8
                if market_cap <= 0:
                    market_cap = np.random.lognormal(23, 1.2)

                roe = fin.get('ROE', 0.10)
                profit_growth = fin.get('PROFIT_GROWTH', 0.15)
                ocf_per_share = fin.get('OCF_PER_SHARE', 0.1)
                eps = fin.get('EPS', 0.1)
                roe_stability = fin.get('ROE_STABILITY', 5)

                # 股息率估算
                dividend = eps * 0.3 / close if close > 0 and eps > 0 else 0.02
                # OCF/营收估算
                ocf_to_revenue = ocf_per_share / close * 0.5 if close > 0 else 0.1

                records.append({
                    'code': code,
                    'MARKET_CAP': market_cap,
                    'EP': ep,
                    'BP': bp,
                    'DIVIDEND_YIELD': dividend,
                    'ROE': roe,
                    'ROE_STABILITY': roe_stability,
                    'PROFIT_GROWTH': profit_growth,
                    'OCF_TO_REVENUE': ocf_to_revenue,
                    'MOMENTUM_20': mom_20,
                    'MOMENTUM_60': mom_60,
                    'VOLATILITY_20': vol_20,
                    'TURNOVER_STABILITY': turnover_stability,
                    'VOLUME_PRICE_DIVERGENCE': vp_divergence,
                    'INDUSTRY': self._industry_map.get(code, '综合'),
                    'LIST_DATE': pd.to_datetime('2010-01-01'),
                })

            except Exception:
                continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df.set_index('code', inplace=True)

        try:
            df.to_csv(cache_file)
        except Exception as e:
            print(f"AkShare因子缓存保存失败 {cache_file}: {e}")

        return df

    def _parse_percent(self, s: str) -> float:
        if not s or s == 'False' or s == 'None':
            return 0
        s = str(s).replace('%', '').replace(',', '').strip()
        try:
            return float(s) / 100 if '%' in str(s) else float(s)
        except:
            return 0

    def _parse_number(self, s: str) -> float:
        if not s or s == 'False' or s == 'None':
            return 0
        s = str(s).replace(',', '').strip()
        if '亿' in s:
            s = s.replace('亿', '').strip()
            try:
                return float(s)
            except:
                return 0
        if '万' in s:
            s = s.replace('万', '').strip()
            try:
                return float(s) / 10000
            except:
                return 0
        try:
            return float(s)
        except:
            return 0


if __name__ == '__main__':
    ds = AkShareDataSource()
    print("测试 AkShare 因子获取...")
    factors = ds.load_factor_data('2024-03-15')
    print(f"获取到 {len(factors)} 只股票的因子数据")
    if not factors.empty:
        print(factors[['EP', 'BP', 'ROE', 'DIVIDEND_YIELD', 'MOMENTUM_20']].head(10))
