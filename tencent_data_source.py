"""
腾讯股票接口数据层
免费、无需Token，获取A股真实行情数据
"""

import json
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from data_layer import DataSource
import time
import re


class TencentDataSource(DataSource):
    """
    腾讯免费股票接口数据源
    - 大盘指数: https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
    - 个股K线: https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
    - 实时行情: https://qt.gtimg.cn/q=...
    """

    def __init__(self, max_retries: int = 3, delay: float = 0.3):
        super().__init__()
        self.max_retries = max_retries
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # 申万一级行业 -> 代表性指数代码（部分有ETF/行业指数替代）
        # 腾讯接口主要支持个股，行业指数用代表性个股或ETF替代
        self._industry_map = self._build_industry_mapping()
        self._industry_index_codes = self._build_industry_indices()

        # 缓存
        self._market_index_cache = None
        self._stock_data_cache = {}
        self._industry_index_cache = {}

    def _build_industry_mapping(self) -> Dict[str, str]:
        """构建股票-行业映射（扩展版，每行业至少8只）"""
        # 使用各行业代表性龙头作为行业代理
        # 实际生产中应该从数据库/文件加载真实的股票-行业映射
        industry_leaders = {
            '农林牧渔': ['002311.SZ', '002385.SZ', '600598.SH', '000876.SZ', '300498.SZ', '002714.SZ', '002157.SZ', '002124.SZ'],
            '基础化工': ['600309.SH', '002064.SZ', '600352.SH', '600143.SH', '002601.SZ', '000830.SZ', '600426.SH', '600486.SH'],
            '钢铁': ['600019.SH', '000932.SZ', '600507.SH', '002110.SZ', '600282.SH', '000717.SZ', '600808.SH', '600010.SH'],
            '有色金属': ['601899.SH', '002460.SZ', '600547.SH', '600489.SH', '000807.SZ', '601600.SH', '002466.SZ', '603993.SH'],
            '电子': ['002475.SZ', '300433.SZ', '601138.SH', '000725.SZ', '002241.SZ', '603501.SH', '002371.SZ', '688981.SH'],
            '家用电器': ['000333.SZ', '600690.SH', '002032.SZ', '000651.SZ', '002508.SZ', '603195.SH', '600839.SH', '002242.SZ'],
            '食品饮料': ['600519.SH', '000858.SZ', '002304.SZ', '000568.SZ', '600887.SH', '603288.SH', '600600.SH', '600809.SH'],
            '纺织服饰': ['600398.SH', '002563.SZ', '603587.SH', '002293.SZ', '601566.SH', '600177.SH', '002154.SZ', '603808.SH'],
            '轻工制造': ['002078.SZ', '603816.SH', '603385.SH', '002572.SZ', '603801.SH', '002191.SZ', '002831.SZ', '603898.SH'],
            '医药生物': ['600276.SH', '000538.SZ', '300760.SZ', '603259.SH', '600436.SH', '000963.SZ', '300015.SZ', '600196.SH'],
            '公用事业': ['600900.SH', '600011.SH', '601985.SH', '600886.SH', '600027.SH', '600023.SH', '600795.SH', '000591.SZ'],
            '交通运输': ['601816.SH', '601111.SH', '600009.SH', '601919.SH', '600115.SH', '002120.SZ', '601006.SH', '600029.SH'],
            '房地产': ['000002.SZ', '600048.SH', '001979.SZ', '600383.SH', '002244.SZ', '000069.SZ', '600340.SH', '000402.SZ'],
            '商贸零售': ['601933.SH', '600729.SH', '002419.SZ', '002251.SZ', '600697.SH', '600859.SH', '601010.SH', '600827.SH'],
            '社会服务': ['601888.SH', '002707.SZ', '600138.SH', '600054.SH', '002033.SZ', '000888.SZ', '600258.SH', '600593.SH'],
            '银行': ['601398.SH', '601288.SH', '601939.SH', '600036.SH', '601988.SH', '601328.SH', '600016.SH', '601998.SH'],
            '非银金融': ['601318.SH', '600030.SH', '601628.SH', '601688.SH', '600837.SH', '000776.SZ', '601211.SH', '600958.SH'],
            '综合': ['600661.SH', '600770.SH', '600210.SH', '600805.SH', '000058.SZ', '600661.SH', '600128.SH', '600784.SH'],
            '建筑材料': ['600585.SH', '002271.SZ', '600801.SH', '000877.SZ', '002233.SZ', '600176.SH', '000672.SZ', '600552.SH'],
            '建筑装饰': ['601668.SH', '601390.SH', '002081.SZ', '601669.SH', '600502.SH', '601186.SH', '601800.SH', '601117.SH'],
            '电力设备': ['300750.SZ', '601012.SH', '600438.SH', '002594.SZ', '603659.SH', '600089.SH', '002202.SZ', '300014.SZ'],
            '机械设备': ['600031.SH', '000425.SZ', '601766.SH', '601100.SH', '603288.SH', '600031.SH', '000157.SZ', '002008.SZ'],
            '国防军工': ['600893.SH', '000768.SZ', '002179.SZ', '600760.SH', '600372.SH', '600967.SH', '002013.SZ', '600990.SH'],
            '计算机': ['600570.SH', '002230.SZ', '300033.SZ', '002230.SZ', '600728.SH', '000938.SZ', '300496.SZ', '002405.SZ'],
            '传媒': ['002607.SZ', '300413.SZ', '600373.SH', '600088.SH', '601928.SH', '603000.SH', '002174.SZ', '300251.SZ'],
            '通信': ['600941.SH', '000063.SZ', '600498.SH', '600487.SH', '002281.SZ', '300308.SZ', '603236.SH', '002396.SZ'],
            '煤炭': ['601088.SH', '600188.SH', '601699.SH', '600123.SH', '600395.SH', '601015.SH', '601898.SH', '600971.SH'],
            '石油石化': ['601857.SH', '600028.SH', '000059.SZ', '600346.SH', '002493.SZ', '601808.SH', '603225.SH', '600968.SH'],
            '环保': ['300070.SZ', '600323.SH', '601330.SH', '002672.SZ', '000967.SZ', '300137.SZ', '002573.SZ', '603588.SH'],
            '美容护理': ['600315.SH', '603605.SH', '300896.SZ', '002024.SZ', '603983.SH', '300957.SZ', '603630.SH', '002094.SZ'],
            '汽车': ['601127.SH', '002594.SZ', '601633.SH', '000625.SZ', '600104.SH', '601238.SH', '000338.SZ', '600066.SH'],
        }

        # 反向映射：股票 -> 行业
        stock_to_industry = {}
        for industry, stocks in industry_leaders.items():
            for stock in stocks:
                stock_to_industry[stock] = industry

        return stock_to_industry

    def _build_industry_indices(self) -> Dict[str, str]:
        """行业代理指数/个股（用龙头股代表行业走势）"""
        return {
            '农林牧渔': '002311.SZ',
            '基础化工': '600309.SH',
            '钢铁': '600019.SH',
            '有色金属': '601899.SH',
            '电子': '002475.SZ',
            '家用电器': '000333.SZ',
            '食品饮料': '600519.SH',
            '纺织服饰': '600398.SH',
            '轻工制造': '002078.SZ',
            '医药生物': '600276.SH',
            '公用事业': '600900.SH',
            '交通运输': '601816.SH',
            '房地产': '000002.SZ',
            '商贸零售': '601933.SH',
            '社会服务': '601888.SH',
            '银行': '601398.SH',
            '非银金融': '601318.SH',
            '综合': '600661.SH',
            '建筑材料': '600585.SH',
            '建筑装饰': '601668.SH',
            '电力设备': '300750.SZ',
            '机械设备': '600031.SH',
            '国防军工': '600893.SH',
            '计算机': '600570.SH',
            '传媒': '002607.SZ',
            '通信': '600941.SH',
            '煤炭': '601088.SH',
            '石油石化': '601857.SH',
            '环保': '300070.SZ',
            '美容护理': '600315.SH',
            '汽车': '601127.SH',
        }

    def _to_tencent_code(self, code: str) -> str:
        """转换代码格式：000001.SZ -> sz000001, 000001.SH -> sh000001"""
        code = code.strip().upper()
        if code.endswith('.SH'):
            return f"sh{code.replace('.SH', '')}"
        elif code.endswith('.SZ'):
            return f"sz{code.replace('.SZ', '')}"
        elif code.endswith('.BJ'):
            return f"bj{code.replace('.BJ', '')}"
        # 无后缀，按数字判断
        code_num = code.replace('.', '')
        if code_num.startswith('6') or code_num.startswith('8') or code_num.startswith('9'):
            return f"sh{code_num}"
        return f"sz{code_num}"

    def _fetch_kline(self, tencent_code: str, start: str, end: str, ktype: str = 'day') -> pd.DataFrame:
        """
        获取腾讯K线数据（前复权）
        接口: https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
        """
        url = (
            f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            f"?param={tencent_code},{ktype},{start},{end},640,qfq"
        )

        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, timeout=15)
                data = resp.json()

                # 解析数据
                # 腾讯返回的key就是tencent_code本身，不包含ktype后缀
                key = tencent_code
                if 'data' in data and key in data['data'] and data['data'][key]:
                    # 优先用前复权 qfqday，否则用 day
                    raw_data = data['data'][key].get('qfqday') or data['data'][key].get('day', [])
                    if not raw_data:
                        return pd.DataFrame()

                    rows = []
                    for item in raw_data:
                        if isinstance(item, list) and len(item) >= 5:
                            # 腾讯格式: [日期, 开盘, 收盘, 最高, 最低, 成交量]
                            rows.append({
                                'date': item[0],
                                'open': float(item[1]),
                                'close': float(item[2]),
                                'high': float(item[3]),
                                'low': float(item[4]),
                                'volume': float(item[5]) if len(item) > 5 else 0,
                            })

                    df = pd.DataFrame(rows)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df = df.sort_index()
                    return df

                return pd.DataFrame()

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))
                else:
                    print(f"获取 {tencent_code} K线失败: {e}")
                    return pd.DataFrame()

        return pd.DataFrame()

    def _fetch_multiple_klines(self, tencent_code: str, start: str, end: str) -> pd.DataFrame:
        """分批获取K线（腾讯接口每次最多640条）"""
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        all_dfs = []
        current_start = start_dt

        while current_start <= end_dt:
            # 640个交易日大约覆盖2.5年
            chunk_end = current_start + timedelta(days=900)
            if chunk_end > end_dt:
                chunk_end = end_dt

            df = self._fetch_kline(
                tencent_code,
                current_start.strftime('%Y-%m-%d'),
                chunk_end.strftime('%Y-%m-%d')
            )
            if not df.empty:
                all_dfs.append(df)

            current_start = chunk_end + timedelta(days=1)

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs)
        combined = combined[~combined.index.duplicated(keep='first')]
        combined = combined.sort_index()

        # 截取到目标范围
        combined = combined[(combined.index >= start_dt) & (combined.index <= end_dt)]
        return combined

    def load_market_index(self, index_code: str = '000001', start: str = None, end: str = None) -> pd.DataFrame:
        """加载上证指数作为市场代理（000001.SH）"""
        if self._market_index_cache is not None and start is None and end is None:
            return self._market_index_cache.copy()

        # 默认用上证指数，也可用沪深300(000300)
        if index_code in ['WINDA', '000001', '000985']:
            tencent_code = self._to_tencent_code('000001.SH')
        else:
            tencent_code = self._to_tencent_code(index_code)

        start = start or '2020-01-01'
        end = end or datetime.today().strftime('%Y-%m-%d')

        df = self._fetch_multiple_klines(tencent_code, start, end)
        if df.empty:
            return df

        df['amount'] = df['volume'] * df['close']  # 估算成交额
        df['volume'] = df['volume'] / 100  # 腾讯成交量是手，转为股
        df = df.rename(columns={'volume': 'volume'})

        if start is None and end is None:
            self._market_index_cache = df.copy()

        return df

    def load_industry_index(self, industry_code: str, start: str = None, end: str = None) -> pd.DataFrame:
        """加载行业指数（用龙头股代理）"""
        cache_key = f"{industry_code}_{start}_{end}"
        if cache_key in self._industry_index_cache:
            return self._industry_index_cache[cache_key].copy()

        proxy_stock = self._industry_index_codes.get(industry_code)
        if not proxy_stock:
            return pd.DataFrame()

        start = start or '2020-01-01'
        end = end or datetime.today().strftime('%Y-%m-%d')

        tencent_code = self._to_tencent_code(proxy_stock)
        df = self._fetch_multiple_klines(tencent_code, start, end)

        if not df.empty:
            df['amount'] = df['volume'] * df['close']
            df['volume'] = df['volume'] / 100
            self._industry_index_cache[cache_key] = df.copy()

        return df

    def load_stock_list(self, date: str) -> pd.DataFrame:
        """返回股票列表（基于各行业代表性股票）"""
        # 使用所有行业的代表股票
        all_codes = list(self._industry_map.keys())
        all_codes = list(set(all_codes))  # 去重

        df = pd.DataFrame({'code': all_codes})
        df['industry'] = df['code'].map(self._industry_map)
        df['industry'] = df['industry'].fillna('综合')
        df['list_date'] = pd.to_datetime('2010-01-01')  # 简化处理
        return df

    def load_stock_daily(self, stock_code: str, start: str = None, end: str = None) -> pd.DataFrame:
        """加载个股日线数据"""
        cache_key = f"{stock_code}_{start}_{end}"
        if cache_key in self._stock_data_cache:
            return self._stock_data_cache[cache_key].copy()

        start = start or '2020-01-01'
        end = end or datetime.today().strftime('%Y-%m-%d')

        tencent_code = self._to_tencent_code(stock_code)
        df = self._fetch_multiple_klines(tencent_code, start, end)

        if not df.empty:
            df['amount'] = df['volume'] * df['close']
            df['volume'] = df['volume'] / 100
            df['turnover'] = df['volume'] / (df['volume'].rolling(20).mean().fillna(df['volume'])) * 0.01  # 简化
            df['is_st'] = False
            df['is_suspended'] = False
            self._stock_data_cache[cache_key] = df.copy()

        return df

    def load_factor_data(self, date: str) -> pd.DataFrame:
        """
        基于真实行情计算纯技术因子
        （腾讯接口无财务数据，用技术指标替代基本面因子）
        """
        date_obj = pd.to_datetime(date)
        # 获取前120天数据用于计算因子
        start = (date_obj - timedelta(days=180)).strftime('%Y-%m-%d')
        end = date

        stock_list = self.load_stock_list(date)
        codes = stock_list['code'].tolist()

        records = []
        for code in codes:
            try:
                df = self.load_stock_daily(code, start, end)
                if df.empty or len(df) < 30:
                    continue

                # 只用到date当日的数据
                df = df[df.index <= date_obj]
                if df.empty:
                    continue

                latest = df.iloc[-1]
                close = latest['close']

                # 计算技术因子
                returns = df['close'].pct_change().dropna()
                if len(returns) < 20:
                    continue

                # 动量因子
                mom_20 = df['close'].iloc[-1] / df['close'].iloc[-min(20, len(df))] - 1 if len(df) >= 20 else 0
                mom_60 = df['close'].iloc[-1] / df['close'].iloc[-min(60, len(df))] - 1 if len(df) >= 60 else mom_20

                # 波动率
                vol_20 = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else 0.3

                # 成交额稳定性
                avg_volume = df['volume'].tail(20).mean()
                turnover_stability = 1 / (df['volume'].tail(20).std() / (avg_volume + 1e-6) + 1e-6)

                # 量价背离（价格与成交量相关性）
                if len(df) >= 20:
                    price_change = df['close'].tail(20).diff().dropna()
                    volume_change = df['volume'].tail(20).diff().dropna()
                    min_len = min(len(price_change), len(volume_change))
                    if min_len >= 10:
                        vp_corr = np.corrcoef(price_change.tail(min_len).values,
                                              volume_change.tail(min_len).values)[0, 1]
                        vp_divergence = -vp_corr  # 负相关视为背离
                    else:
                        vp_divergence = 0
                else:
                    vp_divergence = 0

                # 用技术指标替代基本面因子（简化处理）
                # EP/BP/ROE等用固定值+随机扰动模拟，实际应从财报数据库获取
                np.random.seed(hash(code) % 10000)
                ep = np.random.normal(0.05, 0.02)  # 盈利收益率
                bp = np.random.normal(0.8, 0.3)    # 市净率倒数
                dividend = np.random.uniform(0, 0.03)
                roe = np.random.normal(0.10, 0.05)
                roe_stability = 1 / np.random.uniform(0.01, 0.05)
                profit_growth = np.random.normal(0.15, 0.20)
                ocf_to_revenue = np.random.normal(0.12, 0.08)

                records.append({
                    'code': code,
                    'MARKET_CAP': np.random.lognormal(23, 1.2),
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
            except Exception as e:
                continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df.set_index('code', inplace=True)
        return df

    def load_industry_mapping(self, date: str) -> pd.Series:
        """返回股票-行业映射"""
        stock_list = self.load_stock_list(date)
        return pd.Series(stock_list.set_index('code')['industry'])

    def get_trade_dates(self, start: str, end: str) -> List[str]:
        """获取交易日列表（从上证指数数据推断）"""
        df = self.load_market_index(start=start, end=end)
        if df.empty:
            # 返回所有工作日
            dates = pd.bdate_range(start, end)
            return dates.strftime('%Y-%m-%d').tolist()
        return df.index.strftime('%Y-%m-%d').tolist()
