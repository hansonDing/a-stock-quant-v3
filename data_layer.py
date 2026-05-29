"""
数据层：提供行情、财务、行业数据接口
包含：模拟数据生成器（用于离线演示）+ 真实数据源接入占位
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class DataSource:
    """
    数据接口基类
    实际使用时，继承此类并实现 load_* 方法接入Tushare/聚宽/米筐等
    """

    def __init__(self):
        self._cache = {}

    def load_market_index(self, index_code: str, start: str, end: str) -> pd.DataFrame:
        """加载大盘指数日线：open, high, low, close, volume, amount"""
        raise NotImplementedError

    def load_industry_index(self, industry_code: str, start: str, end: str) -> pd.DataFrame:
        """加载行业指数日线"""
        raise NotImplementedError

    def load_stock_list(self, date: str) -> pd.DataFrame:
        """加载某日期股票列表及基础信息"""
        raise NotImplementedError

    def load_stock_daily(self, stock_code: str, start: str, end: str) -> pd.DataFrame:
        """加载个股日线"""
        raise NotImplementedError

    def load_factor_data(self, factor_name: str, date: str) -> pd.Series:
        """加载某日期横截面因子数据"""
        raise NotImplementedError

    def load_industry_mapping(self, date: str) -> pd.Series:
        """加载股票-行业映射 {stock_code: industry_name}"""
        raise NotImplementedError


class MockDataSource(DataSource):
    """
    模拟数据生成器
    生成具有真实统计特征的模拟数据，用于策略逻辑验证
    """

    def __init__(self, n_stocks: int = 500, random_seed: int = 42):
        super().__init__()
        np.random.seed(random_seed)
        self.n_stocks = n_stocks
        self.trade_dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
        self.trade_dates = self.trade_dates[self.trade_dates.isin(
            pd.bdate_range('2020-01-01', '2024-12-31')
        )]

        # 生成股票代码
        self.stock_codes = [f"{np.random.choice(['600', '601', '000', '002', '300'])}{np.random.randint(100, 999):03d}.SH" 
                           if i % 2 == 0 else 
                           f"{np.random.choice(['000', '002', '300'])}{np.random.randint(100, 999):03d}.SZ"
                           for i in range(n_stocks)]

        # 行业映射
        industries = ['电子', '医药生物', '电力设备', '食品饮料', '计算机', '汽车', '银行', '非银金融',
                     '有色金属', '基础化工', '机械设备', '国防军工', '传媒', '通信', '煤炭', '石油石化',
                     '家用电器', '农林牧渔', '房地产', '交通运输', '公用事业', '建筑材料', '建筑装饰',
                     '钢铁', '纺织服饰', '轻工制造', '商贸零售', '社会服务', '环保', '美容护理', '综合']
        self.industry_map = pd.Series(
            np.random.choice(industries, n_stocks), 
            index=self.stock_codes
        )

        # 上市日期
        self.list_dates = pd.Series(
            [self.trade_dates[0] - timedelta(days=np.random.randint(200, 2000)) 
             for _ in range(n_stocks)],
            index=self.stock_codes
        )

        # 生成市场指数（万得全A模拟）
        self.market_index = self._generate_market_index()

        # 生成风格指数（价值/成长）
        self.value_index, self.growth_index = self._generate_style_indices()

        # 生成行业指数
        self.industry_indices = self._generate_industry_indices()

        # 生成个股数据
        self.stock_data = self._generate_stock_data()

        # 生成财务/因子数据
        self.factor_data = self._generate_factor_data()

    def _generate_market_index(self) -> pd.DataFrame:
        """生成模拟大盘指数（含趋势周期特征）"""
        n = len(self.trade_dates)
        returns = np.random.normal(0.0003, 0.012, n)

        # 注入趋势周期（2020牛市->2021震荡->2022熊市->2023复苏->2024震荡）
        regimes = [
            (0, 120, 0.001, 0.012),      # 2020 牛市
            (120, 300, 0.0002, 0.015),   # 2021 震荡
            (300, 450, -0.001, 0.018),   # 2022 熊市
            (450, 650, 0.0008, 0.014),   # 2023 复苏
            (650, n, 0.0002, 0.013),     # 2024 震荡
        ]
        for start, end, mu, sigma in regimes:
            returns[start:end] = np.random.normal(mu, sigma, end-start)

        price = 4000 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({
            'open': price * (1 + np.random.normal(0, 0.005, n)),
            'high': price * (1 + np.abs(np.random.normal(0, 0.008, n))),
            'low': price * (1 - np.abs(np.random.normal(0, 0.008, n))),
            'close': price,
            'volume': np.random.lognormal(20, 0.5, n),
            'amount': np.random.lognormal(23, 0.5, n)
        }, index=self.trade_dates)
        return df

    def _generate_style_indices(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """生成价值/成长指数（具有风格轮动特征）"""
        n = len(self.trade_dates)

        # 价值指数：低波动，2021-2022占优
        v_returns = np.random.normal(0.0002, 0.010, n)
        v_returns[120:400] += 0.0005  # 价值强势期
        v_price = 3000 * np.exp(np.cumsum(v_returns))

        # 成长指数：高波动，2020、2023占优
        g_returns = np.random.normal(0.0004, 0.015, n)
        g_returns[0:120] += 0.001    # 成长强势期
        g_returns[450:650] += 0.0008 # 成长强势期
        g_price = 3500 * np.exp(np.cumsum(g_returns))

        v_df = pd.DataFrame({'close': v_price}, index=self.trade_dates)
        g_df = pd.DataFrame({'close': g_price}, index=self.trade_dates)

        return v_df, g_df

    def _generate_industry_indices(self) -> Dict[str, pd.DataFrame]:
        """生成31个行业指数"""
        industries = ['电子', '医药生物', '电力设备', '食品饮料', '计算机', '汽车', '银行', '非银金融',
                     '有色金属', '基础化工', '机械设备', '国防军工', '传媒', '通信', '煤炭', '石油石化',
                     '家用电器', '农林牧渔', '房地产', '交通运输', '公用事业', '建筑材料', '建筑装饰',
                     '钢铁', '纺织服饰', '轻工制造', '商贸零售', '社会服务', '环保', '美容护理', '综合']

        result = {}
        for ind in industries:
            n = len(self.trade_dates)
            returns = np.random.normal(0.0003, 0.014, n)
            # 不同行业不同Beta
            beta = np.random.uniform(0.8, 1.2)
            market_returns = self.market_index['close'].pct_change().fillna(0)
            returns += beta * market_returns.values * 0.3

            price = 3000 * np.exp(np.cumsum(returns))
            df = pd.DataFrame({
                'close': price,
                'volume': np.random.lognormal(18, 0.6, n),
                'amount': np.random.lognormal(21, 0.6, n)
            }, index=self.trade_dates)
            result[ind] = df
        return result

    def _generate_stock_data(self) -> Dict[str, pd.DataFrame]:
        """生成个股日线数据"""
        data = {}
        market_returns = self.market_index['close'].pct_change().fillna(0).values

        for code in self.stock_codes:
            n = len(self.trade_dates)
            beta = np.random.uniform(0.7, 1.3)
            alpha = np.random.normal(0, 0.008, n)
            returns = beta * market_returns + alpha + np.random.normal(0, 0.005, n)

            price = 20 * np.exp(np.cumsum(returns))
            df = pd.DataFrame({
                'open': price * (1 + np.random.normal(0, 0.008, n)),
                'high': price * (1 + np.abs(np.random.normal(0, 0.012, n))),
                'low': price * (1 - np.abs(np.random.normal(0, 0.012, n))),
                'close': price,
                'volume': np.random.lognormal(15, 0.8, n),
                'amount': np.random.lognormal(18, 0.8, n),
                'turnover': np.random.uniform(0.01, 0.15, n),
                'is_st': False,
                'is_suspended': False
            }, index=self.trade_dates)
            data[code] = df
        return data

    def _generate_factor_data(self) -> Dict[str, pd.DataFrame]:
        """生成横截面因子数据（每个交易日所有股票的因子值）"""
        factors = {}

        for date in self.trade_dates:
            n = self.n_stocks

            # 市值因子（对数）
            market_cap = np.random.lognormal(23, 1.2, n)

            # 价值因子（与市值负相关，模拟小盘价值）
            ep = np.random.normal(0.05, 0.03, n) + np.random.normal(0, 0.01) * np.log(market_cap)
            bp = np.random.normal(0.8, 0.4, n) + np.random.normal(0, 0.05) * np.log(market_cap)
            dividend = np.random.uniform(0, 0.05, n)

            # 质量因子
            roe = np.random.normal(0.10, 0.06, n)
            roe_std = np.random.uniform(0.01, 0.05, n)
            profit_growth = np.random.normal(0.15, 0.30, n)
            ocf_to_revenue = np.random.normal(0.12, 0.08, n)

            # 动量因子（与近期收益相关）
            mom_20 = np.random.normal(0.02, 0.08, n)
            mom_60 = np.random.normal(0.05, 0.15, n)

            # 波动率因子
            vol_20 = np.random.uniform(0.15, 0.45, n)
            turnover_stability = 1 / np.random.uniform(0.01, 0.10, n)

            # 量价背离（随机）
            vp_divergence = np.random.normal(0, 1, n)

            df = pd.DataFrame({
                'MARKET_CAP': market_cap,
                'EP': ep,
                'BP': bp,
                'DIVIDEND_YIELD': dividend,
                'ROE': roe,
                'ROE_STABILITY': 1 / (roe_std + 1e-6),
                'PROFIT_GROWTH': profit_growth,
                'OCF_TO_REVENUE': ocf_to_revenue,
                'MOMENTUM_20': mom_20,
                'MOMENTUM_60': mom_60,
                'VOLATILITY_20': vol_20,
                'TURNOVER_STABILITY': turnover_stability,
                'VOLUME_PRICE_DIVERGENCE': vp_divergence,
                'INDUSTRY': self.industry_map.values,
                'LIST_DATE': self.list_dates.values
            }, index=self.stock_codes)

            factors[date.strftime('%Y-%m-%d')] = df

        return factors

    # ============== 接口实现 ==============

    def load_market_index(self, index_code: str = 'WINDA', start: str = None, end: str = None) -> pd.DataFrame:
        df = self.market_index.copy()
        if start:
            start_dt = pd.to_datetime(start)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df[df.index >= start_dt]
        if end:
            end_dt = pd.to_datetime(end)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df[df.index <= end_dt]
        return df

    def load_industry_index(self, industry_code: str, start: str = None, end: str = None) -> pd.DataFrame:
        df = self.industry_indices.get(industry_code, pd.DataFrame()).copy()
        if start:
            start_dt = pd.to_datetime(start)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df[df.index >= start_dt]
        if end:
            end_dt = pd.to_datetime(end)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df[df.index <= end_dt]
        return df

    def load_stock_list(self, date: str) -> pd.DataFrame:
        """返回某日期可交易股票列表"""
        date_obj = pd.to_datetime(date)
        codes = []
        for code in self.stock_codes:
            if self.list_dates[code] <= date_obj:
                codes.append(code)

        df = pd.DataFrame({'code': codes})
        df['industry'] = df['code'].map(self.industry_map)
        df['list_date'] = df['code'].map(self.list_dates)
        return df

    def load_stock_daily(self, stock_code: str, start: str = None, end: str = None) -> pd.DataFrame:
        df = self.stock_data.get(stock_code, pd.DataFrame()).copy()
        if start:
            start_dt = pd.to_datetime(start)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df[df.index >= start_dt]
        if end:
            end_dt = pd.to_datetime(end)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df[df.index <= end_dt]
        return df

    def load_factor_data(self, date: str) -> pd.DataFrame:
        return self.factor_data.get(date, pd.DataFrame())

    def load_industry_mapping(self, date: str) -> pd.Series:
        return self.industry_map

    def get_trade_dates(self, start: str, end: str) -> List[str]:
        dates = self.trade_dates[(self.trade_dates >= start) & (self.trade_dates <= end)]
        return dates.strftime('%Y-%m-%d').tolist()
