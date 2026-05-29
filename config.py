"""
A股多因子轮动 × 趋势周期策略 - 全局配置
"""

import numpy as np
from enum import Enum

# ==================== 市场状态定义 ====================
class MarketState(Enum):
    STRONG_BULL = "strong_bull"      # 强趋势牛
    WEAK_BULL = "weak_bull"          # 震荡偏强
    CONSOLIDATION = "consolidation"  # 震荡偏弱
    WEAK_BEAR = "weak_bear"          # 弱趋势熊
    STRONG_BEAR = "strong_bear"      # 强趋势熊

class StyleRegime(Enum):
    VALUE_DOMINANT = "value_dominant"
    GROWTH_DOMINANT = "growth_dominant"
    TRANSITION = "style_transition"

# ==================== 参数配置 ====================
class StrategyConfig:
    """策略核心参数 - 所有超参集中管理"""

    # --- 回测参数 ---
    START_DATE = "2020-01-01"
    END_DATE = "2024-12-31"
    REBALANCE_FREQ = "W-FRI"  # 双周周五（实际用双周需额外过滤）

    # --- 宏观周期 ---
    TREND_MA_SHORT = 20       # 短期均线
    TREND_MA_LONG = 60        # 长期均线
    TREND_MA_SUPER = 120      # 超长期均线（半年线）
    RSI_WINDOW = 14
    STYLE_MA_SHORT = 20       # 风格判断短期均线
    STYLE_MA_LONG = 60        # 风格判断长期均线

    # --- 仓位系数 ---
    POSITION_LIMIT = {
        MarketState.STRONG_BULL: 1.0,
        MarketState.WEAK_BULL: 0.75,
        MarketState.CONSOLIDATION: 0.55,
        MarketState.WEAK_BEAR: 0.35,
        MarketState.STRONG_BEAR: 0.15,
    }

    # --- 行业轮动 ---
    INDUSTRY_MOMENTUM_SHORT = 20   # 行业短期动量
    INDUSTRY_MOMENTUM_LONG = 60    # 行业长期动量
    INDUSTRY_TREND_MA = 120        # 行业趋势均线
    INDUSTRY_VOLUME_WINDOW = 5     # 拥挤度计算窗口
    INDUSTRY_VOLUME_LOOKBACK = 60    # 拥挤度历史分位回看
    INDUSTRY_TOP_N = 10            # 入选行业数量
    INDUSTRY_MAX_WEIGHT = 0.15     # 单个行业权重上限
    INDUSTRY_STOP_MA = 20          # 行业止损均线

    # --- 拥挤度阈值 ---
    CONGESTION_EXTREME = 0.95      # 极度拥挤
    CONGESTION_HIGH = 0.90         # 高度拥挤
    CONGESTION_MID = 0.80          # 中度拥挤

    # --- 多因子 ---
    FACTOR_IC_LOOKBACK = 40        # 因子IC计算回看
    FACTOR_IC_THRESHOLD = 0.02     # 因子失效阈值
    MAD_MULTIPLIER = 5             # MAD去极值倍数

    # --- 动态权重映射 (Value / Growth / Transition) ---
    FACTOR_WEIGHT_MAP = {
        StyleRegime.VALUE_DOMINANT: {
            'EP': 0.20, 'BP': 0.15, 'DIVIDEND_YIELD': 0.10,
            'ROE': 0.20, 'ROE_STABILITY': 0.15, 'PROFIT_GROWTH': 0.10,
            'MOMENTUM_20': 0.05, 'MOMENTUM_60': 0.05,
            'VOLATILITY_20': -0.15, 'TURNOVER_STABILITY': 0.10, 'VOLUME_PRICE_DIVERGENCE': -0.05
        },
        StyleRegime.GROWTH_DOMINANT: {
            'EP': 0.10, 'BP': 0.05, 'DIVIDEND_YIELD': 0.0,
            'ROE': 0.15, 'ROE_STABILITY': 0.05, 'PROFIT_GROWTH': 0.25,
            'MOMENTUM_20': 0.20, 'MOMENTUM_60': 0.15,
            'VOLATILITY_20': -0.10, 'TURNOVER_STABILITY': 0.05, 'VOLUME_PRICE_DIVERGENCE': -0.10
        },
        StyleRegime.TRANSITION: {
            'EP': 0.15, 'BP': 0.10, 'DIVIDEND_YIELD': 0.05,
            'ROE': 0.20, 'ROE_STABILITY': 0.15, 'PROFIT_GROWTH': 0.15,
            'MOMENTUM_20': 0.10, 'MOMENTUM_60': 0.10,
            'VOLATILITY_20': -0.15, 'TURNOVER_STABILITY': 0.10, 'VOLUME_PRICE_DIVERGENCE': -0.05
        }
    }

    # 因子权重约束
    VALUE_WEIGHT_RANGE = (0.20, 0.45)    # 价值类因子合计
    MOMENTUM_WEIGHT_RANGE = (0.10, 0.40) # 动量类因子合计
    MICRO_WEIGHT_RANGE = (-0.25, -0.10)  # 微观结构（负向）

    # --- 选股参数 ---
    STOCK_MIN_LIST_DAYS = 120      # 最小上市天数
    STOCK_MIN_AMOUNT_PCT = 0.20    # 日均成交额后20%剔除
    STOCKS_PER_INDUSTRY = 5        # 每行业选股数量
    STOCK_MAX_WEIGHT = 0.03        # 单只个股权重上限
    MIN_CASH_RATIO = 0.05          # 最小现金比例

    # --- 风控参数 ---
    STOCK_STOP_LOSS = 0.10         # 个股止损回撤（10%）
    PORTFOLIO_STOP_LOSS = 0.05     # 组合止损
    INDUSTRY_STOP_MA_BREAK = 20    # 行业破均线止损
    INDUSTRY_CRASH_LIMIT = 0.10    # 行业5日跌幅止损

    # --- 交易成本 ---
    COMMISSION_RATE = 0.0003         # 佣金 0.03%
    STAMP_TAX_RATE = 0.0005        # 印花税 0.05%（仅卖出）
    SLIPPAGE_RATE = 0.0005         # 滑点 0.05%

    # --- 申万一级行业列表（31个） ---
    SW_INDUSTRY_NAMES = [
        '农林牧渔', '基础化工', '钢铁', '有色金属', '电子', '家用电器', '食品饮料',
        '纺织服饰', '轻工制造', '医药生物', '公用事业', '交通运输', '房地产',
        '商贸零售', '社会服务', '银行', '非银金融', '综合', '建筑材料', '建筑装饰',
        '电力设备', '机械设备', '国防军工', '计算机', '传媒', '通信', '煤炭',
        '石油石化', '环保', '美容护理', '汽车'
    ]
