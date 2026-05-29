"""
回测结果可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class PerformanceVisualizer:
    """绩效可视化"""

    def __init__(self, records_df: pd.DataFrame, engine):
        self.df = records_df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.set_index('date').sort_index()
        self.engine = engine

    def plot_nav_curve(self, save_path: str = None):
        """绘制净值曲线"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

        # 1. 净值曲线
        ax1 = axes[0]
        ax1.plot(self.df.index, self.df['nav'] / 1e6, linewidth=1.5, color='#2E86AB', label='Strategy NAV')
        ax1.set_title('A股多因子轮动 × 趋势周期策略 | 净值曲线', fontsize=14, fontweight='bold')
        ax1.set_ylabel('净值 (百万)', fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 标注关键状态
        for state, color in [('strong_bull', 'red'), ('strong_bear', 'green')]:
            mask = self.df['market_state'] == state
            if mask.any():
                ax1.scatter(self.df.index[mask], self.df['nav'][mask] / 1e6, 
                           c=color, s=20, alpha=0.6, label=state)

        # 2. 市场状态与仓位
        ax2 = axes[1]
        state_map = {s: i for i, s in enumerate(self.df['market_state'].unique())}
        colors = ['#E63946', '#F4A261', '#E9C46A', '#2A9D8F', '#264653']
        for i, (state, idx) in enumerate(state_map.items()):
            mask = self.df['market_state'] == state
            ax2.scatter(self.df.index[mask], [idx]*mask.sum(), 
                      c=colors[i % len(colors)], s=30, label=state, alpha=0.8)
        ax2.set_ylabel('市场状态', fontsize=11)
        ax2.set_yticks(range(len(state_map)))
        ax2.set_yticklabels(list(state_map.keys()))
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. 现金比例
        ax3 = axes[2]
        ax3.fill_between(self.df.index, self.df['cash_ratio'] * 100, alpha=0.4, color='#F4A261')
        ax3.plot(self.df.index, self.df['cash_ratio'] * 100, color='#E76F51', linewidth=1)
        ax3.set_ylabel('现金比例 (%)', fontsize=11)
        ax3.set_xlabel('日期', fontsize=11)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_drawdown(self, save_path: str = None):
        """绘制回撤曲线"""
        cummax = self.df['nav'].cummax()
        drawdown = (self.df['nav'] - cummax) / cummax * 100

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.fill_between(self.df.index, drawdown, 0, alpha=0.4, color='#E63946')
        ax.plot(self.df.index, drawdown, color='#E63946', linewidth=1)
        ax.set_title('策略回撤曲线', fontsize=14, fontweight='bold')
        ax.set_ylabel('回撤 (%)', fontsize=11)
        ax.set_xlabel('日期', fontsize=11)
        ax.grid(True, alpha=0.3)

        # 标注最大回撤
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        ax.annotate(f'最大回撤: {max_dd_val:.1f}%
{max_dd_idx.strftime("%Y-%m-%d")}',
                   xy=(max_dd_idx, max_dd_val), xytext=(max_dd_idx + pd.Timedelta(days=100), max_dd_val + 5),
                   arrowprops=dict(arrowstyle='->', color='black'),
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_industry_rotation(self, save_path: str = None):
        """绘制行业轮动热力图"""
        # 构建行业持仓时间序列
        industry_records = self.engine.industry_records
        all_industries = set()
        for rec in industry_records:
            all_industries.update(rec['selected_industries'])
        all_industries = sorted(list(all_industries))

        industry_matrix = pd.DataFrame(0, 
            index=[pd.to_datetime(rec['date']) for rec in industry_records],
            columns=all_industries
        )

        for rec in industry_records:
            date = pd.to_datetime(rec['date'])
            for ind, w in rec['industry_weights'].items():
                if ind in industry_matrix.columns:
                    industry_matrix.loc[date, ind] = w

        fig, ax = plt.subplots(figsize=(16, 8))
        im = ax.imshow(industry_matrix.T.values, aspect='auto', cmap='YlOrRd', 
                      interpolation='nearest', vmin=0, vmax=0.15)

        ax.set_yticks(range(len(all_industries)))
        ax.set_yticklabels(all_industries, fontsize=9)
        ax.set_xticks(range(0, len(industry_matrix), max(1, len(industry_matrix)//10)))
        ax.set_xticklabels([industry_matrix.index[i].strftime('%Y-%m') 
                           for i in range(0, len(industry_matrix), max(1, len(industry_matrix)//10))], 
                          rotation=45, ha='right')
        ax.set_title('行业配置权重热力图', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='权重')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_factor_weights(self, save_path: str = None):
        """绘制因子权重变化"""
        factor_records = self.engine.factor_weight_records

        # 收集所有因子
        all_factors = set()
        for rec in factor_records:
            all_factors.update(rec['weights'].keys())
        all_factors = sorted(list(all_factors))

        factor_df = pd.DataFrame(index=[pd.to_datetime(rec['date']) for rec in factor_records])
        for f in all_factors:
            factor_df[f] = [rec['weights'].get(f, 0) for rec in factor_records]

        fig, ax = plt.subplots(figsize=(14, 7))
        factor_df.plot(kind='area', stacked=True, ax=ax, alpha=0.7, 
                      colormap='tab20', linewidth=0.5)
        ax.set_title('动态因子权重变化', fontsize=14, fontweight='bold')
        ax.set_ylabel('权重', fontsize=11)
        ax.set_xlabel('日期', fontsize=11)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
