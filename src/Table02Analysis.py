import pandas as pd
import wrds
import config
from datetime import datetime
import unittest
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

"""
Is referenced by Table02Prep. Creates tables to understand the data and figures to understand the different ratios.
"""

def create_summary_stat_table_for_data(datasets, UPDATED=False):
    summary_df = pd.DataFrame()
    for key in datasets.keys():
        dataset = datasets[key].drop(columns=['datadate'])
        info = dataset.describe()
        # 删除 25%, 50%, 75% 行，只保留 {count, mean, std, min, max}
        info = info.drop(['25%', '50%', '75%'])
        numeric_cols = info.select_dtypes(include=['float64', 'int']).columns
        info[numeric_cols] = info[numeric_cols].round(2)
        info.reset_index(inplace=True)
        info['Key'] = key
        info.set_index(['Key', 'index'], inplace=True)
        summary_df = pd.concat([summary_df, info], axis=0)

    summary_df = summary_df.round(2)
    # 改列名以便表格更易读
    summary_df.columns = ['total assets', 'book debt', 'book equity', 'market equity']

    caption = (
        "There are significantly fewer entries for book equity than for other measures "
        "as shown in the count rows. There are also some negatives for book equity, "
        "which is not present for other categories."
    )

    latex_table = summary_df.to_latex(
        index=True, 
        multirow=True, 
        multicolumn=True,
        escape=False, 
        float_format="%.2f", 
        caption=caption, 
        label='tab:Table 2.1'
    )
    # 有时 to_latex 可能插入多余的 \multirow，这里做个替换
    latex_table = latex_table.replace(r'\multirow[t]{5}{*}', '')

    # 将输出文件写到 config.OUTPUT_DIR
    if UPDATED:
        outpath = config.OUTPUT_DIR / "updated_table02_sstable.tex"
    else:
        outpath = config.OUTPUT_DIR / "table02_sstable.tex"

    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print(f"Summary stats LaTeX saved to: {outpath}")


def create_figure_for_data(ratios_dict, UPDATED=False):
    """
    ratios_dict 是一个 dict，每个 key 对应一个 DataFrame 或 Series，显示各组别的比率等。
    此函数生成四个子图 (total_assets, book_debt, book_equity, market_equity)，
    并保存为 PNG 图表。
    """

    # 先把所有 Series/DataFrame 逐个拼接到 concatenated_df
    concatenated_df = pd.concat(
        [s.rename(f"{key}_{s.name}") for key, s in ratios_dict.items()], 
        axis=1
    )

    concatenated_df.sort_index(inplace=True)
    concatenated_df = concatenated_df.apply(pd.to_numeric, errors='coerce')
    concatenated_df.ffill(inplace=True)
    concatenated_df.bfill(inplace=True)

    # 依据列名区分四种指标
    asset_columns = [col for col in concatenated_df.columns if 'total_assets' in col]
    debt_columns = [col for col in concatenated_df.columns if 'book_debt' in col]
    equity_columns = [col for col in concatenated_df.columns if 'book_equity' in col]
    market_columns = [col for col in concatenated_df.columns if 'market_equity' in col]

    # 这里设定每种分类的颜色，若列数 > 4，可以再做扩展
    asset_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    debt_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    equity_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    market_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # 根据 subplot 分别绘图
    for ax, columns, colors, category in zip(
        axes.flatten(),
        [asset_columns, debt_columns, equity_columns, market_columns],
        [asset_colors, debt_colors, equity_colors, market_colors],
        ['total_assets', 'book_debt', 'book_equity', 'market_equity']
    ):
        columns.sort()
        # unique_keys 用于 label
        unique_keys = [col.split('_')[-1] for col in columns]  # 取列名里最后一段当作 key

        for col, color, key in zip(columns, colors, unique_keys):
            ax.plot(
                concatenated_df.index, 
                concatenated_df[col], 
                label=key, 
                color=color
            )
        ax.set_title(f"{category.capitalize()}")
        ax.legend(loc='upper left')
        ax.grid(True)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')

    # 在图下方添加一句说明
    time = datetime.now()
    caption = (
        f"{time}: From the plots above we can observe the trends of the ratios for "
        "each comparison group over time. Missing values have been forward/back filled."
    )
    fig.text(0.5, -0.1, caption, ha='center', fontsize=8)

    # 将图片保存到 config.OUTPUT_DIR
    if UPDATED:
        figpath = config.OUTPUT_DIR / "updated_table02_figure.png"
    else:
        figpath = config.OUTPUT_DIR / "table02_figure.png"

    plt.savefig(figpath, bbox_inches='tight')
    plt.close(fig)  # 生成后关掉，防止后续重复画
    print(f"Figure saved to: {figpath}")