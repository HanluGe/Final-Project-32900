import pandas as pd
import wrds
import config
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

"""
Table02Analysis.py

Provides analysis utilities for Table 02, including:
 - create_summary_stat_table_for_data: produce summary stats in LaTeX
 - create_figure_for_data: plot ratio lines (no rolling average)
 - create_corr_matrix_for_data: build correlation table (same metric across PD, BD, Banks, Cmpust.)

No testing code is included here; see Table02_testing.py for tests.
"""

def create_summary_stat_table_for_data(datasets, UPDATED=False):
    """
    Creates summary statistics (count, mean, std, min, max) for each group's dataset,
    outputs them as a LaTeX table to config.OUTPUT_DIR.
    """
    summary_df = pd.DataFrame()
    for gname, df in datasets.items():
        local_df = df.drop(columns=['datadate'], errors='ignore')
        stats = local_df.describe()
        stats = stats.drop(['25%', '50%', '75%'], errors='ignore')
        numeric_cols = stats.select_dtypes(include=['float64','int']).columns
        stats[numeric_cols] = stats[numeric_cols].round(2)
        stats.reset_index(inplace=True)
        stats['Key'] = gname
        stats.set_index(['Key','index'], inplace=True)
        summary_df = pd.concat([summary_df, stats], axis=0)

    summary_df = summary_df.round(2)
    summary_df.columns = ['total assets','book debt','book equity','market equity']

    cap = """
    There are significantly fewer entries for book equity
    than for other measures as shown in the count rows.
    There are also some negatives for book equity.
    """
    latex = summary_df.to_latex(
        index=True,
        multirow=True,
        multicolumn=True,
        escape=False,
        float_format="%.2f",
        caption=cap,
        label='tab:Table 2.1'
    )
    latex = latex.replace(r'\multirow[t]{5}{*}', '')

    if UPDATED:
        out = config.OUTPUT_DIR / "updated_table02_sstable.tex"
    else:
        out = config.OUTPUT_DIR / "table02_sstable.tex"

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(out, 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f"Summary stats LaTeX saved to: {out}")


def create_figure_for_data(ratio_df, UPDATED=False):
    """
    Plots lines for ratio columns, grouped by subplots:
      - total_assets*
      - book_debt*
      - book_equity*
      - market_equity*

    No rolling average lines; straightforward time series for each ratio.
    """
    ratio_df = ratio_df.copy()
    ratio_df.sort_index(inplace=True)
    ratio_df = ratio_df.apply(pd.to_numeric, errors='coerce').ffill().bfill()

    asset_cols = [c for c in ratio_df.columns if 'total_assets' in c]
    debt_cols  = [c for c in ratio_df.columns if 'book_debt' in c]
    eqty_cols  = [c for c in ratio_df.columns if 'book_equity' in c]
    mkt_cols   = [c for c in ratio_df.columns if 'market_equity' in c]

    fig, axes = plt.subplots(2,2, figsize=(12,8), sharex=True)
    cat_map = [
       ('Total_assets', asset_cols),
       ('Book_debt',   debt_cols),
       ('Book_equity', eqty_cols),
       ('Market_equity', mkt_cols)
    ]
    for ax, (title, cols) in zip(axes.flatten(), cat_map):
        cols.sort()
        for col in cols:
            ax.plot(ratio_df.index, ratio_df[col], label=col)
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')

    time = datetime.now()
    cap = f"{time}: Subplots show ratio lines, no rolling average."
    fig.text(0.5, -0.08, cap, ha='center', fontsize=8)

    if UPDATED:
        figpath = config.OUTPUT_DIR / "updated_table02_figure.png"
    else:
        figpath = config.OUTPUT_DIR / "table02_figure.png"
        
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(figpath, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to: {figpath}")


def create_corr_matrix_for_data(datasets, UPDATED=False):
    """
    Builds correlation matrices for each metric (total_assets, book_debt, book_equity, market_equity)
    across PD, BD, Banks, Cmpust.
    The result is saved as a LaTeX file in config.OUTPUT_DIR.
    """
    group_order = ['PD','BD','Banks','Cmpust.']
    metrics = ['total_assets','book_debt','book_equity','market_equity']
    all_latex = []

    for m in metrics:
        combined_df = pd.DataFrame()
        for g in group_order:
            if g not in datasets:
                continue
            if m not in datasets[g].columns:
                continue
            sub = datasets[g][['datadate', m]].copy()
            sub['datadate'] = pd.to_datetime(sub['datadate'])
            sub[m] = pd.to_numeric(sub[m], errors='coerce')
            sub = sub.dropna(subset=[m])
            sub = sub.drop_duplicates(subset=['datadate'])
            sub.set_index('datadate', inplace=True)
            col_name = f"{m}_{g}"

            if combined_df.empty:
                combined_df[col_name] = sub[m]
            else:
                combined_df = combined_df.join(sub[m].rename(col_name), how='outer')

        combined_df.dropna(how='all', inplace=True)
        if len(combined_df.columns) < 2:
            continue

        corr = combined_df.corr()
        c_latex = corr.to_latex(
            float_format="%.3f",
            caption=f"Correlation of {m} across PD, BD, Banks, Cmpust.",
            label=f"tab:{m}"
        )
        all_latex.append(c_latex)

    final_txt = "\n\n".join(all_latex)
    if UPDATED:
        out = config.OUTPUT_DIR / "updated_table02_corr.tex"
    else:
        out = config.OUTPUT_DIR / "table02_corr.tex"
        
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(out, 'w', encoding='utf-8') as f:
        f.write(final_txt)

    print(f"Correlation matrix LaTeX saved to: {out}")
