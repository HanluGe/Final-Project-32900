"""
Table03Analysis.py

This module generates descriptive tables and figures to analyze the relationships 
among financial ratios and macroeconomic variables. It produces a LaTeX summary table 
and several figures, including:
- Figure 1: Standardized time series of market cap ratio and capital risk factor.
- Figure 2: Original levels of financial ratios (market cap ratio, book capital ratio, and AEM leverage).
- Figure 3: Combined plots of standardized financial ratios and macroeconomic variables.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config  # Use config to standardize paths

def create_summary_stat_table_for_data(dataset, UPDATED=False):
    """
    Creates a summary statistics table for the dataset.
    Input: dataset (DataFrame) with numerical columns.
    Output: Exports a LaTeX file with the summary statistics table to OUTPUT_DIR.
    Drops the 25%, 50%, and 75% quantiles and writes the formatted table.
    """
    summary_df = pd.DataFrame()
    info = dataset.describe()
    info = info.drop(['25%', '50%', '75%'])
    summary_df = pd.concat([summary_df, info], axis=0)
    
    caption = "Summary statistics of capital factors and macro variables"
    latex_table = summary_df.to_latex(index=True, multirow=True, multicolumn=True,
                                      escape=False, float_format="%.2f", caption=caption, label='tab:Table 3.1')
    latex_table = latex_table.replace(r'\multirow[t]{5}{*}', '')
    
    outfile = config.OUTPUT_DIR / ("updated_table03_sstable.tex" if UPDATED else "table03_sstable.tex")
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(latex_table)

def standardize_ratios_and_factors(data):
    """
    Automatically standardizes columns ending with 'ratio' or 'factor' in the DataFrame.
    Input: data (DataFrame) with financial ratio or factor columns.
    Output: DataFrame with additional standardized columns (_std) calculated as (x - mean) / std.
    This enables comparison among variables on a common scale.
    """
    columns_to_standardize = [col for col in data.columns if col.endswith('ratio') or col.endswith('factor')]
    for col in columns_to_standardize:
        standardized_col_name = f'{col}_std'
        data[standardized_col_name] = (data[col] - data[col].mean()) / data[col].std()
    return data

def plot_figure01(ratios, factors, UPDATED=False):
    """
    Plots the standardized market cap ratio and capital risk factor over time.
    Input: ratios (DataFrame) with 'market_cap_ratio'; factors (DataFrame) with 'market_capital_factor'.
    Output: Saves a PNG file to OUTPUT_DIR.
    It concatenates the two series, standardizes them, and plots the standardized values.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    data = pd.concat([ratios[['market_cap_ratio']], factors[['market_capital_factor']]], axis=1)
    data = standardize_ratios_and_factors(data)
    ax.plot(data.index, data['market_cap_ratio_std'], label='Market Cap Ratio')
    ax.plot(data.index, data['market_capital_factor_std'], color='orange', linestyle='--', label='Capital Risk Factor')
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.set_xlabel('Date')
    ax.set_ylabel('Standardized Value')
    ax.set_ylim(-4, 4)
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_title('Intermediary Capital Ratio and Risk Factor of Primary Dealers')
    ax.legend(loc='best')
    
    outfile = config.OUTPUT_DIR / ("updated_table03_figure01.png" if UPDATED else "table03_figure01.png")
    plt.savefig(outfile)
    plt.close()

def plot_figure02(ratios, UPDATED=False):
    """
    Plots the levels of market cap ratio, book capital ratio, and AEM leverage over time.
    Input: ratios (DataFrame) with columns 'market_cap_ratio', 'book_cap_ratio', and 'aem_leverage'.
    Output: Saves a PNG file to OUTPUT_DIR.
    Displays original (unstandardized) values (scaled by 100) using a logarithmic y-scale.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ratios = standardize_ratios_and_factors(ratios)  # For auxiliary standardization; original values remain unchanged.
    ax.plot(ratios.index, ratios['market_cap_ratio']*100, label='Market Capital Ratio')
    ax.plot(ratios.index, ratios['book_cap_ratio']*100, label='Book Capital Ratio', color='green', linestyle='dotted')
    ax.plot(ratios.index, ratios['aem_leverage']*100, label='AEM Leverage', color='orange', linestyle='--')
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.set_xlabel('Date')
    ax.set_yscale('log')
    ax.set_yticks([5, 10, 50, 100])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_title('AEM Leverage and Intermediary Capital Ratio: Level')
    ax.legend(loc='best')
    
    outfile = config.OUTPUT_DIR / ("updated_table03_figure.png" if UPDATED else "table03_figure.png")
    plt.savefig(outfile)
    plt.close()

def plot_figure03(ratios, macro, UPDATED=False):
    """
    Plots the standardized trends of financial ratios and macroeconomic variables over time.
    Input: ratios (DataFrame) with 'market_cap_ratio', 'book_cap_ratio', and 'aem_leverage';
           macro (DataFrame) with macro variables (e.g., 'e/p', 'unemp_rate', 'real_gdp', 'mkt_ret', 'mkt_vol').
    Output: Saves a PNG file to OUTPUT_DIR.
    The function first standardizes both the financial ratios and the macro variables,
    then plots them in two subplots (upper: financial ratios; lower: macro variables) for trend comparison.
    """
    # Standardize financial ratios
    ratios_std = ratios.copy()
    for col in ['market_cap_ratio', 'book_cap_ratio', 'aem_leverage']:
        ratios_std[col] = (ratios_std[col] - ratios_std[col].mean()) / ratios_std[col].std()
    
    # Standardize selected macro variables
    macro_std = macro.copy()
    for col in ['e/p', 'unemp_rate', 'real_gdp', 'mkt_ret', 'mkt_vol']:
        macro_std[col] = (macro_std[col] - macro_std[col].mean()) / macro_std[col].std()
    
    # Create two subplots sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Upper subplot: financial ratios
    ax1.plot(ratios_std.index, ratios_std['market_cap_ratio'], label='Market Capital Ratio', marker='o')
    ax1.plot(ratios_std.index, ratios_std['book_cap_ratio'], label='Book Capital Ratio', marker='s', linestyle='--')
    ax1.plot(ratios_std.index, ratios_std['aem_leverage'], label='AEM Leverage', marker='^', linestyle='-.')
    ax1.set_ylabel('Standardized Financial Ratios')
    ax1.set_title('Standardized Financial Ratios Over Time')
    ax1.legend(loc='best')
    
    # Lower subplot: macroeconomic variables
    ax2.plot(macro_std.index, macro_std['e/p'], label='E/P', marker='o')
    ax2.plot(macro_std.index, macro_std['unemp_rate'], label='Unemployment Rate', marker='s', linestyle='--')
    ax2.plot(macro_std.index, macro_std['real_gdp'], label='Real GDP', marker='^', linestyle='-.')
    ax2.plot(macro_std.index, macro_std['mkt_ret'], label='Market Excess Return', marker='d', linestyle=':')
    ax2.plot(macro_std.index, macro_std['mkt_vol'], label='Market Volatility', marker='v', linestyle='-')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Standardized Macro Variables')
    ax2.set_title('Standardized Macroeconomic Variables Over Time')
    ax2.legend(loc='best')
    
    ax2.xaxis.set_major_locator(mdates.YearLocator(10))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    outfile = config.OUTPUT_DIR / ("updated_table03_figure03.png" if UPDATED else "table03_figure03.png")
    plt.savefig(outfile)
    plt.close()

if __name__ == "__main__":
    # For testing purposes, individual plotting functions can be called here.
    pass
