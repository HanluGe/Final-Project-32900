"""
Table03.py

This module reads manually curated datasets for primary dealers and holding companies,
matches them with link history data, calculates key financial ratios (market cap ratio, 
book cap ratio, and AEM leverage defined as broker-dealer book leverage), and converts 
these ratios into analytical factors. It then merges these results with macroeconomic 
variables to produce outputs (tables and figures) for Table 03 in the intermediary asset 
pricing replication.
"""

import pandas as pd
import wrds
import config
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.seasonal import seasonal_decompose

import Table03Load
from Table03Load import quarter_to_date, date_to_quarter
import Table03Analysis
import Table02Prep

def combine_bd_financials(UPDATED=False):
    """
    Combine broker-dealer financial data from historical sources and, if UPDATED, from recent FRED data.
    Input: UPDATED (bool) flag indicating whether to fetch updated data.
    Output: A DataFrame containing combined broker-dealer financial data.
    """
    bd_financials_historical = Table03Load.load_fred_past()
    bd_financials_historical.index = pd.to_datetime(bd_financials_historical.index)
    
    if UPDATED:
        bd_financials_recent = Table03Load.load_bd_financials()  
        bd_financials_recent.index = pd.to_datetime(bd_financials_recent.index)
        start_date = pd.to_datetime(config.END_DATE)
        bd_financials_recent = bd_financials_recent[bd_financials_recent.index > start_date]
        bd_financials_combined = pd.concat([bd_financials_historical, bd_financials_recent])
    else:
        bd_financials_combined = bd_financials_historical
    
    return bd_financials_combined    

def prep_dataset(dataset, UPDATED=False):
    """
    Prepare the raw financial dataset by removing duplicates, converting quarter strings to dates,
    and aggregating key financial columns by quarter.
    Input: dataset (DataFrame) with raw financial data and UPDATED flag.
    Output: Aggregated DataFrame with summed total_assets, book_debt, book_equity, and market_equity,
    merged with broker-dealer data.
    """
    dataset = dataset.drop_duplicates()
    dataset['datafqtr'] = dataset['datafqtr'].apply(quarter_to_date)
    aggregated_dataset = dataset.groupby('datafqtr').agg({
        'total_assets': 'sum',
        'book_debt': 'sum',
        'book_equity': 'sum',
        'market_equity': 'sum'
    }).reset_index()
    
    bd_financials_combined = combine_bd_financials(UPDATED=UPDATED)
    aggregated_dataset = aggregated_dataset.merge(bd_financials_combined, left_on='datafqtr', right_index=True)

    return aggregated_dataset

def calculate_ratios(data):
    """
    Calculate key financial ratios from aggregated data.
    Input: data (DataFrame) with aggregated columns for assets, debt, equity, etc.
    Output: DataFrame with added columns: market_cap_ratio, book_cap_ratio, and aem_leverage.
    market_cap_ratio = market_equity / (book_debt + market_equity), 
    book_cap_ratio = book_equity / (book_debt + book_equity), 
    aem_leverage = bd_fin_assets / (bd_fin_assets - bd_liabilities).
    """
    data['market_cap_ratio'] = data['market_equity'] / (data['book_debt'] + data['market_equity'])
    data['book_cap_ratio'] = data['book_equity'] / (data['book_debt'] + data['book_equity'])
    data['aem_leverage'] = data['bd_fin_assets'] / (data['bd_fin_assets'] - data['bd_liabilities'])
    return data

def aggregate_ratios(data):
    """
    Aggregates the calculated financial ratios and sets the date as the index.
    Input: data (DataFrame) after ratio calculation.
    Output: DataFrame with columns market_cap_ratio, book_cap_ratio, and aem_leverage, with index set to date.
    The function renames 'datafqtr' to 'date' and sets it as the index.
    """
    data = calculate_ratios(data)
    # Directly use aem_leverage without inversion
    data = data[['datafqtr', 'market_cap_ratio', 'book_cap_ratio', 'aem_leverage']]
    data.rename(columns={'datafqtr': 'date'}, inplace=True)
    data = data.set_index('date')
    return data

def convert_ratios_to_factors(data):
    """
    Converts financial ratios into analytical factors.
    Input: data (DataFrame) with financial ratios.
    Output: DataFrame with factors: market_capital_factor, book_capital_factor, and aem_leverage_factor.
    For each ratio, an AR(1) model is fit and the residuals are used to compute the factor.
    The AEM leverage factor is based on the percentage change of raw leverage minus its seasonal component.
    """
    factors_df = pd.DataFrame(index=data.index)

    # AR(1) for market cap ratio
    cleaned_data = data['market_cap_ratio'].dropna()
    model = AutoReg(cleaned_data, lags=1, trend='c')
    model_fitted = model.fit()
    factors_df['innovations_mkt_cap'] = model_fitted.resid
    factors_df['market_capital_factor'] = factors_df['innovations_mkt_cap'] / data['market_cap_ratio'].shift(1)
    factors_df.drop(columns=['innovations_mkt_cap'], inplace=True)

    # AR(1) for book cap ratio
    cleaned_data = data['book_cap_ratio'].dropna()
    model = AutoReg(cleaned_data, lags=1, trend='c')
    model_fitted = model.fit()
    factors_df['innovations_book_cap'] = model_fitted.resid
    factors_df['book_capital_factor'] = factors_df['innovations_book_cap'] / data['book_cap_ratio'].shift(1)
    factors_df.drop(columns=['innovations_book_cap'], inplace=True)

    # Calculate AEM leverage factor based on raw leverage growth (percentage change) and remove seasonal component.
    factors_df['leverage_growth'] = data['aem_leverage'].pct_change().fillna(0)
    decomposition = seasonal_decompose(factors_df['leverage_growth'], model='additive', period=4)
    factors_df['aem_leverage_factor'] = factors_df['leverage_growth'] - decomposition.seasonal

    return factors_df[['market_capital_factor', 'book_capital_factor', 'aem_leverage_factor']]

def calculate_ep(shiller_cape):
    """
    Process Shiller CAPE data to calculate the earnings-to-price (E/P) ratio.
    Input: shiller_cape (DataFrame) with columns for date and CAPE.
    Output: DataFrame with an additional 'e/p' column computed as 1 / CAPE, with date as index.
    The date is parsed using the format '%Y.%m' and adjusted to month end.
    """
    df = shiller_cape.copy()
    df.columns = ['date', 'cape']
    df['date'] = df['date'].astype(str)
    df['date'] = pd.to_datetime(df['date'], format='%Y.%m') + pd.offsets.MonthEnd(0)
    df = df.set_index('date')
    df['e/p'] = 1 / df['cape']
    return df

def macro_variables(db, UPDATED=False):
    """
    Creates a merged DataFrame of quarterly macroeconomic variables.
    Input: WRDS connection object and UPDATED flag.
    Output: A DataFrame containing macro data from FRED, Shiller, Fama-French factors, and CRSP volatility.
    It renames FRED series, resamples to quarterly, and merges all data on the date index.
    """
    import numpy as np
    import pandas as pd
    
    # Load FRED macroeconomic data and rename columns
    macro_data = Table03Load.load_fred_macro_data()
    macro_data = macro_data.rename(columns={'UNRATE': 'unemp_rate',
                                              'NFCI': 'nfci',
                                              'GDPC1': 'real_gdp',
                                              'A191RL1Q225SBEA': 'real_gdp_growth'})
    macro_data.index = pd.to_datetime(macro_data.index)
    macro_data.rename(columns={'DATE': 'date'}, inplace=True)
    macro_quarterly = macro_data.resample('Q').mean()

    # Load Shiller market data, calculate E/P, and resample quarterly
    shiller_cape = Table03Load.load_shiller_pe()
    shiller_ep = calculate_ep(shiller_cape)
    shiller_quarterly = shiller_ep.resample('Q').mean()

    # Determine the end date for FF factors based on the UPDATED flag and load the data
    if UPDATED:
        end_date_ff = config.UPDATED_END_DATE
    else:
        end_date_ff = config.END_DATE
    ff_facs = Table03Load.fetch_ff_factors(start_date=config.START_DATE.replace("-", ""),
                                           end_date=end_date_ff.replace("-", ""))
    ff_facs_quarterly = ff_facs.to_timestamp(freq='M').resample('Q').last()

    # Load the CRSP value-weighted index data and convert the date column to datetime
    value_wtd_indx = Table03Load.pull_CRSP_Value_Weighted_Index(db)
    value_wtd_indx['date'] = pd.to_datetime(value_wtd_indx['date'])
    
    # Compute market volatility using logarithmic returns:
    # Assume vwretd is a return (in decimal form), then the log return is ln(1 + vwretd)
    log_returns = np.log(1 + value_wtd_indx.set_index('date')['vwretd'])
    annual_vol_quarterly = log_returns.groupby(pd.Grouper(freq='Q')).std().rename('mkt_vol')

    # Merge all macroeconomic data
    macro_merged = shiller_quarterly.merge(macro_quarterly, left_index=True, right_index=True, how='left')
    macro_merged = macro_merged.merge(ff_facs_quarterly[['mkt_ret']], left_index=True, right_index=True)
    macro_merged = macro_merged.merge(annual_vol_quarterly, left_index=True, right_index=True)

    return macro_merged

def create_panelA(ratios, macro):
    """
    Creates Panel A for Table 03 by merging financial ratios with macroeconomic variables.
    Input: ratios (DataFrame with financial ratios) and macro (DataFrame with macro data).
    Output: A DataFrame for Panel A with columns for Market capital, Book capital, AEM leverage, and selected macro variables.
    This panel includes data from 1970-01-01 onward.
    """
    ratios_renamed = ratios.rename(columns={
        'market_cap_ratio': 'Market capital',
        'book_cap_ratio': 'Book capital',
        'aem_leverage': 'AEM leverage'
    })
    macro = macro[['e/p', 'unemp_rate', 'nfci', 'real_gdp', 'mkt_ret', 'mkt_vol']]
    macro_renamed = macro.rename(columns={
        'e/p': 'E/P',
        'unemp_rate': 'Unemployment',
        'nfci': 'Financial conditions',
        'real_gdp': 'GDP',
        'mkt_ret': 'Market excess return',
        'mkt_vol': 'Market volatility'
    })
    panelA = ratios_renamed.merge(macro_renamed, left_index=True, right_index=True)
    ordered_columns = ['Market capital', 'Book capital', 'AEM leverage',
                       'E/P', 'Unemployment', 'Financial conditions', 'GDP', 'Market excess return', 'Market volatility']
    panelA = panelA[ordered_columns]
    panelA = panelA.loc['1970-01-01':]
    return panelA

def create_panelB(factors, macro):
    """
    Creates Panel B for Table 03 by merging analytical factors with macroeconomic variable growth rates.
    Input: factors (DataFrame with analytical factors) and macro (DataFrame with macro data).
    Output: A DataFrame for Panel B with growth rate columns for the factors and macro variables.
    Only data from 1970-01-01 onward is included.
    """
    factors_renamed = factors.rename(columns={
        'market_capital_factor': 'Market capital factor',
        'book_capital_factor': 'Book capital factor',
        'aem_leverage_factor': 'AEM leverage factor'
    })
    macro_growth = np.log(macro / macro.shift(1))
    macro_growth = macro_growth.fillna(0)
    macro_growth = macro_growth.loc['1970-01-01':]
    macro_growth['mkt_ret'] = macro['mkt_ret']
    macro_growth_renamed = macro_growth.rename(columns={
        'e/p': 'E/P growth',
        'unemp_rate': 'Unemployment growth',
        'nfci': 'Financial conditions growth',
        'real_gdp': 'GDP growth',
        'mkt_ret': 'Market excess return',
        'mkt_vol': 'Market volatility growth'
    })
    panelB = factors_renamed.merge(macro_growth_renamed, left_index=True, right_index=True)
    ordered_columns = ['Market capital factor', 'Book capital factor', 'AEM leverage factor',
                       'E/P growth', 'Unemployment growth', 'Financial conditions growth', 'GDP growth', 'Market excess return', 'Market volatility growth']
    panelB = panelB[ordered_columns]
    panelB = panelB.loc['1970-01-01':]
    return panelB

def format_correlation_matrix(corr_matrix):
    """
    Formats the given correlation matrix by masking its lower triangle.
    Input: corr_matrix (DataFrame) containing correlation coefficients.
    Output: The same DataFrame with the lower triangle masked (set to NaN) for better readability.
    """
    corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=0).astype(bool))
    return corr_matrix

def calculate_correlation_panelA(panelA):
    """
    Calculates pairwise correlations for Panel A (levels) data.
    Input: panelA (DataFrame) containing levels of financial ratios and macro variables.
    Output: A correlation DataFrame showing correlations between the main ratios and each macro variable.
    It computes the correlations among the first three columns and then with each macro variable.
    """
    correlation_panelA = format_correlation_matrix(panelA.iloc[:, :3].corr())
    main_cols = panelA[['Market capital', 'Book capital', 'AEM leverage']]
    other_cols = panelA[['E/P', 'Unemployment', 'GDP', 'Financial conditions', 'Market volatility']]
    correlation_results_panelA = pd.DataFrame(index=main_cols.columns)
    for column in other_cols.columns:
        correlation_results_panelA[column] = main_cols.corrwith(other_cols[column])
    return pd.concat([correlation_panelA, correlation_results_panelA.T], axis=0)

def calculate_correlation_panelB(panelB):
    """
    Calculates pairwise correlations for Panel B (factor growth rates) data.
    Input: panelB (DataFrame) containing analytical factor growth rates and macro variable growth rates.
    Output: A correlation DataFrame showing correlations between factor growth rates and macro growth rates.
    It computes the upper triangle and then correlations between the first three columns and the remaining columns.
    """
    correlation_panelB = format_correlation_matrix(panelB.iloc[:, :3].corr())
    main_cols = panelB[['Market capital factor', 'Book capital factor', 'AEM leverage factor']]
    other_cols = panelB[['Market excess return', 'E/P growth', 'Unemployment growth', 'GDP growth', 'Financial conditions growth', 'Market volatility growth']]
    correlation_results_panelB = pd.DataFrame(index=main_cols.columns)
    for column in other_cols.columns:
        correlation_results_panelB[column] = main_cols.corrwith(other_cols[column])
    return pd.concat([correlation_panelB, correlation_results_panelB.T], axis=0)

def format_final_table(corrA, corrB):
    """
    Formats the final correlation table by merging Panel A and Panel B correlation data.
    Input: corrA and corrB (DataFrames) for Panel A and Panel B respectively.
    Output: A combined DataFrame with a MultiIndex for columns for clearer presentation.
    The function reorders and labels columns and rows.
    """
    panelB_renamed = corrB.copy()
    panelB_renamed.columns = corrA.columns
    panelB_column_names = pd.DataFrame([corrB.columns], columns=corrA.columns)
    panelB_column_names.reset_index(drop=True, inplace=True)
    panelB_combined = pd.concat([panelB_column_names, panelB_renamed])
    panelA_title = pd.DataFrame({col: ["Panel A: Correlations of Levels"] for col in corrA.columns})
    panelB_title = pd.DataFrame({col: ["Panel B: Correlations of Factors"] for col in corrA.columns})
    full_table = pd.concat([panelA_title, corrA, panelB_title, panelB_combined])
    return full_table

def convert_and_export_tables_to_latex(corrA, corrB, UPDATED=False):
    """
    Converts correlation tables to LaTeX format and exports the result as a .tex file.
    Input: corrA and corrB (DataFrames) and an UPDATED flag.
    Output: A LaTeX file saved in the directory specified by config.OUTPUT_DIR.
    The function rounds values, formats columns, and writes the LaTeX table to disk.
    """
    corrA = corrA.round(2).fillna('')
    corrB = corrB.round(2).fillna('')
    caption = "Updated" if UPDATED else "Original"
    column_format = 'l' + 'c' * (len(corrA.columns))
    header_row = " & " + " & ".join(corrA.columns) + " \\\\"
    panelA_rows = "\n".join([f"{index} & " + " & ".join(corrA.loc[index].astype(str)) + " \\\\" for index in corrA.index])
    panelB_rows = "\n".join([f"{index} & " + " & ".join(corrB.loc[index].astype(str)) + " \\\\" for index in corrB.index])
    full_latex = rf"""
    \begin{{table}}[htbp]
    \centering
    \caption{{\label{{tab:correlation}}{caption}}}
    \begin{{adjustbox}}{{max width=\textwidth}}
    \small
    \begin{{tabular}}{{{column_format}}}
        \toprule
        Panel A: Correlation of Levels \\
        \midrule
        {header_row}
        \midrule
        {panelA_rows}
        \midrule
        Panel B: Correlations of Factors \\
        \midrule
        {header_row}
        \midrule
        {panelB_rows}
        \bottomrule
    \end{{tabular}}
    \end{{adjustbox}}
    \end{{table}}
    """
    outfile = config.OUTPUT_DIR / ("updated_table03.tex" if UPDATED else "table03.tex")
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(full_latex)

def main(UPDATED=False):
    """
    Main function to execute the entire data processing pipeline for Table 03.
    Input: UPDATED (bool) flag to determine if updated data should be used.
    Output: Generates and exports a formatted correlation table in LaTeX format.
    The function connects to WRDS, processes primary dealer data, calculates ratios and factors,
    merges with macro variables, and exports summary statistics, figures, and correlation matrices.
    """
    db = wrds.Connection(wrds_username=config.WRDS_USERNAME)
    prim_dealers, _ = Table02Prep.prim_deal_merge_manual_data_w_linktable(UPDATED=UPDATED)
    dataset, _ = Table03Load.fetch_data_for_tickers(prim_dealers, db)
    # Use the local prep_dataset defined in Table03.py (not in Table03Analysis)
    prep_datast = prep_dataset(dataset, UPDATED=UPDATED)
    ratio_dataset = aggregate_ratios(prep_datast)
    factors_dataset = convert_ratios_to_factors(ratio_dataset)
    macro_dataset = macro_variables(db, UPDATED=UPDATED)
    panelA = create_panelA(ratio_dataset, macro_dataset)
    panelB = create_panelB(factors_dataset, macro_dataset)
    Table03Analysis.create_summary_stat_table_for_data(panelB, UPDATED=UPDATED)
    Table03Analysis.plot_figure02(ratio_dataset, UPDATED=UPDATED)
    # Generate Figure 3 using standardized data for both financial ratios and macro variables
    Table03Analysis.plot_figure03(ratio_dataset, macro_dataset, UPDATED=UPDATED)
    correlation_panelA = calculate_correlation_panelA(panelA)
    correlation_panelB = calculate_correlation_panelB(panelB)
    formatted_table = format_final_table(correlation_panelA, correlation_panelB)
    convert_and_export_tables_to_latex(correlation_panelA, correlation_panelB, UPDATED=UPDATED)
    print(formatted_table.style.format(na_rep=''))
    
if __name__ == "__main__":
    main(UPDATED=False)
    print("Table 03 has been created and exported to LaTeX format.")
