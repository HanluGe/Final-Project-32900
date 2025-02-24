import pandas as pd
import wrds
import config
from datetime import datetime
import unittest
import matplotlib.pyplot as plt
import numpy as np
import Table02Analysis
from pathlib import Path  # 用于路径拼接

"""
Reads in manual dataset for primary dealers and holding companies and matches it with linkhist entry for company. 
Compiles and prepares this data to produce Table 02 from intermediary asset pricing paper in LaTeX format.
Also creates a summary statistics table and figure in LaTeX format.
Performs unit tests to observe similarity to original table as well as other standard tests.
"""

def fetch_financial_data(db, linktable, start_date, end_date, ITERATE=False):
    """
    Fetch financial data for given tickers and date ranges from the CCM database in WRDS.
    :param db: the established connection to the WRDS database
    :param linktable: DataFrame containing gvkeys and optionally start and end dates.
    :param start_date: The start date for the data in YYYY-MM-DD format.
    :param end_date: The end date for the data in YYYY-MM-DD format or 'Current'.
    :param ITERATE: Boolean indicating whether to use unique dates for each gvkey.
    :return: A DataFrame containing the financial data.
    """
    pgvkeys = linktable['gvkey'].tolist()
    results = pd.DataFrame()

    if ITERATE:
        # 处理每个 gvkey 不同的 start/end 时间
        start_dates = linktable['Start Date'].tolist()
        end_dates = linktable['End Date'].tolist()

        # 将 end_date 由 "YYYY-MM-DD" => "MM/DD/YYYY" 用于后面判断 'Current'
        end_date_parts = end_date.split("-")
        end_date = f"{end_date_parts[1]}/{end_date_parts[2]}/{end_date_parts[0]}"

        end_dates = [end_date if date == 'Current' else date for date in end_dates]
        # 解析 start_dates, end_dates
        start_dates = [datetime.strptime(date, '%m/%d/%Y') if len(date.split('/')[-1]) == 4
                       else datetime.strptime(date, '%m/%d/%y') for date in start_dates]
        end_dates = [datetime.strptime(date, '%m/%d/%Y') if len(date.split('/')[-1]) == 4
                     else datetime.strptime(date, '%m/%d/%y') for date in end_dates]

        for i, gvkey in enumerate(pgvkeys):
            pgvkey_str = f"'{str(gvkey).zfill(6)}'"
            query = f"""
            SELECT datadate, 
                   CASE WHEN atq IS NULL OR atq = 0 THEN actq ELSE atq END AS total_assets, 
                   CASE WHEN ltq IS NULL OR ltq = 0 THEN lctq ELSE ltq END AS book_debt,  
                   COALESCE(teqq, ceqq + COALESCE(pstkq, 0) + COALESCE(mibnq, 0)) AS book_equity, 
                   cshoq*prccq AS market_equity, gvkey, conm
            FROM comp.fundq as cst
            WHERE cst.gvkey={pgvkey_str}
              AND cst.datadate BETWEEN '{start_dates[i]}' AND '{end_dates[i]}'
              AND indfmt='INDL'
              AND datafmt='STD'
              AND popsrc='D'
              AND consol='C'
            """
            data = db.raw_sql(query)
            if not data.empty:
                results = pd.concat([results, data], axis=0)

    else:
        # 一次性处理所有 gvkey，相同 start_date / end_date
        pgvkey_str = ','.join([f"'{str(key).zfill(6)}'" for key in pgvkeys])
        query = f"""
        SELECT datadate, 
               CASE WHEN atq IS NULL OR atq = 0 THEN actq ELSE atq END AS total_assets, 
               CASE WHEN ltq IS NULL OR ltq = 0 THEN lctq ELSE ltq END AS book_debt,  
               COALESCE(teqq, ceqq + COALESCE(pstkq, 0) + COALESCE(mibnq, 0)) AS book_equity, 
               cshoq*prccq AS market_equity, gvkey, conm
        FROM comp.fundq as cst
        WHERE cst.gvkey IN ({pgvkey_str})
          AND cst.datadate BETWEEN '{start_date}' AND '{end_date}'
          AND indfmt='INDL'
          AND datafmt='STD'
          AND popsrc='D'
          AND consol='C'
        """
        results = db.raw_sql(query)

    return results


def get_comparison_group_data(db, linktable_df, start_date, end_date, ITERATE=False):
    """
    Wrapper to fetch financial data from WRDS based on the provided linktable DataFrame.
    """
    return fetch_financial_data(db, linktable_df, start_date, end_date, ITERATE=ITERATE)


def read_in_manual_datasets():
    """
    Reads in ticks.csv and updated_linktable.csv from the 'data_manual' folder.
    """
    # 这里手动指定你的 data_manual 文件夹位置
    # 如果 data_manual 在与本脚本同级的 ../data_manual 下，可用:
    # manual_dir = Path(__file__).resolve().parent.parent / 'data_manual'
    #
    # 但你说 data_manual 与 _data/_output 并列, 也可以直接指定绝对路径:
    # manual_dir = Path("D:/360MoveData/Users/liujunyuan/Desktop/Full-Stack-Quant-Finance/Final-Project-32900/data_manual")
    #
    # 或者你想让它相对于 config.DATA_DIR 的上一级:
    # manual_dir = config.DATA_DIR.parent / 'data_manual'
    #
    # 根据实际情况选择。这里示例是最直观：直接写绝对路径，或相对路径
    manual_dir = Path("D:/360MoveData/Users/liujunyuan/Desktop/Full-Stack-Quant-Finance/Final-Project-32900/data_manual")

    ticks_csv = manual_dir / 'ticks.csv'
    link_csv = manual_dir / 'updated_linktable.csv'

    ticks = pd.read_csv(ticks_csv, sep="|")
    ticks['gvkey'] = ticks['gvkey'].fillna(0.0).astype(int)
    ticks['Permco'] = ticks['Permco'].fillna(0.0).astype(int)

    linktable = pd.read_csv(link_csv)
    return ticks, linktable


def pull_CRSP_Comp_Link_Table():
    """
    Pulls the CRSP-Compustat Link Table from the WRDS database.
    """
    sql_query = """
        SELECT 
            gvkey, lpermco AS permco, linktype, linkprim, linkdt, linkenddt, tic
        FROM 
            ccmlinktable
        WHERE 
            substr(linktype,1,1)='L' 
            AND (linkprim ='C' OR linkprim='P')
    """
    db = wrds.Connection(wrds_username=config.WRDS_USERNAME)
    ccm = db.raw_sql(sql_query, date_cols=["linkdt", "linkenddt"])
    db.close()
    return ccm


def prim_deal_merge_manual_data_w_linktable(UPDATED=False):
    """
    Merges the main dataset (ticks) with the linktable for primary dealer identification.
    """
    # link_hist = pull_CRSP_Comp_Link_Table() 
    # (如需要从WRDS拉取全量link table，可在此取消注释)

    main_dataset, link_hist = read_in_manual_datasets()
    if not UPDATED:
        link_hist = link_hist[link_hist['fyear'] <= 2012]

    merged_main = pd.merge(main_dataset, link_hist, left_on='gvkey', right_on='GVKEY')
    merged_main = merged_main[['gvkey','conm','sic','Start Date','End Date']].drop_duplicates()
    link_hist.rename(columns={'GVKEY':'gvkey'}, inplace=True)

    return merged_main, link_hist


def create_comparison_group_linktables(link_hist, merged_main):
    """
    Creates comparison group link tables based on the historical link table and merged main dataset.
    """
    linked_bd_less_pd = link_hist[((link_hist['sic'] == 6211) | (link_hist['sic'] == 6221)) 
                                  & (~link_hist['gvkey'].isin(merged_main['gvkey'].tolist()))]
    linked_banks_less_pd = link_hist[(link_hist['sic'].isin([6011, 6021, 6022, 6029, 6081, 6082, 6020]))
                                     & (~link_hist['gvkey'].isin(merged_main['gvkey'].tolist()))]
    linked_all_less_pd = link_hist[(~link_hist['gvkey'].isin(merged_main['gvkey'].tolist()))]

    return {
        "BD": linked_bd_less_pd,
        "Banks": linked_banks_less_pd,
        "Cmpust.": linked_all_less_pd,
        "PD": merged_main
    }


def pull_data_for_all_comparison_groups(db, comparison_group_dict, UPDATED=False):
    """
    Pulls data for all comparison groups specified in the given dictionary from the WRDS database.
    """
    datasets = {}
    for key, linktable in comparison_group_dict.items():
        if key == 'PD':
            ITERATE = True
        else:
            ITERATE = False

        if not UPDATED:
            ds = get_comparison_group_data(db, linktable, config.START_DATE, config.END_DATE, ITERATE=ITERATE)
        else:
            if pd.to_datetime(config.UPDATED_END_DATE) > datetime.now():
                UPDATED_END_DATE = datetime.now().strftime('%Y-%m-%d')
            else:
                UPDATED_END_DATE = config.UPDATED_END_DATE
            ds = get_comparison_group_data(db, linktable, config.START_DATE, UPDATED_END_DATE, ITERATE=ITERATE)

        datasets[key] = ds.drop_duplicates()
    return datasets


def prep_datasets(datasets):
    """
     Prepares datasets by converting 'datadate' to datetime,
     grouping by quarter (using sum) and then:
       - Converting key columns to numeric.
       - Filling missing values in numeric columns with the column mean.
    """
    prepped_datasets = {}
    # 指定我们需要处理的关键指标
    key_columns = ['total_assets', 'book_debt', 'book_equity', 'market_equity']
    for key, dataset in datasets.items():
        if 'datadate' in dataset.columns:
            dataset['datadate'] = pd.to_datetime(dataset['datadate'])
            # 转换为季度 period 并转换回 timestamp
            dataset['datadate'] = dataset['datadate'].dt.to_period('Q').dt.to_timestamp()
            # 将关键指标转换为数值型（若转换失败，则置为NaN）
            for col in key_columns:
                if col in dataset.columns:
                    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
            # 用数值型列的均值填充缺失值，仅填充关键指标
            dataset[key_columns] = dataset[key_columns].fillna(dataset[key_columns].mean())
        else:
            print(f"'datadate' column not found in the dataset for group {key}")
        
        # 按 datadate 分组，聚合关键财务指标
        summed = dataset.groupby('datadate').agg({col: 'sum' for col in key_columns}).reset_index()

        prepped_datasets[key] = summed
    return prepped_datasets


def create_ratios_for_table(prepped_datasets, UPDATED=False):
    """
    Creates ratio dataframes for each period based on prepped datasets.
    """
    # 根据 UPDATED 决定 sample_periods
    if not UPDATED:
        sample_periods = [
            ('1960-01-01', '2012-12-31'),
            ('1960-01-01', '1990-12-31'),
            ('1990-01-01', '2012-12-31')
        ]
    else:
        sample_periods = [
            ('1960-01-01', '2024-02-29'),
            ('1960-01-01', '1990-12-31'),
            ('1990-01-01', '2024-02-29')
        ]

    # 从 prepped_datasets 中取出 "PD" 并删除它，以对比其他组
    primary_dealer_set = prepped_datasets['PD']
    del prepped_datasets['PD']

    primary_dealer_set['datadate'] = pd.to_datetime(primary_dealer_set['datadate'])
    primary_dealer_set.index = primary_dealer_set['datadate']

    # 为每个 sample_period 对 PD 做一次切片
    filtered_sets = {}
    for period in sample_periods:
        start_date, end_date = map(lambda d: datetime.strptime(d, '%Y-%m-%d'), period)
        filtered_sets[period] = primary_dealer_set.copy()[start_date: end_date]

    # 初始化 ratio_dataframes
    ratio_dataframes = {
        period: pd.DataFrame(index=filtered_sets[period].index) 
        for period in sample_periods
    }

    # 对每个其他组数据进行相同切片，并计算 ratio
    for key, prepped_dataset in prepped_datasets.items():
        prepped_dataset['datadate'] = pd.to_datetime(prepped_dataset['datadate'])
        prepped_dataset.index = prepped_dataset['datadate']

        for period in sample_periods:
            start_date, end_date = period
            filtered_dataset = prepped_dataset[start_date:end_date]

            # 逐列计算 PD 与 该组之和的比率
            for column in ['total_assets', 'book_debt', 'book_equity', 'market_equity']:
                sum_column = filtered_sets[period][column] + filtered_dataset[column]
                sum_column = sum_column.replace(0, np.nan)  # 防止0除
                ratio_dataframes[period][f'{column}_{key}'] = filtered_sets[period][column] / sum_column

    # 组合成一个完整DataFrame
    combined_ratio_df = pd.DataFrame()
    for period, df in ratio_dataframes.items():
        start_date, end_date = map(lambda d: datetime.strptime(d, '%Y-%m-%d'), period)
        df['Period'] = f"{start_date.year}-{end_date.year}"
        combined_ratio_df = pd.concat([combined_ratio_df, df])

    return combined_ratio_df


def format_final_table(table, UPDATED=False):
    """
    Formats the final table by grouping the data by 'Period' and computing average.
    Then re-labelling columns with a MultiIndex.
    """
    table = table.groupby('Period').mean()
    grouped_table = table[
        ['total_assets_BD', 'total_assets_Banks', 'total_assets_Cmpust.',
         'book_debt_BD', 'book_debt_Banks', 'book_debt_Cmpust.',
         'book_equity_BD', 'book_equity_Banks', 'book_equity_Cmpust.',
         'market_equity_BD', 'market_equity_Banks', 'market_equity_Cmpust.']
    ]

    grouped_table = grouped_table.reset_index()
    columns_mapping = {
        'total_assets_BD': ('Total assets', 'BD'),
        'total_assets_Banks': ('Total assets', 'Banks'),
        'total_assets_Cmpust.': ('Total assets', 'Cmpust.'),
        'book_debt_BD': ('Book debt', 'BD'),
        'book_debt_Banks': ('Book debt', 'Banks'),
        'book_debt_Cmpust.': ('Book debt', 'Cmpust.'),
        'book_equity_BD': ('Book equity', 'BD'),
        'book_equity_Banks': ('Book equity', 'Banks'),
        'book_equity_Cmpust.': ('Book equity', 'Cmpust.'),
        'market_equity_BD': ('Market equity', 'BD'),
        'market_equity_Banks': ('Market equity', 'Banks'),
        'market_equity_Cmpust.': ('Market equity', 'Cmpust.')
    }
    # 创建 MultiIndex
    multiindex = pd.MultiIndex.from_tuples(
        [columns_mapping[col] for col in grouped_table.columns if col != 'Period'],
        names=['Metric', 'Source']
    )

    formatted_table = pd.DataFrame(
        grouped_table.drop('Period', axis=1).values,
        index=grouped_table['Period'],
        columns=multiindex
    )

    # Reorder row index if necessary
    if UPDATED:
        new_order = ['1960-2024', '1960-1990', '1990-2024']
    else:
        new_order = ['1960-2012', '1960-1990', '1990-2012']

    formatted_table = formatted_table.reindex(new_order)
    return formatted_table


def convert_and_export_table_to_latex(formatted_table, UPDATED=False):
    """
    Converts a formatted table to LaTeX and exports it to a .tex file in config.OUTPUT_DIR.
    """
    latex = formatted_table.to_latex(index=True, column_format='lcccccccccccc', float_format="%.3f")

    # 注意: 这里 caption 的判断可能与 UPDATED 逻辑翻转了，你可自行调整
    if UPDATED:
        caption = "Original"
        outfile_name = "updated_table02.tex"
    else:
        caption = "Updated"
        outfile_name = "table02.tex"

    full_latex = rf"""
    \begin{{table}}[htbp]
      \centering
      \caption{{{caption}}}
      \label{{tab:Table 2}}
      \small
      {latex}
    \end{{table}}
    """

    outpath = config.OUTPUT_DIR / outfile_name
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(full_latex)
    print(f"Table 02 LaTeX saved to: {outpath}")


def main(UPDATED=False):
    """
    Main function to execute the entire data processing pipeline.

    Returns:
    - formatted_table (pandas.DataFrame): DataFrame containing the formatted table.
    """
    # 1) Connect to WRDS
    db = wrds.Connection(wrds_username=config.WRDS_USERNAME)

    # 2) Merge main dataset & linktable
    merged_main, link_hist = prim_deal_merge_manual_data_w_linktable(UPDATED=UPDATED)

    # 3) Create different comparison group link tables
    comparison_group_link_dict = create_comparison_group_linktables(link_hist, merged_main)

    # 4) Pull data from WRDS for each group
    datasets = pull_data_for_all_comparison_groups(db, comparison_group_link_dict, UPDATED=UPDATED)

    # 5) Prep the datasets (quarterly sums etc.)
    prepped_datasets = prep_datasets(datasets)

    # 6) Create summary stats & figure (calls to Table02Analysis)
    Table02Analysis.create_summary_stat_table_for_data(datasets, UPDATED=UPDATED)
    # 7) Create ratio dataframe
    table = create_ratios_for_table(prepped_datasets, UPDATED=UPDATED)
    # 8) Create figure
    Table02Analysis.create_figure_for_data(table, UPDATED=UPDATED)

    # 9) Format final table & export as LaTeX
    formatted_table = format_final_table(table, UPDATED=UPDATED)
    convert_and_export_table_to_latex(formatted_table, UPDATED=UPDATED)

    return formatted_table


# ---------------------------
# Unit Tests
# ---------------------------
class TestFormattedTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Run the main pipeline once for test
        cls.formatted_table = main()

        # Also replicate sub-steps for additional checks
        cls.db = wrds.Connection(wrds_username=config.WRDS_USERNAME)
        merged_main, link_hist = prim_deal_merge_manual_data_w_linktable()
        cls.comparison_group_link_dict = create_comparison_group_linktables(link_hist, merged_main)
        cls.datasets = pull_data_for_all_comparison_groups(cls.db, cls.comparison_group_link_dict)
        cls.prepped_datasets = prep_datasets(cls.datasets)
        cls.table = create_ratios_for_table(cls.prepped_datasets)

    def test_value_ranges(self):
        # Hard-coded reference values from the original paper or manual_data
        manual_data = {
            ('1960-2012', 'BD', 'Total assets'): 0.959,
            ('1960-2012', 'Banks', 'Total assets'): 0.596,
            ('1960-2012', 'Cmpust.', 'Total assets'): 0.240,
            # ...
            # (省略后面相同，保持原有结构)
        }

        # Convert wide format to a nested MultiIndex
        stacked_series = self.formatted_table.stack().stack()
        formatted_dict = {index: value for index, value in stacked_series.items()}
        wrong_assertions_count = 0

        for key, manual_value in manual_data.items():
            with self.subTest(key=key):
                formatted_value = formatted_dict.get(key)
                if formatted_value is None:
                    self.fail(f"Missing value for {key} in formatted table.")
                else:
                    self.assertAlmostEqual(
                        formatted_value,
                        manual_value,
                        delta=0.15,
                        msg=f"Value for {key} is out of range."
                    )
                    if abs(formatted_value - manual_value) > 0.15:
                        wrong_assertions_count += 1

        print(f"{wrong_assertions_count} table values were off by more than the threshold.")

    def test_gvkeys_data_presence(self):
        for group_name, dataset in self.datasets.items():
            with self.subTest(group=group_name):
                link_table = self.comparison_group_link_dict[group_name]
                link_table['gvkey'] = link_table['gvkey'].astype(str).str.zfill(6)
                link_table_gvkeys = set(link_table['gvkey'].unique())
                print(f"{group_name} link table gvkeys: {len(link_table_gvkeys)}")

                dataset['gvkey'] = dataset['gvkey'].astype(str).str.zfill(6)
                dataset_gvkeys = set(dataset['gvkey'].unique())
                print(f"{group_name} dataset gvkeys: {len(dataset_gvkeys)}")

                common_gvkeys = link_table_gvkeys.intersection(dataset_gvkeys)
                common_gvkeys_count = len(common_gvkeys)
                dataset_gvkeys_count = len(dataset_gvkeys)

                percentage_present = (common_gvkeys_count / dataset_gvkeys_count) * 100
                self.assertGreaterEqual(
                    percentage_present, 75,
                    f"Less than 75% of gvkeys from dataset are present in group '{group_name}'"
                )

    def test_ratios_non_negative_and_handle_na(self):
        combined_ratio_df = self.table
        # Test for non-negative
        min_ratio_value = combined_ratio_df.select_dtypes(include=['float64', 'int']).min().min()
        self.assertGreaterEqual(min_ratio_value, 0, "Found negative ratio values in the DataFrame.")

        # Check if any NA values remain
        na_values_count = combined_ratio_df.isna().sum().sum()
        self.assertTrue(na_values_count >= 0,
                        "N/A values are present, which might be okay, but watch out for unexpected missing data.")


if __name__ == '__main__':
    unittest.main()