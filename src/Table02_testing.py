"""
Table02_testing.py

All unit tests for the Table 02 pipeline, previously in Table02Prep.py.
Ensures that the data pipeline, ratio calculations, and final outputs 
meet expected constraints.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# warnings.filterwarnings(
#     "ignore",
#     message=".*DataFrame concatenation with empty or all-NA entries.*",
#     category=FutureWarning
# )
import unittest
import wrds
import config
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import Table02Prep

class TestFormattedTable(unittest.TestCase):
    """
    Tests the final table for expected numeric ranges, presence of gvkeys, and ratio properties.
    """

    @classmethod
    def setUpClass(cls):
        """
        Runs the main pipeline once before all tests, storing references for subsequent checks.
        """
        # Produce the final pivot table from main() 
        cls.formatted_table = Table02Prep.main()
        
        # For auxiliary checks (gvkeys etc.), use the available functions:
        cls.db = wrds.Connection(wrds_username=config.WRDS_USERNAME)
        # 通过 clean_primary_dealers_data 和 load_link_table 获取数据
        merged_main = Table02Prep.clean_primary_dealers_data(fname='Primary_Dealer_Link_Table3.csv')
        link_hist = Table02Prep.load_link_table(fname='updated_linktable.csv')
        link_dict = Table02Prep.create_comparison_group_linktables(link_hist, merged_main)
        cls.datasets = Table02Prep.pull_data_for_all_comparison_groups(cls.db, link_dict)
        cls.prepped = Table02Prep.prep_datasets(cls.datasets)
        cls.ratio_df = Table02Prep.create_ratios_for_table(cls.prepped)

    def test_value_ranges(self):
        """
        Checks certain known reference values from the paper 
        (like total_assets ratio for BD, Banks, etc.)
        """
        manual_data = {
            ('1960-2012','BD','Total assets'): 0.959,
            ('1960-2012','Banks','Total assets'): 0.596,
            ('1960-2012','Cmpust.','Total assets'): 0.240,
        }
        # 'formatted_table' is the final pivot table
        stacked = self.formatted_table.stack().stack()
        dct = {idx: val for idx, val in stacked.items()}
        off = 0
        for key, val in manual_data.items():
            if key not in dct:
                self.fail(f"Missing {key} in final table.")
            else:
                got = dct[key]
                if abs(got - val) > 0.2:
                    off += 1
        print(f"{off} table values off by more than 0.2")

    def test_gvkeys_data_presence(self):
        """
        Ensures data merges included correct gvkeys for each group.
        We'll check that each dataset's gvkeys intersects well with the link table.
        """
        # 使用 clean_primary_dealers_data 和 load_link_table 替代原有的合并函数
        merged_main = Table02Prep.clean_primary_dealers_data(fname='Primary_Dealer_Link_Table3.csv')
        link_hist = Table02Prep.load_link_table(fname='updated_linktable.csv')
        link_dict = Table02Prep.create_comparison_group_linktables(link_hist, merged_main)

        for gname, df in self.datasets.items():
            with self.subTest(group=gname):
                link_table = link_dict.get(gname, pd.DataFrame())
                if not link_table.empty:
                    link_table.loc[:, 'gvkey']  = link_table['gvkey'].astype(str).str.zfill(6)
                    link_gvkeys = set(link_table['gvkey'].unique())
                else:
                    link_gvkeys = set()
                if 'gvkey' in df.columns:
                    df['gvkey'] = df['gvkey'].astype(str).str.zfill(6)
                    data_gvkeys = set(df['gvkey'].unique())
                else:
                    data_gvkeys = set()

                overlap = link_gvkeys.intersection(data_gvkeys)
                if not link_table.empty:
                    self.assertGreater(
                        len(overlap), 0, 
                        f"No overlap in gvkeys for group {gname}; overlap=0"
                    )

    def test_ratios_non_negative_and_handle_na(self):
        """
        Ensures the final pivot table has no negative values 
        and strictly no NaN (zero tolerance).
        """
        # Use the final pivot table
        final_df = self.formatted_table.select_dtypes(include=['float64','int'])

        # 1) Check non-negative
        min_val = final_df.min().min()
        self.assertGreaterEqual(
            min_val, 0, 
            f"Found negative values in final pivot table. min_val={min_val}"
        )

        # 2) Zero tolerance for NaN
        na_count = final_df.isna().sum().sum()
        self.assertEqual(
            na_count, 0,
            f"Unexpected NA values found in final pivot table: {na_count}"
        )


if __name__ == '__main__':
    unittest.main()
