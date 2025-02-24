import pandas as pd
import pandas_datareader
import config
from pathlib import Path

# 使用 config.DATA_DIR 作为数据目录
DATA_DIR = Path(config.DATA_DIR)

def load_fred(
    data_dir=DATA_DIR,
    from_cache=True,
    save_cache=False,
    start="1913-01-01",
    end="2023-10-01",
):
    """
    从 FRED 拉取 CPI、GDP 和 GDPC1 数据。
    如果 from_cache 为 True，则从缓存文件中读取（缓存文件为 data_dir/pulled/fred.parquet）。
    否则，从网络拉取数据，并在 save_cache 为 True 时保存缓存。
    """
    if from_cache:
        file_path = data_dir / "pulled" / "fred.parquet"
        df = pd.read_parquet(file_path)
    else:
        df = pandas_datareader.get_data_fred(
            ["CPIAUCNS", "GDP", "GDPC1"], start=start, end=end
        )
        if save_cache:
            file_dir = data_dir / "pulled"
            file_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(file_dir / "fred.parquet")
    return df

def resample_quarterly(df):
    """
    将数据重采样为季度频率。
    """
    df = df.resample('Q').mean()
    return df

macro_series_descriptions = {
    'UNRATE': 'Unemployment Rate (Seasonally Adjusted)',
    'NFCI': 'Chicago Fed National Financial Conditions Index',
    'GDPC1': 'Real Gross Domestic Product',
    'A191RL1Q225SBEA': 'Real Gross Domestic Product Growth',
}

fred_bd_series_descriptions = {
    'BOGZ1FL664090005Q': 'Security Brokers and Dealers; Total Financial Assets, Level',
    'BOGZ1FL664190005Q': 'Security Brokers and Dealers; Total Liabilities, Level',
}

def pull_fred_macro_data(data_dir=DATA_DIR, start="1969-01-01", end="2024-02-29"):
    """
    从 FRED 拉取宏观经济数据（UNRATE, NFCI, GDPC1, A191RL1Q225SBEA），
    并保存为 parquet 文件到 data_dir/pulled/fred_macro.parquet。
    """
    try:
        series_keys = list(macro_series_descriptions.keys())
        df = pandas_datareader.data.get_data_fred(series_keys, start=start, end=end)
        file_dir = data_dir / "pulled"
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / "fred_macro.parquet"
        df.to_parquet(file_path)
        print(f"Data pulled and saved to {file_path}")
    except Exception as e:
        print(f"Failed to pull or save FRED macro data: {e}")

def load_fred_macro_data(data_dir=DATA_DIR, from_cache=True, start="1969-01-01", end="2024-02-29"):
    """
    加载 FRED 宏观数据。如果缓存存在（data_dir/pulled/fred_macro.parquet），则直接读取，
    否则调用 pull_fred_macro_data 拉取并保存后再读取。
    """
    file_path = data_dir / "pulled" / "fred_macro.parquet"
    try:
        if from_cache and file_path.exists():
            df = pd.read_parquet(file_path)
            print("Loaded macro data from cache.")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Cache not found for macro data, pulling data...")
        pull_fred_macro_data(data_dir=data_dir, start=start, end=end)
        df = pd.read_parquet(file_path)
    return df

def pull_fred_bd_data(data_dir=DATA_DIR, start="1969-01-01", end="2024-02-29"):
    """
    从 FRED 拉取证券经纪商和交易商的资产与负债数据，
    并保存为 parquet 文件到 data_dir/pulled/fred_bd.parquet。
    """
    try:
        series_keys = list(fred_bd_series_descriptions.keys())
        df = pandas_datareader.data.get_data_fred(series_keys, start=start, end=end)
        file_dir = data_dir / "pulled"
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / "fred_bd.parquet"
        df.to_parquet(file_path)
        print(f"Data pulled and saved to {file_path}")
    except Exception as e:
        print(f"Failed to pull or save FRED BD data: {e}")

def load_fred_bd_data(data_dir=DATA_DIR, from_cache=True, start="1969-01-01", end="2024-02-29"):
    """
    加载证券经纪商和交易商的 BD 数据。如果缓存存在，则直接读取，
    否则拉取数据后保存并读取。
    """
    file_path = data_dir / "pulled" / "fred_bd.parquet"
    try:
        if from_cache and file_path.exists():
            df = pd.read_parquet(file_path)
            print("Loaded BD data from cache.")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Cache not found for BD data, pulling data...")
        pull_fred_bd_data(data_dir=data_dir, start=start, end=end)
        df = pd.read_parquet(file_path)
    return df

def demo():
    df = load_fred()

if __name__ == "__main__":
    # Pull and save cache of FRED data
    _ = load_fred(start="1913-01-01", end="2023-10-01", data_dir=DATA_DIR, from_cache=False, save_cache=True)
    
    # Pull and save cache of macroeconomic data
    _ = load_fred_macro_data(start="1969-01-01", end="2024-01-01", data_dir=DATA_DIR, from_cache=False)
