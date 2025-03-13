import pandas as pd
import pandas_datareader
import config
from pathlib import Path
from datetime import datetime

# Use config.DATA_DIR as the data directory
DATA_DIR = Path(config.DATA_DIR)

def load_fred(
    data_dir=DATA_DIR,
    from_cache=True,
    save_cache=False,
    start=config.START_DATE,
    end=None,
):
    """
    Fetch CPI, GDP, and GDPC1 data from FRED.
    If from_cache is True, read from a cached file (data_dir/pulled/fred.parquet).
    Otherwise, pull data from the web and save it if save_cache is True.
    """
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d') 
        
    if from_cache:
        file_path = data_dir / "fred" / "fred.parquet"
        df = pd.read_parquet(file_path)
    else:
        df = pandas_datareader.get_data_fred(
            ["CPIAUCNS", "GDP", "GDPC1"], start=start, end=end
        )
        if save_cache:
            file_dir = data_dir / "fred"
            file_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(file_dir / "fred.parquet")
    return df


def resample_quarterly(df):
    """
    Resample data to quarterly frequency.
    """
    df = df.resample('QE').mean()
    return df

macro_series_descriptions = {
    'UNRATE': 'Unemployment Rate (Seasonally Adjusted)',
    'NFCI': 'Chicago Fed National Financial Conditions Index',
    'GDPC1': 'Real Gross Domestic Product'
}

fred_bd_series_descriptions = {
    'BOGZ1FL664090005Q': 'Security Brokers and Dealers; Total Financial Assets, Level',
    'BOGZ1FL664190005Q': 'Security Brokers and Dealers; Total Liabilities, Level',
}


def pull_fred_macro_data(data_dir=DATA_DIR, start=config.START_DATE, end=config.END_DATE):
    """
    Pull macroeconomic data from FRED (UNRATE, NFCI, GDPC1, A191RL1Q225SBEA),
    and save it as a parquet file in data_dir/pulled/fred_macro.parquet.
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


def load_fred_macro_data(data_dir=DATA_DIR, from_cache=True, start=config.START_DATE, end=None):
    """
    Load FRED macro data. If cache exists (data_dir/pulled/fred_macro.parquet), read it.
    Otherwise, call pull_fred_macro_data to fetch and save before reading.
    """
    file_path = data_dir / "pulled" / "fred_macro.parquet"
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d') 
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


def pull_fred_bd_data(data_dir=DATA_DIR, start=config.START_DATE, end=config.END_DATE):
    """
    Pull broker-dealer financial data from FRED (assets and liabilities),
    and save it as a parquet file in data_dir/pulled/fred_bd.parquet.
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


def load_fred_bd_data(data_dir=DATA_DIR, from_cache=True, start=config.START_DATE, end=None):
    """
    Load broker-dealer financial data. If cache exists, read it.
    Otherwise, pull data from FRED, save it, and then read.
    """
    file_path = data_dir / "pulled" / "fred_bd.parquet"
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d') 
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
    _ = load_fred(start=config.START_DATE, end=config.END_DATE, data_dir=DATA_DIR, from_cache=False, save_cache=True)
    
    # Pull and save cache of macroeconomic data
    _ = load_fred_macro_data(start=config.START_DATE, end=config.END_DATE, data_dir=DATA_DIR, from_cache=False)
