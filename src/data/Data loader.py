import pandas as pd
from pathlib import Path

def load_hr_data(filepath=None):
    """
    Load heart rate CSV data into a DataFrame.

    Args:
        filepath (str or Path, optional): Path to CSV file.

    Returns:
        pd.DataFrame: HR data indexed by timestamp.
    """
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "synthetic_hr.csv"
    else:
        filepath = Path(filepath)

    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)

    return df

