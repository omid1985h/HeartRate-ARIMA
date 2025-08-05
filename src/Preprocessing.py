import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def preprocess_hr_simple(df: pd.DataFrame, resample_rule="5T", fillna_method="ffill"):
    """
    Preprocess heart rate data by resampling with mean and filling missing values.

    Args:
        df (pd.DataFrame): Input HR data indexed by timestamp.
        resample_rule (str): Pandas resampling frequency (e.g., '5T' for 5 minutes).
        fillna_method (str): Method to fill missing values ('ffill', 'bfill', etc.).

    Returns:
        pd.DataFrame: Resampled and cleaned HR data.
    """
    df_resampled = df.resample(resample_rule).mean()
    df_clean = df_resampled.fillna(method=fillna_method)
    return df_clean



def adf_test(series):
    """
    Run Augmented Dickey-Fuller test to check stationarity.
    """
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    if result[1] > 0.05:
        print("=> Series is likely non-stationary.")
    else:
        print("=> Series is likely stationary.")

def plot_acf_pacf(series, lags=40):
    """
    Plot ACF and PACF for model order identification.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series, ax=ax[0], lags=lags)
    plot_pacf(series, ax=ax[1], lags=lags)
    ax[0].set_title("ACF")
    ax[1].set_title("PACF")
    plt.tight_layout()
    plt.show()

