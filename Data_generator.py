import os
import numpy as np
import pandas as pd
from pathlib import Path

def generate_synthetic_hr(duration_mins=1440, base_hr=70, noise_std=2, trend_slope=0.001):
    """
    Generate synthetic heart rate (HR) time series data.

    Parameters:
        duration_mins (int): Number of minutes to simulate (default 1 day).
        base_hr (float): Baseline heart rate in bpm.
        noise_std (float): Standard deviation of Gaussian noise.
        trend_slope (float): Linear trend per time step.

    Returns:
        pd.DataFrame: DataFrame with 'timestamp' and 'heart_rate' columns.
    """
    np.random.seed(42)

    timestamps = pd.date_range(start='2024-01-01', periods=duration_mins, freq='T')

    # Linear trend component
    trend = trend_slope * np.arange(duration_mins)

    # Simulated circadian rhythm with daily cycle
    circadian = 5 * np.sin(np.linspace(0, 2 * np.pi, duration_mins))

    # Random noise
    noise = np.random.normal(0, noise_std, duration_mins)

    hr_values = base_hr + trend + circadian + noise

    # Clip heart rate values between plausible physiological limits
    hr_values = pd.Series(hr_values).clip(lower=50, upper=100).values

    df = pd.DataFrame({'timestamp': timestamps, 'heart_rate': hr_values})
    return df

if __name__ == "__main__":
    print("Generating synthetic heart rate data...")

    hr_df = generate_synthetic_hr()

    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    save_path = data_dir / "synthetic_hr.csv"
    hr_df.to_csv(save_path, index=False)

    print(f"Saved synthetic HR data to: {save_path}")

