from src.data.loader import load_hr_data
from src.data.generator import generate_synthetic_hr
from src.preprocessing import preprocess_hr, adf_test, plot_acf_pacf
from src.model import train_arima_model, forecast_arima
from src.diagnostics import plot_forecast, evaluate_forecast, plot_residual_diagnostics

from pathlib import Path

if __name__ == "__main__":
    # Generate & Save data (optional if already exists)
    df_raw = generate_synthetic_hr()
    data_path = Path("data/synthetic_hr.csv")
    data_path.parent.mkdir(exist_ok=True)
    df_raw.to_csv(data_path, index=False)

    # Load
    df = load_hr_data(data_path)

    # Preprocess
    df_clean = preprocess_hr(df, resample_rule="5min")
    adf_test(df_clean["heart_rate"])
    plot_acf_pacf(df_clean["heart_rate"])

    # Train/Test Split
    split_point = int(len(df_clean) * 0.8)
    train, test = df_clean.iloc[:split_point], df_clean.iloc[split_point:]

    # Train
    model = train_arima_model(train["heart_rate"], order=(6,0,6))
    print(f"AIC: {model.aic}")
    print(f"BIC: {model.bic}")
    print(f"Log Likelihood: {model.llf}")


    # Forecast
    forecast = forecast_arima(model, steps=len(test))

    # Evaluate
    evaluate_forecast(test["heart_rate"], forecast)
    plot_forecast(test["heart_rate"], forecast)
    plot_residual_diagnostics(model)

