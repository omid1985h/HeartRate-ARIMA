# Heart Rate Time Series Forecasting Using ARIMA

## Project Overview

This project focuses on modeling and forecasting synthetic heart rate (HR) data exhibiting circadian patterns using ARIMA models. The goal is to analyze how well ARIMA can capture the trend and periodic fluctuations in HR, evaluate its forecasting accuracy, and identify limitations.

---

## Data

- **Type:** Synthetic heart rate data  
- **Duration:** 1440 minutes (1 day) at 1-minute intervals  
- **Characteristics:**  
  - Baseline heart rate ~70 bpm  
  - Added circadian sinusoidal variation with amplitude of 5 bpm  
  - Linear trend with slope 0.0005 bpm per minute  
  - Gaussian noise with standard deviation 2 bpm  

---

## Methodology

1. **Data Generation:** Synthetic HR data with trend + circadian rhythm + noise.  
2. **Preprocessing:** Resampled to 5-minute intervals, forward-filled missing data.  
3. **Stationarity Check:** Augmented Dickey-Fuller (ADF) test to assess stationarity.  
4. **Modeling:**  
   - ARIMA model fitted on training set (80% of data).  
   - Evaluated several orders; best found at ARIMA(6,0,6) based on RMSE and diagnostics.  
5. **Forecasting:** Forecast horizon equal to test set length (20% of data).  
6. **Evaluation:**  
   - Root Mean Squared Error (RMSE)  
   - Residual diagnostics (histogram, Q-Q plot, autocorrelation)

---

## Results

- **ADF Test:**  
  - Statistic: -0.6668  
  - p-value: 0.8552  
  - Interpretation: Series is non-stationary (expected due to trend and seasonality).

- **Model Selection:** ARIMA(6,0,6)  
- **Model Fit Metrics:**  
  - AIC: 643.38  
  - BIC: 691.51  
  - Log Likelihood: -307.69  
- **Forecast Accuracy:** RMSE = 1.19 bpm  
- **Residual Analysis:** Residuals approximate white noise, with minor deviations indicating some structure remains unexplained.

---

## Limitations & Recommendations

- **Non-Stationarity:** Data remains non-stationary, which ARIMA(6,0,6) partially accounts for without differencing (d=0).  
- **Overfitting Risk:** Some AR and MA coefficients in the model are not statistically significant, suggesting potential overfitting.  
- **Lack of Seasonality Modeling:** ARIMA lacks explicit seasonality components; SARIMA or other seasonal models may better capture the circadian rhythm.  
- **Model Improvements:**  
  - Consider SARIMA with seasonal order reflecting 24-hour cycles.  
  - Explore decomposition methods (e.g., STL) before modeling.  
  - Use automated model selection techniques (e.g., auto_arima) to optimize parameters.

---

## Conclusion

While the ARIMA(6,0,6) model provides a reasonable fit and forecasts the main heart rate trends with an RMSE of 1.19 bpm, it does not fully capture seasonal dynamics and residual autocorrelation. Future work should focus on seasonal models and advanced decomposition techniques to improve predictive accuracy for circadian heart rate data.

---


