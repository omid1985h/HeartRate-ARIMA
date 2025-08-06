import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf


def plot_forecast(actual, predicted, title="ARIMA Forecast"):
    """
    Plot actual vs. predicted values.
    """
    plt.figure(figsize=(10,4))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Forecast", linestyle="--")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Heart Rate (bpm)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_forecast(actual, predicted):
    """
    Calculate and print RMSE.
    """
    rmse = mean_squared_error(actual, predicted, squared=False)
    print(f"RMSE: {rmse:.2f}")
    return rmse



def plot_residual_diagnostics(model_fit):
    """
    Check residuals: normality, autocorrelation.
    """
    residuals = model_fit.resid.dropna()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram & KDE
    sns.histplot(residuals, kde=True, ax=ax[0])
    ax[0].set_title("Residual Histogram")

    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax[1])
    ax[1].set_title("Q-Q Plot")

    # ACF
    plot_acf(residuals, ax=ax[2])
    ax[2].set_title("Residual ACF")

    plt.tight_layout()
    plt.show()

