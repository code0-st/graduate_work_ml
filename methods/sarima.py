import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100


def sarima_predict(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), forecast_steps=10, test_ratio=0.25, use_auto_arima=False):
    series = np.array(series)

    split_idx = int(len(series) * (1 - test_ratio))
    train_data = series[:split_idx]
    test_data = series[split_idx:]

    if use_auto_arima:
        model = auto_arima(
            train_data,
            seasonal=True,
            m=seasonal_order[3] if seasonal_order[3] > 0 else 1,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        test_predicted = model.predict(n_periods=len(test_data))
        model.update(test_data)
        forecast = model.predict(n_periods=forecast_steps)
        order_used = model.order
        seasonal_order_used = model.seasonal_order
    else:
        p, d, q = order
        P, D, Q, s = seasonal_order
        model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s), enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit(disp=False)
        test_predicted = fitted_model.predict(start=split_idx, end=split_idx + len(test_data) - 1)
        forecast = fitted_model.predict(start=split_idx + len(test_data), end=split_idx + len(test_data) + forecast_steps - 1)
        order_used = order
        seasonal_order_used = seasonal_order

    metrics = {
        "mse": float(mean_squared_error(test_data, test_predicted)),
        "mae": float(mean_absolute_error(test_data, test_predicted)),
        "mape": float(mean_absolute_percentage_error(test_data, test_predicted)),
        "r2": float(r2_score(test_data, test_predicted)),
        "order_used": order_used,
        "seasonal_order_used": seasonal_order_used,
    }

    return {
        "train": train_data.tolist(),
        "test_true": test_data.tolist(),
        "test_predicted": test_predicted.tolist(),
        "forecast": forecast.tolist(),
        "metrics": metrics
    }


def plot_forecast(result, title="Прогноз (SARIMA)"):
    train = result["train"]
    test_true = result["test_true"]
    test_predicted = result["test_predicted"]
    forecast = result["forecast"]

    plt.figure(figsize=(12, 6))
    plt.plot(train, label="Обучающие данные", color="blue")
    plt.plot(range(len(train), len(train) + len(test_true)), test_true, label="Тестовые (истинные)", color="green")
    plt.plot(range(len(train), len(train) + len(test_predicted)), test_predicted, label="Тестовые (предсказания)", color="orange")
    plt.plot(range(len(train) + len(test_true), len(train) + len(test_true) + len(forecast)), forecast, label="Прогноз", color="red")
    plt.title(title)
    plt.xlabel("Индекс")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True)
    plt.show()


def test_sarima():
    series = generate_mock_series(300)
    result = sarima_predict(
        series,
        order=(1, 1, 1),
        forecast_steps=10,
        use_auto_arima=True
    )
    plot_forecast(result)


# if __name__ == "__main__":
#     test_sarima()