import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100

def linear_regression_predict(series, window_size=10, forecast_steps=10, test_ratio=0.25):
    series = np.array(series)
    split_idx = int(len(series) * (1 - test_ratio))
    train_data = series[:split_idx]
    test_data = series[split_idx:]

    # Обучение модели на скользящих окнах из train
    X_train, y_train = [], []
    for i in range(window_size, len(train_data)):
        X_train.append(train_data[i - window_size:i])
        y_train.append(train_data[i])
    X_train, y_train = np.array(X_train), np.array(y_train)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Предсказания на тестовой части
    test_predicted = []
    full_series = list(train_data) + list(test_data)
    for i in range(len(test_data)):
        start_idx = split_idx + i - window_size
        if start_idx < 0:
            test_predicted.append(train_data[-1])
            continue
        X_test_i = np.array(full_series[start_idx : split_idx + i]).reshape(1, -1)
        y_pred = model.predict(X_test_i)[0]
        test_predicted.append(y_pred)

    # Прогноз на будущее
    forecast_input = list(series[-window_size:])
    forecast = []
    for _ in range(forecast_steps):
        X_forecast = np.array(forecast_input[-window_size:]).reshape(1, -1)
        pred = model.predict(X_forecast)[0]
        forecast.append(pred)
        forecast_input.append(pred)

    metrics = {
        "mse": float(mean_squared_error(test_data, test_predicted)),
        "mae": float(mean_absolute_error(test_data, test_predicted)),
        "mape": float(mean_absolute_percentage_error(test_data, test_predicted)),
        "r2": float(r2_score(test_data, test_predicted)),
    }

    return {
        "train": train_data.tolist(),
        "test_true": test_data.tolist(),
        "test_predicted": test_predicted,
        "forecast": forecast,
        "metrics": metrics
    }

def plot_forecast(result, title="Прогноз (Линейная регрессия)"):
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

def test_linear_regression():
    series = generate_mock_series(length=300, seasonal_period=50, noise_level=0.2)
    result = linear_regression_predict(series, window_size=5, forecast_steps=10)
    plot_forecast(result)
