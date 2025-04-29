import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# from mock import generate_mock_series

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100

def exponential_smoothing_predict(series, alpha=0.2, forecast_steps=10, test_ratio=0.25):
    series = np.array(series)

    # Разделение на обучающую и тестовую части
    split_idx = int(len(series) * (1 - test_ratio))
    train_data = series[:split_idx]
    test_data = series[split_idx:]

    # Вычисление EMA на обучающей части
    ema_values = [train_data[0]]
    for val in train_data[1:]:
        ema_values.append(alpha * val + (1 - alpha) * ema_values[-1])

    # Прогнозирование на тестовой части
    test_predicted = []
    last_value = ema_values[-1]
    for val in test_data:
        pred = alpha * val + (1 - alpha) * last_value
        test_predicted.append(pred)
        last_value = pred

    # Прогноз на будущее
    forecast = []
    for _ in range(forecast_steps):
        pred = last_value
        forecast.append(pred)
        last_value = alpha * pred + (1 - alpha) * last_value

    # Метрики
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

# 🎨 График
def plot_forecast(result, title="Прогноз (EMA)"):
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

# 🧪 Тестовая функция
def test_ema():
    series = generate_mock_series()
    result = exponential_smoothing_predict(series, alpha=0.3, forecast_steps=10)
    plot_forecast(result)

# if __name__ == "__main__":
#     test_ema()