import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100

def moving_average_predict(series, window_size=5, forecast_steps=10, test_ratio=0.25):
    series = np.array(series)

    # Разделение данных
    split_idx = int(len(series) * (1 - test_ratio))
    train_data = series[:split_idx]
    test_data = series[split_idx:]

    # Предсказания на тестовой части
    test_predicted = []
    extended_data = list(train_data)

    for i in range(len(test_data)):
        if len(extended_data) < window_size:
            test_predicted.append(extended_data[-1])
        else:
            sma = np.mean(extended_data[-window_size:])
            test_predicted.append(sma)
        extended_data.append(test_data[i])

    # Прогноз на будущее
    forecast_input = extended_data[-window_size:]
    forecast = []
    for _ in range(forecast_steps):
        pred = np.mean(forecast_input[-window_size:])
        forecast.append(pred)
        forecast_input.append(pred)

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

# 🎨 Отрисовка графика
def plot_forecast(result, title="Прогноз (SMA)"):
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
def test_moving_average():
    series = generate_mock_series()
    result = moving_average_predict(series, window_size=10, forecast_steps=20)
    plot_forecast(result)

# # Для быстрого запуска вручную
# if __name__ == "__main__":
#     test_moving_average()