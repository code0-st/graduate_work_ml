import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100

def linear_regression_predict(series, forecast_steps=10, window_size=5, test_ratio=0.25):
    series = np.array(series)

    # Подготовка данных
    def create_features(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    split_idx = int(len(series) * (1 - test_ratio))
    train_series = series[:split_idx]
    test_series = series[split_idx:]

    X_train, y_train = create_features(train_series, window_size)
    X_test, y_test = create_features(test_series, window_size)

    model = LinearRegression()
    model.fit(X_train, y_train)
    test_predicted = model.predict(X_test)

    # Прогноз на будущее
    last_window = series[-window_size:]
    forecast = []
    current_input = last_window.copy()
    for _ in range(forecast_steps):
        prediction = model.predict([current_input])[0]
        forecast.append(prediction)
        current_input = np.roll(current_input, -1)
        current_input[-1] = prediction

    # Метрики
    metrics = {
        "mse": float(mean_squared_error(y_test, test_predicted)),
        "mae": float(mean_absolute_error(y_test, test_predicted)),
        "mape": float(mean_absolute_percentage_error(y_test, test_predicted)),
        "r2": float(r2_score(y_test, test_predicted)),
    }

    # Восстановим длину обучающей и тестовой части
    train_data = series[:split_idx]
    test_true_full = series[split_idx:]
    test_pred_full = [None] * window_size + test_predicted.tolist()

    return {
        "train": train_data.tolist(),
        "test_true": test_true_full.tolist(),
        "test_predicted": test_pred_full,
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
    series = generate_mock_series()
    result = linear_regression_predict(series, window_size=10, forecast_steps=20)
    plot_forecast(result)

# if __name__ == "__main__":
#     test_linear_regression()