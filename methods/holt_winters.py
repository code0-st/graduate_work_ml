import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100

def holt_winters_predict(series, forecast_steps=10, test_ratio=0.25, seasonal_periods=50, trend='add', seasonal='add'):
    series = np.array(series)

    # Разделение на обучающую и тестовую выборки
    split_idx = int(len(series) * (1 - test_ratio))
    train_data = series[:split_idx]
    test_data = series[split_idx:]

    # Обучение модели
    model = ExponentialSmoothing(train_data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    fitted_model = model.fit()

    # Предсказание на тестовой части
    test_predicted = fitted_model.forecast(len(test_data))

    # Прогноз на будущее
    full_model = ExponentialSmoothing(np.concatenate([train_data, test_data]), trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    full_fitted = full_model.fit()
    forecast = full_fitted.forecast(forecast_steps)

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
        "test_predicted": test_predicted.tolist(),
        "forecast": forecast.tolist(),
        "metrics": metrics
    }

def plot_forecast(result, title="Прогноз (Хольт-Винтерс)"):
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

def test_holt_winters():
    series = generate_mock_series()
    result = holt_winters_predict(series, forecast_steps=20, seasonal_periods=50)
    plot_forecast(result)

# if __name__ == "__main__":
#     test_holt_winters()