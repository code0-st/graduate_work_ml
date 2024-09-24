import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def exponential_smoothing(series, seasonal_periods):
    # Применение метода Хольта-Винтерса с аддитивной сезонностью
    model = ExponentialSmoothing(series, seasonal='add', trend='add', seasonal_periods=seasonal_periods)
    fit = model.fit()

    return fit

def exponential_smoothing_forecast(series, seasonal_periods):
    return exponential_smoothing(series, seasonal_periods).forecast(seasonal_periods)

def plot_series_with_smoothing(time_series, forecast, seasonal_periods):    
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(time_series, label='Оригинальные данные')
    plt.plot(range(len(time_series), len(time_series) + seasonal_periods), forecast, label='Прогноз', color='orange')
    plt.title('Прогнозирование методом Хольта-Винтерса')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_holtwinters(time_series, seasonal_periods):
    forecast = exponential_smoothing_forecast(time_series, seasonal_periods)
    plot_series_with_smoothing(time_series, forecast, seasonal_periods)
