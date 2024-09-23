import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def exponential_smoothing(series, seasonal_periods):
    # Применение метода Хольта-Винтерса с аддитивной сезонностью
    model = ExponentialSmoothing(series, seasonal='add', trend='add', seasonal_periods=seasonal_periods)
    fit = model.fit()

    return fit

def exponential_smoothing_forecast(series, steps):
    return exponential_smoothing(series, steps).forecast(steps)

def plot_series_with_smoothing(time_series, forecast, steps):    
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(time_series, label='Оригинальные данные')
    # TODO: 30 заменить на параметр
    plt.plot(range(len(time_series), len(time_series) + steps), forecast, label='Прогноз', color='orange')
    plt.title('Прогнозирование методом Хольта-Винтерса')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_holtwinters(time_series, steps):
    forecast = exponential_smoothing_forecast(time_series, steps)
    plot_series_with_smoothing(time_series, forecast, steps)

# # Пример временного ряда с сезонностью
# time_series = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
#                115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
#                145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
#                171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
#                196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201]

# # Применение метода Хольта-Винтерса с аддитивной сезонностью
# model = ExponentialSmoothing(time_series, seasonal='add', trend='add', seasonal_periods=12)
# fit = model.fit()

# # Прогнозирование на 12 периодов вперед
# forecast = fit.forecast(steps=12)

# # Построение графика
# plt.figure(figsize=(10, 6))
# plt.plot(time_series, label='Оригинальные данные')
# plt.plot(range(len(time_series), len(time_series) + 12), forecast, label='Прогноз', color='orange')
# plt.title('Прогнозирование методом Хольта-Винтерса')
# plt.xlabel('Время')
# plt.ylabel('Значение')
# plt.legend()
# plt.grid(True)
# plt.show()