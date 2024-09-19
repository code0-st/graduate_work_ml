import numpy as np
import matplotlib.pyplot as plt

def exponential_smoothing(series, alpha):
    """
    Сглаживание временного ряда методом простого экспоненциального сглаживания.

    :param series: Временной ряд (список или numpy array).
    :param alpha: Коэффициент сглаживания (0 < alpha < 1).
    
    :return: Сглаженный временной ряд (numpy array).
    """
    smoothed_series = np.zeros(len(series))
    smoothed_series[0] = series[0]  # Первое значение равно исходному первому значению

    # Применение формулы экспоненциального сглаживания
    for t in range(1, len(series)):
        smoothed_series[t] = alpha * series[t] + (1 - alpha) * smoothed_series[t - 1]
    
    return smoothed_series


def exponential_smoothing_forecast(series, alpha):
    """
    Прогнозирование следующего значения временного ряда методом простого экспоненциального сглаживания.

    :param series: Временной ряд (список или numpy array).
    :param alpha: Коэффициент сглаживания (0 < alpha < 1).
    
    :return: Прогноз на следующий временной шаг.
    """
    # Получаем последнее сглаженное значение ряда
    smoothed_series = exponential_smoothing(series, alpha)
    
    # Прогнозом будет последнее сглаженное значение
    return smoothed_series[-1]


def plot_series_with_smoothing(original_series, smoothed_series):
    """
    Построение графика исходного и сглаженного временного ряда.

    :param original_series: Исходный временной ряд (список или numpy array).
    :param smoothed_series: Сглаженный временной ряд (numpy array).
    """
    plt.figure(figsize=(10, 6))
    
    # График исходного временного ряда
    plt.plot(range(len(original_series)), original_series, label='Оригинальный ряд', color='blue', linestyle='--')
    
    # График сглаженного временного ряда
    plt.plot(range(len(smoothed_series)), smoothed_series, label='Сглаженный ряд', color='orange')
    
    # Настройки графика
    plt.title('Сравнение оригинального и сглаженного временных рядов')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    
    # Показ графика
    plt.show()


# Пример использования:
time_series = [100, 105, 110, 120, 115, 125, 130, 135]
alpha = 0.5

# Сглаживание временного ряда
smoothed_series = exponential_smoothing(time_series, alpha)
print(f"Сглаженный временной ряд: {smoothed_series}")

# Прогнозирование следующего значения
forecast = exponential_smoothing_forecast(time_series, alpha)
print(f"Прогноз на следующий период: {forecast}")

# Построение графика
plot_series_with_smoothing(time_series, smoothed_series)