import numpy as np
import matplotlib.pyplot as plt

def smooth_time_series(series, window_size):
    """
    Сглаживание временного ряда методом скользящего среднего.

    :param series: Исходный временной ряд (список или numpy array).
    :param window_size: Размер окна для вычисления скользящего среднего.
    
    :return: Сглаженный временной ряд (numpy array).
    """
    if len(series) < window_size:
        raise ValueError("Размер временного ряда должен быть больше или равен размеру окна.")
    
    # Применение скользящего среднего с помощью numpy
    smoothed_series = np.convolve(series, np.ones(window_size)/window_size, mode='valid')
    
    return smoothed_series


def moving_average_forecast(series, window_size):
    """
    Прогнозирование временного ряда методом скользящего среднего.

    :param series: Исходный временной ряд (список или numpy array).
    :param window_size: Размер окна для вычисления скользящего среднего.
    
    :return: Прогноз на следующий временной шаг.
    """
    if len(series) < window_size:
        raise ValueError("Размер временного ряда должен быть больше или равен размеру окна.")
    
    # Рассчитываем среднее значение последних 'window_size' элементов
    forecast = np.mean(series[-window_size:])
    
    return forecast

def plot_time_series(original_series, smoothed_series, window_size):
    """
    Построение графика исходного и сглаженного временного ряда.

    :param original_series: Исходный временной ряд (список или numpy array).
    :param smoothed_series: Сглаженный временной ряд (numpy array).
    :param window_size: Размер окна для вычисления скользящего среднего.
    """
    plt.figure(figsize=(10, 6))
    
    # График исходного временного ряда
    plt.plot(range(len(original_series)), original_series, label='Оригинальный ряд', color='blue', linestyle='--')
    
    # График сглаженного временного ряда
    plt.plot(range(window_size-1, len(original_series)), smoothed_series, label='Сглаженный ряд', color='orange')
    
    # Добавление легенды и подписей
    plt.title(f"Сглаживание временного ряда с окном {window_size}")
    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True)
    
    # Показ графика
    plt.show()

# Пример использования:
time_series = [100, 105, 110, 120, 115, 125, 130, 135]
window_size = 2

# Сглаживание временного ряда
smoothed_series = smooth_time_series(time_series, window_size)
print(f"Сглаженный временной ряд: {smoothed_series}")

# Прогнозирование следующего значения
forecast = moving_average_forecast(time_series, window_size)
print(f"Прогноз на следующий период: {forecast}")

# Построение графика
plot_time_series(time_series, smoothed_series, window_size)