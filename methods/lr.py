import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def prepare_data(series, window_size):
    """
    Подготовка данных для обучения модели линейной регрессии.
    
    :param series: Временной ряд (список или numpy array).
    :param window_size: Размер окна для создания признаков (количество предыдущих шагов для прогнозирования следующего).
    
    :return: X, y - массивы для обучения модели.
    """
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

def linear_regression_forecast(series, window_size, forecast_steps):
    """
    Прогнозирование временного ряда с помощью линейной регрессии.
    
    :param series: Временной ряд (список или numpy array).
    :param window_size: Размер окна (количество предыдущих шагов для прогнозирования следующего).
    :param forecast_steps: Количество шагов, на которые нужно сделать прогноз.
    
    :return: Прогнозируемые значения (список).
    """
    # Преобразование списка в numpy array
    series = np.array(series)
    
    # Подготовка данных
    X, y = prepare_data(series, window_size)
    
    # Обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X, y)
    
    # Начальные данные для прогноза (последнее окно данных)
    last_window = series[-window_size:].reshape(1, -1)
    forecast = []
    
    # Прогнозирование
    for _ in range(forecast_steps):
        next_value = model.predict(last_window)[0]  # Прогноз следующего значения
        forecast.append(next_value)
        
        # Обновляем окно данных
        last_window = np.roll(last_window, -1)
        last_window[0, -1] = next_value

    return forecast

# Пример использования:
time_series = [100, 102, 105, 107, 105, 108, 110, 108, 110, 115]
window_size = 3
forecast_steps = 5

# Прогнозирование
forecast = linear_regression_forecast(time_series, window_size, forecast_steps)
print(f"Прогнозируемые значения: {forecast}")

# Построение графика
plt.figure(figsize=(10, 6))

# Оригинальный временной ряд
plt.plot(range(len(time_series)), time_series, label='Оригинальный ряд', color='blue', marker='o')

# Прогнозируемые значения
plt.plot(range(len(time_series), len(time_series) + forecast_steps), forecast, label='Прогноз', color='orange', marker='x')

plt.title('Прогнозирование временного ряда с помощью линейной регрессии')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.show()
