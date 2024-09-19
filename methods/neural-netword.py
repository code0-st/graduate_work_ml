import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import MinMaxScaler

# Подготовка данных
def prepare_data(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

# Пример использования
np.random.seed(42)  # Для воспроизводимости
time_series = np.sin(np.arange(0, 200, 0.1)) + np.random.normal(scale=0.5, size=2000)  # Пример данных, синусоид с шумом
window_size = 60

# Нормализация данных
scaler = MinMaxScaler(feature_range=(0, 1))
time_series_scaled = scaler.fit_transform(time_series.reshape(-1, 1))

# Подготовка данных
X, y = prepare_data(time_series_scaled, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Преобразование данных для Conv1D и LSTM

# Создание модели
model = Sequential()

# Добавляем 1D-Сверточный слой
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(window_size, 1)))
model.add(MaxPooling1D(pool_size=2))

# Добавляем LSTM слой
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))  # Добавляем Dropout для регуляризации
model.add(LSTM(units=50))
model.add(Dropout(0.2))

# Полносвязный слой для вывода прогноза
model.add(Dense(1))

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model.fit(X, y, epochs=50, batch_size=64)

# Прогнозирование
predicted = model.predict(X)
predicted = scaler.inverse_transform(predicted)  # Обратное масштабирование

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(time_series, label='Оригинальные данные')
plt.plot(range(window_size, len(predicted) + window_size), predicted, label='Прогнозируемые данные', color='orange')
plt.title('Прогнозирование котировок с использованием CNN + LSTM')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.show()