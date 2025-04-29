import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
import tensorflow as tf

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def neural_network_predict(series, window_size=20, forecast_steps=10, test_ratio=0.25, epochs=20, batch_size=8):
    series = np.array(series)
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    split_index = int(len(series_scaled) * (1 - test_ratio))
    train = series_scaled[:split_index]
    test = series_scaled[split_index:]

    # Обучающая выборка
    X_train, y_train = create_sequences(train, window_size)

    # Тестовая выборка — захватываем часть train в начало
    combined = np.concatenate((train[-window_size:], test))  # <-- ключевой момент
    X_test, y_test = create_sequences(combined, window_size)

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Создание модели
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(window_size, 1)),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Прогноз на тесте
    test_pred_scaled = model.predict(X_test).flatten()
    test_pred = scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
    y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Прогноз на будущее
    forecast_input = combined[-window_size:]
    forecast = []
    for _ in range(forecast_steps):
        x = forecast_input[-window_size:].reshape(1, window_size, 1)
        next_val_scaled = model.predict(x, verbose=0).flatten()[0]
        forecast.append(next_val_scaled)
        forecast_input = np.append(forecast_input, next_val_scaled)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    train_original = scaler.inverse_transform(train.reshape(-1, 1)).flatten()

    metrics = {
        "mse": float(mean_squared_error(y_test_true, test_pred)),
        "mae": float(mean_absolute_error(y_test_true, test_pred)),
        "mape": float(mean_absolute_percentage_error(y_test_true, test_pred)),
        "r2": float(r2_score(y_test_true, test_pred)),
    }

    return {
        "train": train_original.tolist(),
        "test_true": y_test_true.tolist(),
        "test_predicted": test_pred.tolist(),
        "forecast": forecast.tolist(),
        "metrics": metrics
    }

def plot_forecast(result, title="Прогноз (CNN + LSTM)"):
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

def test_neural_network():
    series = generate_mock_series()
    result = neural_network_predict(series)
    plot_forecast(result)

# if __name__ == "__main__":
#     test_neural_network()