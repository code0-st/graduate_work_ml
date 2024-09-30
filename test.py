from flask import Flask, jsonify
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={'/train': {"origins": "http://localhost:3000"}})


# Генерация данных с умеренной случайностью (тот же набор данных)
def generate_data():
    np.random.seed(0)
    time = np.arange(60)
    season_length = 7
    trend = 0.08 * time
    seasonality = 10 + np.sin(2 * np.pi * time / season_length) * 0.5
    random_noise = np.random.normal(0, 0.1, len(time))
    data = trend + seasonality + random_noise
    return data

# Функция для создания лагов
def create_lagged_features(data, lag):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i-lag:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Функция, выполняющая прогноз
def forecast_model():
    # Данные и их масштабирование
    data = generate_data()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    # Разделение данных
    train_data = data[:40]
    test_data = data[40:]
    
    # Параметры
    lag = 5

    # Создание обучающих данных
    X_train, y_train = create_lagged_features(scaled_data[:40], lag)
    X_test, y_test = create_lagged_features(scaled_data[40-lag:], lag)

    # Преобразование формы
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Построение модели
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Обучение модели
    model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=2)

    # Прогноз для тестовой выборки
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

    # Прогноз на 10 шагов вперед
    X_forecast = scaled_data[-lag:].reshape((1, lag, 1))
    forecast_scaled = []
    for i in range(10):
        pred_scaled = model.predict(X_forecast)[0][0]
        forecast_scaled.append(pred_scaled)
        X_forecast = np.append(X_forecast[:, 1:, :], [[[pred_scaled]]], axis=1)
    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

    # Метрики
    mse = mean_squared_error(test_data[:len(y_pred)], y_pred)
    mae = mean_absolute_error(test_data[:len(y_pred)], y_pred)
    r2 = r2_score(test_data[:len(y_pred)], y_pred)

    # Возвращаем результаты
    return {
        "real_data": data.tolist(),
        "predictions": y_pred.tolist(),
        "test_data": test_data.tolist(),
        "forecast": forecast.tolist(),
        "metrics": {
            "MSE": mse,
            "MAE": mae,
            "R2": r2
        }
    }

# Flask-эндпоинт для получения прогноза
@app.route('/train', methods=['POST'])
def get_forecast():
    results = forecast_model()
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='localhost', port=8080)
