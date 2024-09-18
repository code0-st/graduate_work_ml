import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten


def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        end_idx = i + n_steps
        if end_idx > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_idx], sequence[end_idx]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def create_model():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(Flatten())
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_model(df, model, n_steps, epochs):
    TEST_SIZE = 40
    TRAIN_SIZE = df.shape[0] - TEST_SIZE
    n_features = 1

    train = df[:TRAIN_SIZE]['<CLOSE>']
    print(train)

    train_x, train_y = split_sequence(train.values, n_steps)
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], n_features))

    model.fit(train_x, train_y, epochs=epochs, verbose=1)
    return model


def inference_model(df, model, n_steps, count):
    TEST_SIZE = 40
    n_features = 1
    test = df[-(TEST_SIZE + n_steps):]['<CLOSE>'].values
    dates = list(df[-TEST_SIZE:]['<DATE>'].values)
    test_x, test_y = split_sequence(test, n_steps)
    nn_out = list()
    nn_out_additional = list()
    for temp in test_x:
        temp = temp.reshape(1, temp.shape[0], n_features)
        res = model.predict(temp, verbose=0)
        nn_out.append(res[0][0])

    temp_slice = df[-n_steps:]['<CLOSE>'].values
    for i in range(count):
        temp = temp_slice.reshape(1, temp_slice.shape[0], n_features)
        res = model.predict(temp, verbose=0)
        nn_out_additional.append(res[0][0])
        temp_slice = np.append(temp_slice, res[0][0])
        temp_slice = np.delete(temp_slice, 0)

    nn_metrics = metrics(test_y, nn_out)

    return [dates, nn_out + nn_out_additional, nn_metrics]


def metrics(real, forecast):
    real = np.array(real)
    forecast = np.array(forecast)
    return {
        'MAD': round(abs(real - forecast).mean(), 4),
        'MSE': round(((real - forecast) ** 2).mean(), 4),
        'MAPE': round((abs(real - forecast) / real).mean(), 4),
        'MPE': round(((real - forecast) / real).mean(), 4),
        'Стандартная ошибка': round(((real - forecast) ** 2).mean() ** 0.5, 4),
    }
