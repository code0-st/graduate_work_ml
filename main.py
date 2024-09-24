import json
import enum

from flask import Flask, jsonify, request
from flask_cors import CORS

from csv_utils import read_file, plot_data, get_column_values
from methods import ema, sma, holtwinters, lr, neuralnetwork

@enum.unique
class EPredictMethod(enum.Enum):
    sma = 'SMA'
    ema = 'EMA'
    holt_winters = 'HoltWinters'
    linear_regression = 'LinearRegression'
    neural_network = 'NeuralNetwork'
    arima = 'ARIMA'

app = Flask(__name__)

# Cross Origin Resource Sharing (CORS) handling
CORS(app, resources={'/files': {"origins": "http://localhost:3000"}})
CORS(app, resources={'/train': {"origins": "http://localhost:3000"}})


@app.route('/files', methods=['POST'])
def file_post_request():
    uploaded_file = request.files['data']
    uploaded_file.save(uploaded_file.filename)
    return jsonify({'success': True, 'name': uploaded_file.filename})

@app.route('/train', methods=['POST'])
def train_post_request():
    # TODO: Подумать над общими данными в запросе 
    # Например, кол-во шагов для прогноза и тд
    # Также добавить больше конфигурационных параметров для отдельных методов
    payload = json.loads(request.data)
    close_values = get_column_values(read_file(payload['name']), column='<CLOSE>')
    dates_values = get_column_values(read_file(payload['name']), column='<DATE>')
    
    # TODO: Нужно унифицировать ответ
    # test_result = proxy_method(close_prices, payload)

    return jsonify({'status': 'OK', 'close_values': close_values.tolist(), 'dates_values': dates_values.tolist()})

def proxy_method(time_series, payload):
    match payload['method']:
        case EPredictMethod.sma.value:
            return sma.test_sma(time_series, payload['window_size'])
        case EPredictMethod.ema.value:
            return ema.test_ema(time_series, payload['alpha'])
        case EPredictMethod.holt_winters.value:
            return holtwinters.test_holtwinters(time_series, payload['seasonal_periods'])
        case EPredictMethod.linear_regression.value:
            return lr.test_lr(time_series, payload['window_size'], payload['steps'])
        case EPredictMethod.neural_network.value:
            return neuralnetwork.test_neural_network(time_series, payload['window_size'])
        case EPredictMethod.arima.value:
            return None

# def predict_sma(time_series):
#     sma.test_sma(time_series, 6)

# def predict_ema(time_series):
#     ema.test_ema(time_series, 0.7)

# def predict_holrwinters(time_series): 
#     holtwinters.test_holtwinters(time_series, 30)

# def predict_lr(time_series):
#     lr.test_lr(time_series, 15, 5)

# def predict_neuralnetwork(time_series):
#     neuralnetwork.test_neural_network(time_series, 15)
    

if __name__ == "__main__":
    app.run(host='localhost', port=8080)
