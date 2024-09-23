from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from model import create_model, train_model, inference_model
import pandas as pd
from csv_utils import read_file, plot_data, get_column_values

# methods
from methods import ema, sma, holtwinters, lr, neuralnetwork

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
    data = json.loads(request.data)
    df = pd.read_csv(data['name'], sep=';')
    print(df.head())
    model = create_model()
    trained_model = train_model(df, model, int(data['steps']), int(data['epochs']))
    inferenced_model = inference_model(df, trained_model, int(data['steps']), int(data['count']))
    return jsonify({'status': 'OK', 'dates': str(inferenced_model[0]), "values": str(inferenced_model[1]), "metrics": inferenced_model[2]})


def main():
    file_path = 'GC.csv'
    data = read_file(file_path)
    # plot_data(data, column='<CLOSE>')
    close_prices = get_column_values(data, column='<CLOSE>')

    predict_neuralnetwork(close_prices)


def predict_sma(time_series):
    sma.test_sma(time_series, 6)

def predict_ema(time_series):
    ema.test_ema(time_series, 0.7)

def predict_holrwinters(time_series): 
    holtwinters.test_holtwinters(time_series, 30)

def predict_lr(time_series):
    lr.test_lr(time_series, 15, 5)

def predict_neuralnetwork(time_series):
    neuralnetwork.test_neural_network(time_series, 15)
    

if __name__ == "__main__":
    main()
    # app.run(host='localhost', port=8080)
