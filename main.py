import os

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd

from enums.forecast_methods import ForecastMethod

from methods.moving_average import moving_average_predict
from methods.exponential_smoothing import exponential_smoothing_predict
from methods.holt_winters import holt_winters_predict
from methods.linear_regression import linear_regression_predict
from methods.neural_network import neural_network_predict
from methods.arima import arima_predict
from methods.sarima import sarima_predict

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
CORS(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Загрузка файла
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'message': 'File uploaded successfully', 'filename': filename})
    return jsonify({'error': 'Invalid file type'}), 400

# Получение списка файлов
@app.route('/files', methods=['GET'])
def list_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return jsonify({'files': files})

# Получение содержимого файла
@app.route('/file/<filename>', methods=['GET'])
def get_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

# Прогнозирование
@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        method = ForecastMethod(json_data["method"])
        filename = json_data["filename"]
        params = json_data.get("params", {})
    except Exception as e:
        return jsonify({"error": f"Ошибка валидации запроса: {str(e)}"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "Файл не найден"}), 404

    try:
        df = pd.read_csv(filepath, encoding='cp1251', sep=';')
        df.columns = df.columns.str.strip()

        if "<CLOSE>" not in df.columns:
            return jsonify({"error": "В файле не найден столбец '<CLOSE>'"}), 400

        series = df["<CLOSE>"].dropna().tolist()
    except Exception as e:
        return jsonify({"error": f"Ошибка чтения файла: {str(e)}"}), 500

    try:
        if method == ForecastMethod.SMA:
            result = moving_average_predict(series, **params)
        elif method == ForecastMethod.EMA:
            result = exponential_smoothing_predict(series, **params)
        elif method == ForecastMethod.HOLT_WINTERS:
            result = holt_winters_predict(series, **params)
        elif method == ForecastMethod.LINEAR_REGRESSION:
            result = linear_regression_predict(series, **params)
        elif method == ForecastMethod.NEURAL_NETWORK:
            result = neural_network_predict(series, **params)
        elif method == ForecastMethod.ARIMA:
            result = arima_predict(series, **params)
        elif method == ForecastMethod.SARIMA:
            result = sarima_predict(series, **params)
        else:
            return jsonify({"error": "Метод не поддерживается"}), 400
    except Exception as e:
        return jsonify({"error": f"Ошибка выполнения метода: {str(e)}"}), 500

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='localhost', port=8080)