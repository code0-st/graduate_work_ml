from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from model import create_model, train_model, inference_model
import pandas as pd
from csv_utils import read_file, plot_data, get_column_values

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
    # Укажи путь к твоему CSV файлу
    file_path = 'GC_240101_240920.csv'
    
    # Чтение файла с котировками
    data = read_file(file_path)
    
    # Построение графика по ценам закрытия
    plot_data(data, column='<CLOSE>')
    
    # Получение списка значений для столбца с ценами закрытия
    close_prices = get_column_values(data, column='<CLOSE>')
    
    # Вывод первых 5 значений для проверки
    if close_prices is not None:
        print("Первые 5 значений закрытия цен:", close_prices[:5])

if __name__ == "__main__":
    main()
    # app.run(host='localhost', port=8080)
