import pandas as pd
import matplotlib.pyplot as plt

def read_file(file_path):
    """
    Чтение данных из CSV файла с котировками.
    """
    try:
        data = pd.read_csv(file_path, delimiter=';')
        print("Файл успешно загружен.")
        return data
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None

def plot_data(data, column='<CLOSE>'):
    """
    Построение графика для выбранного столбца (по умолчанию – закрытие цены).
    """
    if data is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(data['<DATE>'], data[column], label=column)
        plt.xlabel('Дата')
        plt.ylabel('Значение')
        plt.title(f'График {column}')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Необходимо передать корректные данные.")

def get_column_values(data, column='<CLOSE>'):
    """
    Получение значений для выбранного столбца (по умолчанию – закрытие цены).
    """
    if data is not None:
        return data[column].values
    else:
        print("Необходимо передать корректные данные.")
        return None