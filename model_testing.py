#!/usr/bin/env python3
import os
import glob
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error


def load_testing_data(folder):
    files = glob.glob(os.path.join(folder, '*.csv'))
    data_frames = []
    for file in files:
        df = pd.read_csv(file)
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)


if __name__ == '__main__':
    # Загружаем предобработанные тестовые данные
    test_df = load_testing_data('test_processed')

    # Загружаем сохранённую модель
    model_path = os.path.join('model', 'linear_regression.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Модель не найдена по пути: {model_path}')

    model = joblib.load(model_path)

    # Используем day как входной признак, а temperature_scaled – как истинное значение
    X_test = test_df[['day']]
    y_true = test_df['temperature_scaled']

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_true, y_pred)
    print(f'Среднеквадратичная ошибка на тестовом наборе: {mse:.4f}')
