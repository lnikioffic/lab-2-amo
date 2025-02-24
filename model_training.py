#!/usr/bin/env python3
import os
import glob
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


def load_training_data(folder):
    files = glob.glob(os.path.join(folder, '*.csv'))
    data_frames = []
    for file in files:
        df = pd.read_csv(file)
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)


if __name__ == '__main__':
    # Загружаем предобработанные тренировочные данные
    train_df = load_training_data('train_processed')

    # Используем day как входной признак, а temperature_scaled – как целевую переменную
    X = train_df[['day']]
    y = train_df['temperature_scaled']

    # Создаём и обучаем модель линейной регрессии
    model = LinearRegression()
    model.fit(X, y)

    # Сохраняем модель
    os.makedirs('model', exist_ok=True)
    model_path = os.path.join('model', 'linear_regression.pkl')
    joblib.dump(model, model_path)
    print(f'Модель обучена и сохранена: {model_path}')
