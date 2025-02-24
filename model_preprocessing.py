#!/usr/bin/env python3
import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_and_save(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files = glob.glob(os.path.join(input_folder, '*.csv'))
    scaler = StandardScaler()

    for file in files:
        df = pd.read_csv(file)
        # Например, нормализуем столбец температуры
        df['temperature_scaled'] = scaler.fit_transform(df[['temperature']])

        # Сохраняем предобработанные данные
        base_name = os.path.basename(file)
        output_path = os.path.join(output_folder, base_name)
        df.to_csv(output_path, index=False)
        print(f'Предобработанный файл сохранён: {output_path}')


if __name__ == '__main__':
    # Предобработка тренировочных данных
    preprocess_and_save('train', 'train_processed')
    # Предобработка тестовых данных
    preprocess_and_save('test', 'test_processed')
