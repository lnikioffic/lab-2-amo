#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

# Создаем папки, если их нет
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

def generate_data(n_days=100, add_noise=False, add_anomaly=False):
    # Простой процесс: изменение температур от 10 до 30 градусов по Цельсию
    days = np.arange(1, n_days+1)
    # Линейный тренд + синусоида для сезонности
    temperature = 10 + (days / n_days) * 20 + 5 * np.sin(2 * np.pi * days/30)
    
    if add_noise:
        noise = np.random.normal(0, 1.5, n_days)
        temperature += noise
    
    if add_anomaly:
        # Вставляем аномалию в случайное место
        idx = np.random.randint(0, n_days)
        temperature[idx] += np.random.choice([15, -15])
    
    return pd.DataFrame({'day': days, 'temperature': temperature})

# Генерируем несколько наборов данных
# 3 наборa для обучения и 2 для тестирования
for i in range(1, 4):
    df = generate_data(n_days=100, add_noise=True, add_anomaly=(i % 2 == 0))
    file_path = os.path.join('train', f'data_train_{i}.csv')
    df.to_csv(file_path, index=False)
    print(f'Сохранён тренировочный набор: {file_path}')

for i in range(1, 3):
    df = generate_data(n_days=100, add_noise=True, add_anomaly=(i % 2 == 1))
    file_path = os.path.join('test', f'data_test_{i}.csv')
    df.to_csv(file_path, index=False)
    print(f'Сохранён тестовый набор: {file_path}')
