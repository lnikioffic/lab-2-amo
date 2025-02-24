#!/bin/bash
set -x

venv\Scripts\activate

# Запуск скрипта генерации данных
python data_creation.py

# Запуск скрипта предобработки данных
python model_preprocessing.py

# Запуск скрипта обучения модели
python model_training.py

# Запуск скрипта тестирования модели
python model_testing.py