#!/bin/bash
set -x

if [ ! -d "venv" ]; then
    echo "Создаём виртуальное окружение..."
    python3 -m venv venv
fi

source venv/bin/activate

pip install --upgrade pip
pip insatll -r requirements.txt

# Запуск скрипта генерации данных
python data_creation.py

# Запуск скрипта предобработки данных
python model_preprocessing.py

# Запуск скрипта обучения модели
python model_training.py

# Запуск скрипта тестирования модели
python model_testing.py