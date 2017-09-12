from keras.models import Sequential, model_from_json  # модель сети в которой слои соединены друг с другом
from keras.layers import Dense  # модель слоя в котором все нейроны одной сети соединены с нейроном следующей
from keras.utils import np_utils  # утилиты для работы с массивами
import h5py
from create_model import *

def save_model(model):
    # Генерируем описание модели в формате json
    model_json = model.to_json()
    # Записываем модель в файл
    json_file = open("model.json", "w")
    json_file.write(model_json)
    model.save_weights("model.h5")
    json_file.close()


def load_model():
    # Загружаем данные об архитектуре сети из файла json
    json_file = open("model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    # Создаем модель на основе загруженных данных
    model = model_from_json(loaded_model_json)
    # Загружаем веса в модель
    model.load_weights("model.h5")
    return model

def test_model(model):
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Точность модели на тестовых данных: %.2f%%" % (scores[1] * 100))
