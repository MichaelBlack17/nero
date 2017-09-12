import numpy as np # модуль для работы с массивами
from keras.models import Sequential  # модель сети в которой слои соединены друг с другом
from keras.layers import Dense  # модель слоя в котором все нейроны одной сети соединены с нейроном следующей
from keras.utils import np_utils  # утилиты для работы с массивами
from keras.datasets import mnist  # подключение базы для обучения
import h5py

from prepair_data import *

x_train, y_train, x_test, y_test = prepair()


# создаем модель сети
nero = Sequential()

# добавление уровней в сеть
nero.add(Dense(800, input_dim=784, init="normal", activation="relu"))
# 800 нейронов в слое
# каждый нейрон имеет 784 входа
# значение весов задано нормальным распределением
# функция активации - relu

nero.add(Dense(10, init="normal", activation="softmax"))

# компиляция модели
nero.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
# optimizer - метод обучения. SGD - метод стахастического градиентного спуска
# loss - мера ошибки. categorical_crossentropy - категория
# metrics - оптимизация.accuracy - по точности

# печать характеристик сети
print(nero.summary())

# обучаем сеть
nero.fit(x_train, y_train, batch_size=50, nb_epoch=150, verbose=1)
#размер минивыборки для опитмизации 50. через каждые 50 изображений меняются веса
#100 эпох. 100 раз прогоним иблиотеку с картинками
# verbose=1 - печать диагностической информации на момент обучения

nero.save('model.h5')