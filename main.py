import numpy as np # модуль для работы с массивами
from keras.models import Sequential, load_model  # модель сети в которой слои соединены друг с другом
from keras.layers import Dense  # модель слоя в котором все нейроны одной сети соединены с нейроном следующей
from keras.utils import np_utils  # утилиты для работы с массивами
from keras.datasets import mnist  # подключение базы для обучения
from scipy.misc import imread

from PIL import Image
from img_convert import *
from prepair_data import *

x_train, y_train, x_test, y_test = prepair()

#загрузка сохраненной модели
nero = load_model('model.h5')

#компиляция загруженной модели
nero.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])


#print(nero.summary())


scores = nero.evaluate(x_test, y_test, verbose=1)
print(" Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))


#проверка работы сети на собственных данных
i = 0
while i < 10:
    name = str(i) + '.bmp'

    img = Image.open(name)
    img.load()
    test = img_to_arr(img)

    pr = nero.predict(test)
    print('Test number: ', str(i))
    print("Answer     : ",np.argmax(pr))

    print('-----------------------------------------')
    i += 1
