from keras.utils import np_utils  # утилиты для работы с массивами
from keras.datasets import mnist  # подключение базы для обучения
from img_convert import *
from matplotlib import pyplot as plt


def prepair():
    # seed для повторяемости результатов(параметр генератора случайных чисел)
    np.random.seed(42)

    # подготовка данных для обучения
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    plt.imshow(x_train[0])
    # x_train массив с картинками для обучения
    # y_train массив с ответами

    # изменение размера массива
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    # нормализация данных
    x_train = x_train.astype('float32')
    x_train /= 255
    x_test = x_test.astype('float32')
    x_test /= 255

    # преобразовываем метки в категории
    y_train = np_utils.to_categorical(y_train, 10)
    y_test  = np_utils.to_categorical(y_test, 10)
    # 0 => [1,0,0,0,0,0,0,0,0,0]
    # 1 => [0,1,0,0,0,0,0,0,0,0]
    # 2 => [0,0,1,0,0,0,0,0,0,0]

    return x_train, y_train, x_test, y_test

prepair()
