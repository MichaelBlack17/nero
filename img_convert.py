from PIL import Image
import numpy as np


def img_to_arr(img):
    (width, height) = img.size
    rez = np.zeros(28*28)

    i = 0
    while i < height:
        j = 0
        while j < width:
            r, g, b = img.getpixel((i, j))
            rez[j * height + i] = 255 - r
            j += 1
        i += 1

    rez = rez.reshape(1, 784)
    rez = rez.astype('float32')
    rez /= 255
    return rez

img = Image.open('6.bmp')
img.load()

test = img_to_arr(img)
