from keras.models import load_model
from PIL import Image
import cv2
import numpy as np
import string
model = load_model(r"Sign_Language_model.h5")
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
resize_img = cv2.resize(img, (80, 80))
reshape_img = np.reshape(resize_img, (-1, 80, 80, 1))


def standardize(x):
    x_stan = x.astype('float32')
    m = x_stan.mean()
    s = x_stan.std()
    x_stan = (x_stan - m) / s
    return x_stan


x = standardize(reshape_img)
prediction = np.argmax(model.predict(x), axis=-1)
print(prediction[0])
CATEGORIES = list(string.ascii_uppercase)
CATEGORIES.extend(("nothing","del","space"))
print(CATEGORIES[prediction[0]])
