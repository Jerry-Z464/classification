# -*- coding: utf-8 -*-
import tensorflow as tf
from cv2 import cv2 as cv
tf.keras.__version__
import os
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import optimizers
import sys

label_dict = {0: '空调', 1: '照相机', 2: '吊灯', 3: '风扇',
                  4: '榨汁机', 5: '电饭煲',
                  6: '插座', 7: '扫地机器人', 8: '抽油烟机', 9: '洗衣机'}

def cnnModel():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def LoadImage(img):
    img = cv.resize(img, dsize=(48, 48), interpolation=cv.INTER_AREA)
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    return img


def jiadian(img):
    model = cnnModel()
    weight_file = u"./Helectrical_2.h5"
    if os.path.exists(weight_file):
        model.load_weights(weight_file)  
    image = LoadImage(img)
    res = model.predict_classes(image)
    class_pic = label_dict[int(res)]
    return class_pic



