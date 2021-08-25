import tensorflow as tf
import cv2 as cv

tf.keras.__version__

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from split import makeDatasets
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import optimizers


# # 绘制数据集中的图片
# def plot_image(image):
#     b, g, r = cv2.split(image)
#     img_rgb = cv2.merge([r, g, b])
#
#     fig = plt.gcf()
#     fig.set_size_inches(2, 2)
#     plt.imshow(img_rgb)
#     plt.show()


# 载入数据集
def loadDatesets(path):
    try:
        trainImages = np.load("./48/Helectrical_train_images.npy")
        testImages = np.load("./48/Helectrical_test_images.npy")
        trainLabels = np.load("./48/Helectrical_train_labels.npy")
        testLabels = np.load("./48/Helectrical_test_labels.npy")
    except:
        trainImages, trainLabels, testImages, testLabels = makeDatasets(path)

    return trainImages, trainLabels, testImages, testLabels

# 创建模型
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
    model.summary()

    return model

# 绘制训练过程曲线变化图
def drawtrainning(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()


path = r'./Helectrical'
trainImages, trainLabels, testImages, testLabels = loadDatesets(path)
logdir = os.path.join("callbacks")
output_model_file = os.path.join(logdir, "train.ckpt")
output_model_file = output_model_file.replace('\\', '/')
print(output_model_file)

trainImages = trainImages / 255.0
testImages = testImages / 255.0

model = cnnModel()


#加载模型权重
try:
    model.load_weights("save_weights_1/Helectrical.h5")
    print("Load the existing model parameters successfully, continue training")
except:
    print("No model parameter file, start to train")
# optimizer='adam'
#对模型进行训练
# opt = optimizers.Adam(lr=1e-5)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x=trainImages, y=trainLabels, epochs=10, batch_size=220, validation_split=0.1)
# 模型存放路径
save_path = './save_weights_1/Helectrical.h5'
model.save_weights(save_path)

results = model.evaluate(testImages, testLabels)
print("test loss:", results[0], "test acc:", results[1])

y = model.predict_classes(testImages)
print("测试结果啊:", y[20:40])
#print(np.argmax(y[7]))
print("测试集标签:", testLabels[20:40])


# #加载模型权重
# try:
#     #model.load_weights(output_model_file)
#     print("Load the existing model parameters successfully, continue training")
# except:
#     print("No model parameter file, start to train")
#
# #对模型进行训练
# opt = optimizers.Adam(lr=1e-3)
# model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
# # 创建一个保存模型的回调函数,每1个周期保存一次权重
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=output_model_file,
#     monitor='val_loss',
#     verbose=1,
#     save_best_only=True,
#     save_weights_only=True,
#     mode='min')
# history = model.fit(trainImages, trainLabels, epochs=10, batch_size=8,
#                     validation_split=0.1)
#
# # 模型存放路径
# save_path = './save_weights/'
# model.save_weights(save_path)
#
# drawtrainning(history)
#
# results = model.evaluate(testImages, testLabels)
# print("test loss:", results[0], "test acc:", results[1])
#
# y = model.predict_classes(testImages)
# print("测试结果啊:", y[20:40])
# #print(np.argmax(y[7]))
# print("测试集标签:", testLabels[20:40])




