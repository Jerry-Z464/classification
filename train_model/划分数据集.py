import tensorflow as tf

tf.__version__

import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


def makeDatasets(path):
    allImages = []
    allLabels = []
    subdirs = os.listdir(path)
    subdirs.sort()
    print(subdirs)
    classes = len(subdirs)

    for subdir in range(classes):
        for index in os.listdir(os.path.join(path, subdirs[subdir])):
            # rint(index)
            imagePath = os.path.join(path, subdirs[subdir], index)
            imagePath = imagePath.replace('\\', '/')
            img = cv2.imread(imagePath)
            img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_AREA)
            allImages.append(img)
            allLabels.append(subdir)

    c = list(zip(allImages, allLabels))
    random.shuffle(c)
    allImages, allLabels = zip(*c)

    trainNum = int(0.9 * len(allImages))
    print(trainNum)
    trainImages, trainLabels = allImages[:trainNum], allLabels[:trainNum]
    testImages, testLabels = allImages[trainNum:], allLabels[trainNum:]

    print(np.array(trainImages).shape)
    print(np.array(testImages).shape)
    print(np.array(trainLabels).shape)
    print(np.array(testLabels).shape)

    np.save("Helectrical_train_images.npy", trainImages)
    np.save("Helectrical_test_images.npy", testImages)
    np.save("Helectrical_train_labels.npy", trainLabels)
    np.save("Helectrical_test_labels.npy", testLabels)

    return trainImages, trainLabels, testImages, testLabels

# if __name__ == "__main__":
#     path = r'./Helectrical'
#     makeDatasets(path)


# 绘制数据集中的图片
def plot_image(image):
    b, g, r = cv2.split(image)
    img_rgb = cv2.merge([r, g, b])

    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(img_rgb)
    plt.show()




# 载入数据集
def loadDatesets(path):
    try:
        trainImages = np.load("Helectrical_train_images.py")
        testImages = np.load("Helectrical_test_images.py")
        trainLabels = np.load("Helectrical_train_labels.py")
        testLabels = np.load("Helectrical_test_labels.py")
    except:
        trainImages, trainLabels, testImages, testLabels = makeDatasets(path)

    return trainImages, trainLabels, testImages, testLabels




# 创建模型
from tensorflow.keras import datasets, layers, models


def cnnModel():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(257, activation='softmax'))
    model.summary()

    return model


# In[51]:


# 绘制训练过程曲线变化图
def drawtrainning(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')


# In[52]:


# 主函数trainImages, trainLabels, testImages, testLabels = loadDatesets(path)
if __name__ == "__main__":
    path = r'./Helectrical'
    trainImages, trainLabels, testImages, testLabels = loadDatesets(path)
    logdir = os.path.join("callbacks")
    output_model_file = os.path.join(logdir, "train.ckpt")
    print(output_model_file)

    



trainImages = trainImages / 255.0
testImages = testImages / 255.0



model = cnnModel()


# 加载模型权重
try:
    model.load_weights(output_model_file)
    print("Load the existing model parameters successfully, continue training")
except:
    print("No model parameter file, start to train")

# In[56]:


#     from tensorflow.keras import optimizers
#     opt=optimizers.Adam(lr=1e-8)
#     model.compile(optimizer=opt,
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])


#     # 创建一个保存模型的回调函数,每1个周期保存一次权重

#     cp_callback = tf.keras.callbacks.ModelCheckpoint(
#                         filepath=output_model_file,
#                         monitor='val_loss',
#                         verbose=1,
#                         save_best_only=True,
#                         save_weights_only=True,
#                         mode='min'
#                         )


#     history = model.fit(trainImages, trainLabels, epochs=10, batch_size=8,
#                     validation_split=0.1,callbacks=[cp_callback])

# #     # 模型存放路径
# #     save_path = './save_weights/'
# #     model.save_weights(save_path)


# In[57]:


#     drawtrainning(history)


# In[58]:


#     results=model.evaluate(testImages,testLabels)
#     print("test loss:", results[0],"test acc:", results[1])




y = model.predict(testImages)
print(np.argmax(y[7]))
print(testLabels[7])