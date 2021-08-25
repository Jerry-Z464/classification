import os
import cv2
import numpy as np
import random

# # 载入数据集
# def loadDatesets(path):
#     try:
#         trainImages = np.load("./48/Helectrical_train_images.npy")
#         testImages = np.load("./48/Helectrical_test_images.npy")
#         trainLabels = np.load("./48/Helectrical_train_labels.npy")
#         testLabels = np.load("./48/Helectrical_test_labels.npy")
#     except:
#         trainImages, trainLabels, testImages, testLabels = makeDatasets(path)
#
#     return trainImages, trainLabels, testImages, testLabels
# path = r'./Helectrical'
# trainImages, trainLabels, testImages, testLabels = loadDatesets(path)
# print(trainImages.shape)
# print(testImages.shape)
# print(trainLabels.shape)
# print(testLabels.shape)


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
            print(index)
            imagePath = os.path.join(path, subdirs[subdir], index)
            imagePath = imagePath.replace('\\', '/')
            img = cv2.imread(imagePath)
            img = cv2.resize(img, dsize=(48, 48), interpolation=cv2.INTER_AREA)
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

    np.save("./48/Helectrical_train_images.npy", trainImages)
    np.save("./48/Helectrical_test_images.npy", testImages)
    np.save("./48/Helectrical_train_labels.npy", trainLabels)
    np.save("./48/Helectrical_test_labels.npy", testLabels)

    return trainImages, trainLabels, testImages, testLabels


if __name__ == "__main__":
    path = r'./Helectrical'
    makeDatasets(path)
