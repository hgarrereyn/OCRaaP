
import cv2
import os
import numpy as np


class Dataset:
    """An object that facilitates data loading and formating."""

    def __init__(self):
        self.loaded = False

        self.train_images = []
        self.train_labels = []

        self.test_images = []
        self.test_labels = []

        self.unknown_images = []

    def loadData(self, train, test, categories):
        """
        Load training and testing data given a list of categories.

        For example,
        `dataset.loadData(train='train/', test='test/', categories=['cat1','cat2'])`
        will look in:

        - train/cat1/
        - train/cat2/
        - test/cat1/
        - test/cat2/
        """

        print('* Loading image data...')

        n = len(categories)

        for i in range(n):
            cat = categories[i]

            print('\t- ' + cat)

            trainpath = train + cat
            testpath = test + cat

            vec = np.zeros(n)
            vec[i] = 1

            for ftrain in os.listdir(trainpath):
                img = cv2.imread(trainpath + '/' + ftrain, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (40, 40), None)
                self.train_images.append(img)
                self.train_labels.append(vec.copy())

            for ftest in os.listdir(testpath):
                img = cv2.imread(testpath + '/' + ftest, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (40, 40), None)
                self.test_images.append(img)
                self.test_labels.append(vec.copy())

        self.test_images = np.array(self.test_images)
        self.test_labels = np.array(self.test_labels)
        self.test_images = self.test_images.reshape((-1, 40, 40, 1))

        self.train_images = np.array(self.train_images)
        self.train_labels = np.array(self.train_labels)
        self.train_images = self.train_images.reshape((-1, 40, 40, 1))


    def loadUnknown(self, images):
        self.unknown_images = []

        for img in images:
            img = cv2.resize(img, (40, 40), None)
            self.unknown_images.append(img)

        self.unknown_images = np.array(self.unknown_images)
        self.unknown_images = self.unknown_images.reshape((-1, 40, 40, 1))

    def getTrainBatch(self, m):
        index = np.random.choice(np.arange(len(self.train_images)), m, replace=False)

        return (self.train_images[index],
        self.train_labels[index])
