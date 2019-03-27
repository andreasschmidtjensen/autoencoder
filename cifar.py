import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


class Cifar:
    def __init__(self, directory):
        self.directory = directory
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.labels = None

    def __unpickle(self, file):
        with open(f'{self.directory}/{file}', 'rb') as fo:
            return pickle.load(fo, encoding='bytes')

    def load(self):
        pd_tr = pd.DataFrame()
        tr_y = pd.DataFrame()

        for i in range(1, 6):
            data = self.__unpickle(f'data_batch_{i}')
            pd_tr = pd_tr.append(pd.DataFrame(data[b'data']))
            tr_y = tr_y.append(pd.DataFrame(data[b'labels']))
            pd_tr['labels'] = tr_y

        self.train_x = np.asarray(pd_tr.iloc[:, :3072])
        self.train_x = np.array([Cifar.create_image(i) for i in self.train_x])

        self.train_y = np.asarray(pd_tr['labels'])

        self.test_x = np.asarray(self.__unpickle('test_batch')[b'data'])
        self.test_x = np.array([Cifar.create_image(i) for i in self.test_x])
        self.test_y = np.asarray(self.__unpickle('/test_batch')[b'labels'])

        self.labels = self.__unpickle('/batches.meta')[b'label_names']

    @staticmethod
    def create_image(arr):
        r = arr[0:1024].reshape(32, 32) / 255.0
        g = arr[1024:2048].reshape(32, 32) / 255.0
        b = arr[2048:].reshape(32, 32) / 255.0

        img = np.dstack((r, g, b))

        return img

    def plot(self, index, train=True):
        x = self.train_x if train else self.test_x
        y = self.train_y if train else self.test_y

        img = x[index]
        #img = Cifar.create_image(arr)

        title = re.sub('[!@#$b]', '', str(self.labels[y[index]]))
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)
        ax.imshow(img, interpolation='bicubic')
        ax.set_title('Category = ' + title, fontsize=15)
