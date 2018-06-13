#!/usr/bin/env python3
# coding: utf-8
# File: preprocessing.py
# Author: lxw
# Date: 5/30/18 3:58 PM
"""
Reference:
[Introduction to CNN Keras - 0.997 (top 6%)](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)
这个文件中采用的方法都很不错，值得学习
原文中还有动态调整学习率的实现代码
"""

# import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def fetch_data_df():
    """
    :return: return train_df and test_df without Normalization.
    """
    train_df = pd.read_csv("../data/input/train.csv")
    test_df = pd.read_csv("../data/input/test.csv")
    return train_df, test_df


def data_analysis(train_df, test_df):
    sns.set(style="white", context="notebook", palette="deep")

    Y_train = train_df["label"]
    X_train = train_df.drop(labels=["label"], axis=1)

    # free some space
    del train_df

    # 1. 查看样本数据分布情况(各个label数据是否均匀分布)
    sns.countplot(Y_train)
    plt.show()
    print(Y_train.value_counts())

    # 2. Check for null and missing values
    print(X_train.isnull().any().describe())
    print(test_df.isnull().any().describe())
    # fillna() if missing values occur.

    # 3. Normalization(IMPORTANT)
    # We perform a grayscale normalization to reduce the effect of illumination's
    # differences. Moreover the CNN converge faster on [0..1] data than on [0..255].
    X_train = X_train / 255.0
    test = test_df / 255.0

    X_train = X_train.values.reshape(-1, 28, 28, 1)
    test = test.values.reshape(-1, 28, 28, 1)

    # NOTE: Be carefull with some unbalanced dataset a simple random split could cause inaccurate
    # evaluation during the validation. To avoid that, you could use `stratify` option.
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=1)

    plt.imshow(X_train[0][:, :, 0])  # X_train[0].shape: (28, 28, 1)
    plt.show()


def data_preparation(train_df):
    y = train_df["label"]
    y = np_utils.to_categorical(y)  # shape: (42000, 10)

    X = train_df.iloc[:, 1:]  # <DataFrame>. shape: (42000, 784)
    X = X / 255.0    # Normalization(CNN converge faster on [0..1] data than on [0..255].)

    """
    X = X.values.reshape(-1, 28, 28)  # NOTE: (42000, 28, 28). 处理成这种形式, 在下面的神经网络中会报如下的错误:
    Error 1: ValueError: Input 0 is incompatible with layer Conv2d_1: expected ndim=4, found ndim=3
    or
    Error 2: ValueError: Error when checking input: expected Conv2d_1_input to have 4 dimensions, but
    got array with shape (29400, 28, 28)
    """
    X = X.values.reshape(-1, 28, 28, 1)  # NO: X.reshape(-1, 28, 28, 1). X: <DataFrame>. X.values: <ndarray>
    '''
    # 不要像下面这样scaler: 内存会爆掉(65G内存都会爆掉)
    for column in X:
        """
        # NOTE: 使用X_train[column] = MinMaxScaler(X_train[column])或
        # X.loc[:, column] = MinMaxScaler(X.loc[:, column])会报下面的警告
        # A value is trying to be set on a copy of a slice from a DataFrame.
        # Try using .loc[row_indexer,col_indexer] = value instead
        X_train[column] = MinMaxScaler(X_train[column])
        """
        X.loc[:, column] = MinMaxScaler(X.loc[:, column])
    '''

    # X = X[:1000, :]  # TODO: DEBUG
    # y = y[:1000, :]  # TODO: DEBUG
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)
    # X_train.shape: (29400, 28, 28, 1). X_val.shape: (12600, 28, 28, 1)
    return X_train, X_val, y_train, y_val


def data_augmentation(X_train):
    # Data augmentation(对效果的提升非常大).
    # 应该在Normalization之后进行, 这样Normalization的数据会比较少, 而且能够保证最终所有的数据都是Normalization的
    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image. datagen.zoom_range: [0.9, 1.1]
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)  # 本行代码返回值是: None
    return datagen


if __name__ == "__main__":
    train_df, test_df = fetch_data_df()

    data_analysis(train_df, test_df)

    X_train, X_val, y_train, y_val = data_preparation(train_df)
