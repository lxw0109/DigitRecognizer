#!/usr/bin/env python3
# coding: utf-8
# File: CNN_keras.py
# Author: lxw
# Date: 5/30/18 8:50 AM

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def data_preparation():
    train_df = pd.read_csv("../data/input/train.csv")
    y = train_df["label"]
    y = np_utils.to_categorical(y)  # shape: (42000, 10)
    X = train_df.iloc[:, 1:]  # <DataFrame>. shape: (42000, 784)

    # X = X.values.reshape(-1, 28, 28)  # NOTE: (42000, 28, 28). 处理成这种形式, 在下面的神经网络中会报如下的错误:
    """
    Error 1: ValueError: Input 0 is incompatible with layer Conv2d_1: expected ndim=4, found ndim=3
    or
    Error 2: ValueError: Error when checking input: expected Conv2d_1_input to have 4 dimensions, but
    got array with shape (29400, 28, 28)
    """
    X = X.values.reshape(-1, 28, 28, 1)  # NO: X.reshape(-1, 28, 28, 1). X: <DataFrame>. X.values: <ndarray>
    X = X / 255.0    # Normalization(CNN converge faster on [0..1] data than on [0..255].)
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

    # X = X[:1000, :]  # DEBUG
    # y = y[:1000, :]  # DEBUG
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)
    # X_train.shape: (29400, 28, 28, 1). X_val.shape: (12600, 28, 28, 1)
    return X_train, X_val, y_train, y_val


def model_training(input_shape=(28, 28, 1), num_classes=10):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape, name="conv2d_1"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2d_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_3"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation="relu", name="dense_4"))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation="softmax", name="dense_softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    """
    The most important function is the optimizer. This function will iteratively improve parameters(filters kernel
    values, weights and bias of neurons, ...) in order to minimise the loss. **We could also have used Stochastic
    Gradient Descent ('sgd') optimizer, but it is slower than RMSprop**.
    """
    return model


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = data_preparation()
    model = model_training()
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    hist_obj = model.fit(X_train, y_train, batch_size=1024, epochs=1000, verbose=1, validation_data=(X_val, y_val), callbacks=[early_stopping])
    model.save("../data/model/cnn.model")

    # 绘制训练集和验证集的曲线
    """
    plt.plot(hist_obj.history["acc"], label="Training Accuracy", color="green", linewidth=2)
    plt.plot(hist_obj.history["loss"], label="Training Loss", color="red", linewidth=1)
    plt.plot(hist_obj.history["val_acc"], label="Validation Accuracy", color="purple", linewidth=2)
    plt.plot(hist_obj.history["val_loss"], label="Validation Loss", color="blue", linewidth=1)
    plt.grid(True)  # 设置网格形式
    plt.xlabel("epoch")
    plt.ylabel("acc-loss")  # 给x, y轴加注释
    plt.legend(loc="upper right")  # 设置图例显示位置
    plt.show()
    """

    test_df = pd.read_csv("../data/input/test.csv")
    X_test = test_df.values  # <ndarray>. shape: (12600, 784). essential.
    X_test = X_test / 255.0    # Normalization
    X_test = X_test.reshape(-1, 28, 28, 1)  # (28000, 28, 28, 1).
    predicted = model.predict(X_test)  # shape: (28000, 10)
    # 把categorical数据转为numeric值，得到分类结果
    preds = list()
    row, col = predicted.shape
    for i in range(row):
        max_index = -1
        max_prop = -1.0
        for j in range(col):
            if predicted[i][j] > max_prop:
                max_index, max_prop = j, predicted[i][j]
        preds.append(max_index)

    np.savetxt("../data/output/cnn_submission.csv", np.c_[range(1, len(X_test) + 1), preds], delimiter=",",
               header="ImageId,Label", comments="", fmt="%d")

    score = model.evaluate(X_val, y_val, verbose=0)    # score: [0.5269775622282991, 0.9260317460695903]
    print("Validation Loss:", score[0])
    print("Validation accuracy:", score[1])
    """
    epochs 60:
    Validation Loss: 0.0650922215245861
    Validation accuracy: 0.988253968216124
    """

