#!/usr/bin/env python3
# coding: utf-8
# File: CNN_keras.py
# Author: lxw
# Date: 5/30/18 8:50 AM

import pandas as pd

from keras import Sequential
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
    y_train = train_df["label"]
    y_train = np_utils.to_categorical(y_train)
    X_train = train_df.iloc[:, 1:]
    for column in X_train:
        X_train[column] = MinMaxScaler(X_train[column])

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=1)
    return X_train, X_val, y_train, y_val


def model_training(input_shape=(28, 28), num_classes=10):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = data_preparation()
    model = model_training()
    model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_val, y_val))

    test_df = pd.read_csv("../data/input/test.csv")
    for column in test_df:
        test_df[column] = MinMaxScaler(test_df[column])
    predicted = model.predict(test_df)
    print("predict result:", predicted)
    score = model.evaluate(X_val, y_val, verbose=0)
    print("Validation Loss:", score[0])
    print("Validation accuracy:", score[2])

