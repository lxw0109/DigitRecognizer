#!/usr/bin/env python3
# coding: utf-8
# File: CNN_keras.py
# Author: lxw
# Date: 5/30/18 8:50 AM

import itertools
import numpy as np
import pickle
import tensorflow as tf

from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import load_model

from sklearn.metrics import confusion_matrix

# self-defined module
from preprocessing import fetch_data_df
from preprocessing import data_preparation
from preprocessing import data_augmentation


def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=input_shape, padding="Same", activation="relu", name="conv2d_1"))
    model.add(Conv2D(32, kernel_size=(5, 5), padding="Same", activation="relu", name="conv2d_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_3"))  # strides: default None. If None, it will default to pool_size.
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(5, 5), padding="Same", activation="relu", name="conv2d_4"))
    model.add(Conv2D(64, kernel_size=(5, 5), padding="Same", activation="relu", name="conv2d_5"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_6"))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), padding="Same", activation="relu", name="conv2d_7"))
    model.add(Conv2D(128, kernel_size=(3, 3), padding="Same", activation="relu", name="conv2d_8"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_9"))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), padding="Same", activation="relu", name="conv2d_10"))
    model.add(Conv2D(128, kernel_size=(3, 3), padding="Same", activation="relu", name="conv2d_11"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_12"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu", name="dense_13"))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation="softmax", name="dense_softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def plot_loss_acc_curve():
    import matplotlib.pyplot as plt
    history = None
    with open("../data/output/history_1024.pkl", "rb") as f:
        history = pickle.load(f)
    if not history:
        return

    # 绘制训练集和验证集的loss和accuracy曲线
    plt.plot(history["acc"], label="Training Accuracy", color="green", linewidth=1)
    plt.plot(history["loss"], label="Training Loss", color="red", linewidth=1)
    plt.plot(history["val_acc"], label="Validation Accuracy", color="purple", linewidth=1)
    plt.plot(history["val_loss"], label="Validation Loss", color="blue", linewidth=1)
    plt.grid(True)  # 设置网格形式
    plt.xlabel("epoch")
    plt.ylabel("acc-loss")  # 给x, y轴加注释
    plt.legend(loc="upper right")  # 设置图例显示位置
    plt.show()


'''
def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
'''


def model_train_val(X_train, y_train, X_val, y_val):
    model = build_model()
    BATCH_SIZE = 1024
    EPOCHS = 300
    # Set a learning rate annealer
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    lr_reduction = ReduceLROnPlateau(monitor="val_loss", patience=5, verbose=1, factor=0.2, min_lr=1e-5)
    # 检查最好模型: 只要有提升, 就保存一次
    model_path = "../data/model/best_model_{epoch:02d}_{val_loss:.2f}.hdf5"  # 保存到多个模型文件
    checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    datagen = data_augmentation(X_train)
    hist_obj = model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), epochs=EPOCHS,
                                   validation_data=(X_val, y_val), verbose=2, steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                                   callbacks=[early_stopping, lr_reduction, checkpoint])
    """
    hist_obj = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                         validation_data=(X_val, y_val), callbacks=[early_stopping])
    """
    with open(f"../data/output/history_{BATCH_SIZE}.pkl", "wb") as f:
        pickle.dump(hist_obj.history, f)


def model_predict(test_df, X_val, y_val):
    model = load_model("../data/model/cnn.model")
    X_test = test_df.values  # <ndarray>. shape: (12600, 784). essential.
    X_test = X_test / 255.0    # Normalization
    X_test = X_test.reshape(-1, 28, 28, 1)  # (28000, 28, 28, 1).
    predicted = model.predict(X_test)  # shape: (28000, 10)
    # 把categorical数据转为numeric值，得到分类结果
    predicted = np.argmax(predicted, axis=1)
    np.savetxt("../data/output/cnn_submission.csv", np.c_[range(1, len(X_test) + 1), predicted], delimiter=",",
               header="ImageId,Label", comments="", fmt="%d")
    """
    predicted = pd.Series(predicted, name="Label")
    submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), predicted], axis=1)
    submission.to_csv("cnn_submission.csv", index=False)
    """

    score = model.evaluate(X_val, y_val, verbose=0)    # score: [0.5269775622282991, 0.9260317460695903]
    print("Validation Loss:", score[0])
    print("Validation accuracy:", score[1])

    """
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)
    confusion_mtx = confusion_matrix(y_true, y_pred_classes)
    plot_confusion_matrix(confusion_mtx, classes=range(10))
    """


if __name__ == "__main__":
    # For reproducibility
    np.random.seed(1)
    tf.set_random_seed(1)

    train_df, test_df = fetch_data_df()
    X_train, X_val, y_train, y_val = data_preparation(train_df)
    model_train_val(X_train, y_train, X_val, y_val)

    # plot_loss_acc_curve()

    # model_predict(test_df, X_val, y_val)

