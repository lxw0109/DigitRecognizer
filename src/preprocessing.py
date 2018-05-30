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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

from sklearn.model_selection import train_test_split


sns.set(style="white", context="notebook", palette="deep")

train = pd.read_csv("../data/input/train.csv")
test = pd.read_csv("../data/input/test.csv")

Y_train = train["label"]
# Drop 'label' column
X_train = train.drop(labels=["label"], axis=1)

# free some space
del train

# 1. 查看样本数据分布情况(各个label数据是否均匀分布)
"""
g = sns.countplot(Y_train)
plt.show()

print(Y_train.value_counts())
"""

# 2. Check for null and missing values
"""
print(X_train.isnull().any().describe())
print(test.isnull().any().describe())
# fillna() if missing values occur.
"""

# 3. Normalization(IMPORTANT)
# We perform a grayscale normalization to reduce the effect of illumination's
# differences. Moreover the CNN converge faster on [0..1] data than on [0..255].
X_train = X_train / 255.0
test = test / 255.0

X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# NOTE: Be carefull with some unbalanced dataset a simple random split could cause inaccurate
# evaluation during the validation. To avoid that, you could use stratify option.
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=1)

plt.imshow(X_train[0][:, :, 0])  # X_train[0].shape: (28, 28, 1)

print()

