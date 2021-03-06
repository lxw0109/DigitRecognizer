#!/usr/bin/env python3
# coding: utf-8
# File: xgboost_method.py
# Author: lxw
# Date: 5/29/18 1:47 PM
"""
References:
1. [xgboost入门与实战(实战调参篇)](https://blog.csdn.net/sb19931201/article/details/52577592)
2. [Python Package Introduction](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
"""


import numpy as np
import pandas as pd
import time
import xgboost as xgb


from sklearn.model_selection import train_test_split


def digit_recognizer():
    start_time = time.time()

    # 1. data preparation
    train = pd.read_csv("../data/input/train.csv")
    test = pd.read_csv("../data/input/test.csv")

    training, validation = train_test_split(train, test_size=0.3, random_state=1)

    y = training.label
    X = training.drop(["label"], axis=1)
    val_y = validation.label
    val_X = validation.drop(["label"], axis=1)

    # xgb矩阵赋值
    xgb_val = xgb.DMatrix(val_X, label=val_y)
    xgb_train = xgb.DMatrix(X, label=y)
    xgb_test = xgb.DMatrix(test)

    params = {
        "booster": "gbtree",
        "objective": "multi:softmax",  # 多分类的问题
        "num_class": 10,  # 类别数, 与 multisoftmax 并用
        "gamma": 0.2,  # 用于控制是否后剪枝的参数, 越大越保守, 一般0.1、0.2这样子
        "max_depth": 8,  # 构建树的深度，越大越容易过拟合
        "lambda": 3,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合
        "subsample": 0.7,  # 随机采样训练样本
        "colsample_bytree": 0.5,  # 生成树时进行的列采样
        "min_child_weight": 3,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # 假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting
        "silent": 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        "eta": 0.01,  # 如同学习率
        "seed": 1000,
        "nthread": 8,  # cpu 线程数
        # "eval_metric": "auc"
    }
    plst = list(params.items())
    num_rounds = 5000  # 迭代次数(下面设置了early_stopping)
    watchlist = [(xgb_train, "train"), (xgb_val, "val")]

    # 2. 训练模型并保存
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
    model.save_model("../data/model/xgb.model")  # 用于存储训练出的模型
    print("best best_ntree_limit", model.best_ntree_limit)

    preds = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)

    np.savetxt("../data/output/xgb_submission.csv", np.c_[range(1, len(test) + 1), preds], delimiter=",",
               header="ImageId,Label", comments="", fmt="%d")

    cost_time = time.time() - start_time
    print("XGBoost run successfully!\nCost time:{}s".format(cost_time))


def model_plot():
    import matplotlib.pyplot as plt
    bst = xgb.Booster({"nthread": 4})  # init model
    bst.load_model("../data/model/xgb.model")  # load data
    # xgb.plot_importance(bst)
    # plt.show()
    # To plot the output tree via matplotlib, use plot_tree, specifying the ordinal number of the target tree.
    xgb.plot_tree(bst, num_trees=2)
    plt.show()
    # When using IPython, you can use the to_graphviz function, which converts the target tree to a graphviz instance.
    # The graphviz instance is automatically rendered in IPython.
    xgb.to_graphviz(bst, num_trees=2)



if __name__ == "__main__":
    digit_recognizer()
    # model_plot()
