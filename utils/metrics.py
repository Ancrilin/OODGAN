# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: metrics
@time: 2021/1/18 20:24
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


def binary_accuracy(predicts, labels) -> float:
    """
    compute binary accuracy
    :param predicts: [num, 1]
    :param labels: [num, 1]
    :param threshold: float
    :return:
    """
    return accuracy_score(labels, predicts)


def binary_recall_fscore(predicts, labels):
    """
    compute binary accuracy
    :param predicts: [num, 1]
    :param labels: [num, 1]
    :param threshold: (recall, fscore)
    :return:
    """

    return precision_recall_fscore_support(labels, predicts)


def ind_class_accuracy(predicts: torch.Tensor, labels: torch.Tensor, oos_index=0):
    """忽略oos的正确率"""
    _labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    _predicts = predicts.numpy() if isinstance(predicts, torch.Tensor) else predicts

    sample_weight = (_labels != oos_index).astype(int)
    return accuracy_score(_labels, _predicts, sample_weight=sample_weight)


def accuracy(labels: torch.Tensor, predicts: torch.Tensor):
    """正确率"""
    _labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    _predicts = predicts.numpy() if isinstance(predicts, torch.Tensor) else predicts

    return accuracy_score(_labels, _predicts)


def binary_classification_report(labels, predicts, output_dict=False):
    '''
    输出precision, accuracy, f1-score, support
    :param predicts: 预测标签
    :param labels: 真实标签
    :return: report
    '''
    predicts = predicts.numpy() if isinstance(predicts, torch.Tensor) else predicts
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    return classification_report(labels, predicts, labels=[0, 1], target_names=['ood', 'ind'], output_dict=output_dict)


def plot_confusion_matrix(y_true, y_pred, save_path, classes=['ood', 'ind'], cmap=plt.cm.Blues):
    """Plot a confusion matrix using ground truth and predictions."""
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #  Figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Axis
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    ax.set_xticklabels([''] + classes)
    ax.set_yticklabels([''] + classes)
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Values
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:d} ({cm_norm[i, j] * 100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # Display
    plt.savefig(save_path + '/confusion_matrix.png')
    plt.show()


def cal_eer(y_true, y_score):
    '''
    Returns the equal error rate for a binary classifier output.
    '''
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


import operator
def ErrorRateAt95Recall_t(labels, scores):
    recall_point = 0.95
    # Sort label-score tuples by the score in descending order.
    temp = zip(labels, scores)
    # operator.itemgetter(1)按照第二个元素的次序对元组进行排序，reverse=True是逆序，即按照从大到小的顺序排列
    # sorted_scores.sort(key=operator.itemgetter(1), reverse=True)
    sorted_scores = sorted(temp, key=operator.itemgetter(1), reverse=True)

    # Compute error rate
    # n_match表示测试集正样本数目
    n_match = sum(1 for x in sorted_scores if x[0] == 1)
    n_thresh = recall_point * n_match
    tp = 0
    count = 0
    for label, score in sorted_scores:
        count += 1
        if label == 1:
            tp += 1
        if tp >= n_thresh:
            break
    return float(count - tp) / count

# 平均数
def get_mean(array: list):
    return np.mean(array)


# 中位数
def get_median(array: list):
    return np.median(array)


# 众数
def get_mode(array: list):
    return stats.mode(array)[0][0]


if __name__ == '__main__':
    a = [1, 1, 2, 2, 4, 5, 5, 5, 6, 7, 7, 8, 8, 8, 8, 8, 9, 10, 10, 10, 10, 11, 12]
    b = [0.806891922,
    0.727703576,
    0.69545793,
    0.744614097,
    0.708913915,
    0.764925373,
    0.753732913,
    0.785562633,
    0.807453416,
]
    print(np.mean(a))
    print(np.median(a))

    from scipy import stats
    print(stats.mode(a)[0][0])
    print(np.mean(b))