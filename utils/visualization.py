# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: visualization
@time: 2021/1/19 13:38
"""

import os
from typing import Dict, List
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc,  confusion_matrix
from sklearn.manifold import TSNE

from data_processor.base_processor import BaseProcessor


def scatter_plot(data: np.ndarray, processor: BaseProcessor):
    """
    二维散点图
    data: [x, y, label]
    """
    # prepare id to label map
    id_to_label_map = {i: label for i, label in enumerate(processor.id_to_label)}
    # id_to_label_map = {i: label for i, label in enumerate(id_to_label)}
    print('id_to_label_map', type(id_to_label_map), id_to_label_map)
    id_to_label_map[-1] = 'GAN'
    id_to_label_map[0] = 'OOD'
    id_to_label_map[1] = 'ID'
    print('id_to_label_map', type(id_to_label_map), id_to_label_map)
    print('id_to_label_map', id_to_label_map)
    df = pd.DataFrame(data, columns=['x', 'y', 'label'])
    df['label'] = df['label'].apply(lambda x: id_to_label_map[int(x)])

    fig, ax = plt.subplots()
    for group_name, group_data in df.groupby('label'):
        ax.scatter(group_data['x'], group_data['y'], alpha=0.5, label=str(group_name))
    ax.legend()
    return fig


def draw_curve(x, y, title, save_path):
    """
    :param x: array
    :param y: scale (iteration)
    :param title: title of curve
    :param save_path: save path
    :return: none
    """
    x = x.cpu().numpy if isinstance(x, torch.Tensor) else np.array(x)
    y = np.arange(y)
    plt.title(title)
    plt.xlabel('epoch')
    plt.plot(y, x)
    save_path += '/curve'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_path + '/' + title + '.png')
    plt.show()


def plot_roc(labels: List[float], predict_probs: Dict[str, List[float]]):
    for name, predict_prob in predict_probs.items():
        false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.plot(false_positive_rate, true_positive_rate, label='{}AUC = {:0.4f}'.format(name, roc_auc))

    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()

def my_plot_roc(y_true, y_score, save_path):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, label='AUC = {:0.4f}'.format(roc_auc))
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.savefig(save_path)
    plt.show()

def plot_train_test(train_feature, test_feature, save_path):
    dim_train_feature = TSNE(2, verbose=1).fit_transform(train_feature)
    dim_test_feature = TSNE(2, verbose=1).fit_transform(test_feature)
    plt.title('train_test_distribution')
    plt.scatter(dim_train_feature[:, 0], dim_train_feature[:, 1], color='red', label='train_feature', s=3)
    plt.scatter(dim_test_feature[:, 0], dim_test_feature[:, 1], color='blue', label='test_feature', s=3)
    plt.legend()
    plt.savefig(save_path + 'train_test_distribution.png')

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