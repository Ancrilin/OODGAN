# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: roc
@time: 2021/4/11 16:05
"""
import numpy as np
import torch
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from sklearn.metrics import roc_curve, auc

def process_npy(path):
    data = np.load(path, allow_pickle=True)
    data[0] = np.array(data[0])
    return data


if __name__ == '__main__':
    GQO_GAN_LS_ED_path = 'test_result/GQOGAN(LS+ED)_2.npy'
    GQO_GAN_LS_path = 'test_result/GQOGAN(LS).npy'
    gan_id_path = 'test_result/ood.npy'
    gan_id_ood_path = 'test_result/GAN_id_ood_2.npy'
    cnn_path = 'test_result/cnn.npy'
    bert_path = 'test_result/bert_s123.npy'
    biLSTM_path = 'test_result/biLSTM.npy'
    GQO_GAN_LS_ED = process_npy(GQO_GAN_LS_ED_path)
    GQO_GAN_LS = process_npy(GQO_GAN_LS_path)
    gan_id = process_npy(gan_id_path)
    gan_id_ood = process_npy(gan_id_ood_path)
    cnn = process_npy(cnn_path)
    bi = process_npy(biLSTM_path)
    bert = process_npy(bert_path)

    GQO_GAN_LS_ED_false_positive_rate, GQO_GAN_LS_ED_true_positive_rate, GQO_GAN_LS_ED_thresholds = roc_curve(GQO_GAN_LS_ED[0], GQO_GAN_LS_ED[1])
    GQO_GAN_LS_ED_roc_auc = auc(GQO_GAN_LS_ED_false_positive_rate, GQO_GAN_LS_ED_true_positive_rate)
    plt.plot(GQO_GAN_LS_ED_false_positive_rate, GQO_GAN_LS_ED_true_positive_rate, label='{}'.format('GQO-GAN$_{LS}$'))

    GQO_GAN_LS_false_positive_rate, GQO_GAN_LS_true_positive_rate, GQO_GAN_LS_thresholds = roc_curve(GQO_GAN_LS[0], GQO_GAN_LS[1])
    GQO_GAN_LS_roc_auc = auc(GQO_GAN_LS_false_positive_rate, GQO_GAN_LS_true_positive_rate)
    plt.plot(GQO_GAN_LS_false_positive_rate, GQO_GAN_LS_true_positive_rate, label='{}'.format('GQO-GAN$_{LS&ED}$'))

    gan_id_false_positive_rate, gan_id_true_positive_rate, gan_id_thresholds = roc_curve(gan_id[0], gan_id[1])
    gan_id_roc_auc = auc(gan_id_false_positive_rate, gan_id_true_positive_rate)
    plt.plot(gan_id_false_positive_rate, gan_id_true_positive_rate, label='{}'.format('GAN$_{id}$'))

    gan_id_ood_false_positive_rate, gan_id_ood_true_positive_rate, gan_id_ood_thresholds = roc_curve(gan_id_ood[0], gan_id_ood[1])
    gan_id_ood_roc_auc = auc(gan_id_ood_false_positive_rate, gan_id_ood_true_positive_rate)
    plt.plot(gan_id_ood_false_positive_rate, gan_id_ood_true_positive_rate, label='{}'.format('GAN$_{id&ood}$'))

    bi_false_positive_rate, bi_true_positive_rate, bi_thresholds = roc_curve(bi[0], bi[1])
    bi_roc_auc = auc(bi_false_positive_rate, bi_true_positive_rate)
    plt.plot(bi_false_positive_rate, bi_true_positive_rate, label='{}'.format('BiLSTM'))

    cnn_false_positive_rate, cnn_true_positive_rate, cnn_thresholds = roc_curve(cnn[0], cnn[1])
    cnn_roc_auc = auc(cnn_false_positive_rate, cnn_true_positive_rate)
    plt.plot(cnn_false_positive_rate, cnn_true_positive_rate, label='{}'.format('CNN'))

    bert_false_positive_rate, bert_true_positive_rate, bert_thresholds = roc_curve(bert[0], bert[1])
    bert_roc_auc = auc(bert_false_positive_rate, bert_true_positive_rate)
    plt.plot(bert_false_positive_rate, bert_true_positive_rate, label='{}'.format('BERT'))

    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.savefig('roc_curve.jpg')
    plt.show()