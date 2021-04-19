# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: roc_3
@time: 2021/4/12 15:57
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
    GQO_GAN_LS_ED_path = 'test_result/GQOGAN(LS+ED)_3.npy'
    GQO_GAN_LS_path = 'test_result/GQOGAN(LS)_3.npy'
    gan_id_path = 'test_result/ood.npy'
    gan_id_ood_path = 'test_result/GAN_id_ood_4.npy'
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

    plt.figure()
    p1 = plt.subplot()
    # p2 = plt.subplot()
    GQO_GAN_LS_ED_false_positive_rate, GQO_GAN_LS_ED_true_positive_rate, GQO_GAN_LS_ED_thresholds = roc_curve(GQO_GAN_LS_ED[0], GQO_GAN_LS_ED[1])
    GQO_GAN_LS_ED_roc_auc = auc(GQO_GAN_LS_ED_false_positive_rate, GQO_GAN_LS_ED_true_positive_rate)
    p1.plot(GQO_GAN_LS_ED_false_positive_rate, GQO_GAN_LS_ED_true_positive_rate, label='{}'.format('GQO-GAN$_{LS&ED}$'))
    # p2.plot(GQO_GAN_LS_ED_false_positive_rate, GQO_GAN_LS_ED_true_positive_rate, label='{}'.format('GQO-GAN$_{LS&ED}$'))

    GQO_GAN_LS_false_positive_rate, GQO_GAN_LS_true_positive_rate, GQO_GAN_LS_thresholds = roc_curve(GQO_GAN_LS[0], GQO_GAN_LS[1])
    GQO_GAN_LS_roc_auc = auc(GQO_GAN_LS_false_positive_rate, GQO_GAN_LS_true_positive_rate)
    p1.plot(GQO_GAN_LS_false_positive_rate, GQO_GAN_LS_true_positive_rate, label='{}'.format('GQO-GAN$_{LS}$'))
    # p2.plot(GQO_GAN_LS_false_positive_rate, GQO_GAN_LS_true_positive_rate, label='{}'.format('GQO-GAN$_{LS}$'))

    gan_id_false_positive_rate, gan_id_true_positive_rate, gan_id_thresholds = roc_curve(gan_id[0], gan_id[1])
    gan_id_roc_auc = auc(gan_id_false_positive_rate, gan_id_true_positive_rate)
    p1.plot(gan_id_false_positive_rate, gan_id_true_positive_rate, label='{}'.format('GAN$_{id}$'))
    # p2.plot(gan_id_false_positive_rate, gan_id_true_positive_rate, label='{}'.format('GAN$_{id}$'))

    gan_id_ood_false_positive_rate, gan_id_ood_true_positive_rate, gan_id_ood_thresholds = roc_curve(gan_id_ood[0], gan_id_ood[1])
    gan_id_ood_roc_auc = auc(gan_id_ood_false_positive_rate, gan_id_ood_true_positive_rate)
    p1.plot(gan_id_ood_false_positive_rate, gan_id_ood_true_positive_rate, label='{}'.format('GAN$_{id&ood}$'))
    # p2.plot(gan_id_ood_false_positive_rate, gan_id_ood_true_positive_rate, label='{}'.format('GAN$_{id&ood}$'))

    bi_false_positive_rate, bi_true_positive_rate, bi_thresholds = roc_curve(bi[0], bi[1])
    bi_roc_auc = auc(bi_false_positive_rate, bi_true_positive_rate)
    p1.plot(bi_false_positive_rate, bi_true_positive_rate, label='{}'.format('BiLSTM'))
    # p2.plot(bi_false_positive_rate, bi_true_positive_rate, label='{}'.format('BiLSTM'))

    cnn_false_positive_rate, cnn_true_positive_rate, cnn_thresholds = roc_curve(cnn[0], cnn[1])
    cnn_roc_auc = auc(cnn_false_positive_rate, cnn_true_positive_rate)
    p1.plot(cnn_false_positive_rate, cnn_true_positive_rate, label='{}'.format('CNN'))
    # p2.plot(cnn_false_positive_rate, cnn_true_positive_rate, label='{}'.format('CNN'))

    bert_false_positive_rate, bert_true_positive_rate, bert_thresholds = roc_curve(bert[0], bert[1])
    bert_roc_auc = auc(bert_false_positive_rate, bert_true_positive_rate)
    p1.plot(bert_false_positive_rate, bert_true_positive_rate, label='{}'.format('BERT'))
    # p2.plot(bert_false_positive_rate, bert_true_positive_rate, label='{}'.format('BERT'))

    p1.plot([0, 1], [0, 1], 'r--')
    p1.axis([0.0, 1.0, 0.0, 1.0])
    p1.set_title('ROC')
    p1.legend(loc='lower right')
    p1.set_ylabel('TPR', fontsize=14)
    p1.set_xlabel('FPR', fontsize=14)
    # p2.plot([0, 1], [0, 1], 'r--')
    # p2.axis([0.0, 0.3, 0.5, 1.0])
    # p2.set_title('ROC')
    # p2.legend(loc='lower right')
    # p2.set_ylabel('TPR', fontsize=14)
    # p2.set_xlabel('FPR', fontsize=14)
    # # 方框
    # bx0 = 0.05
    # bx1 = 0.3
    # by0 = 0.5
    # by1 = 0.95
    # box_x = [bx0, bx1, bx1, bx0, bx0]
    # box_y = [by0, by0, by1, by1, by0]
    # p1.plot(box_x, box_y, "purple", linestyle='-.', linewidth=2)
    # # plot patch lines
    # xy = (0.3, 0.5)
    # xy2 = (0.05, 0.65)
    # con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
    #                       axesA=p2, axesB=p1, linestyle='-', color='purple')
    # p2.add_artist(con)
    #
    # xy = (0.3, 0.95)
    # xy2 = (0.05, 0.93)
    # con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
    #                       axesA=p2, axesB=p1, linestyle='-', color='purple')
    # p2.add_artist(con)
    plt.show()
