# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: vote
@time: 2021/2/26 16:13
"""

import numpy as np
import torch
import os
import json
import utils.tools as tools
import utils.metrics as metrics

def get_data(path):
    path_list = os.listdir(path)
    data = []
    for path_seed in path_list:
        # test_result_path = path + "/" + path_seed + "/test_result.npy"
        if os.path.isfile(os.path.join(path, path_seed)):
            continue
        test_result_path = os.path.join(path, path_seed, "test_result.npy")
        # print(np.load(test_result_path, allow_pickle=True))
        data.append(np.expand_dims(np.load(test_result_path, allow_pickle=True)[1], axis=1))
    # print(np.shape(data))
    data = np.concatenate(data, axis=1)
    # print(np.shape(data))
    return data


def soft_voting(data):
    # print(data)
    print(np.sum(data, axis=1))
    score = np.sum(data, axis=1) / len(data[0])
    # print(score)
    label = tools.convert_to_int_by_threshold(score)
    # print(label)
    return label, score


if __name__ == '__main__':
    all_binary_y = np.load('../data/smp/all_binary_y.npy')
    path_2 = '../output/only-maxlen-oodp/only-maxlen_20-gan_oodp/only-maxlen_20-oodp-smp_123/test_result.npy'

    path_3 = '../output/only-maxlen-oodp/only-maxlen_20-gan_oodp'

    data_1 = [[0.95, 0.85, 0.45], [0.25, 0.15, 0.546], [0.54, 0.78965, 0.6534]]
    score_1 = [0.9716266989707947, 0.9715043902397156, 0.9709370732307434, 0.9563643336296082]
    score_2 = [0.9599705934524536, 0.9613510370254517, 0.9614847302436829, 0.8296872973442078]
    score_1 = np.expand_dims(score_1, axis=1)
    score_2 = np.expand_dims(score_2, axis=1)
    print(score_1)
    print(score_2)
    score = [score_1, score_2]
    print(score)
    print(np.concatenate(score, axis=1))

    path = '../output/base-gan-oodp-alpha0.99_manual'
    data = get_data(path)
    label, score = soft_voting(data)
    print("label")
    print(label)
    print('score')
    print(score)
    report = metrics.binary_classification_report(all_binary_y, label)
    print(report)
    oos_ind_precision, oos_ind_recall, oos_ind_fscore, _ = metrics.binary_recall_fscore(label, all_binary_y)
    print('oos_ind_precision', oos_ind_precision)
    print('oos_ind_recall', oos_ind_recall)
    print('oos_ind_fscore', oos_ind_fscore)
    fpr95 = tools.ErrorRateAt95Recall(all_binary_y, score)
    print('fpr95', fpr95)
    eer = metrics.cal_eer(all_binary_y, score)
    print('eer', eer)
    auc = tools.roc_auc_score(all_binary_y, score)
    print('auc', auc)
