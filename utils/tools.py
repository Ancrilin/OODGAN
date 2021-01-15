# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: tools
@time: 2021/1/15 15:37
"""

import json
import os
import random
from sklearn.metrics import roc_auc_score
import numpy as np

import numpy
import pandas as pd
import torch


def check_manual_seed(seed):
    """ If manual seed is not specified, choose a random one and communicate it to the user."""

    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    # 添加随机数种子
    numpy.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False

    print('Using manual seed: {seed}'.format(seed=seed))


def convert_to_int_by_threshold(array, threshold=0.5):
    """
    根据阈值，将浮点数转为0/1
    :param array: numpy数组或者pytorch.Tensor对象
    :param threshold: 阈值
    :return: 0/1数组
    """
    _array = array.numpy() if isinstance(array, torch.Tensor) else array
    _array = (_array > threshold).astype(int)
    return _array


def output_cases(texts, ground_truth, predicts, path, processor, logit=None):
    """
    生成csv case
    :param texts: 文本 list
    :param ground_truth: 真正标签 数组
    :param predicts: 预测 数组
    :param path: 保存路径
    :param processor: 数据集处理器
    :return:
    """
    ground_truth = ground_truth.numpy().astype(int) if isinstance(ground_truth, torch.Tensor) else ground_truth
    predicts = predicts.numpy().astype(int) if isinstance(
        predicts, torch.Tensor) else predicts
    if logit != None:
        df = pd.DataFrame(data={
            'text': texts,
            'ground_truth': ground_truth,
            'predict': predicts,
            'score': logit
        })
    else:
        df = pd.DataFrame(data={
            'text': texts,
            'ground_truth': ground_truth,
            'predict': predicts
        })
    df['ground_truth_label'] = df['ground_truth'].apply(lambda i: processor.id_to_label[i])
    df['predict_label'] = df['predict'].apply(lambda i: processor.id_to_label[i])
    df['ground_truth_is_ind'] = df['ground_truth'] != 0
    df['predict_is_ind'] = df['predict'] != 0

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if logit == None:
        df.to_csv(path, columns=['text',
                                 'ground_truth', 'predict',
                                 'ground_truth_label', 'predict_label',
                                 'ground_truth_is_ind', 'predict_is_ind'],
                  index=False)
    else:
        df.to_csv(path, columns=['text',
                                 'score',
                                 'ground_truth', 'predict',
                                 'ground_truth_label', 'predict_label',
                                 'ground_truth_is_ind', 'predict_is_ind'],
                  index=False)



if __name__ == '__main__':
    pass