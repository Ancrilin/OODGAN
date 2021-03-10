# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: distribution
@time: 2021/2/23 15:56
"""

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def show_test_len(data_path):
    result = {}
    with open(data_path, 'r', encoding='utf-8') as fp:
        source = json.load(fp)
        for type in source:
            n = 0
            n_id = 0
            n_ood = 0
            text_len = {}
            all_text_len = []
            for line in source[type]:
                if line['domain'] == 'chat':
                    n_ood += 1
                else:
                    n_id += 1
                n += 1
                text_len[len(line['text'])] = text_len.get(len(line['text']), 0) + 1
                all_text_len.append(len(line['text']))
            print(type, n)
            print('ood', n_ood)
            print('id', n_id)
            print(sorted(text_len.items(), key=lambda d: d[0], reverse=False))
            result[type] = {'num': n, 'ood': n_ood, 'id': n_id, 'text_len': sorted(text_len.items(), key=lambda d: d[0], reverse=False),
                            'all_len': all_text_len}
    return result

def get_oos_data_info(data_path):
    """

    Args:
        data_path: url

    Returns: {'train':{'num':..., 'ood':..., 'id':..., 'tex_len': [(), ()], 'all_len': [],
                  'val':....,
                'test':...}

    """
    result = {}
    with open(data_path, 'r', encoding='utf-8') as fp:
        source = json.load(fp)
        for type in source:
            n = 0
            n_id = 0
            n_ood = 0
            text_len = {}
            all_text_len = []
            for line in source[type]:
                if line[1] == 'oos':
                    n_ood += 1
                else:
                    n_id += 1
                n += 1
                text_len[len(line[0])] = text_len.get(len(line[0]), 0) + 1
                all_text_len.append(len(line[0]))
            result[type] = {'num': n, 'ood': n_ood, 'id': n_id,
                            'text_len': sorted(text_len.items(), key=lambda d: d[0], reverse=False),
                            'all_len': all_text_len}
    return result

def probability_distribution(data, density=False):
    plt.figure()  # 初始化一张图
    x = data
    width = 40
    n, bins, patches = plt.hist(x, bins=width, range=(0, width), color='blue', alpha=0.5, density=density)
    # print(n)
    # print(bins)
    # print(patches)
    plt.grid(alpha=0.5, linestyle='-.')  # 网格线，更好看
    plt.xlabel('Life Cycle /Month')
    plt.ylabel('Number of Events')
    plt.title(r'Life cycle frequency distribution histogram of events in New York')  # +citys[i])
    plt.xticks(np.arange(0, width, 2))
    plt.plot(bins[0:width] + ((bins[1] - bins[0]) / 2.0), n, color='red')  # 利用返回值来绘制区间中点连线
    plt.show()

if __name__ == '__main__':
    data_path = '../data/oos-eval/binary_undersample.json'
    data = get_oos_data_info(data_path)
    print(data['train'])
    print(data['val'])
    print(data['test'])
    train_all_len = data['test']['all_len']
    train_all_len_ln = np.log(data['test']['all_len'])
    plt.figure()
    plt.hist(train_all_len_ln, bins=20)
    plt.grid(alpha=0.5, linestyle='-.')
    plt.show()
