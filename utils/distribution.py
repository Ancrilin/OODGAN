# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: distribution
@time: 2021/2/23 15:56
"""

import numpy as np
import json

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


if __name__ == '__main__':
    data_path = '../data/smp/binary_smp_full.json'
    data = show_test_len(data_path)
    print(data['train'])