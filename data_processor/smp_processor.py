# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: smp_processor
@time: 2021/1/14 15:28
"""

from data_processor.base_processor import BertProcessor

import os
import json
from config import Config
from configparser import SectionProxy
import numpy as np
from scipy import stats


class SMP_Processor(BertProcessor):
    def __init__(self, bertConfig, maxlen = 32):
        super().__init__(bertConfig, maxlen)

    def convert_to_ids(self, dataset: list) -> list:
        ids_data = []
        for line in dataset:
            ids_data.append(self.parse_line(line))
        return ids_data

    def read_dataset(self, path: str, data_types: list):
        """
        读取数据集文件
        :param path: 路径
        :param data_type: [type1, type2]
        :return dataset list
        """
        # load dataset
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for data_type in data_types:
            for line in data[data_type]:
                dataset.append(line)
        return dataset

    def load_label(self, path):
        """load label"""
        with open(path, 'r', encoding='utf-8') as f:
            self.id_to_label = json.load(f)
            self.label_to_id = {label: i for i, label in enumerate(self.id_to_label)}

    def parse_line(self, line: dict) -> list:
        """
        :param line: [text, label]
        :return: [text_ids, mask, type_ids, label_ids]
        """
        text = line['text']
        label = line['domain']

        ids = self.parse_text_to_bert_token(text) + [self.parse_label(label)]
        return ids

    def parse_text(self, text) -> (list, list, list):
        """
        将文本转为ids
        :param text: 字符串文本
        :return: [token_ids, mask, type_ids]
        """
        return self.parse_text_to_bert_token(text)

    def parse_label(self, label):
        """
        讲label转为ids
        :param label: 文本label
        :return: ids
        """
        return self.label_to_id[label]

    def remove_minlen(self, dataset, minlen):
        n_dataset = []
        for i, line in enumerate(dataset):
            if len(line['text']) >= minlen:
                n_dataset.append(line)
        return n_dataset

    def remove_maxlen(self, dataset, maxlen):
        n_dataset = []
        for i, line in enumerate(dataset):
            if len(line['text']) <= maxlen:
                n_dataset.append(line)
        return n_dataset

    def weight_minlen(self, dataset, minlen):
        n_dataset = []
        for i, line in enumerate(dataset):
            if len(line['text']) >= minlen:
                n_dataset.append(line)
        return n_dataset

    def weight_maxlen(self, dataset, maxlen):
        n_dataset = []
        for i, line in enumerate(dataset):
            if len(line['text']) <= maxlen:
                n_dataset.append(line)
        return n_dataset

    def get_smp_data_info(self, data_path):
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
                    if line['domain'] == 'oos':
                        n_ood += 1
                    else:
                        n_id += 1
                    n += 1
                    text_len[len(line['text'])] = text_len.get(len(line['text']), 0) + 1
                    all_text_len.append(len(line['text']))
                result[type] = {'num': n, 'ood': n_ood, 'id': n_id,
                                'text_len': sorted(text_len.items(), key=lambda d: d[0], reverse=False),
                                'all_len': all_text_len}
        return result

    def get_conf_intveral(self, data: list, alpha, logarithm=False):
        """
        置信区间
        Args:
            data:

        Returns: (a, b)
        a: 置信上界; b: 置信下界

        alpha : array_like of float
            Probability that an rv will be drawn from the returned range.
            Each value should be in the range [0, 1].
        """
        data = np.array(data)
        if logarithm:   # 对数正态分布
            data = np.log(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        # from scipy import stats
        skew = stats.skew(data)  # 求偏度
        kurtosis = stats.kurtosis(data)  # 求峰度
        conf_intveral = stats.norm.interval(alpha, loc=mean, scale=std) # 置信区间
        if logarithm:
            conf_intveral = np.exp(conf_intveral)
        return conf_intveral


if __name__ == '__main__':
    from config import BertConfig
    os.chdir('..')
    bertConfig = BertConfig('config/bert.ini')
    bertConfig = bertConfig('bert-base-chinese')
    processor = SMP_Processor(bertConfig, maxlen=32)
    label_path = 'data/smp/smp_full.label'
    processor.load_label(label_path)
    print(processor.label_to_id)
    print(processor.id_to_label)
    print(processor.parse_line({'text':"不是吧", 'domain':'oos'}))
    print(bertConfig.hidden_size)
    file_path = 'data/smp/true_smp_full.json'
    data = processor.get_smp_data_info(file_path)
    print(data['train'])
    print(data['val'])
    print(data['test'])