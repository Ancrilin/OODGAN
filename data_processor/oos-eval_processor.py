# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: oos-eval_processor
@time: 2021/1/29 10:03
"""

from data_processor.base_processor import BertProcessor

import os
import json
from config import Config
from configparser import SectionProxy


class OOS_Eval_Processor(BertProcessor):
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


if __name__ == '__main__':
    from config import BertConfig
    os.chdir('..')
    bertConfig = BertConfig('config/bert.ini')
    bertConfig = bertConfig('bert-base-uncased')
    processor = OOS_Eval_Processor(bertConfig, maxlen=32)
    label_path = 'data/smp/smp_full.label'
    processor.load_label(label_path)
    print(processor.label_to_id)
    print(processor.id_to_label)
    print(bertConfig.hidden_size)