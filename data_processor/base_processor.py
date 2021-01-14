# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: base_processor
@time: 2021/1/14 15:24
"""

from transformers import BertTokenizer
import os
from configparser import ConfigParser, SectionProxy


class BaseProcessor:
    label_to_id = dict()
    id_to_label = list()

    def __init__(self):
        pass

    def convert_to_ids(self, data: list) -> list:
        raise NotImplementedError()

    def read_dataset(self, path: str):
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(self.parse_line(line))
        return dataset

    def parse_line(self, line) -> list:
        raise NotImplementedError()

    def parse_text(self, text):
        raise NotImplementedError()

    def parse_label(self, label):
        raise NotImplementedError()


class BertProcessor(BaseProcessor):
    CLS, SEP, PAD, UNK = '[CLS]', '[SEP]', '[PAD]', '[UNK]'

    """BERT的预处理类"""

    def __init__(self, config: SectionProxy, maxlen=32):
        super().__init__()
        self.config = config
        self.maxlen = maxlen
        self.tokenizer = BertTokenizer(os.path.join(config['PreTrainModelDir'], config['VocabFile']),
                                       do_lower_case=True,
                                       unk_token=self.UNK,
                                       sep_token=self.SEP,
                                       pad_token=self.PAD,
                                       cls_token=self.CLS
                                       )

    def convert_to_ids(self, data: list) -> list:
        pass

    def read_dataset(self, path: str):
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(self.parse_line(line))
        return dataset

    def parse_text_to_bert_token(self, text) -> (list, list, list):
        """
        将文本数据转为ids， 返回 (token_ids, mask, type_ids)
        :param text:
        :return:
        """
        token = [self.CLS] + self.tokenizer.tokenize(text) + [self.SEP]
        mask = []
        type_ids = [0] * self.maxlen

        token_ids = self.tokenizer.convert_tokens_to_ids(token)

        # padding
        if len(token) < self.maxlen:
            mask = [1] * len(token_ids) + [0] * (self.maxlen - len(token))
            token_ids += ([0] * (self.maxlen - len(token)))
        else:
            mask = [1] * self.maxlen
            token_ids = token_ids[:self.maxlen]

        return [token_ids, mask, type_ids]


if __name__ == '__main__':
    from config import BertConfig
    print(os.chdir('..'))
    bertConfig = BertConfig('config/bert.ini')
    bertConfig = bertConfig('bert-base-chinese')
    bertProcessor = BertProcessor(bertConfig)
    print(bertProcessor.parse_text_to_bert_token('不是吧'))