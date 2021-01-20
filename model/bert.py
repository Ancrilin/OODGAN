# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: bert
@time: 2021/1/14 19:19
"""

from configparser import SectionProxy

import torch
from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """BERT分类器"""

    def __init__(self, config: SectionProxy, num_labels):
        super(BertClassifier, self).__init__()
        self.config = config
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(config['PreTrainModelDir'])
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)


    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # print('input')
        # print(type(input_ids))
        # print(input_ids)
        # sequence_output, pooled_output = self.bert(input_ids, attention_mask, token_type_ids)
        bert_result = self.bert(input_ids, attention_mask, token_type_ids, return_dict=False)
        # sequence_output = result.last_hidden_state
        # pooled_output = result.pooler_output
        # print('result')
        # print(type(result))
        # print(result)
        last_hidden_state, pooled_output = bert_result
        logits = self.classifier(pooled_output)
        # logits = self.classifier(result[1])
        return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    import os
    import sys
    import numpy as np
    from config import BertConfig

    os.chdir('..')
    bertConfig = BertConfig('config/bert.ini')
    bertConfig = bertConfig('bert-base-chinese')
    print(bertConfig['PreTrainModelDir'])
    model = BertClassifier(bertConfig, 1)
    print(model)
