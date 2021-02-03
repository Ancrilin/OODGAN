# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: bert_2
@time: 2021/2/3 15:35
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
        # self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.discriminator = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )


    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        bert_result = self.bert(input_ids, attention_mask, token_type_ids, return_dict=False)
        last_hidden_state, pooled_output = bert_result
        discriminator_output = self.discriminator(pooled_output)
        # logits = self.classifier(pooled_output)
        return discriminator_output

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True