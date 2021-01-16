# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: dataset
@time: 2021/1/16 10:14
"""

from torch.utils.data import Dataset
import torch
import numpy as np


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = np.array(dataset)

    def __getitem__(self, index: int):
        token_ids, mask_ids, type_ids, label_ids = self.dataset[index]
        return (torch.tensor(token_ids, dtype=torch.long),
                torch.tensor(mask_ids, dtype=torch.long),
                torch.tensor(type_ids, dtype=torch.long),
                torch.tensor(label_ids, dtype=torch.float32),
                )

    def __len__(self) -> int:
        return len(self.dataset)