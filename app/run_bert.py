# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: run_bert
@time: 2021/1/15 15:32
"""

import argparse
import os
import pickle

import pandas as pd
import torch
import tqdm
from torch.utils.data.dataloader import DataLoader
from transformers.optimization import AdamW
from sklearn.metrics import roc_auc_score
from utils.tools import check_manual_seed

if torch.cuda.is_available():
    device = 'cuda:0'
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    device = 'cpu'
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor


def check_args(args):
    """to check if the args is ok"""
    if not (args.do_train or args.do_eval or args.do_test):
        raise argparse.ArgumentError('You should pass at least one argument for --do_train or --do_eval or --do_test')
    if args.gradient_accumulation_steps < 1 or args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise argparse.ArgumentError('Gradient_accumulation_steps should >=1 and train_batch_size%gradient_accumulation_steps == 0')


def main(args):
    check_args(args)
    check_manual_seed(args.seed)

if __name__ == '__main__':
    # 创建一个解析器
    parser = argparse.ArgumentParser()

    # 添加参数
    parser.add_argument('--output_dir', required=True,
                        help='The output directory saving model and logging file.')

    parser.add_argument('--seed', required=False, type=int, default=123,
                        help='Random seed.')

    # 解析参数
    args = parser.parse_args()

    from logger import Logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(os.path.join(args.output_dir, 'train.log'))
    main(args)