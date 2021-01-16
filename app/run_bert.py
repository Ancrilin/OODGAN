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
from config import BertConfig, Config
from data_processor.smp_processor import SMP_Processor
from model.bert import BertClassifier
from data_utils.dataset import MyDataset

# 检测设备
if torch.cuda.is_available():
    device = 'cuda:0'
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    device = 'cpu'
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor


# 检测参数
def check_args(args):
    """to check if the args is ok"""
    if not (args.do_train or args.do_eval or args.do_test):
        raise argparse.ArgumentError('You should pass at least one argument for --do_train or --do_eval or --do_test')
    if args.gradient_accumulation_steps < 1 or args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise argparse.ArgumentError('Gradient_accumulation_steps should >=1 and train_batch_size%gradient_accumulation_steps == 0')


def main(args):
    """

    Args:
        args:

    Returns:

    """
    # 检查参数ars， 及设置Random seed
    logger.info('Checking...')
    check_args(args)
    check_manual_seed(args.seed)

    if torch.cuda.is_available():
        logger.info('The number of GPU: ' + str(torch.cuda.device_count()))
        logger.info('Running in cuda:' + str(torch.cuda.current_device()) + ' - ' + torch.cuda.get_device_name(0))
    else:
        logger.info('Running in cpu.')

    # 读取bert配置, bert.ini
    logger.info('Loading config...')
    bert_config = BertConfig('config/bert.ini')
    bert_config = bert_config(args.bert_type)

    # 读取data配置, data.ini
    data_config = Config('config/data.ini')
    data_config = data_config(args.dataset)

    logger.info("dataset: " + args.dataset + "  data_file: " + args.data_file)
    # Prepare data processor
    data_path = os.path.join(data_config['DataDir'], data_config[args.data_file])  # 把目录和文件名合成一个路径
    label_path = data_path.replace('.json', '.label')  #获取label.json文件路径

    # 实例化数据处理类
    processor = SMP_Processor(bert_config, maxlen=32)
    processor.load_label(label_path)    #加载label, 生成label_to_ids与ids_to_label

    n_class = len(processor.id_to_label)
    # 实例化bert encoder
    model = BertClassifier(bert_config, n_class)  # Bert encoder
    logger.info("fine_tune: " + str(args.fine_tune))
    if args.fine_tune:
        model.unfreeze_bert_encoder()
    else:
        model.freeze_bert_encoder()
    # 加载到设备
    model.to(device)

    # 全局迭代步长
    global_step = 0

    def train(train_dataset, dev_dataset):
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps,
                                      shuffle=True,
                                      num_workers=2)

    if args.do_train:
        # 读取数据
        # 格式为[{'text': '\ufeff打开熊猫看书', 'domain': 'in'}]
        text_train_set = processor.read_dataset(data_path, ['train'])
        text_dev_set = processor.read_dataset(data_path, ['val'])

        # 文本转换为ids
        # 格式为[[token_ids], [mask], [type_ids], label_to_id]
        train_features = processor.convert_to_ids(text_train_set)
        # 调用__getitem__
        # 格式为([token_ids], [mask_ids], [type_ids], label_ids), 类型转换为torch.tensor
        train_dataset = MyDataset(train_features)
        dev_features = processor.convert_to_ids(text_dev_set)
        dev_dataset = MyDataset(dev_features)

        # train
        train(train_dataset, dev_dataset)



if __name__ == '__main__':
    # 创建一个解析器
    parser = argparse.ArgumentParser()

    # 添加参数
    parser.add_argument('--output_dir', required=True,
                        help='The output directory saving model and logging file.')

    parser.add_argument('--seed', required=False, type=int, default=123,
                        help='Random seed.')

    parser.add_argument('--dataset',
                        choices={'oos-eval', 'smp'}, required=True,
                        help='Which dataset will be used.')

    parser.add_argument('--data_file', required=True, type=str,
                        help="""Which type of dataset to be used, 
                            i.e. binary_undersample.json, binary_wiki_aug.json. Detail in config/data.ini""")

    parser.add_argument('--bert_type',
                        choices={'bert-base-uncased', 'bert-large-uncased', 'bert-base-chinese', 'chinese-bert-wwm'},
                        required=True,
                        help='Type of the pre-trained BERT to be used.')

    # ------------------------action------------------------ #
    parser.add_argument('--do_train', action='store_true',
                        help='Do training step')

    parser.add_argument('--do_eval', action='store_true',
                        help='Do evaluation on devset step')

    parser.add_argument('--do_test', action='store_true',
                        help='Do validation on testset step')

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='Number of updates steps to accumulate before performing a backward/update pass')

    parser.add_argument('--train_batch_size', default=32, type=int,
                        help='Batch size for training.')

    parser.add_argument('--fine_tune', action='store_true', default=True,
                        help='Whether to fine tune BERT during training.')

    # 解析参数
    args = parser.parse_args()

    from logger import Logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(os.path.join(args.output_dir, 'train.log'))
    logger.info(os.getcwd())
    os.chdir('../')
    main(args)
