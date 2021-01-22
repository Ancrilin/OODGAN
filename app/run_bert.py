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
import numpy as np

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
from utils.tools import EarlyStopping, ErrorRateAt95Recall, save_model, load_model, save_result, output_cases, save_feature
import utils.metrics as metrics

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
    logger.info('Using manual seed: {seed}'.format(seed=args.seed))

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

    logger.info("dataset: " + args.dataset)
    logger.info("data_file: " + args.data_file)
    # Prepare data processor
    data_path = os.path.join(data_config['DataDir'], data_config[args.data_file])  # 把目录和文件名合成一个路径
    label_path = data_path.replace('.json', '.label')  #获取label.json文件路径

    # 实例化数据处理类
    processor = SMP_Processor(bert_config, maxlen=32)
    processor.load_label(label_path)    #加载label, 生成label_to_ids 与 ids_to_label

    n_class = len(processor.id_to_label)
    # config = vars(args)  # 返回参数字典
    config = args.__dict__
    config['model_save_path'] = os.path.join(args.output_dir, 'save', 'bert.pt')
    config['n_class'] = n_class
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
        n_sample = len(train_dataloader)
        early_stopping = EarlyStopping(args.patience, logger=logger)

        # Loss function
        classified_loss = torch.nn.CrossEntropyLoss().to(device)

        # Optimizers
        optimizer = AdamW(model.parameters(), args.lr)

        iteration = 0
        train_loss = []
        if dev_dataset:
            valid_loss = []
            valid_ind_class_acc = []

        nonlocal global_step
        for i in range(args.n_epoch):

            model.train()
            total_loss = 0

            for sample in tqdm.tqdm(train_dataloader):
                # 数据加载到设备
                sample = (i.to(device) for i in sample)
                token, mask, type_ids, y = sample
                batch = len(token)

                logits = model(token, mask, type_ids)
                print('logit')
                print(logits)
                print('y')
                print(y.long())
                loss = classified_loss(logits, y.long())
                total_loss += loss.item()
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                # bp and update parameters
                if (global_step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            logger.info('[Epoch {}] Train: train_loss: {}'.format(i, total_loss / n_sample))
            logger.info('-' * 30)

            train_loss.append(total_loss / n_sample)
            iteration += 1

            # 验证集
            if dev_dataset:
                logger.info('#################### eval result at step {} ####################'.format(global_step))
                eval_result = eval(dev_dataset)

                valid_loss.append(eval_result['loss'])
                valid_ind_class_acc.append(eval_result['ind_class_acc'])

                # 1 表示要保存模型
                # 0 表示不需要保存模型
                # -1 表示不需要模型，且超过了patience，需要early stop
                signal = early_stopping(eval_result['accuracy'])
                if signal == -1:
                    break
                elif signal == 0:
                    pass
                elif signal == 1:
                    save_model(model, path=config['model_save_path'], model_name='bert')


                logger.info('valid_eer: {}'.format(eval_result['eer']))
                logger.info('valid_oos_ind_precision: {}'.format(eval_result['oos_ind_precision']))
                logger.info('valid_oos_ind_recall: {}'.format(eval_result['oos_ind_recall']))
                logger.info('valid_oos_ind_f_score: {}'.format(eval_result['oos_ind_f_score']))
                logger.info('valid_auc: {}'.format(eval_result['auc']))
                logger.info('valid_fpr95: {}'.format(eval_result['fpr95']))
                logger.info('valid_accuracy: {}'.format(eval_result['accuracy']))
                logger.info('report')
                logger.info(eval_result['report'])

        from utils.visualization import draw_curve
        draw_curve(train_loss, iteration, 'train_loss', args.output_dir)
        if dev_dataset:
            draw_curve(valid_loss, iteration, 'valid_loss', args.output_dir)
            draw_curve(valid_ind_class_acc, iteration, 'valid_ind_class_accuracy', args.output_dir)

        if args.patience >= args.n_epoch:
            save_model(model, path=config['model_save_path'], model_name='bert')

    def eval(dataset):
        dev_dataloader = DataLoader(dataset, batch_size=args.predict_batch_size, shuffle=False, num_workers=2)
        n_sample = len(dev_dataloader)
        result = dict() # eval result

        model.eval()

        classified_loss = torch.nn.CrossEntropyLoss().to(device)

        all_pred = []
        all_logit = []
        total_loss = 0

        for sample in tqdm.tqdm(dev_dataloader):
            sample = (i.to(device) for i in sample)
            token, mask, type_ids, y = sample
            batch = len(token)

            with torch.no_grad():
                logit = model(token, mask, type_ids)
                all_logit.append(logit) # 推断值
                all_pred.append(torch.argmax(logit, 1)) # 预测label
                total_loss += classified_loss(logit, y.long())

        all_y = LongTensor(dataset.dataset[:, -1].astype(int)).cpu()  # [length, n_class] 原始真实label
        all_binary_y = (all_y != 0).long()  # [length, 1] label 0 is oos    二分类ood， ind 真实label

        all_pred = torch.cat(all_pred, 0).cpu() # 预测值拼接
        all_logit = torch.cat(all_logit, 0).cpu() # 推断值拼接

        y_score = all_logit.softmax(1)[:, 1].tolist()
        eer = metrics.cal_eer(all_binary_y, y_score)
        oos_ind_precision, oos_ind_recall, oos_ind_fscore, _ = metrics.binary_recall_fscore(
            all_pred, all_y)
        ind_class_acc = metrics.ind_class_accuracy(all_pred, all_y)
        fpr95 = ErrorRateAt95Recall(all_binary_y, y_score)

        report = metrics.binary_classification_report(all_y, all_pred)

        result['eer'] = eer
        result['ind_class_acc'] = ind_class_acc
        result['loss'] = total_loss / n_sample # avg loss
        result['y_score'] = y_score
        result['all_binary_y'] = all_binary_y
        result['oos_ind_precision'] = oos_ind_precision
        result['oos_ind_recall'] = oos_ind_recall
        result['oos_ind_f_score'] = oos_ind_fscore
        result['auc'] = roc_auc_score(all_binary_y, y_score)
        result['fpr95'] = fpr95
        result['report'] = report
        result['accuracy'] = metrics.binary_accuracy(all_binary_y, all_pred)

        return result

    def test(dataset):
        load_model(model, path=config['model_save_path'], model_name='bert')
        test_dataloader = DataLoader(dataset, batch_size=args.predict_batch_size, shuffle=False, num_workers=2)
        n_sample = len(test_dataloader)
        result = dict()
        model.eval()

        classified_loss = torch.nn.CrossEntropyLoss().to(device)

        all_pred = []
        all_logit = []
        total_loss = 0

        for sample in tqdm.tqdm(test_dataloader):
            sample = (i.to(device) for i in sample)
            token, mask, type_ids, y = sample
            batch = len(token)

            with torch.no_grad():
                logit = model(token, mask, type_ids)
                all_logit.append(logit)  # 推断值
                all_pred.append(torch.argmax(logit, 1))  # 预测label
                total_loss += classified_loss(logit, y.long())

        all_y = LongTensor(dataset.dataset[:, -1].astype(int)).cpu()  # [length, n_class] 原始真实label
        all_binary_y = (all_y != 0).long()  # [length, 1] label 0 is oos    二分类ood， ind 真实label

        all_pred = torch.cat(all_pred, 0).cpu()  # 预测值拼接
        all_logit = torch.cat(all_logit, 0).cpu()  # 推断值拼接

        y_score = all_logit.softmax(1)[:, 1].tolist()
        eer = metrics.cal_eer(all_binary_y, y_score)
        oos_ind_precision, oos_ind_recall, oos_ind_fscore, _ = metrics.binary_recall_fscore(
            all_pred, all_y)
        ind_class_acc = metrics.ind_class_accuracy(all_pred, all_y)
        fpr95 = ErrorRateAt95Recall(all_binary_y, y_score)

        report = metrics.binary_classification_report(all_y, all_pred)

        result['all_y'] = all_y.tolist()
        result['all_pred'] = all_pred.tolist()
        result['test_logit'] = all_logit.tolist()

        result['eer'] = eer
        result['ind_class_acc'] = ind_class_acc
        result['loss'] = total_loss / n_sample  # avg loss
        result['y_score'] = y_score
        result['all_binary_y'] = all_binary_y
        result['oos_ind_precision'] = oos_ind_precision
        result['oos_ind_recall'] = oos_ind_recall
        result['oos_ind_f_score'] = oos_ind_fscore
        result['auc'] = roc_auc_score(all_binary_y, y_score)
        result['fpr95'] = fpr95
        result['report'] = report
        result['accuracy'] = metrics.binary_accuracy(all_binary_y, all_pred)

        return result

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

    if args.do_eval:
        logger.info('#################### eval result at step {} ####################'.format(global_step))
        text_dev_set = processor.read_dataset(data_path, ['val'])
        dev_features = processor.convert_to_ids(text_dev_set)
        dev_dataset = MyDataset(dev_features)

        eval_result = eval(dev_dataset)
        logger.info('eval_eer: {}'.format(eval_result['eer']))
        logger.info('eval_oos_ind_precision: {}'.format(eval_result['oos_ind_precision']))
        logger.info('eval_oos_ind_recall: {}'.format(eval_result['oos_ind_recall']))
        logger.info('eval_oos_ind_f_score: {}'.format(eval_result['oos_ind_f_score']))
        logger.info('eval_auc: {}'.format(eval_result['auc']))
        logger.info('eval_fpr95: {}'.format(eval_result['fpr95']))
        logger.info('eval_accuracy: {}'.format(eval_result['accuracy']))
        logger.info('report')
        logger.info(eval_result['report'])

    if args.do_test:
        text_test_set = processor.read_dataset(data_path, ['test'])
        test_features = processor.convert_to_ids(text_test_set)
        test_dataset = MyDataset(test_features)
        test_result = test(test_dataset)

        save_result(test_result, os.path.join(args.output_dir, 'test_result'))
        logger.info('test_eer: {}'.format(test_result['eer']))
        logger.info('test_oos_ind_precision: {}'.format(test_result['oos_ind_precision']))
        logger.info('test_oos_ind_recall: {}'.format(test_result['oos_ind_recall']))
        logger.info('test_oos_ind_f_score: {}'.format(test_result['oos_ind_f_score']))
        logger.info('test_auc: {}'.format(test_result['auc']))
        logger.info('test_fpr95: {}'.format(test_result['fpr95']))
        logger.info('test_accuracy: {}'.format(test_result['accuracy']))
        logger.info('report')
        logger.info(test_result['report'])

        save_result(test_result, os.path.join(args.output_dir, 'test_result'))

        # 输出错误cases
        texts = [line['text'] for line in text_test_set]
        output_cases(texts, test_result['all_y'], test_result['all_pred'],
                     os.path.join(args.output_dir, 'test_cases.csv'), processor, test_result['test_logit'])

        # confusion matrix
        metrics.plot_confusion_matrix(test_result['all_y'], test_result['all_pred'],
                              args.output_dir)


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

    parser.add_argument('--predict_batch_size', default=16, type=int,
                        help='Batch size for evaluating and testing.')

    parser.add_argument('--patience', default=10, type=int,
                        help='Number of epoch of early stopping patience.')

    parser.add_argument('--fine_tune', action='store_true', default=True,
                        help='Whether to fine tune BERT during training.')

    parser.add_argument('--n_epoch', default=500, type=int,
                        help='Number of epoch for training.')

    parser.add_argument('--lr', type=float, default=4e-5,
                        help="Learning rate for Discriminator.")

    # 解析参数
    args = parser.parse_args()

    # os.chdir('../')

    # Log
    from logger import Logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(os.path.join(args.output_dir, 'train.log'))
    logger.info(os.getcwd())
    main(args)
