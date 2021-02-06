# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: run_gan_2
@time: 2021/2/4 19:57
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
from transformers import BertModel
from sklearn.metrics import roc_auc_score
from utils.tools import check_manual_seed
from config import BertConfig, Config
from data_processor.smp_processor import SMP_Processor
from data_processor.oos_eval_processor import OOS_Eval_Processor
from data_utils.dataset import MyDataset
from utils.tools import EarlyStopping, ErrorRateAt95Recall, save_gan_model, load_gan_model, save_result, save_model, output_cases, save_feature, std_mean
import utils.metrics as metrics
from model.gan_2 import Discriminator, Generator
import utils.tools as tools
from sklearn.manifold import TSNE
from utils import visualization
import json
from data_processor.entity_processor import EntityProcessor

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

gross_result = {}
gross_result['type'] = ['ind', 'oos']

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

    logger.info('device: ' + device)

    logger.info('fake_sample_weight: ' + str(args.fake_sample_weight))
    logger.info('train_batch_size: ' + str(args.train_batch_size))
    logger.info('predict_batch_size: ' + str(args.predict_batch_size))

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
    if args.dataset == 'smp':
        processor = SMP_Processor(bert_config, maxlen=32)
    else:
        processor = OOS_Eval_Processor(bert_config, maxlen=32)
    processor.load_label(label_path)    #加载label, 生成label_to_ids 与 ids_to_label

    n_class = len(processor.id_to_label)
    # config = vars(args)  # 返回参数字典
    config = args.__dict__
    config['n_class'] = n_class
    config['gan_save_path'] = os.path.join(args.output_dir, 'save', 'gan.pt')
    config['bert_save_path'] = os.path.join(args.output_dir, 'save', 'bert.pt')

    logger.info('n_class: ' + str(n_class))
    logger.info('id_to_label: ' + str(processor.id_to_label))
    logger.info('label_to_id: ' + str(processor.label_to_id))

    # 输出数据集分布
    # with open(data_path, 'r', encoding='utf-8') as fp:
    #     source = json.load(fp)
    #     for type in source:
    #         n = 0
    #         n_id = 0
    #         n_ood = 0
    #         text_len = {}
    #         for line in source[type]:
    #             if line['domain'] == 'chat':
    #                 n_ood += 1
    #             else:
    #                 n_id += 1
    #             n += 1
    #             text_len[len(line['text'])] = text_len.get(len(line['text']), 0) + 1
    #         print(type, n)
    #         print('ood', n_ood)
    #         print('id', n_id)
    #         print(sorted(text_len.items(), key=lambda d: d[0], reverse=False))

    logger.info('BertPreTrainModelDir: ' + bert_config['PreTrainModelDir'])

    D = Discriminator(config)
    G = Generator(config)
    E = BertModel.from_pretrained(bert_config['PreTrainModelDir'])  # Bert encoder

    logger.info("fine_tune: " + str(args.fine_tune))
    if args.fine_tune:
        for param in E.parameters():
            param.requires_grad = True
    else:
        for param in E.parameters():
            param.requires_grad = False

    D.to(device)
    G.to(device)
    E.to(device)

    # 全局迭代步长
    global_step = 0

    def train(train_dataset, dev_dataset):
        logger.info('-------------------------------------------------')
        logger.info('training...')
        logger.info('Loading train_dataloader...')
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.train_batch_size,
                                      shuffle=True,
                                      num_workers=2)
        n_sample = len(train_dataloader)
        early_stopping = EarlyStopping(args.patience, logger=logger)

        # Loss function
        # adversarial_loss = torch.nn.BCELoss().to(device)
        adversarial_loss = torch.nn.CrossEntropyLoss().to(device)
        classified_loss = torch.nn.CrossEntropyLoss().to(device)

        logger.info('G_lr: ' + str(args.G_lr))
        logger.info('D_lr: ' + str(args.D_lr))
        logger.info('bert_lr: ' + str(args.bert_lr))
        # Optimizers
        optimizer_G = torch.optim.Adam(G.parameters(), lr=args.G_lr)  # optimizer for generator
        optimizer_D = torch.optim.Adam(D.parameters(), lr=args.D_lr)  # optimizer for discriminator
        optimizer_E = AdamW(E.parameters(), args.bert_lr)

        G_total_train_loss = []
        D_total_fake_loss = []
        D_total_real_loss = []
        FM_total_train_loss = []
        D_total_class_loss = []
        valid_detection_loss = []
        valid_oos_ind_recall = []
        valid_oos_ind_f_score = []
        valid_ind_class_acc = []
        valid_oos_ind_precision = []

        iteration = 0

        nonlocal global_step
        for i in range(args.n_epoch):

            # Initialize model state
            G.train()
            D.train()
            E.train()

            G_train_loss = 0
            D_fake_loss = 0
            D_real_loss = 0
            FM_train_loss = 0
            D_class_loss = 0

            for sample in tqdm.tqdm(train_dataloader):
                # 数据加载到设备
                sample = (i.to(device) for i in sample)
                token, mask, type_ids, y = sample
                batch = len(token)

                # the label used to train generator and discriminator.
                true_label = LongTensor(batch, 1).fill_(1.0).detach()
                fake_label = LongTensor(batch, 1).fill_(0.0).detach()

                # bert encoder real_sample
                optimizer_E.zero_grad()
                sequence_output, pooled_output = E(token, mask, type_ids, return_dict=False)
                real_feature = pooled_output

                # train D on real
                optimizer_D.zero_grad()
                real_f_vector, discriminator_output, classification_output = D(real_feature, return_feature=True)
                # discriminator_output = discriminator_output.squeeze()
                # print('discriminator_output', discriminator_output.size())
                # print(discriminator_output)
                real_loss = adversarial_loss(discriminator_output, (y != 0.0).long())  # chat=0 ood=0 ind=1
                if n_class > 2:  # 大于2表示除了训练判别器还要训练分类器 binary 只训练判别器
                    class_loss = classified_loss(classification_output, y.long())
                    real_loss += class_loss
                    D_class_loss += class_loss.detach()
                # print('real loss', real_loss.size())
                # print(real_loss)
                real_loss.backward()

                # train D on fake
                z = FloatTensor(np.random.normal(0, 1, (batch, args.G_z_dim))).to(device)
                fake_feature = G(z).detach()
                fake_discriminator_output = D.detect_only(fake_feature)
                fake_loss = args.fake_sample_weight * adversarial_loss(fake_discriminator_output, fake_label.squeeze())
                fake_loss.backward()
                optimizer_D.step()

                if args.fine_tune:
                    optimizer_E.step()

                # train G
                optimizer_G.zero_grad()
                z = FloatTensor(np.random.normal(0, 1, (batch, args.G_z_dim))).to(device)
                fake_f_vector, D_decision = D.detect_only(G(z), return_feature=True)
                gd_loss = adversarial_loss(D_decision, true_label.squeeze())
                fm_loss = torch.abs(torch.mean(real_f_vector.detach(), 0) - torch.mean(fake_f_vector, 0)).mean()
                g_loss = gd_loss + 0 * fm_loss  # 简单除去FM项损失
                g_loss.backward()
                optimizer_G.step()

                global_step += 1

                D_fake_loss += fake_loss.detach()
                D_real_loss += real_loss.detach()
                G_train_loss += g_loss.detach() + fm_loss.detach()
                FM_train_loss += fm_loss.detach()

            # logger.info('[Epoch {}] Train: D_fake_loss: {}'.format(i, D_fake_loss / n_sample))
            # logger.info('[Epoch {}] Train: D_real_loss: {}'.format(i, D_real_loss / n_sample))
            # logger.info('[Epoch {}] Train: D_class_loss: {}'.format(i, D_class_loss / n_sample))
            # logger.info('[Epoch {}] Train: G_train_loss: {}'.format(i, G_train_loss / n_sample))
            # logger.info('[Epoch {}] Train: FM_train_loss: {}'.format(i, FM_train_loss / n_sample))
            # logger.info('---------------------------------------------------------------------------')

            D_total_fake_loss.append(D_fake_loss / n_sample)
            D_total_real_loss.append(D_real_loss / n_sample)
            D_total_class_loss.append(D_class_loss / n_sample)
            G_total_train_loss.append(G_train_loss / n_sample)
            FM_total_train_loss.append(FM_train_loss / n_sample)

            iteration += 1

            # 验证集
            if dev_dataset:
                logger.info('#################### eval result at step {} ####################'.format(global_step))
                eval_result = eval(dev_dataset)

                valid_detection_loss.append(eval_result['detection_loss'])
                valid_oos_ind_precision.append(eval_result['oos_ind_precision'])
                valid_oos_ind_recall.append(eval_result['oos_ind_recall'])
                valid_oos_ind_f_score.append(eval_result['oos_ind_f_score'])

                # 1 表示要保存模型
                # 0 表示不需要保存模型
                # -1 表示不需要模型，且超过了patience，需要early stop
                signal = early_stopping(-eval_result['eer'])
                if signal == -1:
                    break
                elif signal == 0:
                    pass
                elif signal == 1:
                    save_gan_model(D, G, config['gan_save_path'])
                    if args.fine_tune:
                        save_model(E, path=config['bert_save_path'], model_name='bert')


                # logger.info('valid_eer: {}'.format(eval_result['eer']))
                # logger.info('valid_oos_ind_precision: {}'.format(eval_result['oos_ind_precision']))
                # logger.info('valid_oos_ind_recall: {}'.format(eval_result['oos_ind_recall']))
                # logger.info('valid_oos_ind_f_score: {}'.format(eval_result['oos_ind_f_score']))
                # logger.info('valid_auc: {}'.format(eval_result['auc']))
                # logger.info('valid_fpr95: {}'.format(eval_result['fpr95']))
                # logger.info('valid_accuracy: {}'.format(eval_result['accuracy']))
                # logger.info('\n' + eval_result['report'])

        from utils.visualization import draw_curve
        draw_curve(D_total_fake_loss, iteration, 'D_total_fake_loss', args.output_dir)
        draw_curve(D_total_real_loss, iteration, 'D_total_real_loss', args.output_dir)
        draw_curve(D_total_class_loss, iteration, 'D_total_class_loss', args.output_dir)
        draw_curve(G_total_train_loss, iteration, 'G_total_train_loss', args.output_dir)
        draw_curve(FM_total_train_loss, iteration, 'FM_total_train_loss', args.output_dir)

        if dev_dataset:
            draw_curve(valid_detection_loss, iteration, 'valid_detection_loss', args.output_dir)
            # draw_curve(valid_ind_class_acc, iteration, 'valid_ind_class_accuracy', args.output_dir)

        if args.patience >= args.n_epoch:
            tools.save_gan_model(D, G, config['gan_save_path'])
            if args.fine_tune:
                save_model(E, path=config['bert_save_path'], model_name='bert')

    def eval(dataset):
        logger.info('-------------------------------------------------')
        logger.info('evaluating...')
        logger.info('Loading eval_dataloader...')
        dev_dataloader = DataLoader(dataset, batch_size=args.predict_batch_size, shuffle=False, num_workers=2)
        n_sample = len(dev_dataloader)
        result = dict() # eval result

        # Loss function
        # detection_loss = torch.nn.BCELoss().to(device)
        detection_loss = torch.nn.CrossEntropyLoss().to(device)
        classified_loss = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

        G.eval()
        D.eval()
        E.eval()

        all_detection_preds = []
        all_class_preds = []
        all_detection_logits = []

        for sample in tqdm.tqdm(dev_dataloader):
            sample = (i.to(device) for i in sample)
            token, mask, type_ids, y = sample
            batch = len(token)

            # -------------------------evaluate D------------------------- #
            # BERT encode sentence to feature vector

            with torch.no_grad():
                sequence_output, pooled_output = E(token, mask, type_ids, return_dict=False)
                real_feature = pooled_output

                # 大于2表示除了训练判别器还要训练分类器
                if n_class > 2:
                    f_vector, discriminator_output, classification_output = D(real_feature, return_feature=True)
                    all_detection_logits.append(discriminator_output)
                    all_detection_preds.append(torch.argmax(discriminator_output, 1))
                    all_class_preds.append(classification_output)

                # 只预测判别器
                else:
                    f_vector, discriminator_output = D.detect_only(real_feature, return_feature=True)
                    all_detection_logits.append(discriminator_output)
                    all_detection_preds.append(torch.argmax(discriminator_output, 1))

        all_y = LongTensor(dataset.dataset[:, -1].astype(int)).cpu()  # [length, n_class]
        all_binary_y = (all_y != 0).long()  # [length, 1] label 0 is oos    二分类ood， ind 真实label

        all_detection_preds = torch.cat(all_detection_preds, 0).cpu()  # [length, 1]
        all_detection_binary_preds = tools.convert_to_int_by_threshold(all_detection_preds.squeeze())  # [length, 1]
        all_detection_logits = torch.cat(all_detection_logits, 0).cpu() # 拼接 list

        # 计算损失
        all_detection_preds = all_detection_preds.squeeze()
        detection_loss = detection_loss(all_detection_logits, all_binary_y.long())
        result['detection_loss'] = detection_loss

        if n_class > 2:
            class_one_hot_preds = torch.cat(all_class_preds, 0).detach().cpu()  # one hot label
            class_loss = classified_loss(class_one_hot_preds, all_y)  # compute loss
            all_class_preds = torch.argmax(class_one_hot_preds, 1)  # label
            class_acc = metrics.ind_class_accuracy(all_class_preds, all_y, oos_index=0)  # accuracy for ind class
            logger.info(metrics.classification_report(all_y, all_class_preds, target_names=processor.id_to_label))


        y_score = all_detection_preds.squeeze().tolist()
        eer = metrics.cal_eer(all_binary_y, y_score)
        oos_ind_precision, oos_ind_recall, oos_ind_fscore, _ = metrics.binary_recall_fscore(
            all_detection_binary_preds, all_y)
        ind_class_acc = metrics.ind_class_accuracy(all_detection_binary_preds, all_y)
        fpr95 = ErrorRateAt95Recall(all_binary_y, y_score)

        report = metrics.binary_classification_report(all_y, all_detection_binary_preds)

        result['eer'] = eer
        result['ind_class_acc'] = ind_class_acc
        result['loss'] = detection_loss / n_sample # avg loss
        result['y_score'] = y_score
        result['all_binary_y'] = all_binary_y
        result['oos_ind_precision'] = oos_ind_precision
        result['oos_ind_recall'] = oos_ind_recall
        result['oos_ind_f_score'] = oos_ind_fscore
        result['auc'] = roc_auc_score(all_binary_y, y_score)
        result['fpr95'] = fpr95
        result['report'] = report
        result['accuracy'] = metrics.binary_accuracy(all_detection_binary_preds, all_binary_y)

        return result

    def test(dataset):
        # load BERT and GAN
        load_gan_model(D, G, config['gan_save_path'])
        test_dataloader = DataLoader(dataset, batch_size=args.predict_batch_size, shuffle=False, num_workers=2)
        n_sample = len(test_dataloader)
        result = dict()
        G.eval()
        D.eval()
        E.eval()

        # detection_loss = torch.nn.BCELoss().to(device)
        detection_loss = torch.nn.CrossEntropyLoss().to(device)
        classified_loss = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

        all_detection_preds = []
        all_class_preds = []
        all_detection_logits = []
        all_features = []

        for sample in tqdm.tqdm(test_dataloader):
            sample = (i.to(device) for i in sample)
            token, mask, type_ids, y = sample
            batch = len(token)

            # -------------------------evaluate D------------------------- #
            # BERT encode sentence to feature vector

            with torch.no_grad():
                sequence_output, pooled_output = E(token, mask, type_ids, return_dict=False)
                real_feature = pooled_output

                # 大于2表示除了训练判别器还要训练分类器
                if n_class > 2:
                    f_vector, discriminator_output, classification_output = D(real_feature, return_feature=True)
                    all_detection_logits.append(discriminator_output)
                    all_detection_preds.append(torch.argmax(discriminator_output, 1))
                    all_class_preds.append(classification_output)

                # 只预测判别器
                else:
                    f_vector, discriminator_output = D.detect_only(real_feature, return_feature=True)
                    all_detection_logits.append(discriminator_output)
                    all_detection_preds.append(torch.argmax(discriminator_output, 1))
                    if args.do_vis:
                        all_features.append(f_vector)

        all_y = LongTensor(dataset.dataset[:, -1].astype(int)).cpu()  # [length, n_class] 原始真实label
        all_binary_y = (all_y != 0).long()  # [length, 1] label 0 is oos    二分类ood， ind 真实label

        all_detection_preds = torch.cat(all_detection_preds, 0).cpu()  # [length, 1]
        all_detection_binary_preds = tools.convert_to_int_by_threshold(all_detection_preds.squeeze())  # [length, 1]
        all_detection_logits = torch.cat(all_detection_logits, 0).cpu()  # 拼接 list

        # 计算损失
        all_detection_preds = all_detection_preds.squeeze()
        detection_loss = detection_loss(all_detection_logits, all_binary_y.long())
        result['detection_loss'] = detection_loss

        if n_class > 2:
            class_one_hot_preds = torch.cat(all_class_preds, 0).detach().cpu()  # one hot label
            class_loss = classified_loss(class_one_hot_preds, all_y)  # compute loss
            all_class_preds = torch.argmax(class_one_hot_preds, 1)  # label
            class_acc = metrics.ind_class_accuracy(all_class_preds, all_y, oos_index=0)  # accuracy for ind class
            logger.info(metrics.classification_report(all_y, all_class_preds, target_names=processor.id_to_label))

        y_score = all_detection_preds.squeeze().tolist()
        eer = metrics.cal_eer(all_binary_y, y_score)
        oos_ind_precision, oos_ind_recall, oos_ind_fscore, _ = metrics.binary_recall_fscore(
            all_detection_binary_preds, all_y)
        ind_class_acc = metrics.ind_class_accuracy(all_detection_binary_preds, all_y)
        fpr95 = ErrorRateAt95Recall(all_binary_y, y_score)

        report = metrics.binary_classification_report(all_y, all_detection_binary_preds)

        result['eer'] = eer
        result['ind_class_acc'] = ind_class_acc
        result['loss'] = detection_loss / n_sample  # avg loss
        result['y_score'] = y_score
        result['all_binary_y'] = all_binary_y
        result['oos_ind_precision'] = oos_ind_precision
        result['oos_ind_recall'] = oos_ind_recall
        result['oos_ind_f_score'] = oos_ind_fscore
        result['auc'] = roc_auc_score(all_binary_y, y_score)
        result['fpr95'] = fpr95
        result['report'] = report
        result['accuracy'] = metrics.binary_accuracy(all_binary_y, all_detection_binary_preds)
        result['all_y'] = all_y
        result['all_detection_binary_preds'] = all_detection_binary_preds
        result['all_detection_preds'] = all_detection_preds

        if n_class > 2:
            result['class_loss'] = class_loss
            result['class_acc'] = class_acc

        if args.do_vis:
            all_features = torch.cat(all_features, 0).cpu().numpy()
            result['all_features'] = all_features

        return result

    def get_fake_feature(num_output):
        """
        生成一定数量的假特征
        """
        G.eval()
        fake_features = []
        start = 0
        batch = args.predict_batch_size
        with torch.no_grad():
            while start < num_output:
                end = min(num_output, start + batch)
                z = FloatTensor(np.random.normal(0, 1, size=(end - start, args.G_z_dim)))
                fake_feature = G(z)
                f_vector, _ = D.detect_only(fake_feature, return_feature=True)
                fake_features.append(f_vector)
                start += batch
            return torch.cat(fake_features, 0).cpu().numpy()

    if args.do_train:
        # 读取数据
        # 格式为[{'text': '\ufeff打开熊猫看书', 'domain': 'in'}]
        text_train_set = processor.read_dataset(data_path, ['train'])
        text_dev_set = processor.read_dataset(data_path, ['val'])

        # 去除训练集中的ood数据
        if args.remove_oodp and args.dataset == "smp":
            logger.info('remove ood data in train_dataset')
            text_train_set = [sample for sample in text_train_set if sample['domain'] != 'chat']    # chat is ood data

        # 挖去实体词汇
        if args.remove_entity and args.dataset == "smp":
            logger.info('remove entity in train_dataset')
            entity_processor = EntityProcessor('data/smp/训练集 全知识标记.xlsx')
            # logger.info(entity_processor.compiled)
            text_train_set, num = entity_processor.remove_smp_entity(text_train_set)
            logger.info('the number of solved entity data: ' + str(num))

        if args.minlen != -1 and args.dataset == "smp":
            logger.info('remove minlen data')
            logger.info('minlen: ' + str(args.minlen))
            previous_len = len(text_train_set)
            logger.info('previous len: ' + str(previous_len))
            text_train_set = processor.remove_minlen(dataset=text_train_set, minlen=args.minlen)
            removed_len = len(text_train_set)
            logger.info('removed len: ' + str(removed_len))
            logger.info('the number of removed minlen data: ' + str(previous_len - removed_len))

        if args.maxlen != -1 and args.dataset == "smp":
            logger.info('remove maxlen data')
            logger.info('maxlen: ' + str(args.maxlen))
            previous_len = len(text_train_set)
            logger.info('previous len: ' + str(previous_len))
            text_train_set = processor.remove_maxlen(text_train_set, maxlen=args.maxlen)
            removed_len = len(text_train_set)
            logger.info('removed len: ' + str(removed_len))
            logger.info('the number of removed maxlen data: ' + str(previous_len - removed_len))

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
        logger.info('\n' + eval_result['report'])

        gross_result['eval_oos_ind_precision'] = eval_result['oos_ind_precision']
        gross_result['eval_oos_ind_recall'] = eval_result['oos_ind_recall']
        gross_result['eval_oos_ind_f_score'] = eval_result['oos_ind_f_score']
        gross_result['eval_eer'] = eval_result['eer']
        gross_result['eval_fpr95'] = ErrorRateAt95Recall(eval_result['all_binary_y'], eval_result['y_score'])
        gross_result['eval_auc'] = eval_result['auc']

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
        logger.info('\n' + test_result['report'])

        save_result(test_result, os.path.join(args.output_dir, 'test_result'))

        gross_result['test_oos_ind_precision'] = test_result['oos_ind_precision']
        gross_result['test_oos_ind_recall'] = test_result['oos_ind_recall']
        gross_result['test_oos_ind_f_score'] = test_result['oos_ind_f_score']
        gross_result['test_eer'] = test_result['eer']
        gross_result['test_fpr95'] = ErrorRateAt95Recall(test_result['all_binary_y'], test_result['y_score'])
        gross_result['test_auc'] = test_result['auc']

        # 输出错误cases
        if args.dataset == "smp":
            texts = [line['text'] for line in text_test_set]
        else:
            texts = [line[0] for line in text_test_set]
        output_cases(texts, test_result['all_y'], test_result['all_detection_binary_preds'],
                     os.path.join(args.output_dir, 'test_cases.csv'), processor, test_result['all_detection_preds'])

        # confusion matrix
        metrics.plot_confusion_matrix(test_result['all_y'], test_result['all_detection_binary_preds'],
                              args.output_dir)

    if args.do_vis:
        # [2 * length, feature_fim]
        features = np.concatenate([test_result['all_features'], get_fake_feature(len(test_dataset) // 2)], axis=0)
        features = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(features)  # [2 * length, 2]
        # [2 * length, １]
        if n_class > 2:
            labels = np.concatenate([test_result['all_y'], np.array([-1] * (len(test_dataset) // 2))], 0).reshape(
                (-1, 1))
        else:
            labels = np.concatenate([test_result['all_binary_y'], np.array([-1] * (len(test_dataset) // 2))],
                                    0).reshape((-1, 1))
        # [2 * length, 3]
        data = np.concatenate([features, labels], 1)
        fig = visualization.scatter_plot(data, processor)
        fig.savefig(os.path.join(args.output_dir, 'plot.png'))
        fig.show()

    gross_result['seed'] = args.seed
    if args.result != 'no':
        pd_result = pd.DataFrame(gross_result)
        if args.seed == 16:
            pd_result.to_csv(args.result + '_gross_result.csv', index=False)
        else:
            pd_result.to_csv(args.result + '_gross_result.csv', index=False, mode='a', header=False)
        if args.seed == 35085:
            std_mean(args.result + '_gross_result.csv')


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

    # binary_smp_full base
    # binary_smp_full_v2 自己排除知识
    # binary_smp_full_v3 知识库排除
    # binary_smp_full_v4 知识库+自己

    parser.add_argument('--data_file', required=True, type=str,
                        help="""Which type of dataset to be used, 
                            i.e. binary_undersample.json, binary_wiki_aug.json. Detail in config/data.ini""")

    parser.add_argument('--bert_type',
                        choices={'bert-base-uncased', 'bert-large-uncased', 'bert-base-chinese', 'chinese-bert-wwm'},
                        required=True,
                        help='Type of the pre-trained BERT to be used.')

    # ------------------------Discriminator------------------------ #
    parser.add_argument('--D_Wf_dim', default=512, type=int,
                        help='The Dimension of Wf for Discriminator.')

    # ------------------------Generator------------------------ #
    parser.add_argument('--G_z_dim', default=512, type=int,
                        help='The Dimension of z (noise) for Generator.')

    parser.add_argument('--feature_dim', default=768, type=int,
                        help='The Dimension of feature vector for Generator output and Discriminator input.')

    # ------------------------action------------------------ #
    parser.add_argument('--do_train', action='store_true',
                        help='Do training step')

    parser.add_argument('--do_eval', action='store_true',
                        help='Do evaluation on devset step')

    parser.add_argument('--do_test', action='store_true',
                        help='Do validation on testset step')

    parser.add_argument('--do_vis', action='store_true',
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

    parser.add_argument('--remove_oodp', action='store_true', default=False,
                        help='Whether to remove ood data.')

    parser.add_argument('--remove_entity', action='store_true', default=False,
                        help='Whether to remove entity in data.')

    parser.add_argument('--minlen', default=-1, type=int,
                        help='minlen')
    parser.add_argument('--maxlen', default=-1, type=int,
                        help='maxlen')

    parser.add_argument('--n_epoch', default=500, type=int,
                        help='Number of epoch for training.')

    parser.add_argument('--D_lr', type=float, default=1e-5, help="Learning rate for Discriminator.")
    parser.add_argument('--G_lr', type=float, default=1e-5, help="Learning rate for Generator.")
    parser.add_argument('--bert_lr', type=float, default=2e-4, help="Learning rate for Generator.")

    parser.add_argument('--fake_sample_weight', type=float, default=1.0, help="Weight of fake sample loss for Discriminator.")

    parser.add_argument('--save_model', action='store_true', default=False,
                        help='Whether to save model.')
    parser.add_argument('--result', type=str, default="no")

    # 解析参数
    args = parser.parse_args()

    # os.chdir('../')

    # Log
    from logger import Logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(os.path.join(args.output_dir, 'train.log'))
    logger.info(os.getcwd())
    logger.info('mkdir output_dir: ' + args.output_dir)
    main(args)
    if not args.save_model:
        logger.info('Delete model...')
        tools.removeDir(os.path.join(args.output_dir, 'save',))
    logger.info('Ending')
