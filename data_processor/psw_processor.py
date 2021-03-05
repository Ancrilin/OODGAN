# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: text_processor
@time: 2021/3/5 15:08
"""

import jieba
import re


def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8')as fp:
        sw = [line.strip() for line in fp]  # strip() 方法用于移除字符串头尾指定的字符（默认为空格）
    return sw

'''
    全模式 ：把句子中所有的可以成词的词语都扫描出来
    jieba.cut(text, cut_all=True)
    
    精确模式 ：将句子最精确地切开，适合文本分析
    jieba.cut(text, cut_all=False)  # 默认模式
    
    搜索引擎模式 ：在精确模式的基础上，对长词再次切分，适合用于搜索引擎分词
    jieba.cut_for_search(txt)
'''

# 中文分词并且去停用词
def seg_word(sentence, stopwords, punctuation=False):
    # file_userDict = 'dict.txt'  # 自定义的词典
    # jieba.load_userdict(file_userDict)

    sentence_seged = jieba.cut(sentence.strip())
    # stopwords = load_stopwords()
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '/t':
                outstr += word
                # outstr += " "
    if punctuation:
        outstr = remove_punctuation(outstr)
    return outstr


def smp_psw(dataset, stopwords_path, punctuation=False):
    stopwords = load_stopwords(stopwords_path)
    n_dataset = []
    for line in dataset:
        line['text'] = seg_word(line['text'], stopwords, punctuation)
        n_dataset.append(line)
    return n_dataset


def remove_punctuation(text):
    punctuation = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
    return re.sub(punctuation, "", text)


if __name__ == '__main__':
    sw_path = '../data/smp/stopwords/hit_stopwords.txt'
    sw = load_stopwords(sw_path)
    print(sw)
    text_1 = '打开会说话的汤姆猫'
    text_2 = '打开天天动听并开始播放音乐'
    text_3 = '"+蚂=蚁！花!呗/期?免,息★.---《平凡的世界》：了*解一（#@）个“普通人”波涛汹涌的内心世界！"'
    outstr = seg_word(text_3, sw, punctuation=True)
    print(type(outstr))
    print(outstr)