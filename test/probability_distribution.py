# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: probability_distribution
@time: 2021/2/21 10:06
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 按照固定区间长度绘制频率分布直方图
# bins_interval 区间的长度
# margin        设定的左边和右边空留的大小
def probability_distribution(data, bins_interval=1, margin=1):
    bins = range(min(data), max(data) + bins_interval - 1, bins_interval)
    print("bins" + str(len(bins)))
    # for i in range(0, len(bins)):
    #     print(bins[i])
    plt.xlim(min(data) - margin, max(data) + margin)
    plt.title("train dataset")
    plt.xlabel('length')
    plt.ylabel('frequency')
    # 频率分布normed=True，频次分布normed=False
    prob,left,rectangle = plt.hist(x=data, bins=bins, density=False, histtype='bar', color=['r'])
    for x, y in zip(left, prob):
        # 字体上边文字
        # 频率分布数据 normed=True
        plt.text(x + bins_interval / 2, y + 0.003, '%.2f' % y, ha='center', va='top')
        # 频次分布数据 normed=False
        # plt.text(x + bins_interval / 2, y + 0.25, '%.2f' % y, ha='center', va='top')
    plt.show()