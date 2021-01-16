# -*- coding: utf-8 -*-

"""
@author: YuKI
@contact: 1162236967@qq.com
@file: config
@time: 2021/1/14 14:46
"""

from configparser import ConfigParser, SectionProxy
import json
import os


class Config:
    config_parser = ConfigParser()

    # 读取配置文件
    def __init__(self, path):
        self.parse(path)

    def parse(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.config_parser.read_file(f)

    # 获取配置中的某个section
    def __call__(self, section: str) -> SectionProxy:
        """pass the section name and return config for a section"""
        "like dict"
        return self.config_parser[section]


class BertConfig(Config):
    # 读取配置文件
    def __init__(self, path):
        super().__init__(path)

    # 获取配置中的某个section
    # SectionProxy类型可以用数组下标获取section内的键值
    def __call__(self, section) -> SectionProxy:
        # 要先转成dict，python3.6的bug，3.7修复
        param = super().__call__(section)

        file = os.path.join(param['PreTrainModelDir'] + '/', 'config.json')
        # print(file)
        # print(param)
        with open(file, 'r', encoding='utf-8') as f:
            for k, v in json.load(f).items():
                param.__setattr__(k, v)
        # print(param)
        return param


if __name__ == '__main__':
    config = Config('config/data.ini')
    print(config('smp')['DataDir'])
    bertConfig = BertConfig('config/bert.ini')
    bertConfig = bertConfig('bert-base-chinese')
    print(bertConfig.get('VocabFile'))