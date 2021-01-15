# coding:utf-8

import logging


class Logger:
    def __init__(self, log_save_path):
        # 创建logger实例
        logger = logging.getLogger(__name__)
        # 默认为WARNGING， 设置输出级别为INFO
        logger.setLevel(level=logging.INFO)

        # output log to file
        handler = logging.FileHandler(log_save_path)
        formatter = logging.Formatter('%(asctime)s - %(filename)s - %(name)s - %(levelname)s : %(message)s')
        handler.setFormatter(formatter)

        # output log to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)

        # set them to logging attributes
        # 为logger实例增加一个处理器
        logger.addHandler(handler)
        logger.addHandler(console)

        #
        logger.removeHandler("stderr")

        # set self attribute
        self.log = logger

    def info(self, message):
        self.log.info(message)

    def debug(self, message):
        self.log.debug(message)

    def warning(self, message):
        self.log.warning(message)
