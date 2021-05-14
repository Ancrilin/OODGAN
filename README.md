# OODGAN
使用的数据集为哈工大第二届的SMP数据集

app文件夹内是运行文件，run_sh内是已经设置了参数的sh运行脚本文件

*patience* 为提前结束训练后再训练几次

*pseudo_sample_weight* 参数设置伪样本的权重

*result* 参数为是否输出结果到excel文件中

*remove_oodp* 为是否除去OOD样本

*entity_mode* 为是否实体词删除模式

*minlen* 为删除短句子的长度

*maxlen* 为删除长句子的长度

*alpha* 为正态分布的置信度

*manual_knowledge* 为是否应用手动标记的实体词

*remove_punctuation* 为是否除去标点符号

*stopwords* 为是否除去停用词

*logarithm* 为是否使用对数正态分布

