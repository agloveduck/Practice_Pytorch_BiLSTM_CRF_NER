ORIGIN_DIR = './input/origin/'  # 大写变量可以当成常量来使用
ANNOTATION_DIR = './output/annotation/'

TRAIN_SAMPLE_PATH = './output/train_sample.txt'
TEST_SAMPLE_PATH = './output/test_sample.txt'

VOCAB_PATH = './output/vocab.txt'
LABEL_PATH = './output/label.txt'

WORD_PAD = '<PAD>'  # 填充字符 补长
WORD_UNK = '<UNK>'  # 没有见过的词

WORD_PAD_ID = 0  # vocab.txt <PAD>的下标
WORD_UNK_ID = 1
LABEL_O_ID = 0  # label.txt o的下标

VOCAB_SIZE = 3000  # 词表大小
EMBEDDING_DIM = 100  # 词向量维度
HIDDEN_SIZE = 256   # LSTM 输出隐层的大小
TARGET_SIZE = 31  # 经过全连接层输出向量的维度 label种类31个
LR = 1e-4  # 学习率
EPOCH = 100  # 所有数据被训练的总轮数

MODEL_DIR = './output/model/'  # 模型训练好后的保存位置

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'