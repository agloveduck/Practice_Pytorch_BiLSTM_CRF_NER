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
EMBEDDING_DIM = 100
HIDDEN_SIZE = 256
TARGET_SIZE = 31
LR = 1e-4
EPOCH = 100

MODEL_DIR = './output/model/'

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'