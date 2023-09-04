from glob import glob  # 用于读取文件 使用正则的格式来写读取文件的格式（.txt /.ann）
import os
import random
import pandas as pd
from config import *


# 根据标注文件生成对应关系
def get_annotation(ann_path):
    with open(ann_path,encoding='utf-8') as file:  # 打开文件
        anns = {}  # 定义字典来存储 下标和BIO 实体类型
        for line in file.readlines():
            # print(line.split('\t'))  # 根据数据形式进行切分
            # exit()
            arr = line.split('\t')[1].split()  # 按照tab键来进行切分
            name = arr[0]  # 实体类型
            start = int(arr[1])  # 实体对应开始下标
            end = int(arr[-1])   # 实体对应结束下标 左开右闭
            # 标注太长，可能有问题
            if end - start > 50:
                continue
            anns[start] = 'B-' + name
            for i in range(start + 1, end):
                anns[i] = 'I-' + name
        return anns


def get_text(txt_path):  # 打开txt文件
    with open(txt_path,encoding='utf-8') as file:
        return file.read()


# 建立文字和标签对应关系
def generate_annotation():
    for txt_path in glob(ORIGIN_DIR + '*.txt'):  # 获取txt文件路径
        ann_path = txt_path[:-3] + 'ann'  # 获取ann文件路径
        anns = get_annotation(ann_path)  # 字典形式
        text = get_text(txt_path)      # 字符串形式
        # 建立文字和标注对应
        # DataFrame 是一个表格型的数据结构 这里创建两列 word列对应字符 label列初始化全标注为O
        df = pd.DataFrame({'word': list(text), 'label': ['O'] * len(text)})
        df.loc[anns.keys(), 'label'] = list(anns.values())  # 将实体类型为B I 的赋值给label列
        # 导出文件
        file_name = os.path.split(txt_path)[1]  # os.path.split(txt_path) 文件夹路径切割
        df.to_csv(ANNOTATION_DIR + file_name, header=None, index=None)


# 拆分训练集和测试集
def split_sample(test_size=0.3):
    files = glob(ANNOTATION_DIR + '*.txt')
    random.seed(0)
    random.shuffle(files)
    n = int(len(files) * test_size)
    test_files = files[:n]
    train_files = files[n:]
    # 合并文件
    merge_file(train_files, TRAIN_SAMPLE_PATH)
    merge_file(test_files, TEST_SAMPLE_PATH)


def merge_file(files, target_path):
    with open(target_path, 'a',encoding='utf-8') as file:
        for f in files:
            text = open(f).read()
            file.write(text)


# 生成词表
def generate_vocab():
    df = pd.read_csv(TRAIN_SAMPLE_PATH, usecols=[0], names=['word'])
    vocab_list = [WORD_PAD, WORD_UNK] + df['word'].value_counts().keys().tolist()
    vocab_list = vocab_list[:VOCAB_SIZE]
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab = pd.DataFrame(list(vocab_dict.items()))
    vocab.to_csv(VOCAB_PATH, header=None, index=None)


# 生成标签表
def generate_label():
    df = pd.read_csv(TRAIN_SAMPLE_PATH, usecols=[1], names=['label'])
    label_list = df['label'].value_counts().keys().tolist()
    label_dict = {v: k for k, v in enumerate(label_list)}
    label = pd.DataFrame(list(label_dict.items()))
    label.to_csv(LABEL_PATH, header=None, index=None)


if __name__ == '__main__':
    # anns = get_annotation('./input/origin/0.ann')
    # print(anns)  # 测试根据后缀名为ann的文件进行标注 返回一个字典 数据内容类似于 1845: 'B-Disease', 1846: 'I-Disease'
    # 建立文字和标签对应关系
    generate_annotation()
    #
    # # 拆分训练集和测试集
    # split_sample()
    #
    # # 生成词表
    # generate_vocab()
    #
    # # 生成标签表
    # generate_label()
