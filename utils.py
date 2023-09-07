import torch
from torch.utils import data
from config import *
import pandas as pd
from seqeval.metrics import classification_report
# 切完的字符大小可能不一样 在使用pytorch dataloader 转tensor时会报错
# 同一个batch内要padding到一样长度，不同batch之间可以不一样(并行运算不影响)
# 后续模型CRF阶段计算损失时，可以通过MASK 将填充的PAD数值忽略掉 来消除填充PAD的影响

#  加载词表和标签表
def get_vocab():
    df = pd.read_csv(VOCAB_PATH, names=['word', 'id'])
    return list(df['word']), dict(df.values)


def get_label():
    df = pd.read_csv(LABEL_PATH, names=['label', 'id'])
    return list(df['label']), dict(df.values)  # 返回一个元组 （一个列表 一个字典）


class Dataset(data.Dataset):

    def __init__(self, type='train', base_len=50):  # base_len为切分句子字符长度
        super().__init__()  # 继承pytorch 的 dataset
        self.base_len = base_len
        sample_path = TRAIN_SAMPLE_PATH if type == 'train' else TEST_SAMPLE_PATH
        self.df = pd.read_csv(sample_path, names=['word', 'label'])
        _, self.word2id = get_vocab()
        _, self.label2id = get_label()
        self.get_points()
        # print(self.word2id)
        # exit()
    def get_points(self):  # 计算切割点 points列表记录切分点 0，50，100... 如果那个位置是o 不是后移一位来判断
        self.points = [0]
        i = 0
        while True:
            if i + self.base_len >= len(self.df):
                self.points.append(len(self.df))
                break
            if self.df.loc[i + self.base_len, 'label'] == 'O':
                i += self.base_len
                self.points.append(i)
            else:
                i += 1

    def __len__(self):  # 计算可以切出几段句子
        return len(self.points) - 1

    def __getitem__(self, index):  # 取单个句子 文本数字化 字符和label 标签
        df = self.df[self.points[index]:self.points[index + 1]]  # 根据index切片句子 base_len个字一组 一般
        word_unk_id = self.word2id[WORD_UNK]
        label_o_id = self.label2id['O']
        input = [self.word2id.get(w, word_unk_id) for w in df['word']]  # 根据词表 来找将句子里的字 如果找到 则为word2id里这个这个下标 否则为没见过字的下标
        target = [self.label2id.get(l, label_o_id) for l in df['label']]  # 根据label表 找对应label的id O对应0
        return input, target


def collate_fn(batch):  # 数据校对处理 batch 是一个list
    # print(batch[0])  # 输出值是一个元组 包含两个list 第一个list对应input 每个元素即切分句子的字对应词表里的id  第二个list 对应target 每个元素即该字对应label的id
    # exit()
    # print(batch)  # 一个list list里元素为元组 每个元组如batch[0]所示
    # print(len(batch))

    batch.sort(key=lambda x: len(x[0]), reverse=True)  # 按照句子长度从大到小排序  其他句子填充到和他一样长
    max_len = len(batch[0][0])  # 获取最大长度
    # print(max_len)
    # exit()
    input = []
    target = []
    mask = []
    for item in batch:
        pad_len = max_len - len(item[0])  # 最大长度-本身长度 即为填充的长度
        input.append(item[0] + [WORD_PAD_ID] * pad_len)
        target.append(item[1] + [LABEL_O_ID] * pad_len)
        mask.append([1] * len(item[0]) + [0] * pad_len)  # 本身有字符的地方填1 没有后面填0
    return torch.tensor(input), torch.tensor(target), torch.tensor(mask).bool()  # mask转bool crf要求


def extract(label, text):
    i = 0
    res = []
    while i < len(label):
        if label[i] != 'O':
            prefix, name = label[i].split('-')
            start = end = i
            i += 1
            while i < len(label) and label[i] == 'I-' + name:
                end = i
                i += 1
            res.append([name, text[start:end + 1]])
        else:
            i += 1
    return res


def report(y_true, y_pred):
    return classification_report(y_true, y_pred)
    

if __name__ == '__main__':
    # res = get_label()
    # print(res)
    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=100, collate_fn=collate_fn)
    print(iter(loader).__next__())  # collate_fn的输出 输出元组 三个tensor元素 input target mask
