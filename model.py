import torch.nn as nn
from config import *
from torchcrf import CRF
import torch

class Model(nn.Module):  # 定义模型
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, WORD_PAD_ID)  # embedding层
        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True,  # 设置双向
        )  # lstm层
        self.linear = nn.Linear(2 * HIDDEN_SIZE, TARGET_SIZE)  # linear层 双向所以×2
        self.crf = CRF(TARGET_SIZE, batch_first=True)

    def _get_lstm_feature(self, input):
        out = self.embed(input)
        out, _ = self.lstm(out)
        return self.linear(out)

    def forward(self, input, mask):  #  返回的是一个tensor结构
        out = self._get_lstm_feature(input)
        # return out
        return self.crf.decode(out, mask)
        # return self.crf._viterbi_decode(out, mask)

    def loss_fn(self, input, target, mask):  # 损失函数定义
        y_pred = self._get_lstm_feature(input)
        return -self.crf.forward(y_pred, target, mask, reduction='mean')  # 每一个batch算一个平均损失

if __name__ == '__main__':
    model = Model()
    input = torch.randint(0, 3000, (100, 50))  # 词表范围0-3000 batchsize seq长度
    # print(input.shape)  # torch.Size([100, 50])
    # print(model)  # 打印层级结构
    # exit()
    print(model(input, None).shape)  # 最终输出数据的层级结构 不加CRF时 torch.Size([100, 50, 31]) 100 batchsize seq长度 label种类