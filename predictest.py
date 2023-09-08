from utils import *
from model import *
from config import *

if __name__ == '__main__':
    # 模型预测
    text = '每个糖尿病患者,无论是病情轻重,不论是注射胰岛素,还是口服降糖药,都必须合理地控制饮食。'  # 真实场景可能是从前端拿来的
    _, word2id = get_vocab()
    input = torch.tensor([[word2id.get(w, WORD_UNK_ID) for w in text]])  # 词典里没有的字用 WORD_UNK_ID兜底
    mask = torch.tensor([[1] * len(text)]).bool()  #因为不需要padding 所以mask每一位都取1

    model = torch.load(MODEL_DIR + 'model_99.pth', map_location=DEVICE)
    y_pred = model(input, mask)
    id2label, _ = get_label()

    label = [id2label[l] for l in y_pred[0]]
    print(text) # 输入的句子
    print(label) # 该句子对应的label BIO标注法
    # 信息提取

    info = extract(label, text)
    print(info)
    # dataset = Dataset()
    # loader = data.DataLoader(
    #     dataset,
    #     batch_size=100,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    # )
