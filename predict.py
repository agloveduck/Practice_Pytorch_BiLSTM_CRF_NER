from utils import *
from model import *
from config import *

if __name__ == '__main__':
    # text = '每个糖尿病患者,无论是病情轻重,不论是注射胰岛素,还是口服降糖药,都必须合理地控制饮食。'
    # _, word2id = get_vocab()
    # input = torch.tensor([[word2id.get(w, WORD_UNK_ID) for w in text]])
    # mask = torch.tensor([[1] * len(text)]).bool()
    #
    # model = torch.load(MODEL_DIR + 'model_40.pth', map_location=DEVICE)
    # y_pred = model(input, mask)
    # id2label, _ = get_label()
    #
    # label = [id2label[l] for l in y_pred[0]]
    # print(text)
    # print(label)
    #
    # info = extract(label, text)
    # print(info)
    dataset = Dataset()
    loader = data.DataLoader(
        dataset,
        batch_size=100,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)

    for e in range(EPOCH):
        for b,(input,target,mask) in enumerate(loader):
            y_pred = model(input,mask)
            loss = model.loss_fn(input,target,mask)

            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器迭代
            # 三步走完应该有一个损失下降的过程

            if b % 10 == 0:
                print('>>epoch:',e,'loss:',loss.item())  # 训练过程中只要loss是在下降就行

            torch.save(model,MODEL_DIR+f'model_{e}.pth')
            # print(y_pred)  # 二维list
            # exit()