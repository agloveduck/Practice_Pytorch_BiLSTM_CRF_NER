from utils import *
from model import *
from config import *

if __name__ == '__main__':
    dataset = Dataset('test')  # 加载测试集
    loader = data.DataLoader(dataset, batch_size=200, collate_fn=collate_fn)

    with torch.no_grad():
        model = torch.load(MODEL_DIR + 'model_99.pth', map_location=DEVICE)  # 加载train好的model
        # 测试准确率思路  经过crf后，list里每一个list长度不一致
        y_true_list = []  # 真实值  大的一维list
        y_pred_list = []  # 所有数据预测值  大的一维list
        #
        # id2label, _ = get_label()

        for b, (input, target, mask) in enumerate(loader):
            # 返回的target是padding后的 但是 crf处理后的预测值是没有padding的
            y_pred = model(input, mask)  #  list里各元素长度不一致
            loss = model.loss_fn(input, target, mask)
            # print(len(target[1]))  # eg. 74
            # print(len(y_pred[1]))  # eg. 63
            # exit()

            # print('>> batch:', b, 'loss:', loss.item())
        
            # 拼接返回值
            for lst in y_pred:
                y_pred_list += lst
            for y,m in zip(target, mask):
                y_true_list += y[m==True].tolist()  # 统计真实list值时 需要将填充值 去掉
            print(len(y_pred_list))
            print(len(y_true_list))
            exit()
        #     for lst in y_pred:
        #         y_pred_list.append([id2label[i] for i in lst])
        #     for y,m in zip(target, mask):
        #         y_true_list.append([id2label[i] for i in y[m==True].tolist()])
        #
        # print(report(y_true_list, y_pred_list))


        # # 整体准确率
        y_true_tensor = torch.tensor(y_true_list)  # 把这个转换为 tensor
        y_pred_tensor = torch.tensor(y_pred_list)
        accuracy = (y_true_tensor == y_pred_tensor).sum()/len(y_true_tensor)
        print('>> total:', len(y_true_tensor), 'accuracy:', accuracy.item())
        