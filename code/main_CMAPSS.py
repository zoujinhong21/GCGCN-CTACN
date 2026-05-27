import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models import CombinedModel
from Pre_CMAPSS import MyDataG, MyDataG2, seed


np.set_printoptions(threshold=np.inf)

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


EPOCH = 30         # 50,100,150,200
BATCH_SIZE = 256   # 64,128,256,384,512
LR = 0.01

my_result1 = []
my_result2 = []


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return True



def test_path(all=True, num=1):
    if all:
        file_path = r"..\dataset\CMAPSS\FD002\test_FD002.csv"
    else:
        file_path = r'..\dataset\CMAPSS\FD002\test_FD002_' + str(num) + '.csv'
    return file_path


file_path = test_path()
training_dataset = MyDataG(r"..\dataset\CMAPSS\FD002\train_FD002.csv"
                           )
test_dataset = MyDataG2(file_path)

train_loader = DataLoader(
    dataset=training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


cuda_avail = torch.cuda.is_available()
# cuda_avail = False
model = CombinedModel(num_nodes=14,
    in_features=1,
    hidden_features1=128,
    hidden_features2=64,
    out_features=16,
    K=2,
    )
if cuda_avail:
    model.to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)  # 0.0001

loss_func = torch.nn.MSELoss()
loss_func_mae = torch.nn.L1Loss()

train_list = []
train_loss_list = []
test_list = []
test_loss_list = []

model_save_path = r"..\model_save_directory\CMAPSS\FD002"
mkdir(model_save_path)


# 模型保存
def save_models(epoch):
    torch.save(model.state_dict(), model_save_path + "/model_{}.model".format(epoch))
    print("Checkpoint saved")


def calculate_score(Ot, Pt):
    score_elements = np.where(
        Ot - Pt < 0,
        np.exp(-(Ot - Pt) / 10) - 1,
        np.exp((Ot - Pt) / 13) - 1,
    )

    score = np.sum(score_elements)
    return score

def test():
    model.eval()
    test_loss = 0.0
    true_rul = np.array([])
    prediction_rul = np.array([])
    for step, (b_x, b_y) in enumerate(test_loader):
        if cuda_avail:
            b_x = b_x.to("cuda")
            b_y = b_y.to("cuda")

        outputs = model(b_x)
        loss = loss_func(outputs, b_y)
        test_loss += loss.item() * b_x.size(0)

        true_rul = np.append(true_rul, b_y.cpu().detach().numpy())
        prediction_rul = np.append(prediction_rul, outputs.cpu().detach().numpy())

    test_loss = test_loss / len(test_dataset)
    Score = calculate_score(true_rul, prediction_rul)
    return test_loss, Score


def train():
    for epoch in range(EPOCH):
        model.train()
        train_loss = 0.0

        if epoch <= 10:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
        elif 20 >= epoch >= 11:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00001

        for step, (b_x, b_y) in enumerate(train_loader):
            if cuda_avail:
                b_x = b_x.to("cuda")
                b_y = b_y.to("cuda")

            outputs = model(b_x)
            loss = loss_func(outputs, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)

        train_loss = train_loss / len(training_dataset)
        train_loss_list.append(train_loss)

        test_loss, Score = test()
        test_loss_list.append(test_loss)


        if epoch in [29,EPOCH-1]:
            a = round(math.sqrt(test_loss), 2)
            b = round(Score, 2)
            my_result1.append(a)
            my_result2.append(b)
        else:
            pass

        save_models(epoch)
        print(
            f"Epoch: {epoch}, TrainLoss: {train_loss}, TestLoss: {test_loss}, RMSE: {math.sqrt(test_loss)}, Score: {Score}")
    print('TrainLoss_list: ', train_loss_list)




if __name__ == "__main__":

    train()


print("Final test result(s):")
print(f'{my_result1[0]}\t{my_result2[0]}')
# print(f'{my_result1[1]}\t{my_result2[1]}\t{train_time[0]}\t{test_time[0]}')
# for i in range(0,len(my_result1)+1):
#     print(f'{my_result1[i]}')