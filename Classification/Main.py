import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


print("Loading data ...")

data_root="timit_11/"

train=np.load(data_root+"train_11.npy")
train_label=np.load(data_root+"train_label_11.npy")
test=np.load(data_root+"test_11.npy")


print("Size of train data: {}".format(train.shape))
print("Size of train label: {}".format(train_label.shape))
print("Size of test data: {}".format(test.shape))

print("Success loading data")


import torch   
from torch.utils.data import Dataset

class TIMITDataset(Dataset):
    def __init__(self,X, Y=None):
        self.data=torch.from_numpy(X).float()
        if Y is not None :
            Y=Y.astype(int)
            # self.label=torch.LongTensor(Y)
            self.label=torch.from_numpy(Y).long()
        else:
            self.label=None

        super().__init__()
        

    def __getitem__(self, key):
        if self.label is not None:
            return self.data[key], self.label[key]
        else:
            return self.data[key]

    def __len__(self):
        # return self.data.shape[0]
        return len(self.data)

import torch.nn as nn
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(429,1024)
        self.layer2=nn.Linear(1024,512)
        self.layer3=nn.Linear(512,128)
        self.out=nn.Linear(128,39)

        self.act_fn=nn.Sigmoid()


    def forward(self,x):
        x=self.layer1(x)
        x=self.act_fn(x)

        x=self.layer2(x)
        x=self.act_fn(x)

        x=self.layer3(x)
        x=self.act_fn(x)

        x=self.out(x)

        return x


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available() :
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

same_seeds(0)
VAL_RAT=0.2
BATCH_SIZE=512

percent=int(train.shape[0]*(1-VAL_RAT))


train_x, train_y,val_X, val_y= train[:percent], train_label[:percent], train[percent:], train_label[percent:]

train_set=TIMITDataset(train_x,train_y)
val_set=TIMITDataset(val_X,val_y)

from torch.utils.data import DataLoader
train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
val_loader=DataLoader(val_set,batch_size=BATCH_SIZE,shuffle=False)


import gc
del train, train_label, train_x, train_y, val_X, val_y
gc.collect()



def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


device=get_device()
print("Device: {}".format(device))

num_epoch=20
learning_rate=0.0001

model_path='./model.ckpt'

model=Classifier().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), learning_rate)



best_acc=0.0
train_acc_history=[]
val_acc_history=[]
train_loss_history=[]
val_loss_history=[]

for epoch in range(num_epoch):
    train_acc=0
    train_loss=0.0

    model.train()
    for i,data in enumerate(train_loader):
        inputs, labels=data
        inputs, labels=inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs=model(inputs)
        batch_loss=criterion(outputs,labels)
        batch_loss.backward()
        optimizer.step()
        _, train_pred=torch.max(outputs, 1)

        train_acc+=(labels.cpu()==train_pred.cpu()).sum().item()
        train_loss+=batch_loss.item()

    epoch_train_acc=train_acc/len(train_set)
    epoch_train_loss=train_loss/len(train_loader)
    print("epoch: %d, Train Acc: %.2f, Train Loss: %.2f"%(epoch, epoch_train_acc, epoch_train_loss))

    val_acc=0
    val_loss=0.0
    if len(val_set)>0 :
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels=data
                inputs, labels=inputs.to(device), labels.to(device)
                outputs=model(inputs)
                batch_loss=criterion(outputs,labels)
                _, val_pred=torch.max(outputs, 1)
                val_acc+=(labels.cpu()==val_pred.cpu()).sum().item()
                val_loss+=batch_loss.item()

        epoch_val_acc=val_acc/len(val_set)
        epoch_val_loss=val_loss/len(val_loader)
        print("epoch: %d, Val Acc: %.2f, Val Loss: %.2f"%(epoch, epoch_val_acc, epoch_val_loss))

        train_acc_history.append(epoch_train_acc)
        val_acc_history.append(epoch_val_acc)
        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)

        if val_acc> best_acc :
            best_acc=val_acc
            torch.save(model.state_dict(), model_path)
            print("The best torch is saved")
    # else:
        # print("epoch: %d, Train Acc: %.2f, Train Loss: %.2f"%(epoch, train_acc/len(train_set), train_loss/len(train_loader)))


if len(val_set)==0:
    torch.save(model.state_dict(), model_path)
    print("The model is saved")
else:
    epochs=range(1, num_epoch+1)
    fig, axes=plt.subplots(2, 1, figsize=(10, 10))

    axes[0].plot(epochs, train_acc_history, label="Train Accuracy")
    axes[0].plot(epochs, val_acc_history, label="Validation Accuracy")
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(epochs, train_loss_history, label="Train Loss")
    axes[1].plot(epochs, val_loss_history, label="Validation Loss")
    axes[1].set_title("Training and Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    figure_path=Path(__file__).resolve().parent/"training_curves.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print("Training curves are saved to {}".format(figure_path))
