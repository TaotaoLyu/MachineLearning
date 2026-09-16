


model_path='./model.ckpt'


import numpy as np


print("Loading data ...")

data_root="timit_11/"

test=np.load(data_root+"test_11.npy")

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

BATCH_SIZE=512

from torch.utils.data import DataLoader
test_set=TIMITDataset(test)
test_loader=DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=False)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


device=get_device()
print("Device: {}".format(device))
model=Classifier().to(device)
model.load_state_dict(torch.load(model_path))

pred=[]
model.eval()
with torch.no_grad():    
    for data in test_loader:
        inputs=data
        inputs=inputs.to(device)
        outputs=model(inputs)
        _,o_pred=torch.max(outputs,1)
        for y in o_pred.cpu().numpy():
            pred.append(y) 

with open("result.csv","w") as f:
    f.write("id, Class\n")
    for i, Class in enumerate(pred):
        f.write("{},{}\n".format(i,Class)) 