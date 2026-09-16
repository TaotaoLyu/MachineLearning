
BATCH_SIZE=512
model_path="./model.pkt"


import torchvision.transforms as transforms

train_tf=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

test_tf=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])


from torchvision.datasets import DatasetFolder
from PIL import Image
train_set=DatasetFolder("food-11/training/labeled",loader=lambda x: Image.open(x), extensions="jpg",transform=train_tf)
unlabeled_train_set=DatasetFolder("food-11/training/unlabeled",loader=lambda x: Image.open(x), extensions="jpg",transform=train_tf)
val_set=DatasetFolder("food-11/validation",loader=lambda x: Image.open(x), extensions="jpg",transform=test_tf)
test_set=DatasetFolder("food-11/testing",loader=lambda x: Image.open(x), extensions="jpg",transform=test_tf)

from torch.utils.data import DataLoader, Dataset
train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
# train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
val_loader=DataLoader(val_set,batch_size=BATCH_SIZE,shuffle=False)
test_loader=DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=False)


import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_=nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4,4,0),
        )
        self.fc_=nn.Sequential(
            nn.Linear(256*8*8,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,11)
        )

    def forward(self,x):
        x=self.cnn_(x)
        x=x.flatten(1)
        x=self.fc_(x)
        return x


import torch.cuda
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

device=get_device()
print("Device: {}".format(device))


class PseudoLabeledDataset(Dataset):
    def __init__(self, dataset, indices, labels):
        self.dataset = dataset
        self.indices = indices
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.dataset[self.indices[index]]
        return image, self.labels[index]


def get_pseudo_labels(dataset, model:Classifier, threshold=0.65):
    m_dataloader=DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=False)
    was_training = model.training
    model.eval()

    softmax=nn.Softmax(dim=-1)
    selected_indices = []
    selected_labels = []
    batch_start = 0

    for data in m_dataloader:
        image, _=data

        with torch.no_grad():
            logits=model(image.to(device))

        prob=softmax(logits)
        max_prob, pseudo_label = torch.max(prob, dim=1)
        selected = max_prob > threshold

        selected_in_batch = selected.nonzero(as_tuple=True)[0].cpu().tolist()
        selected_indices.extend(batch_start + i for i in selected_in_batch)
        selected_labels.extend(pseudo_label[selected].cpu().tolist())
        batch_start += image.size(0)

    if was_training:
        model.train()
    return PseudoLabeledDataset(dataset, selected_indices, selected_labels)


model=Classifier().to(device)
criteriton=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters(),lr=0.0003,weight_decay=1e-5)

n_epochs=80

do_semi=True

from torch.utils.data import ConcatDataset

best_acc=0
for epoch in range(n_epochs):
    if do_semi and best_acc>=0.35:
        unlabled_set=get_pseudo_labels(unlabeled_train_set,model)
        new_dataset=ConcatDataset([unlabled_set,train_set])
        train_loader=DataLoader(new_dataset,batch_size=BATCH_SIZE,shuffle=True)

    model.train()

    train_acc=[]
    train_loss=[]
    for data in train_loader:
        image,label=data
        optimizer.zero_grad()
        outputs=model(image.to(device))

        batch_loss=criteriton(outputs,label.to(device))
        batch_loss.backward()
        optimizer.step()

        acc=(outputs.argmax(dim=-1)==label.to(device)).float().mean()
        train_acc.append(acc)
        train_loss.append(batch_loss.item())

    print("epoch: %d, Train Acc: %.2f, Train Loss: %.2f"%(epoch,sum(train_acc)/len(train_acc),sum(train_loss)/len(train_loss)))


    # val
    val_acc=[]
    val_loss=[]
    model.eval()
    for data in val_loader:
        image, label=data
        with torch.no_grad() :
            logits=model(image.to(device))
        acc=(logits.argmax(dim=-1)==label.to(device)).float().mean()
        val_acc.append(acc)
        loss=criteriton(logits,label.to(device))
        val_loss.append(loss.item())

    acc=sum(val_acc)/len(val_acc)
    loss=sum(val_loss)/len(val_loss)
    print("Val Acc: %.2f, Val Loss: %.2f"%(acc,loss))

    if acc>best_acc:
        best_acc=acc
        torch.save(model.state_dict(),model_path)
        print("The best model is successly saved")


    

