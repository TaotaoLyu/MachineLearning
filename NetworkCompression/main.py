import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
from tqdm import tqdm


device="cuda" if torch.cuda.is_available() else "cpu"

train_tfm=transforms.Compose([
    transforms.Resize((142,142)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomCrop(128),
    transforms.ToTensor(),
])

test_tfm=transforms.Compose([
    transforms.Resize((142,142)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])

batch_size=64

train_set=DatasetFolder("DATA/food-11/training/labeled", loader=lambda x:Image.open(x), extensions="jpg", transform=train_tfm)
valid_set=DatasetFolder("DATA/food-11/validation", loader=lambda x:Image.open(x), extensions="jpg", transform=test_tfm)
unlabled_set=DatasetFolder("DATA/food-11/training/unlabeled", loader=lambda x:Image.open(x), extensions="jpg", transform=train_tfm)
test_set=DatasetFolder("DATA/food-11/testing", loader=lambda x:Image.open(x), extensions="jpg", transform=test_tfm)

train_loader=DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_loader=DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader=DataLoader(test_set, batch_size=batch_size, shuffle=False)


# [in_ch, H, W]=>[out_ch, H', W']   
def NormalConv(in_channels, out_channels, kernel_size):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)

# [int_ch, H, W]=>[out_ch, H', W'] 
def GroupConv(in_channels, out_channels, kernel_size, group_size):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, groups=group_size)

# [ch, H, W]=>[ch, H', W']
def DeepwiseConv(in_channels, kernel_size):
    return nn.Conv2d(in_channels, in_channels, kernel_size, padding=1, groups=in_channels)

# [in_ch, H, W]=> [out_ch, H, W]
def PointwiseConv(in_channels, out_channel):
    return nn.Conv2d(in_channels, out_channel, 1)

# [in_ch, H, W]=> [out_ch, H, W]
def PointwiseGroupConv(in_channels, out_channels, group_size):
    return nn.Conv2d(in_channels, out_channels, 1, groups=group_size)

# [in_ch, H, W]=>[out_ch, H', W']   
def DwPwConv(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        DeepwiseConv(in_channels, kernel_size),
        PointwiseConv(in_channels, out_channels)
    )


class StudentNet(nn.Module):
    def __init__(self):
      super(StudentNet, self).__init__()



      # ---------- TODO ----------
      # Modify your model architecture

      self.cnn = nn.Sequential(
        # nn.Conv2d(3, 32, 3), 
        DwPwConv(3, 32, 3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        # nn.Conv2d(32, 32, 3),  
        DwPwConv(32, 32, 3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),     

        # nn.Conv2d(32, 64, 3), 
        DwPwConv(32, 64, 3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),     

        # nn.Conv2d(64, 100, 3), 
        DwPwConv(64, 100, 3),
        nn.BatchNorm2d(100),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),
        
        # Here we adopt Global Average Pooling for various input size.
        nn.AdaptiveAvgPool2d((1, 1)),
      )
      self.fc = nn.Sequential(
        nn.Linear(100, 11),
      )
      
    def forward(self, x):
      out = self.cnn(x)
      out = out.view(out.size()[0], -1)
      return self.fc(out)

from torchsummary import summary

student_net=StudentNet().to(device)
summary(student_net, (3, 148, 148), device=device)

import torch.nn.functional as F

def loss_fn_kd(outputs, labels, teacher_outputs, alpha=0.5):
    T = 4.0

    # 1. 学习真实标签
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)

    # 2. Teacher 的软标签
    teacher_probs = F.softmax(teacher_outputs / T, dim=1)

    # 3. Student 的对数概率
    student_log_probs = F.log_softmax(outputs / T, dim=1)

    # 4. KL 散度
    soft_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='batchmean'
    ) * (alpha * T * T)

    return hard_loss + soft_loss

teacher_net = torch.load(
    "teacher_net.ckpt", map_location=device, weights_only=False
)
teacher_net.eval()

teacher_net=teacher_net.to(device)

def Train():
    global unlabled_set, train_loader
    do_semi=True

    def get_pseudo_labels(dataset, model):
        loader=DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        pseudo_labels=[]
        for batch in tqdm(loader):
            img, _ =batch
            with torch.no_grad():
               logits=model(img.to(device))
               pseudo_labels.append(logits.argmax(dim=-1).detach().cpu())


        pseudo_labels=torch.cat(pseudo_labels)

        for idx, ((img, _), pseudo_label) in enumerate(zip(dataset.samples, pseudo_labels)):
           dataset.samples[idx]=(img, pseudo_label.item())

        return dataset

    if do_semi:
        unlabled_set=get_pseudo_labels(unlabled_set, teacher_net)
        concat_dataset=ConcatDataset([train_set, unlabled_set])
        train_loader=DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

    criterion=nn.CrossEntropyLoss()

    optimizer=torch.optim.Adam(student_net.parameters(), lr=0.0003, weight_decay=1e-5)
    n_epochs=80

    for epoch in range(n_epochs):
        student_net.train()
        train_loss=[]
        train_accs=[]

        for batch in tqdm(train_loader):
            imgs, labels=batch
            logits=student_net(imgs.to(device))

            with torch.no_grad():
               soft_labels=teacher_net(imgs.to(device))

            loss=loss_fn_kd(logits, labels.to(device), soft_labels)

            optimizer.zero_grad()
            loss.backward()
            grad_norm=nn.utils.clip_grad_norm_(student_net.parameters(), max_norm=10)
            optimizer.step()

            acc=(logits.argmax(dim=-1)==labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss=sum(train_loss)/len(train_loss)
        train_accs=sum(train_accs)/len(train_accs)

        print(f"[ Train | {epoch+1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_accs:.5f}")

        student_net.eval()
        valid_loss=[]
        valid_accs=[]

        for batch in tqdm(valid_loader):
            imgs, labels=batch

            with torch.no_grad():
                logits=student_net(imgs.to(device))
                soft_labels=teacher_net(imgs.to(device))

            loss=loss_fn_kd(logits, labels.to(device), soft_labels)
            acc=(logits.argmax(dim=-1)==labels.to(device)).float().detach().cpu().view(-1).numpy()

            valid_loss.append(loss.item())
            valid_accs+=list(acc)

        valid_loss=sum(valid_loss)/len(valid_loss)
        valid_acc=sum(valid_accs)/len(valid_accs)

        print(f"[ Valid | {epoch+1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

Train()
