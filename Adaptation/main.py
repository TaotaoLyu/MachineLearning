import matplotlib.pyplot as plt



def no_axis_show(img, title='', cmap=None):
    fig=plt.imshow(img, interpolation='nearest', cmap=cmap)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)


titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'television', 'dog', 'dolphin', 'spider']
plt.figure(figsize=(18, 18))
for i in range(10):
    plt.subplot(1, 10, i+1)
    fig=no_axis_show(plt.imread(f'Data/train_data/{i}/{500*i}.bmp'), title=titles[i])

plt.savefig('train_samples.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(18, 18))
for i in range(10):
  plt.subplot(1, 10, i+1)
  fig = no_axis_show(plt.imread(f'Data/test_data/0/' + str(i).rjust(5, '0') + '.bmp'))

plt.savefig('test_samples.png', dpi=300, bbox_inches='tight')
plt.close()


import cv2
plt.figure(figsize=(18, 18))
original_img = plt.imread(f'Data/train_data/0/0.bmp')
plt.subplot(1, 5, 1)
no_axis_show(original_img, title='original')

gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
plt.subplot(1, 5, 2)
no_axis_show(gray_img, title='gray scale', cmap='gray')

gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
plt.subplot(1, 5, 2)
no_axis_show(gray_img, title='gray scale', cmap='gray')

canny_50100 = cv2.Canny(gray_img, 50, 100)
plt.subplot(1, 5, 3)
no_axis_show(canny_50100, title='Canny(50, 100)', cmap='gray')

canny_150200 = cv2.Canny(gray_img, 150, 200)
plt.subplot(1, 5, 4)
no_axis_show(canny_150200, title='Canny(150, 200)', cmap='gray')

canny_250300 = cv2.Canny(gray_img, 250, 300)
plt.subplot(1, 5, 5)
no_axis_show(canny_250300, title='Canny(250, 300)', cmap='gray')
plt.savefig('edge_detection_samples.png', dpi=300, bbox_inches='tight')
plt.close()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_device():
    return 'cuda:1' if torch.cuda.is_available() else 'cpu'

device=get_device()


source_transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=(0,)),
    transforms.ToTensor(),
])

target_transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=(0,)),
    transforms.ToTensor(),
])

source_dataset = ImageFolder('Data/train_data', transform=source_transform)
target_dataset = ImageFolder('Data/test_data', transform=target_transform)

source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
test_dataset = ImageFolder('Data/test_data', transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv=nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    #   2^5=32

    def forward(self, x):
        x=self.conv(x).flatten(1)
        return x

class LabelPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer=nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512,512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c=self.layer(h)
        return c

class DomainClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer=nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),


            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y=self.layer(h)

        return y

feature_extractor=FeatureExtractor().to(device)
label_predictor=LabelPredictor().to(device)
domain_classifier=DomainClassifier().to(device)

class_criterion=nn.CrossEntropyLoss()
domain_criterion=nn.BCEWithLogitsLoss()

optimizer_F=optim.Adam(feature_extractor.parameters())
optimizer_C=optim.Adam(label_predictor.parameters())
optimizer_D=optim.Adam(domain_classifier.parameters())


def train_epoch(source_dataloader, target_dataloader, lamb):
    running_D_loss, running_F_loss=0.0, 0.0
    total_hit, total_num=0.0, 0.0

    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data=source_data.to(device)
        source_label=source_label.to(device)
        target_data=target_data.to(device)

        mixed_data=torch.cat([source_data, target_data], dim=0)
        domain_label=torch.zeros(mixed_data.size(0), 1).to(device)
        domain_label[:source_data.size(0)]=1

        # step1: train domain classifier
        # [batch, 1, 32, 32]
        feature=feature_extractor(mixed_data)
        # [batch, 512, 512]
        domain_output=domain_classifier(feature.detach())
        # [batch, 1]
        loss=domain_criterion(domain_output, domain_label)
        optimizer_D.zero_grad()
        running_D_loss+=loss.item()
        loss.backward()
        optimizer_D.step()

        #step2: train feature extractor and label predictor
        class_output=label_predictor(feature[:source_data.size(0)])
        domain_output=domain_classifier(feature)

        loss=class_criterion(class_output, source_label)-lamb*domain_criterion(domain_output, domain_label)
        running_F_loss+=loss.item()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        total_hit+=torch.sum(class_output.argmax(dim=1)==source_label).item()
        total_num+=source_data.size(0)
        print(i, end='\r')

    return running_D_loss/(i+1), running_F_loss/(i+1), total_hit/total_num

# for epoch in range(200):
#     lamb=2/(1+np.exp(-10*epoch/200))-1
#     D_loss, F_loss, acc=train_epoch(source_dataloader, target_dataloader, lamb=0.1)
#     torch.save(feature_extractor.state_dict(), f'checkpoints/feature_extractor_{epoch}.pth')
#     torch.save(label_predictor.state_dict(), f'checkpoints/label_predictor_{epoch}.pth')
#     print(f'epoch: {epoch}, D_loss: {D_loss:.4f}, F_loss: {F_loss:.4f}, acc: {acc:.4f}')


def save_test_predictions(test_dataloader, num_images=10, output_path='test_predictions.png'):
    feature_extractor.eval()
    label_predictor.eval()

    # Sample distinct images from the entire test dataset.
    dataset = test_dataloader.dataset
    if num_images <= 0 or len(dataset) == 0:
        raise ValueError('num_images must be positive and the test dataset must not be empty')
    indices = torch.randperm(len(dataset))[:min(num_images, len(dataset))].tolist()
    images = torch.stack([dataset[index][0] for index in indices])
    with torch.no_grad():
        features = feature_extractor(images.to(device))
        logits = label_predictor(features)
        predictions = logits.argmax(dim=1).cpu().tolist()

    rows = (len(images) + 4) // 5
    plt.figure(figsize=(15, 3 * rows))
    for i, (index, img, label) in enumerate(zip(indices, images, predictions)):
        plt.subplot(rows, 5, i + 1)
        no_axis_show(img.squeeze(0).numpy(), title=f'Image {index}\nPred: {titles[label]}', cmap='gray')
        print(f'Test image {index}: label={label}, name={titles[label]}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved test predictions to {output_path}')
    return predictions


feature_extractor.load_state_dict(torch.load(
    'checkpoints/feature_extractor_199.pth', map_location=device, weights_only=True
))
label_predictor.load_state_dict(torch.load(
    'checkpoints/label_predictor_199.pth', map_location=device, weights_only=True
))
save_test_predictions(test_dataloader)
