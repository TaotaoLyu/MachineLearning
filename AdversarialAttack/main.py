import torch
import torch.nn as nn

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=8

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

cifar_10_mean=(0.491, 0.482, 0.447)
cifar_10_std=(0.202, 0.199, 0.201)

mean=torch.tensor(cifar_10_mean).to(device).view(3,1,1)
std=torch.tensor(cifar_10_std).to(device).view(3,1,1)

epsilon=8/255/std
alpha=0.8/255/std

root='./Data'


from torchvision.transforms import transforms

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_10_mean, cifar_10_std)
])

from torch.utils.data import Dataset, DataLoader
import glob
import os
from PIL import Image
class AdvDataset(Dataset):
    def __init__(self, data_dir, transform):
        super().__init__()
        self.images=[]
        self.labels=[]
        self.names=[]

        for i, class_dir in enumerate(sorted(glob.glob(f'{data_dir}/*'))):
            images=sorted(glob.glob(f'{class_dir}/*'))
            self.images+=images
            self.labels+=([i]*len(images))
            self.names+=[os.path.relpath(imgs, data_dir) for imgs in images]

        self.transform=transform

    def __getitem__(self, index):
        image=self.transform(Image.open(self.images[index]))
        label=self.labels[index]
        return image, label

    def __getname__(self):
        return self.names

    def __len__(self):
        return len(self.images)

adv_set=AdvDataset(root, transform=transform)
adv_names=adv_set.__getname__()
adv_loader=DataLoader(adv_set, batch_size=batch_size, shuffle=False)

print(f'number of images = {adv_set.__len__()}')

def epoch_begin(model, loader, loss_fn):
    model.eval()
    train_acc, train_loss=0.0, 0.0
    for x, y in loader:
        x, y =x.to(device), y.to(device)
        yp=model(x)
        loss=loss_fn(yp, y)
        train_acc+=(yp.argmax(dim=1)==y).sum().item()
        train_loss+=loss.item()*x.shape[0]


    return train_acc/len(loader.dataset), train_loss/len(loader.dataset)


def fgsm(model, x, y, loss_fn, epsilon=epsilon):
    x_adv=x.detach().clone()
    x_adv.requires_grad=True
    loss=loss_fn(model(x_adv), y)
    loss.backward()

    x_adv=x_adv+epsilon*x_adv.grad.detach().sign()

    x_adv = ((x_adv * std + mean).clamp(0, 1) - mean) / std

    return x_adv.detach()


def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20):
    """Target ship for non-ship images, and dog for ship images."""
    target_y=torch.full_like(y, classes.index('ship'))
    target_y[y == classes.index('ship')]=classes.index('dog')
    x_adv=x.detach().clone()
    lower=torch.maximum(x.detach()-epsilon, -mean/std)
    upper=torch.minimum(x.detach()+epsilon, (1-mean)/std)
    for _ in range(num_iter):
        x_adv.requires_grad_(True)
        loss=loss_fn(model(x_adv), target_y)
        grad=torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            # Minimize the target-label loss for a targeted attack.
            x_adv.sub_(alpha*grad.sign())
            x_adv.clamp_(min=lower, max=upper)
        x_adv=x_adv.detach()
    return x_adv


import numpy as np
def gen_adv_examples(model, loader, attack, loss_fn):
    model.eval()
    adv_names=[]
    train_acc, train_loss=0.0, 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x_adv=attack(model,x, y, loss_fn)
        yp=model(x_adv)
        loss=loss_fn(yp, y)
        train_acc+=(yp.argmax(dim=1)==y).sum().item()
        train_loss+=loss.item()*x.shape[0]

        adv_ex=((x_adv)*std+mean).clamp(0, 1)
        adv_ex=(adv_ex*255).clamp(0, 255)
        adv_ex=adv_ex.detach().cpu().data.numpy().round()
        adv_ex=adv_ex.transpose((0,2,3,1)) #(bs, c, h, w) => (bs, h, w, c)
        adv_examples=adv_ex if i==0 else np.r_[adv_examples, adv_ex]

    return adv_examples, train_acc/len(loader.dataset), train_loss/len(loader.dataset)

import shutil
def create_dir(data_dir, adv_dir, adv_examples, adv_names):
    if os.path.exists(adv_dir) is not True:
        _=shutil.copytree(data_dir, adv_dir)
    for example, name in zip(adv_examples, adv_names):
        im=Image.fromarray(example.astype(np.uint8))
        im.save(os.path.join(adv_dir,name))



from pytorchcv.model_provider import get_model as ptcv_get_model

model=ptcv_get_model('resnet110_cifar10', pretrained=True).to(device)
loss_fn=nn.CrossEntropyLoss()

benign_acc, benign_loss=epoch_begin(model, adv_loader, loss_fn)
print(f'benign_acc = {benign_acc:.5f}, benign_loss = {benign_loss:.5f}')


adv_examples, fgsm_acc, fgsm_loss=gen_adv_examples(model, adv_loader, fgsm, loss_fn)
print(f'fgsm_acc={fgsm_acc:.5f}, fgsm_loss={fgsm_loss:.5f}')

create_dir(root, 'fgsm', adv_examples, adv_names)

adv_examples, ifgsm_acc, ifgsm_loss=gen_adv_examples(model, adv_loader, ifgsm, loss_fn)
print(f'ifgsm_acc={ifgsm_acc:.5f}, ifgsm_loss={ifgsm_loss:.5f}')

create_dir(root, 'ifgsm', adv_examples, adv_names)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

comparisons = [('benign', root), ('FGSM', 'fgsm'), ('I-FGSM (targeted)', 'ifgsm')]
fig, axes = plt.subplots(len(classes), len(comparisons), figsize=(15, 25))
for row, cls_name in enumerate(classes):
    path = f'{cls_name}/{cls_name}1.png'
    for col, (label, image_dir) in enumerate(comparisons):
        ax = axes[row, col]
        with Image.open(os.path.join(image_dir, path)) as im:
            with torch.no_grad():
                logit = model(transform(im).unsqueeze(0).to(device))[0]
                predict = logit.argmax(-1).item()
                prob = logit.softmax(-1)[predict].item()
            ax.set_title(f'{label}: {cls_name}1.png\n{classes[predict]}: {prob:.2%}')
            ax.axis('off')
            ax.imshow(np.array(im))
plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'comparison.png')
plt.savefig(output_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'Comparison image saved to: {output_path}')
