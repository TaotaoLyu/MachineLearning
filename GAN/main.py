workspace_dir='.'

import random
import torch
import numpy as np

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


same_seeds(2026)

import glob
import os
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import Dataset
class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        super().__init__()
        self.transform=transform
        self.fnames=fnames
        self.num_samples=len(self.fnames)


    def __getitem__(self, index):
        fname=self.fnames[index]

        img=torchvision.io.read_image(fname)

        img=self.transform(img)

        return img

    def __len__(self):
        return self.num_samples


def get_dataset(root):
    fnames=glob.glob(os.path.join(root, "*"))

    compose=[
        transforms.ToPILImage(),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]

    tranform=transforms.Compose(compose)

    dataset=CrypkoDataset(fnames, tranform)

    return dataset

import matplotlib.pyplot as plt

dataset=get_dataset(os.path.join(workspace_dir, "faces"))

# images=[dataset[i] for i in range(16)]
# print(type(images[0]))
# print(images[0].shape)
# grid_img=torchvision.utils.make_grid(images, nrow=4)
# plt.figure(figsize=(10,10))
# plt.imshow(grid_img.permute(1,2,0))
# plt.savefig("1.png")


images=[(dataset[i]+1)/2 for i in range(16)]
print(type(images[0]))
print(images[0].shape)
grid_img=torchvision.utils.make_grid(images, nrow=4)
plt.figure(figsize=(10,10))
plt.imshow(grid_img.permute(1,2,0))
plt.savefig("2.png")



def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm')!=-1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


import torch.nn as nn
class Generator(nn.Module):
    # input shape (Batch, in_dim)
    # Output shape (Batch, 3, 64, 64)

    def __init__(self, in_dim, dim=64):
        super().__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2, padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )

        self.l1=nn.Sequential(
            nn.Linear(in_dim, dim*8*4*4, bias=False),
            nn.BatchNorm1d(dim*8*4*4),
            nn.ReLU()
        )

        self.l2_5=nn.Sequential(
            dconv_bn_relu(dim*8, dim*4),
            dconv_bn_relu(dim*4, dim*2),
            dconv_bn_relu(dim*2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x):
        y=self.l1(x)
        y=y.view(y.size(0), -1, 4, 4)
        # [Batch, dim*8, 4, 4]
        y=self.l2_5(y)
        # [Batch, 3, 64, 64]
        return y


class Discriminator(nn.Module):
    def __init__(self, in_dim, dim=64):
        super().__init__()
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2)
            )

        self.ls=nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2),
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim*2),
            conv_bn_lrelu(dim*2, dim*4),
            conv_bn_lrelu(dim*4, dim*8),
            nn.Conv2d(dim*8, 1, 4),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, x):
        y=self.ls(x)
        y=y.view(-1)
        return y

from torch.autograd import Variable
batch_size=64
z_dim=100
z_sample=Variable(torch.randn(100, z_dim)).cuda()
lr=1e-4

n_epoch=30
n_critic=1

log_dir=os.path.join(workspace_dir, 'logs')
ckpt_dir=os.path.join(workspace_dir, 'checkpoints')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

G=Generator(in_dim=z_dim).cuda()
D=Discriminator(3).cuda()
G.train()
D.train()

criterion=nn.BCELoss()

opt_D=torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
opt_G=torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))


from torch.utils.data import DataLoader

dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

import qqdm
steps=0
for e, epoch in enumerate(range(n_epoch)):
    progress_bar=qqdm.qqdm(dataloader)
    for i, data in enumerate(progress_bar):
        imgs=data
        imgs=data.cuda()

        bs=imgs.size(0)
        z=Variable(torch.randn(bs,z_dim)).cuda()
        r_imgs=Variable(imgs).cuda()
        f_imgs=G(z)

        r_label=torch.ones((bs)).cuda()
        f_label=torch.zeros((bs)).cuda()


        r_logit=D(r_imgs.detach())
        f_logit=D(f_imgs.detach())

        r_loss=criterion(r_logit, r_label)
        f_loss=criterion(f_logit, f_label)
        loss_D=(r_loss+f_loss)/2


        D.zero_grad()
        loss_D.backward()
        opt_D.step()

        if steps%n_critic==0:
            z=Variable(torch.randn(bs, z_dim)).cuda()
            f_imgs=G(z)
            f_logit=D(f_imgs)

            loss_G=criterion(f_logit, r_label)

            G.zero_grad()
            loss_G.backward()

            opt_G.step()


        steps+=1


        progress_bar.set_infos({
            'Loss_D':round(loss_D.item(), 4),
            'Loss_G':round(loss_G.item(), 4),
            'epoch':e+1,
            'step':steps
        })

    G.eval()
    with torch.no_grad():
        f_imgs=(G(z_sample).detach().cpu()+1)/2.0
    save_name=os.path.join(log_dir, f"Epoch_{e+1:03d}.png")
    torchvision.utils.save_image(f_imgs, save_name, nrow=10)
    print(f"Saved some samples into {save_name}")
    G.train()

    if (e+1)%5==0 or e==0:
        torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G.pth'))
        torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D.pth'))