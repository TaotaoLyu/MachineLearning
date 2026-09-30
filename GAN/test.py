import torch
import torchvision
workspace_dir='.'
import torch.nn as nn

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm')!=-1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



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

z_dim=100
G=Generator(z_dim)
D=Discriminator(3)

def print_model_info(name, model):
    total_params=sum(p.numel() for p in model.parameters())
    trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{name} structure:\n{model}')
    print(f'{name} parameters: {total_params:,} total, {trainable_params:,} trainable\n')

print_model_info('Generator', G)
print_model_info('Discriminator', D)

# import os
# log_dir=os.path.join(workspace_dir, 'logs')
# ckpt_dir=os.path.join(workspace_dir, 'checkpoints')
# G.load_state_dict(torch.load(os.path.join(ckpt_dir, 'G.pth')))
# G.eval()
# G.cuda()

# from torch.autograd import Variable

# n_outputs=1000
# z_sample=Variable(torch.randn(n_outputs, z_dim)).cuda()
# imgs_sample=(G(z_sample).data+1)/2.0
# log_dir=os.path.join(workspace_dir, 'logs')
# filename=os.path.join(log_dir, 'results.png')
# torchvision.utils.save_image(imgs_sample, filename, nrow=10)
