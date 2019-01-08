#############################################
###### This script is made by Doi Kento #####
###### University of Tokyo              #####
#############################################

# add the module path
import os
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data_utils
from torch.distributions import Normal
import torchvision.datasets as datasets
import  torchvision.transforms as transforms

from torchcv.transforms import NPSegRandomFlip, NPSegRandomRotate

from src import Generator, Discriminator
from src import SATDataset

# get argument
parser = argparse.ArgumentParser(description='DCGAN for mnist')
parser.add_argument('data', type=str, default='data',
        help='directory of training data')
parser.add_argument('--batchsize', type=int, default=100,
        help='batch size (default: 100)')
parser.add_argument('--epochs', type=int, default=60,
        help='epochs (default:60)')
parser.add_argument('--log', type=str, default='result',
        help='directory of training data (default: result)')
parser.add_argument('--mu', type=float, default=0,
        help='for model initialization (default: 0)')
parser.add_argument('--sigma', type=float, default=0.02,
    help='for model initialization (default: 0.02)')
parser.add_argument('--lr', type=float, default=0.00005,
        help='learning rate (default: 0.0005)')
parser.add_argument('--momentum', type=float, default=0.5,
        help='momentum (default: 0.5)')
parser.add_argument('--ndf', type=int, default=128,
        help='number of discriminator feature map (default: 128)')
parser.add_argument('--ngf', type=int, default=128,
        help='number of generator feature map (default: 128)')
args = parser.parse_args()

# prepare for experiments
Path(args.log).mkdir()
with Path(args.log).joinpath('arguments.json').open("w") as f:
    json.dump(OrderedDict(sorted(args.items(), key=lambda x: x[0])),
            f, indent=4)

device = torch.device('cuda')

# data loader
trans = transforms.Compose([
    NPSegRandomFlip(),
    NPSegRandomRotate()
])
train_dataset = SATDataset(args.data, phase='train', transform=trans)
train_loader = data_utils.DataLoader(train_dataset,
        args.batchsize, shuffle=True, num_workers=2, drop_last=True)
test_dataset = SATDataset(args.data, phase='val')
test_loader = data_utils.DataLoader(test_dataset,
        args.batchsize, num_workers=2, drop_last=True)

# random generator
def generate_z(batch_size):
    return torch.randn((args.batch_size, 50))

# prepare network
D = Discriminator(ndf=args.ndf).to(device)
G = Generator(50, ngf=args.ngf).to(device)
## initialization the network parameters
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv' or 'Linear') != -1:
        init.normal(m.weight, mean=args.mu, std=args.sigma)
D.apply(weights_init)
G.apply(weights_init)

# criterion
criterion = nn.Softplus()

# prepare optimizer
d_optimizer = optim.Adam(D.parameters(), lr=args.lr)
g_optimizer = optim.Adam(G.parameters(), lr=args.lr)

# train
training_history = np.zeros((4, args.epochs))
for epoch in tqdm(range(args.epochs)):
    running_d_loss = 0
    running_g_loss = 0
    running_d_true = 0
    running_d_fake = 0

    for data in train_loader:
        # update D
        d_optimizer.zero_grad()

        x = data[0]
        x.sub_(127.5).div_(127.5).to(device)
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        z = generate_z(args.batchsize).to(device)
        
        d_real = D(x)
        d_fake = D(G(z))

        d_loss = criterion(-d_real) + criterion(d_fake)
        running_d_loss += d_loss
        d_loss.backward()
        d_optimizer.step()
        
        # update G
        g_optimizer.zero_grad()

        z = generate_z(args.batch_size).to(device)

        g_loss = criterion(-G(z))
        running_g_loss += g_loss
        g_loss.backward()
        g_optimizer.step()

    running_d_loss = running_d_loss / len(train_loader)
    running_g_loss = running_g_loss / len(train_loader)
    training_history[0, i] = running_d_loss
    training_history[1, i] = running_g_loss
    print('\n' + '*' * 40, flush=True)
    print('epoch: {}'.format(i+1), flush=True)
    print('train loss: {}'.format(running_d_loss + running_g_loss), flush=True)
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # update D
            x = data[0]
            x.sub_(127.5).div_(127.5).to(device)
            x = F.pad(x, (2, 2, 2, 2), mode='reflect')
            z = generate_z(args.batchsize).to(device)
            
            d_real = D(x)
            d_fake = D(G(z))

            d_loss = criterion(-d_real) + criterion(d_fake)
            running_d_loss += d_loss
            
            # update G
            g_optimizer.zero_grad()

            z = generate_z(args.batch_size).to(device)

            g_loss = criterion(-G(z))
            running_g_loss += g_loss

    running_d_loss = running_d_loss / len(train_loader)
    running_g_loss = running_g_loss / len(train_loader)
    training_history[0, i] = running_d_loss
    training_history[1, i] = running_g_loss
    print('test loss: {}'.format(running_d_loss + running_g_loss), flush=True)

    generated_img = G(Variable(torch.rand((100, 50)).to(device))).data.cpu().numpy().reshape(100, 4, 32, 32).transpose(0, 2, 3, 1)[:,:,:,:3]
    generated_img = np.clip((generated_img + 1) * 127.5, 0, 255).astype(np.uint8)
    if (i+1) % 5 == 0:
        for k in range(100):
            plt.subplot(10,10,k+1)
            plt.imshow(generated_img[k], vmin=-1, vmax=1, cmap='gray')
            plt.axis('off')
        plt.savefig('{}/generated_img_epoch{}.png'.format(args.output_dir, i+1))
        # save model weights
        torch.save(D.state_dict(), os.path.join(args.output_dir, 'D_ep{}.pt'.format(i+1)))
        torch.save(G.state_dict(), os.path.join(args.output_dir, 'G_ep{}.pt'.format(i+1)))
        
plt.close()

# plot training history
plt.plot(np.arange(args.epochs), training_history[0], label='Train D Loss')
plt.plot(np.arange(args.epochs), training_history[1], label='Train G Loss')
plt.plot(np.arange(args.epochs), training_history[0], label='Test D Loss')
plt.plot(np.arange(args.epochs), training_history[1], label='Test G Loss')
plt.legend()
plt.savefig('{}/loss.png'.format(args.log))
plt.close()
