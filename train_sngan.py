#############################################
###### This script is made by Doi Kento #####
###### University of Tokyo              #####
#############################################

# add the module path
import os
import argparse
import json
from pathlib import Path
from collections import OrderedDict

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
import torchvision.transforms as transforms
from torchvision.utils import save_image

from src import SATDataset
from src import SNGANProjectionDiscriminator
from src import SNGANGenerator
from src import DisLoss, GenLoss

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
Path(args.log).mkdir(parents=True)
with Path(args.log).joinpath('arguments.json').open("w") as f:
    json.dump(OrderedDict(sorted(vars(args).items(), key=lambda x: x[0])),
            f, indent=4)
Path(args.log).joinpath('img').mkdir()

device = torch.device('cuda')

# data loader
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 4, [0.5] * 5)
    ])
train_dataset = SATDataset(args.data, phase='train', transform=trans)
train_loader = data_utils.DataLoader(train_dataset,
        args.batchsize, shuffle=True, num_workers=2, drop_last=True)
test_dataset = SATDataset(args.data, phase='val', transform=trans)
test_loader = data_utils.DataLoader(test_dataset,
        args.batchsize, num_workers=2, drop_last=True)

# random generator
def generate_z(batchsize):
    return torch.randn((batchsize, 100))

# prepare network
D = SNGANProjectionDiscriminator(num_classes=6,
        ndf=args.ndf).to(device)
G = SNGANGenerator(num_classes=6,
        ngf=args.ngf).to(device)

## initialization the network parameters
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv' or 'Linear') != -1:
        init.normal(m.weight, mean=args.mu, std=args.sigma)
D.apply(weights_init)
G.apply(weights_init)

# criterion
dis_criterion = DisLoss(loss_type='hinge')
gen_criterion = GenLoss(loss_type='hinge')

# prepare optimizer
d_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))
g_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))

# train
training_history = np.zeros((4, args.epochs))
print('start training!!!')
for epoch in tqdm(range(args.epochs)):
    running_dis_loss = 0
    running_gen_loss = 0
    running_d_true = 0
    running_dis_fake = 0

    for data in train_loader:
        x = data[0].to(device)
        y = data[1].to(device)

        # update G
        g_optimizer.zero_grad()

        z = generate_z(args.batchsize).to(device)

        dis_fake = D(G(z, y), y)
        gen_loss = gen_criterion(dis_fake)
        running_gen_loss += gen_loss.item()
        gen_loss.backward()
        g_optimizer.step()

        # update D
        for i in range(3):
            d_optimizer.zero_grad()

            x = F.pad(x, (2, 2, 2, 2), mode='reflect')
            z = generate_z(args.batchsize).to(device)
            
            dis_real = D(x, y)
            dis_fake = D(G(z, y), y)

            dis_loss = dis_criterion(dis_fake, dis_real)
            running_dis_loss += dis_loss.item()
            dis_loss.backward()
            d_optimizer.step()

    running_dis_loss = running_dis_loss / 5 / len(train_loader)
    running_gen_loss = running_gen_loss / len(train_loader)
    training_history[0, epoch] = running_dis_loss
    training_history[1, epoch] = running_gen_loss
    print('\n' + '*' * 40, flush=True)
    print('epoch: {}'.format(epoch+1), flush=True)
    print('train loss: {}'.format(running_dis_loss + running_gen_loss),
            flush=True)
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # update D
            x = data[0].to(device)
            x = F.pad(x, (2, 2, 2, 2), mode='reflect')
            z = generate_z(args.batchsize).to(device)
            
            dis_real = D(x, y)
            dis_fake = D(G(z, y), y)

            dis_loss = dis_criterion(dis_fake, dis_real)
            running_dis_loss += dis_loss.item()
            
            # update G
            g_optimizer.zero_grad()

            z = generate_z(args.batchsize).to(device)

            dis_fake = D(G(z, y), y)
            gen_loss = gen_criterion(dis_fake)
            running_gen_loss += gen_loss.item()

    running_dis_loss = running_dis_loss / len(test_loader)
    running_gen_loss = running_gen_loss / len(test_loader)
    training_history[2, epoch] = running_dis_loss
    training_history[3, epoch] = running_gen_loss
    print('test loss: {}'.format(running_dis_loss + running_gen_loss),
            flush=True)

    if (epoch+1) % 5 == 0:
        for i in range(6):
            generated_img = G(generate_z(100).to(device),
                torch.LongTensor([i]*100).to(device))
            generated_img = generated_img.data.cpu()[:, :3, ...]
            plt.figure(figsize=(10, 10))
            save_image(generated_img,
                    str(Path(args.log).joinpath(
                        'img/{}_{}.png'.format(epoch+1, i+1))),
                    nrow=10)
        # save model weights
        torch.save(D.state_dict(),
                os.path.join(args.log, 'D_ep{}.pt'.format(i+1)))
        torch.save(G.state_dict(),
                os.path.join(args.log, 'G_ep{}.pt'.format(i+1)))
        
    # plot training history
    plt.plot(np.arange(args.epochs), training_history[0], label='Train D Loss')
    plt.plot(np.arange(args.epochs), training_history[1], label='Train G Loss')
    plt.plot(np.arange(args.epochs), training_history[2], label='Test D Loss')
    plt.plot(np.arange(args.epochs), training_history[3], label='Test G Loss')
    plt.legend()
    plt.savefig('{}/loss.png'.format(args.log))
    plt.close()
