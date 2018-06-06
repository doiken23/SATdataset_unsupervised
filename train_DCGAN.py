#############################################
###### This script is made by Doi Kento #####
###### University of Tokyo              #####
#############################################

# add the module path
import sys
sys.path.append('../pytorch_toolbox')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data_utils
from torch.distributions import Normal
import torchvision.datasets as datasets
import  torchvision.transforms as transforms

import models
from sat_dataset import SAT_Dataset
import numpy_transforms as np_transforms

import os
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import Discriminator, Generator

# get argument
parser = argparse.ArgumentParser(description='DCGAN for mnist')
parser.add_argument('data_path', type=str, default='data', help='directory of training data')
parser.add_argument('--batch_size', type=int, default=100, help='batch size (default: 100)')
parser.add_argument('--epochs', type=int, default=60, help='epochs (default:60)')
parser.add_argument('--output_dir', type=str, default='result', help='directory of training data (default: result)')
parser.add_argument('--mu', type=float, default=0, help='for model initialization (default: 0)')
parser.add_argument('--sigma', type=float, default=0.02, help='for model initialization (default: 0.02)')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate (default: 0.0005)')
parser.add_argument('--momentum', type=float, default=0.5, help='momentum (default: 0.5)')
parser.add_argument('--ndf', type=int, default=128, help='number of discriminator feature map (default: 128)')
parser.add_argument('--ngf', type=int, default=128, help='number of generator feature map (default: 128)')
args = parser.parse_args()

# make output dir
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

# random generator
def generate_z(batch_size):
    return torch.randn((args.batch_size, 50))

# data loader
trans = transforms.Compose([
        np_transforms.Numpy_Flip(),
        np_transforms.Numpy_Rotate(),
])
dataset = SAT_Dataset(args.data_path, phase='train', transform=trans)
data_loader = data_utils.DataLoader(dataset, args.batch_size, shuffle=True, num_workers=1)

# prepare network
D = Discriminator(ndf=args.ndf).cuda()
G = Generator(50, ngf=args.ngf).cuda()
## initialization the network parameters
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv' or 'Linear') != -1:
        init.normal(m.weight, mean=args.mu, std=args.sigma)
D.apply(weights_init)
G.apply(weights_init)

# criterion
criterion = nn.BCELoss()

# prepare optimizer
d_optimizer = optim.Adam(D.parameters(), lr=args.lr)
g_optimizer = optim.Adam(G.parameters(), lr=args.lr)

# train
training_history = np.zeros((4, args.epochs))
for i in tqdm(range(args.epochs)):
    running_d_loss = 0
    running_g_loss = 0
    running_d_true = 0
    running_d_fake = 0

    for j, data in enumerate(data_loader):
        # update D
        d_optimizer.zero_grad()

        x = data[0]
        x = Variable(x.float().cuda())
        x.sub_(127.5).div_(127.5)
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        z = Variable(generate_z(args.batch_size).cuda())
        target = Variable(torch.ones((args.batch_size, 1)).float().cuda())
        
        
        d_x = D(x)
        d_g = D(G(z))
        running_d_true += (d_x.data.cpu().numpy() > 0.5).sum()
        running_d_fake += (d_g.data.cpu().numpy() < 0.5).sum()

        d_loss = criterion(d_x, target) + criterion(1 - d_g, target)
        running_d_loss += d_loss * args.batch_size
        d_loss.backward()
        d_optimizer.step()
        
        # update G
        g_optimizer.zero_grad()

        z = Variable(generate_z(args.batch_size).cuda())

        g_loss = - criterion(1 - D(G(z)), target)
        running_g_loss += g_loss * args.batch_size
        g_loss.backward()
        g_optimizer.step()

    running_d_loss = running_d_loss / len(dataset)
    running_g_loss = running_g_loss / len(dataset)
    training_history[0, i] = running_d_loss
    training_history[1, i] = running_g_loss
    running_d_true = running_d_true / len(dataset)
    running_d_fake = running_d_fake / len(dataset)
    training_history[2, i] = running_d_true
    training_history[3, i] = running_d_fake
    
    print('\n' + '*' * 40, flush=True)
    print('epoch: {}'.format(i+1), flush=True)
    print('real acc: {}'.format(running_d_true), flush=True)
    print('fake acc: {}'.format(running_d_fake), flush=True)
    generated_img = G(Variable(torch.rand((100, 50)).cuda())).data.cpu().numpy().reshape(100, 4, 32, 32).transpose(0, 2, 3, 1)[:,:,:,:3]
    if (i+1) % 5 == 0:
        for k in range(100):
            plt.subplot(10,10,k+1)
            plt.imshow(generated_img[k], vmin=-1, vmax=1, cmap='gray')
            plt.axis('off')
        plt.savefig('{}/generated_img_epoch{}.png'.format(args.output_dir, i+1))
        
plt.close()

# plot training history
plt.plot(np.arange(args.epochs), training_history[0], label='Discriminator Loss')
plt.plot(np.arange(args.epochs), training_history[1], label='Generator Loss')
plt.legend()
plt.savefig('{}/loss.png'.format(args.output_dir))
plt.close()

plt.plot(np.arange(args.epochs), training_history[2], label='Accuracy of real data')
plt.plot(np.arange(args.epochs), training_history[3], label='Accuracy of fale data')
plt.legend()
plt.savefig('{}/accuracy.png'.format(args.output_dir))
plt.close()
