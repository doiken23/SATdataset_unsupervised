#############################################
###### This script is made by Doi Kento #####
###### University of Tokyo              #####
#############################################

# add the module path
import sys
sys.path.append('../pytorch_toolbox')

# import torch module
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import models, datasets
from torchvision import transforms
from autoencoders import AutoEncoder
from sat_dataset import SAT_Dataset
from numpy_transforms import *

# import python module
import numpy as np
import time
import os
import argparse
import datetime
from pathlib import Path

def get_argument():
    # get the argment
    parser = argparse.ArgumentParser(description='training of Pytorch ResNet for SAT dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default:256)')
    parser.add_argument('--epochs', type=int, default=200, help='number of the epoch to train (default:200)')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate for training (default:0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default:0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay (default:0.0001)')
    parser.add_argument('--dropout_ratio', type=float, default=0, help='drop out ratio (default:0)')
    parser.add_argument('--milestones', type=list, default=[100], help='epoch that reduce the learning rate (default:[100])')
    parser.add_argument('data_path', type=str, help='the path of training dataset (mat)')
    parser.add_argument('embedding_dimension', type=int, default=6, help='dimension of embedded feature (default:6)')
    parser.add_argument('outdir_path', type=str, help='directory path of outputs')
    args = parser.parse_args()
    return args


def main(args):
    # Loading the dataset
    trans = transforms.Compose([Numpy_Flip(), Numpy_Rotate()])
    train_dataset = SAT_Dataset(args.data_path, phase='train')
    val_dataset   = SAT_Dataset(args.data_path, phase='val')

    train_loader  = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader    = data_utils.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2)
    loaders       = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    
    print("Complete the preparing dataset")

    # Setting the network
    c, w, h = train_dataset[0][0].shape
    net = AutoEncoder(c * w * h, args.embedding_dimension)
    net.cuda()
    # Initialize the network parameters
    print("Initialize the network parameters")
    net.initialization()

    # Define a Loss function and optimizer
    criterion = nn.MSELoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    # initialize the best accuracy and best model weights
    best_model_wts = net.state_dict()
    best_loss = 0.0

    # initialize the loss and accuracy history
    loss_history = {"train": [], "val":[] }

    # Train the network
    start_time = time.time()
    for epoch in range(args.epochs):
        scheduler.step()
        print('* ' * 20)
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('* ' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            # initialize the runnig loss and corrects
            running_loss = 0.0

            for i, data in enumerate(loaders[phase]):
                # get the input
                inputs, _ = data
                inputs = inputs.float() / 255.0
                n = inputs.size()[0]
                inputs = inputs.view(n, -1)

                # wrap the in valiables
                if phase == 'train':
                    inputs = Variable(inputs.cuda())
                else:
                    inputs = Variable(inputs.cuda(), volatile=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                _, outputs = net(inputs)
                loss = criterion(outputs, inputs)

                # backward + optimize if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statuctics
                running_loss += loss.data[0]

            epoch_loss = running_loss / dataset_sizes[phase] * args.batch_size
            loss_history[phase].append(epoch_loss)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # copy the best model
            if phase == 'val' and epoch_loss < best_loss:
                best_model_wts = net.state_dict()

    elapsed_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net, loss_history

def write_parameters(args):
    import csv
    fout = open(Path(args.outdir_path).joinpath('experimental_settings.csv'), "wt")
    csvout = csv.writer(fout)
    print('*' * 50)
    print('Parameters')
    print('*' * 50)
    for arg in dir(args):
        if not arg.startswith('_'):
            csvout.writerow([arg,  str(getattr(args, arg))])
            print('%-25s %-25s' % (arg , str(getattr(args, arg))))

if __name__ == '__main__':
    # get the arguments and write the log
    args = get_argument()
    write_parameters(args)

    # train the network and output the result
    model_weights, loss_history = main(args)
    torch.save(model_weights.state_dict(), Path(args.outdir_path).joinpath('weight.pth'))
    training_history = np.zeros((2, args.epochs))
    for i, phase in enumerate(["train", "val"]):
        training_history[i] = loss_history[phase]
    np.save(Path(args.outdir_path ).joinpath('training_history_{}.npy'.format(datetime.date.today())), training_history)
