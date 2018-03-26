import torch
import torch.utils.data as data_utils
from torch.autograd import Variable
import autoencoders
from sat_dataset import SAT_Dataset

import argparse
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import time


def get_arguments():
    parser = argparse.ArgumentParser(description='extracting the latent feature and saving.')
    parser.add_argument('dataset', type=str, help='path of dataset')
    parser.add_argument('ae', type=str, help='name of AE')
    parser.add_argument('ed', type=int, help='embedding dimension')
    parser.add_argument('weight', type=str, help='path of network paramter')
    parser.add_argument('out_dir', type=str, help='name of output ddirectory')
    parser.add_argument('--batch_size', type=int, default=512, help='mini batch size')
    args = parser.parse_args()
    
    return args

def main(args):
    # make dataset
    train_dataset = SAT_Dataset(args.dataset, phase='train')
    val_dataset   = SAT_Dataset(args.dataset, phase='val')
    test_dataset  = SAT_Dataset(args.dataset, phase='test')

    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2)
    val_loader   = data_utils.DataLoader(val_dataset,   batch_size=args.batch_size, num_workers=2)
    test_loader  = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, num_workers=2)
    loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    n, h, w = train_dataset[0][0].shape
    print('Complete making dataset')

    # setting network
    if args.ae == 'AE':
        net = autoencoders.AutoEncoder(n*h*w, args.ed)
    elif args.ae == 'SAE':
        net = autoencoders.Stacked_AutoEncoder(n*h*w, args.ed)
    elif args.ae == 'CAE':
        net = autoencoders.Convolutional_AutoEncoder(n, args.ed)
    else:
        print('choose the AutoEncoder from AE, SAE and CAE.')

    th = torch.load(args.weight)
    net.load_state_dict(th)

    net.cuda()
    print('Complete setting the network')

    # extract features from images
    for phase in ['train', 'val', 'test']:
        for i, data in enumerate(tqdm(loaders[phase])):
            # make input data
            inputs, _ = data
            inputs = inputs.float().cuda() / 255
            if args.ae in ['AE', 'SAE']:
                inputs = inputs.view(args.batch_size, -1)

            # wrap input with Variable
            inputs = Variable(inputs, volatile=True)
            
            # extract features
            encoded  = net(inputs)[0]

            # convert to numpy
            encoded = encoded.data.cpu().numpy()
            if i == 0:
                extracted_features = encoded
            else:
                extracted_features = np.vstack((extracted_features, encoded))

        # save the extraced features
        name = 'encoded_feature_{}.npy'.format(phase)
        np.save(Path(args.out_dir).joinpath(name), extracted_features)

if __name__ == '__main__':
    start = time.time()
    args = get_arguments()
    main(args)
    elapsed_time = time.time() - start
    print('elapsed time: {} [s]'.format(elapsed_time))
