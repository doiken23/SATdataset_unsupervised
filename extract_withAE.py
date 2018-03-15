import torch
from torch.autograd import Variable
import autoencoders

import numpy as np
from pathlib import Path
import os
from tqdm import tqdm


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

    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2)
    val_loader   = data_utile.DataLoader(val_dataset,   batch_size=args.batch_size, num_workers=2)
    loaders = {'train': train_loader, 'val': val_loader}

    n, h, w = dataset[0].shape
    print('Complete making dataset')

    # setting network
    if (args.ae == 'AE') or (argss.ae == 'SAE'):
        net = autoencoders.AutoEncoder(n*h*w, args.ed)
    elif args.ae == 'CAE':
        net = autoencoders.Convolutional_AutoEncoder(n, args.ed)
    else:
        print('choose the AutoEncoder from AE, SAE and CAE.')

    th = torch.load(args.weight)
    net.load_state_dict(th)

    net.cuda()

    # extract features from images
    for phase in ['train', 'val']:
        extracted_features = np.empty((0, 0))
        for data in loaders(phase):
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
            encoded = torch.Tensor.numpy(encoded.data())
            extracted_features = np.vstack((extracted_features, encoded))

        # save the extraced features
        name = 'encoded_feature_{}.npy'.format(phase)
        np.save(Path(out_dir).joinpath(name), extracted_features)

if __ name__ == '__main__':
    start = time.time()
    args = get_arguments()
    main(args)
    elapsed_time = time.time() - start
    print('elapsed time: {} [s]'.format(elapsed_time))
