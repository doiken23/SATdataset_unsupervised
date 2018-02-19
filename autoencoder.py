##############################################
##### This code is written by Doi Kento. #####
##############################################

import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_size, embedding_dimension):
        # initialization of class
        super(AutoEncoder, self).__init__()

        # define the network
        self.encoder = nn.Sequential(
                        nn.Linear(input_size, 500),
                        nn.ReLU(),
                        nn.Linear(500, 500),
                        nn.ReLU(),
                        nn.Linear(500, 2000),
                        nn.ReLU(),
                        nn.Linear(2000, embedding_dimension)
                        )
        self.decoder = nn.Sequential(
                        nn.Linear(embedding_dimension, 2000),
                        nn.ReLU(),
                        nn.Linear(2000, 500),
                        nn.ReLU(),
                        nn.Linear(500, 500),
                        nn.ReLU(),
                        nn.Linear(500, input_size)
                        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded

class Large_AutoEncoder(nn.Module):
    def __init__(self, input_size, embedding_dimension):
        # initialization of class
        super(Large_AutoEncoder, self).__init__()

        # define the network
        self.encoder = nn.Sequential(
                        nn.Linear(input_size, 1000),
                        nn.ReLU(),
                        nn.Linear(1000, 1000),
                        nn.ReLU(),
                        nn.Linear(1000, 4000),
                        nn.ReLU(),
                        nn.Linear(4000, embedding_dimension)
                        )
        self.decoder = nn.Sequential(
                        nn.Linear(embedding_dimension, 4000),
                        nn.ReLU(),
                        nn.Linear(4000, 2000),
                        nn.ReLU(),
                        nn.Linear(2000, 2000),
                        nn.ReLU(),
                        nn.Linear(2000, input_size)
                        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded
