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
