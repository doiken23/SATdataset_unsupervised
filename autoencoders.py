##############################################
##### This code is written by Doi Kento. #####
##############################################

import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_size, embedding_dimension):
        # initialization of class
        super(AutoEncoder, self).__init__()

        # define the network
        self.layer1 = nn.Linear(input_size, 32)
        self.layer2 = nn.Linear(32, input_size)
        self.encoder = nn.Sequential(
                        self.layer1,
                        nn.ReLU()
                        )
        self.decoder = nn.Sequential(
                        self.layer2,
                        nn.Sigmoid()
                        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded

    def initialization(self, mean=0, std=0.01):
        nn.init.normal(self.layer1.weight, mean, std)
        nn.init.normal(self.layer2.weight, mean, std)

class Stacked_AutoEncoder(nn.Module):
    def __init__(self, input_size, embedding_dimension):
        # initialization of class
        super(Stacked_AutoEncoder, self).__init__()

        # define the network
        self.l1 = nn.Linear(input_size, 500)
        self.l2 = nn.Linear(500, 500)
        self.l3 = nn.Linear(500, 2000)
        self.l4 = nn.Linear(2000, embedding_dimension)

        self.l5 = nn.Linear(embedding_dimension, 2000)
        self.l6 = nn.Linear(2000, 500)
        self.l7 = nn.Linear(500, 500)
        self.l8 = nn.Linear(500, input_size)


        self.encoder = nn.Sequential(
                        self.l1,
                        nn.ReLU(),
                        self.l2,
                        nn.ReLU(),
                        self.l3,
                        nn.ReLU(),
                        self.l4)

        self.decoder = nn.Sequential(
                        self.l5,
                        nn.ReLU(),
                        self.l6,
                        nn.ReLU(),
                        self.l7,
                        nn.ReLU(),
                        self.l8)

        self.stack1 = nn.Sequential(
                        self.l1,
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        self.l8)

        self.stack2 = nn.Sequential(
                        self.l1,
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        self.l2,
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        self.l7,
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        self.l8)
        
        self.stack3 = nn.Sequential(
                        self.l1,
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        self.l2,
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        self.l3,
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        self.l6,
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        self.l7,
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        self.l8)

        self.stack4 = nn.Sequential(self.encoder, self.decoder)

        self.stacks = [self.stack1, self.stack2, self.stack3, self.stack4]
        self.layers = [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7, self.l8]

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        stack_output = []
        for stack in self.stacks:
            stack_output.append(stack(x))

        return encoded, decoded, stack_output

    def initialization(self, mean=0, std=0.01):
        for layer in self.layers:
            nn.init.normal(layer.weight, mean, std)


class Convolutional_AutoEncoder(nn.Module):
    def  __init__(self, embedding_dimension):
        super(Convolutional_AutoEncoder, self).__init__()

        # define the network
        # encoder
        self.conv1 = nn.Sequential(nn.ZeroPad2d((1,2,1,2)),
                              nn.Conv2d(1, 32, kernel_size=5, stride=2),
                              nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d((1,2,1,2)),
                              nn.Conv2d(32, 64, kernel_size=5, stride=2),
                              nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
                              nn.ReLU())
        self.fc1 = nn.Conv2d(128, 10, kernel_size=3)

        # decoder
        self.fc2 = nn.Sequential(nn.ConvTranspose2d(10, 128, kernel_size=3),
                            nn.ReLU())
        self.conv3d = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
                               nn.ReLU())
        self.conv2d = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
                               nn.ReLU())
        self.conv1d = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2)

    def forward(self, x):
        encoded = self.fc1(self.conv3(self.conv2(self.conv1(x))))

        decoded = self.fc2(encoded)
        decoded = self.conv3d(decoded)
        decoded = self.conv2d(decoded)[:,:,1:-2,1:-2]
        decoded = self.conv1d(decoded)[:,:,1:-2,1:-2]
        decoded = nn.Sigmoid()(decoded)

        return encoded, decoded
