import torch
import torch.nn as nn
import torch.nn.functional as F

def g_conv_unit(input_features, output_features):
    conv_unit = nn.Sequential(nn.ConvTranspose2d(input_features, output_features, 4, stride=2, padding=1),
                	nn.BatchNorm2d(output_features),
                    nn.ReLU())
    return conv_unit

def d_conv_unit(input_features, output_features):
    conv_unit = nn.Sequential(nn.Conv2d(input_features, output_features, 4, stride=2, padding=1),
                    nn.BatchNorm2d(output_features),
                    nn.LeakyReLU(0.2))
    return conv_unit

class Discriminator(nn.Module):
    
    def __init__(self, ndf=32):
        super(Discriminator, self).__init__()

        self.conv1 = d_conv_unit(4, ndf)
        self.conv2 = d_conv_unit(ndf, ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(4)
        self.linear = nn.Linear(ndf * 4, 1)

    def forward(self, x):
        n = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avg_pool(x)
        x = x.view(n, -1)
        x = self.linear(x)
        x = F.sigmoid(x)

        return x

class Generator(nn.Module):
    
    def __init__(self, input_dim=100, ngf=32):
        super(Generator, self).__init__()

        self.ngf = ngf
        self.project = nn.Linear(input_dim, 4*4 * ngf * 4, bias=False)
        self.batch_norm1d = nn.BatchNorm1d(4*4* ngf * 4)
        self.conv1 = g_conv_unit(ngf * 4, ngf * 2)
        self.conv2 = g_conv_unit(ngf * 2 , ngf)
        self.conv3 = nn.ConvTranspose2d(ngf, 4, 4, stride=2, padding=1)
        
    def forward(self, x):
        n = x.size()[0]
        x = F.relu(self.batch_norm1d(self.project(x)))
        x = x.view((n, self.ngf*4, 4, 4))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.tanh(x)

        return x
