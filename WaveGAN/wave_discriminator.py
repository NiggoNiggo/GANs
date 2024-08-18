from Base_Models.custom_layers import DownScaleConv1d, PhaseShuffle
from torch import nn
from torch.nn.functional import linear
import torch


class WaveDiscriminator(nn.Module):
    def __init__(self,num_layers,c,d):
        super().__init__()
        self.d = d
        self.c = c
        self.num_layers = num_layers
        
        layers = []
        in_channels = [self.c,1,2,4,8]
        out_channels = [1,2,4,8,16]
        for num in range(self.num_layers):
            current_layer = DownScaleConv1d(in_channels=in_channels[num]*self.d if (num != 0) else self.c,
                                         out_channels=out_channels[num]*self.d,
                                         kernel_size=25,
                                         stride=4,
                                         padding=11,
                                         batchnorm=False,
                                         last_layer=(num == self.num_layers-1))
            
            layers.append(current_layer)
            layers.append(PhaseShuffle(2))
        self.model = nn.Sequential(*layers)
        self.fc = nn.Linear(256 * self.d,1)
        
    def forward(self,x):
        # print(self.model)
        x =  self.model(x)
        x = x.reshape(x.size(0), 256 * 64)
        x = self.fc(x)
        return x
    

    def __repr__(self):
        return "Discriminator_WaveGAN_"