from Base_Models.custom_layers import DownScaleConv1d, PhaseShuffle
from torch import nn
import torch


class WaveDiscriminator(nn.Module):
    def __init__(self,
                 num_samples:int,
                 c:int,
                 d:int):
        super().__init__()
        self.d = d
        self.c = c
        self.num_samples = num_samples
        layers = []
        if self.num_samples == 65536:
            in_channels = [1,1,2,4,8,16]
            out_channels = [1,2,4,8,16,32]
            num_layers = len(in_channels)
            model_complexity = 32   
            self.linear_shape = 16*model_complexity*self.d
            
        if self.num_samples == 16384:
            in_channels = [self.c,1,2,4,8]
            out_channels = [1,2,4,8,16]
            num_layers = len(in_channels)
            model_complexity = 16#16#1
            self.linear_shape = 16*model_complexity*self.d

        for num in range(num_layers):
            factor_in = self.d*in_channels[num]
            factor_out = self.d*out_channels[num]
            # print(factor_in,factor_out)
            current_layer = DownScaleConv1d(in_channels=factor_in if (num != 0) else self.c,
                                         out_channels=factor_out,
                                         kernel_size=25,
                                         stride=4,
                                         padding=11,
                                         batchnorm=False,
                                         last_layer=(num == num_layers-1))
            
            layers.append(current_layer)
            layers.append(PhaseShuffle(2))
        self.model = nn.Sequential(*layers)
        self.fc = nn.Linear(self.linear_shape,1)
    

    def forward(self,x):
        # print(self.model)
        x =  self.model(x)
        print(x.shape)
        x = x.reshape(x.size(0), self.linear_shape)
        x = self.fc(x)
        return x
    

    def __repr__(self):
        return "Discriminator_WaveGAN_"

class ConditionalWavediscriminator(WaveDiscriminator):
    def __init__(self,
                 num_samples:int,
                 c:int,
                 d:int,
                 num_classes:int):
        super().__init__(num_samples=num_samples,
                         c=c,
                         d=d)
        self.num_classes = num_classes
        self.embedding = nn.Embedding(self.num_classes,self.num_classes)
        first_layer = self.model[0]
        factor_out = first_layer.out_channels
        conditional_in_channels = self.num_samples + self.num_classes
        self.model[0] = DownScaleConv1d(
            in_channels=1,
            out_channels=factor_out,
            kernel_size=25,
            stride=4,
            padding=11,
            batchnorm=False,
            last_layer=False
        )

    def forward(self, x, label):
        x = torch.cat((x.view(x.size(0), -1), self.embedding(label)), -1)
        x = x.unsqueeze(1)  
        x = x.view(x.size(0), 1, -1)  
        x = self.model(x)
        x = x.reshape(x.size(0), self.linear_shape)
        x = self.fc(x)
        return x