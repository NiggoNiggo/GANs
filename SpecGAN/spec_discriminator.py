from Base_Models.custom_layers import DownscaleLayer, ReshapeLayer
from torch import nn

class SpecDiscriminator(nn.Module):
    def __init__(self,num_layers,c,d):
        super().__init__()
        self.d = d
        self.c = c
        self.num_layers = num_layers
        
        layers = []
        in_channels = [self.c,1,2,4,8]
        out_channels = [1,2,4,8,16]
        for num in range(self.num_layers):
            current_layer = DownscaleLayer(in_channels=in_channels[num]*self.d if (num != 0) else self.c,
                                         out_channels=out_channels[num]*self.d,
                                         kernel_size=5,
                                         stride=2,
                                         padding=2,
                                         batchnorm=False,
                                         last_layer=(num == self.num_layers-1))
            layers.append(current_layer)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(256*self.d,1))
        self.model = nn.Sequential(*layers)
        
    def forward(self,x):
        return self.model(x)

    def __repr__(self):
        return "Discriminator_SpecGAN_"