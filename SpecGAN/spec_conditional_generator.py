from Base_Models.custom_layers import UpscaleLayer, ReshapeLayer
from torch import nn
from SpecGAN.spec_generator import SpecGenerator
import torch

class ConditionalSpecGenerator(nn.Module):
    def __init__(self,
                 num_layers:int,
                 num_labels:int,
                 c:int,
                 d:int):
        super().__init__()
        self.d = d
        self.c = c
        self.num_labels = num_labels 
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_labels,num_labels)
        self.img_size = (128,128)
        layers = [
        nn.Linear(100+num_labels,256*self.d),
        nn.Unflatten(1, (16 * d, 4, 4)),
        nn.ReLU(True)
        ]
        in_channels = [16,8,4,2,1]
        out_channels = [8,4,2,1,self.c]
        for num in range(self.num_layers):
            current_layer = UpscaleLayer(in_channels=in_channels[num]*self.d,
                                         out_channels=out_channels[num]*self.d if (num != self.num_layers-1) else self.c,
                                         kernel_size=5,
                                         stride=2,
                                         padding=2,
                                         output_padding=1,
                                         batchnorm=False,
                                         last_layer=(num == self.num_layers-1))
            layers.append(current_layer)
        self.model = nn.Sequential(*layers)
        
        
        
    def forward(self,labels,noise):
        c = self.embedding(labels)
        x = torch.cat([noise, c], dim=1)
        return self.model(x)
    
    def __repr__(self):
        return "Generator_Conditional_SpecGAN_"
    
