from Base_Models.custom_layers import UpscaleConvTranspose1d
from torch import nn
import torch

class WaveGenerator(nn.Module):
    def __init__(self,
                 in_channels:int,
                 len_samples:int,
                 c:int,
                 d:int):
        super().__init__()
        self.len_samples = len_samples
        self.in_channels = in_channels
        self.d = d
        self.c = c


        if self.len_samples  == 16384:
            in_channels = [1,2,4,8,16]
            out_channels = [2,4,8,16,16]
            num_layers = len(in_channels)
            model_complexity = 1
            multiplicator = self.d*model_complexity
            layers = [nn.Linear(self.in_channels,multiplicator*16),
                        nn.Unflatten(1, (multiplicator,16)),
                        nn.ReLU(True)
                        ]
        elif self.len_samples == 65536:
            in_channels = [1,2,4,8,16,16]
            out_channels = [2,4,8,16,16,16]
            num_layers = len(in_channels)
            model_complexity = 32#1#32
            multiplicator = (self.d*model_complexity)
            layers = [nn.Linear(self.in_channels,16*multiplicator),
                        nn.Unflatten(1, (multiplicator,16)),
                        nn.ReLU(True)
                        ]

        for num in range(num_layers):
            factor_in = int(multiplicator// in_channels[num])
            factor_out = int(multiplicator// out_channels[num])
            current_layer = UpscaleConvTranspose1d(in_channels=factor_in,
                                         out_channels=factor_out if (num != num_layers-1) else self.c,
                                         kernel_size=25,
                                         stride=4,
                                         padding=11,
                                         output_padding=1,
                                         batchnorm=False,
                                         last_layer=(num == num_layers-1))
            layers.append(current_layer)
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        x = self.model(x)
        return x
    

    def __repr__(self):
        return "Generator_WaveGAN_"
    

class ConditionalWaveGenerator(WaveGenerator):
    def __init__(self,
                 in_channels:int,
                 len_samples:int,
                 c:int,
                 d:int,
                 len_classes:int
                 ):
        super().__init__(len_samples=len_samples,
                         in_channels=in_channels+len_classes,
                         c=c,
                         d=d)
         
        self.len_classes = len_classes
        self.embedding = nn.Embedding(100,self.len_classes)
    
    def __repr__(self):
        return "Generator_CWaveGAN_"
    
    def forward(self,x,label):
        embedding = self.embedding(label)
        x = torch.concat([embedding,x],dim=1)
        x = self.model(x)
        return x