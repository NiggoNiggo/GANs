from Base_Models.custom_layers import UpscaleConvTranspose1d, ReshapeLayer
from torch import nn

class WaveGenerator(nn.Module):
    def __init__(self,
                 len_samples:int,
                 num_layers:int,
                 c:int,
                 d:int):
        super().__init__()
        self.num_layers = num_layers
        self.len_samples = len_samples
        self.d = d
        self.c = c
        # out_channels = [16,8,4,2,1,self.c]


        if self.len_samples  == 16384:
            in_channels = [ 16// (2**i) for i in range(self.num_layers)]
            out_channels = [16 // (2**i) for i in range(self.num_layers-1)] + [self.c]
            layers = [nn.Linear(100,256*self.d), 
                        nn.Unflatten(1, (16*self.d, 16)),
                        nn.ReLU(True)
                        ]
        elif self.len_samples == 32768:
            in_channels = [ 32// (2**i) for i in range(self.num_layers)]
            out_channels = [32 // (2**i) for i in range(self.num_layers-1)] + [self.c]
            layers = [nn.Linear(100,256*self.d), 
                        nn.Unflatten(1, (32*self.d, 32)),
                        nn.ReLU(True)
                        ]

        for num in range(self.num_layers):
            current_layer = UpscaleConvTranspose1d(in_channels=in_channels[num]*self.d,
                                         out_channels=out_channels[num]*self.d if (num != self.num_layers-1) else self.c,
                                         kernel_size=25,
                                         stride=4,
                                         padding=11,
                                         output_padding=1,
                                         batchnorm=False,
                                         last_layer=(num == self.num_layers-1))
            layers.append(current_layer)
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        x = self.model(x)
        return x
    

    def __repr__(self):
        return "Generator_WaveGAN_"