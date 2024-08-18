from Base_Models.custom_layers import UpscaleConvTranspose1d, ReshapeLayer
from torch import nn

class WaveGenerator(nn.Module):
    def __init__(self,
                 num_layers,
                 c,
                 d):
        super().__init__()
        self.num_layers = num_layers
        self.d = d
        self.c = c
        in_channels = [16,8,4,2,1]
        out_channels = [8,4,2,1,self.c]

        layers = [nn.Linear(100,256*self.d), 
                    nn.Unflatten(1, (16*self.d, 16)),
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