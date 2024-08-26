from torch import nn
from Base_Models.custom_layers import UpscaleLayer

class Generator(nn.Module):
    def __init__(self,
                 num_layers:int,
                 in_channels:list,
                 out_channels:list,
                 kernel_sizes:list,
                 strides:list,
                 paddings:list,
                 batchnorm:bool=True
                 ):
        super().__init__()
        self.layers = []
        for num in range(num_layers):
            #get the correct argument for last layer
            last_layer = True if num == num_layers-1 else False
            layer = UpscaleLayer(in_channels=in_channels[num],
                                 out_channels=out_channels[num],
                                 kernel_size=kernel_sizes[num],
                                 stride=strides[num],
                                 padding=paddings[num],
                                 batchnorm=batchnorm,
                                 last_layer=last_layer)
            self.layers.append(layer)
        self.model = nn.Sequential(*self.layers)
    
    def __repr__(self):
        return "Generator_DCGAN_"
    
    def forward(self,x):
        return self.model(x)
    

