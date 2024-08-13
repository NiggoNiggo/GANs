from torch import nn
import torch
from Base_Models.custom_layers import UpscaleLayer

class Generator(nn.Module):
    def __init__(self,
                 num_labels:int,
                 num_layers:int,
                 in_channels:list,
                 out_channels:list,
                 kernel_sizes:list,
                 strides:list,
                 paddings:list,
                 batchnorm:bool=True,
                 img_size:tuple=(64,64)
                 ):
        super().__init__()
        self.layers = []
        self.img_size = img_size
        self.embedding = nn.Embedding(num_labels,num_labels)
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
        return "Generator_CGAN_"
    
    def forward(self,labels,noise):
        x = torch.cat((self.embedding(labels),noise),-1)
        x = self.model(x)
        x = x.view(x.size(0),*self.img_size)
        return x