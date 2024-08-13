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
    

if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = Generator(num_layers=5,
                    in_channels=[100,512,256,128,64],
                    out_channels=[512,256,128,64,3],
                    kernel_sizes=[4,4,4,4,4],
                    strides=[1,2,2,2,2],
                    paddings=[0,1,1,1,1]).to(device)
    
    latent_space = torch.rand((1,100,1,1)).to(device)
    print(latent_space.shape)
    output = gen(latent_space)
    print(output)
    print(output.shape)