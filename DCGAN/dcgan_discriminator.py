from torch import nn
from Base_Models.custom_layers import DownscaleLayer

class Discriminator(nn.Module):
    def __init__(self,
                 num_layers:int,
                 in_channels:list,
                 out_channels:list,
                 kernel_sizes:list,
                 strides:list,
                 paddings:list
                 ):
        super().__init__()
        self.layers = []
        for num in range(num_layers):
            last_layer = (num == num_layers - 1)
            #get the correct argument for last layer
            layer = DownscaleLayer(in_channels=in_channels[num],
                                 out_channels=out_channels[num],
                                 kernel_size=kernel_sizes[num],
                                 stride=strides[num],
                                 padding=paddings[num],
                                 last_layer=last_layer)
            self.layers.append(layer)
        self.model = nn.Sequential(*self.layers)
    
    def __repr__(self):
        return "Discriminator_DCGAN_"
    
    def forward(self, x):
        x = self.model(x)
        return x.view(-1, 1).squeeze(1)  # Flatten output
    

if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    disc = Discriminator(num_layers=4,
                    in_channels=[3,64,128,256,512],
                    out_channels=[64,128,256,512,1],
                    kernel_sizes=[4,4,4,4,4],
                    strides=[1,2,2,2,1],
                    paddings=[1,1,1,1,0]).to(device)
    
    latent_space = torch.rand((1,3,64,64)).to(device)
    print(disc)
    print(latent_space.shape)
    output = disc(latent_space)
    print(output)
    print(output.shape)