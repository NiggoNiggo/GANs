from Base_Models.custom_layers import UpscaleLayer, ReshapeLayer
from torch import nn

class SpecGenerator(nn.Module):
    def __init__(self,num_layers,c,d):
        super().__init__()
        self.d = d
        self.c = c
        self.num_layers = num_layers
        
        layers = [
        nn.Linear(100,256*self.d),
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
        
    def forward(self,x):
        return self.model(x)

    def __repr__(self):
        return "Generator_SpecGAN_"
    
if __name__ == "__main__":
    import torch 
    spec = SpecGenerator(d=64,c=3)
    latent = torch.rand(1,100)
    print(spec(latent).shape)