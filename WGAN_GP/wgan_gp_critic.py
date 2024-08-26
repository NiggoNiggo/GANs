from torch import nn


from Base_Models.custom_layers import CriticLayer


class Critiker(nn.Module):
    def __init__(self,
                 num_layers:int,
                 in_channels:list,
                 out_channels:list,
                 kernel_sizes:list,
                 strides:list,
                 paddings:list
                 ):
        super().__init__()
        # print("now critik")
        self.layers = []
        for num in range(num_layers):
            last_layer = (num == num_layers - 1)
            layer = CriticLayer(in_channels=in_channels[num],
                                 out_channels=out_channels[num],
                                 kernel_size=kernel_sizes[num],
                                 stride=strides[num],
                                 padding=paddings[num],
                                 last_layer=last_layer,
                                 batchnorm=False)
            self.layers.append(layer)
        self.model = nn.Sequential(*self.layers)
        # print(self.model)
    
    def __repr__(self):
        return "Critiker_WGANGP_"
    
    def forward(self, x):
        x = self.model(x)
        return x.view(-1, 1).squeeze(1)  # Flatten output
    
        
