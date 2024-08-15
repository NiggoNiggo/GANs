from torch import nn

class UpscaleLayer(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 stride:int,
                 padding:int,
                 output_padding:int=0,
                 batchnorm:int=True,
                 last_layer=None,
                    ):
        super().__init__()
        #if last layer change to Tanh activation else stay at ReLU
        activation_function = nn.Tanh() if last_layer else nn.ReLU()
        #create a layer 
        layers = [#add convTranspose2d layer
                nn.ConvTranspose2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                        bias=False)
                ]
        #if not last layer add batchnorm
        if not last_layer and batchnorm:
            #add Batchnorm
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            #add a Leaky Relu
        layers.append(activation_function)
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self,x):
        # print(x.shape,"upscale")
        return self.layers(x)
    

class DownscaleLayer(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 stride:int,
                 padding:int,
                 batchnorm:bool=True,
                 last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batchnorm = batchnorm
        self.layers = self.build_layer()
        
    
    def build_layer(self):
        layers =  [
                nn.Conv2d(in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        bias=False)
                ]
        if not self.last_layer and self.batchnorm:                           #add Batchnorm
            layers.append(nn.BatchNorm2d(num_features=self.out_channels))
        
        activation_funktion = self.get_activation_last_layer()
        # print(activation_funktion)
        if activation_funktion:
            layers.append(activation_funktion)
        return nn.Sequential(*layers)
        
    
    def get_activation_last_layer(self):
        activation_function = nn.Sigmoid() if self.last_layer else nn.LeakyReLU(0.2,True)
        return activation_function
        
    def forward(self,x):
        # print(x.shape, "vorher")
        x = self.layers(x)
        # print(x.shape,"later")
        return x
    

class CriticLayer(DownscaleLayer):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 stride:int,
                 padding:int,
                 batchnorm:bool=True,
                 last_layer=False):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         batchnorm=batchnorm,
                         last_layer=last_layer)
        self.last_layer = last_layer
        self.layers = self.build_layer()
    
    def get_activation_last_layer(self):
        activation_function = None if self.last_layer else nn.LeakyReLU(0.2,True)
        return activation_function

        
    def forward(self,x):
        return self.layers(x)
    

class ReshapeLayer(nn.Module):
    def __init__(self,
                 output_shape):
        super().__init__()
        self.output_shape = output_shape
    
    def forward(self,x):
        x = x.reshape(self.output_shape)
        return x

