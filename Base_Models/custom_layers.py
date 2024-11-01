from torch import nn
import torch

class UpscaleLayer(nn.Module):
    """Upscale layer is a custom implementation to create a Layer that upsamples the signal with the neccessary 
        parameters (almost like pytorch). A *2D Convolutional Transpose Layer* is initialized.

        Parameters
        ----------
        in_channels : int
            In channels to the Concolutional layer
        out_channels : int
            out channels from the Convolutional Layer
        kernel_size : int
            Kernel size of the Convolutional Layer
        stride : int
            Stride of the Convolutional Layer
        padding : int
            Padding of the Convolutional Layer
        output_padding : int, optional
            Outout padding of the convolutional Layer, by default 0
        batchnorm : int, optional
            True if batchnorn is needed else False, by default True
        last_layer : _type_, optional
            If Last Layer sometimes the activation function may differ, by default None
        """
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
        return self.layers(x)
    

class DownscaleLayer(nn.Module):
    """DownscaleLayer down samples with a Covolutional Layer. This class takes nearly the same arguments as torch 
    *conv2d Layer* and additional the option if it is the last layer, baucause often there are some modified activations.
    Batchnorm is optional and can be implementet as well.

    Parameters
    ----------
    in_channels : int
        In channels to the Concolutional layer
    out_channels : int
        out channels from the Convolutional Layer
    kernel_size : int
        Kernel size of the Convolutional Layer
    stride : int
        Stride of the Convolutional Layer
    padding : int
        Padding of the Convolutional Layer
    output_padding : int, optional
        Outout padding of the convolutional Layer, by default 0
    batchnorm : int, optional
        True if batchnorn is needed else False, by default True
    last_layer : _type_, optional
        If Last Layer sometimes the activation function may differ, by default None
    """
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
        if activation_funktion:
            layers.append(activation_funktion)
        return nn.Sequential(*layers)
        
    
    def get_activation_last_layer(self):
        activation_function = nn.Sigmoid() if self.last_layer else nn.LeakyReLU(0.2,True)
        return activation_function
        
    def forward(self,x):
        x = self.layers(x)
        return x
    

class CriticLayer(DownscaleLayer):
    """This Layer is predefined for a Wasserstein Criticer Layer. This class is a subclass of the *DownscaleLayer* and therfore
        takes the same arguments.

        Parameters
        ----------
        in_channels : int
            In channels to the Concolutional layer
        out_channels : int
            out channels from the Convolutional Layer
        kernel_size : int
            Kernel size of the Convolutional Layer
        stride : int
            Stride of the Convolutional Layer
        padding : int
            Padding of the Convolutional Layer
        output_padding : int, optional
            Outout padding of the convolutional Layer, by default 0
        batchnorm : int, optional
            True if batchnorn is needed else False, by default True
        last_layer : _type_, optional
            If Last Layer sometimes the activation function may differ, by default None
        """
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
    """ReshapeLayer, is able to reshape the given input into the given shape and outputs the input in the new shape

        Parameters
        ----------
        output_shape : tuple
            Desired shape of the input data
        """
    def __init__(self,
                 output_shape:tuple):
        super().__init__()
        self.output_shape = output_shape
    
    def forward(self,x):
        x = x.reshape(self.output_shape)
        return x


class UpscaleConvTranspose1d(nn.Module):
    """UpscaleConvTransposed1d, up samples the data in one dimension. This is like the UpscaleCon2d Layer with the only
        difference, that it take a 1d data as input (like Audio raw files)

        Parameters
        ----------
        in_channels : int
            In channels to the Concolutional layer
        out_channels : int
            out channels from the Convolutional Layer
        kernel_size : int
            Kernel size of the Convolutional Layer
        stride : int
            Stride of the Convolutional Layer
        padding : int
            Padding of the Convolutional Layer
        output_padding : int, optional
            Outout padding of the convolutional Layer, by default 0
        batchnorm : int, optional
            True if batchnorn is needed else False, by default True
        last_layer : _type_, optional
            If Last Layer sometimes the activation function may differ, by default None
        """
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 stride:int,
                 padding:int,
                 output_padding:int,
                 batchnorm:bool=False,
                 last_layer:bool=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batchnorm = batchnorm
        self.output_padding = output_padding
        self.last_layer = last_layer
        self.layer = self.build_layer()


    def build_layer(self):
        layers =  [
                nn.ConvTranspose1d(in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        output_padding=self.output_padding,
                        bias=True)
                ]
        if not self.last_layer and self.batchnorm:                           #add Batchnorm
            layers.append(nn.BatchNorm1d(num_features=self.out_channels))
        
        activation_funktion = self.get_activation_last_layer()
        if activation_funktion:
            layers.append(activation_funktion)
        return nn.Sequential(*layers)

    def get_activation_last_layer(self):
        activation = nn.Tanh() if self.last_layer else nn.ReLU(True)
        return activation


    def forward(self,x):
        # print(x.shape,"in")
        x = self.layer(x)
        # print(x.shape,"out")
        return x


class DownScaleConv1d(nn.Module):
    """DownscaleConvTransposed1d, down samples the data in one dimension. This is like the DownscaleCon2d Layer with the only
        difference, that it take a 1d data as input (like Audio raw files)

        Parameters
        ----------
        in_channels : int
            In channels to the Concolutional layer
        out_channels : int
            out channels from the Convolutional Layer
        kernel_size : int
            Kernel size of the Convolutional Layer
        stride : int
            Stride of the Convolutional Layer
        padding : int
            Padding of the Convolutional Layer
        batchnorm : int, optional
            True if batchnorn is needed else False, by default True
        last_layer : _type_, optional
            If Last Layer sometimes the activation function may differ, by default None
        """
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 stride:int,
                 padding:int,
                 batchnorm:bool=False,
                 last_layer:bool=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batchnorm = batchnorm
        self.last_layer = last_layer
        self.layers = self.build_layer()

    def build_layer(self):
        layers =  [
                nn.Conv1d(in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        bias=True)
                ]
        if not self.last_layer and self.batchnorm:                           #add Batchnorm
            layers.append(nn.BatchNorm1d(num_features=self.out_channels))
            #bs, channels, length
        
        activation_funktion = self.get_activation_last_layer()
        if activation_funktion:
            layers.append(activation_funktion)
        return nn.Sequential(*layers)

    def get_activation_last_layer(self):
        activation = nn.LeakyReLU(0.2,True) 
        return activation
    
    def forward(self,x):
        # print(x.shape,"in")
        x = self.layers(x)
        # print(x.shape,"out")
        return x



class PhaseShuffle(nn.Module):

    def __init__(self,n:int):
        """PhaseShuffle, is from WaveGAN to shuffle the input data from the Discriminator and avoid artefacts (this is my own 
        createn, so no guarantee, that is correctly implementet)

        Parameters
        ----------
        n : int
            value of Phaseshuffel
        """
        super().__init__()
        self.n = n

    def forward(self,x):
        shift = torch.randint(-self.n, self.n + 1, (1,)).item()
        if shift == 0:
            return x
        
        elif shift > 0:
            return nn.functional.pad(x, (shift, 0), mode='reflect')[:, :, :-shift]
        else:
            return nn.functional.pad(x, (0, -shift), mode='reflect')[:, :, -shift:]