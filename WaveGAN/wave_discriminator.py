from Base_Models.custom_layers import DownScaleConv1d, PhaseShuffle
from torch import nn
import torch


class WaveDiscriminator(nn.Module):
    """WaveDiscriminator is a Discriminator network for a waveGAN.

        Parameters
        ----------
        len_samples : int
            length of audio signal in samples must be 16384 or 65536
        c : int
            Channels of the audio signal, in most cases default == 1
        d : int
            model complexity to add some neuros per layer
        """
    def __init__(self,
                 num_samples:int,
                 c:int,
                 d:int):
        super().__init__()
        self.d = d
        self.c = c
        self.num_samples = num_samples
        layers = []
        
        num_layers = len(in_channels)
        #1 s of audio
        if self.num_samples == 16384:
            in_channels = [self.c,1,2,4,8]
            out_channels = [1,2,4,8,16]
            model_complexity = 16#16#1
            self.linear_shape = 16*model_complexity*self.d

        #4 s of audio 
        if self.num_samples == 65536:
            in_channels = [1,1,2,4,8,16]
            out_channels = [1,2,4,8,16,32]
            model_complexity = 32   
            #shape of the fully conected layer at the beginnign of the network
            self.linear_shape = 16*model_complexity*self.d

        for num in range(num_layers):
            factor_in = self.d*in_channels[num]
            factor_out = self.d*out_channels[num]
            current_layer = DownScaleConv1d(in_channels=factor_in if (num != 0) else self.c,
                                         out_channels=factor_out,
                                         kernel_size=25,
                                         stride=4,
                                         padding=11,
                                         batchnorm=False,
                                         last_layer=(num == num_layers-1))
            
            layers.append(current_layer)
            layers.append(PhaseShuffle(2))
        self.model = nn.Sequential(*layers)
        #output shape just one value that is the wasserstein disctance of the distributiion of genrated and train data 
        self.fc = nn.Linear(self.linear_shape,1)
    

    def forward(self,x):
        # print(self.model)
        x =  self.model(x)
        # print(x.shape)
        x = x.reshape(x.size(0), self.linear_shape)
        x = self.fc(x)
        return x
    

    def __repr__(self):
        return "Discriminator_WaveGAN_"

class ConditionalWavediscriminator(WaveDiscriminator):
    """conditionalWaveDiscriminator is a Discriminator network for a waveGAN with respect to a condition.

        Parameters
        ----------
        len_samples : int
            length of audio signal in samples must be 16384 or 65536
        c : int
            Channels of the audio signal, in most cases default == 1
        d : int
            model complexity to add some neuros per layer
        num_classes : int
            amount of classes 
        """
    def __init__(self,
                 num_samples:int,
                 c:int,
                 d:int,
                 num_classes:int):
        super().__init__(num_samples=num_samples,
                         c=c,
                         d=d)
        self.num_classes = num_classes
        self.embedding = nn.Embedding(self.num_classes,self.num_classes)
        first_layer = self.model[0]
        factor_out = first_layer.out_channels
        conditional_in_channels = self.num_samples + self.num_classes
        self.model[0] = DownScaleConv1d(
            in_channels=1,
            out_channels=factor_out,
            kernel_size=25,
            stride=4,
            padding=11,
            batchnorm=True,
            last_layer=False
        )

    def forward(self, x, label):
        x = torch.cat((x.view(x.size(0), -1), self.embedding(label)), -1)
        x = x.unsqueeze(1)  
        x = x.view(x.size(0), 1, -1)  
        x = self.model(x)
        x = x.reshape(x.size(0), self.linear_shape)
        x = self.fc(x)
        return x