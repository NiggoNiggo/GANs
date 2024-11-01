from Base_Models.custom_layers import UpscaleConvTranspose1d
from torch import nn
import torch

class WaveGenerator(nn.Module):
    """WaveGenerator is a generator network for a waveGAN.

        Parameters
        ----------
        in_channels : int
            in channels the latent space dimension    
        len_samples : int
            length of audio signal in samples must be 16384 or 65536
        c : int
            Channels of the audio signal, in most cases default == 1
        d : int
            model complexity to add some neuros per layer
        """
    def __init__(self,
                 in_channels:int,
                 len_samples:int,
                 c:int,
                 d:int):
        super().__init__()
        self.len_samples = len_samples
        self.in_channels = in_channels
        self.d = d
        self.c = c

        num_layers = len(in_channels)
        #this is for 1 seconds of audio data 
        if self.len_samples  == 16384:
            in_channels = [1,2,4,8,16]
            out_channels = [2,4,8,16,16]
            model_complexity = 1
            multiplicator = self.d*model_complexity
            layers = [nn.Linear(self.in_channels,multiplicator*16),
                        nn.Unflatten(1, (multiplicator,16)),
                        nn.ReLU(True)
                        ]
            #this is for 4 seconds od audio data
        elif self.len_samples == 65536:
            in_channels = [1,2,4,8,16,16]
            out_channels = [2,4,8,16,16,16]
            model_complexity = 32#1#32
            multiplicator = (self.d*model_complexity)
            layers = [nn.Linear(self.in_channels,16*multiplicator),
                        nn.Unflatten(1, (multiplicator,16)),
                        nn.ReLU(True)
                        ]

        for num in range(num_layers):
            #make the factor of the in and output channels
            factor_in = int(multiplicator// in_channels[num])
            factor_out = int(multiplicator// out_channels[num])
            current_layer = UpscaleConvTranspose1d(in_channels=factor_in,
                                         out_channels=factor_out if (num != num_layers-1) else self.c,
                                         kernel_size=25,
                                         stride=4,
                                         padding=11,
                                         output_padding=1,
                                         batchnorm=False,
                                         last_layer=(num == num_layers-1))
            layers.append(current_layer)
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        x = self.model(x)
        return x
    

    def __repr__(self):
        return "Generator_WaveGAN_"
    

class ConditionalWaveGenerator(WaveGenerator):
    """ConditionalWaveGenerator is a generator network for a waveGAN. This is a subclass of WaveGenerator.
    With an additionl parameter the amount of classes.

        Parameters
        ----------
        in_channels : int
            in channels the latent space dimension    
        len_samples : int
            length of audio signal in samples must be 16384 or 65536
        c : int
            Channels of the audio signal, in most cases default == 1
        d : int
            model complexity to add some neuros per layer
        len_classes : int
            amount of classes in the model
        """
    def __init__(self,
                 in_channels:int,
                 len_samples:int,
                 c:int,
                 d:int,
                 len_classes:int
                 ):
        super().__init__(len_samples=len_samples,
                         in_channels=in_channels+len_classes,
                         c=c,
                         d=d)
         
        self.len_classes = len_classes
        #embeddubg the label 
        self.embedding = nn.Embedding(in_channels,self.len_classes)
    
    def __repr__(self):
        return "Generator_CWaveGAN_"
    
    def forward(self,x,label):
        embedding = self.embedding(label)
        x = torch.concat([embedding,x],dim=1)
        x = self.model(x)
        return x