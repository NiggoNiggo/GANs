from Base_Models.custom_layers import CriticLayer
from torch import nn
import torch

class ConditionalSpecDiscriminator(nn.Module):
    def __init__(self,
                 num_layers:int,
                 c:int,
                 d:int,
                 num_labels:int):
        super().__init__()
        self.d = d
        self.c = c
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_labels,num_labels)
        self.num_labels = num_labels
        layers = []
        in_channels = [self.c+self.num_labels,1,2,4,8]
        out_channels = [1,2,4,8,16]
        for num in range(self.num_layers):
            current_layer = CriticLayer(in_channels=in_channels[num]*self.d if (num != 0) else self.c+self.num_labels,
                                        out_channels=out_channels[num]*self.d,
                                        kernel_size=5,
                                        stride=2,
                                        padding=2,
                                        batchnorm=False,
                                        last_layer=(num == self.num_layers-1))
            layers.append(current_layer)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(256*self.d,1))
        self.model = nn.Sequential(*layers)
        
    def forward(self, labels,img):
        batch_size = img.size(0)
        print(img.shape,"image")
        # Labels auf die Bildgröße bringen und mit dem Bild kombinieren
        c = self.embedding(labels).view(batch_size, self.num_labels,1,1)
        print(c.shape,"embedded")
        c = c.expand(batch_size, self.num_labels, 128, 128)
        print(c.shape,"expanded")
        x = torch.cat([img, c], dim=1)  # Verknüpft entlang der Channel-Dimension
        print(x.shape,"concat")
        return self.model(x)

    def __repr__(self):
        return "Discriminator_Conditional_SpecGAN_"