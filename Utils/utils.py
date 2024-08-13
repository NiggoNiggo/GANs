import torch
from torch import nn
from torchsummary import summary

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



def print_torchinfo(model:torch.nn.Module,
                    batch_size:int,
                    img_size:int,
                    model_type:str="gen",
                    *args):
    if model_type == "disc":
        print(summary(model,(batch_size,img_size,img_size)))
    elif model_type == "gen":
        print(summary(model,(batch_size,laten_space,1,1)))