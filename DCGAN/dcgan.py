from DCGAN.dcgan_discriminator import Discriminator
from DCGAN.dcgan_generator import Generator
from Base_Models.gan_base import GanBase
from Base_Models.image_data_loader import CustomDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm.auto import tqdm
from torch import optim, nn
import torchvision.transforms as T

from Utils.utils import init_weights


class DCGAN(GanBase):
    def __init__(self,
                params:dict,
                name:str):
        super().__init__(params=params,
                         name=name)
        #init loss save dict
        self.init_models()
        self.loss_values["loss_d"] = []
        self.loss_values["loss_g"] = []
    
    def init_models(self):
        self.disc = Discriminator(num_layers=self.params.num_layers,
                                  in_channels=[3, 64, 128, 256, 512, 1024],
                                  out_channels=[64, 128, 256, 512, 1024, 1],
                                  kernel_sizes=[4, 4, 4, 4, 4, 4],
                                  strides=[2, 2, 2, 2, 2, 1],
                                  paddings=[1, 1, 1, 1, 1, 0]).to(self.device)

        self.gen = Generator(num_layers=self.params.num_layers,
                                  in_channels=[100, 1024, 512, 256, 128, 64],
                                  out_channels=[1024, 512, 256, 128, 64, 3],
                                  kernel_sizes=[4, 4, 4, 4, 4, 4],
                                  strides=[1, 2, 2, 2, 2, 2],
                                  paddings=[0, 1, 1, 1, 1, 1],
                                  batchnorm=False).to(self.device)

        self.disc.apply(init_weights)
        self.gen.apply(init_weights)

        self.optim_gen = optim.Adam(self.gen.parameters(),
                                    lr=self.params.lr_dcgan,
                                    betas=(0.5,0.999))

        self.optim_disc = optim.Adam(self.disc.parameters(),
                                    lr=self.params.lr_dcgan,
                                    betas=(0.5,0.999))

        self.loss_fn = nn.BCELoss()

        transforms = T.Compose([T.ToTensor(),
                                T.Resize(self.params.img_size),
                                T.CenterCrop(self.params.img_size)])
        
        self.dataloader = DataLoader(CustomDataset(self.params.data_path,transforms),
                                     batch_size=self.params.batchsize,
                                     shuffle=True)

        
    def train_one_epoch(self):
        #printe output von data loader
        for idx, data in tqdm(enumerate(self.dataloader)):
            # save current batch idx as a class variable
            self.current_batch_idx = idx
            #train discriminator with real batch
            #access data from batch
            real = data.to(self.device)
            
            #make the label of the real data for every pixel
            label_real = torch.full((real.size(0),), 1., dtype=torch.float, device=self.device)
            label_fake = torch.full((real.size(0),), 0., dtype=torch.float, device=self.device)
            
            #zero gradients
            self.disc.zero_grad()
            #predict real image
            pred_real = self.disc(real)
            #calculate loss pred_real and real label
            loss_real = self.loss_fn(pred_real,label_real)
            #backward discriminator
            loss_real.backward()
            # loss_real_disc = pred_real.mean().item()
            
            #train Discriminator with fake batch
            noise = torch.randn(real.size(0),self.params.latent_space,1,1,device=self.device)
            fake = self.gen(noise)
            #predict the fake labels
            pred_fake = self.disc(fake.detach())
            #calc loss between fake predictions and fake label
            loss_fake = self.loss_fn(pred_fake,label_fake)
            loss_fake.backward()
            
            #calc full discriminator loss
            loss_d = loss_fake + loss_real
            # loss_fake_disc = pred_fake.mean().item()
            #optimizer step
            self.optim_disc.step() 
            
            #train generator
            self.gen.zero_grad()
            
            pred_fake_gen = self.disc(fake)
            loss_g = self.loss_fn(pred_fake_gen,label_real)
            loss_g.backward()
            self.optim_gen.step()
            
            # loss_gen = pred_fake_gen.mean().item()
            
            loss_d = round(loss_d.detach().cpu().item(),5)
            loss_g = round(loss_g.detach().cpu().item(),5)
            

            if idx % 100 == 0:
                self.print_stats(epoch=self.epoch,
                                loss_d=loss_d,
                                loss_g=loss_g)
                self.predict(self.epoch)
            #save loss values to make it plotable
            self.loss_values["loss_d"].append(loss_d)
            self.loss_values["loss_g"].append(loss_g)
        self.save_models(self.gen,self.disc)


if __name__ == "__main__":
    cdgan = DCGAN()