from Base_Models import gan_base
from WGAN_GP.wgan_pg import WGAN
from tqdm.auto import tqdm
import torch
import os
import torchvision.utils as vutils
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

class SpecGAN(WGAN):
    def __init__(self,
                gen,
                disc,
                optim_gen,
                optim_disc,
                dataloader,
                params:dict,
                device:str,
                name:str,
                lam:int,
                n_critic:int,
                alpha:float,
                betas:tuple,
                conditional:bool=False):
        super().__init__(gen=gen,
                         disc=disc,
                         optim_disc=optim_disc,
                         optim_gen=optim_gen,
                         dataloader=dataloader,
                         params=params,
                         device=device,
                         name=name,
                         lam=lam,n_critic=n_critic,
                         alpha=alpha,betas=betas,
                         conditional=conditional)
        #hier die condition mit einbringen
    
    def train_one_epoch(self,conditional):
        #printe output von data loader
        for idx, (data,label) in tqdm(enumerate(self.data_loader)):
            batch_size = data.size(0)
            # self.disc.to(self.device)
            # save current batch idx as a class variable
            self.current_batch_idx = idx
            #set löabel and data to device
            real,label = data.to(self.device),label.to(self.device)
            for num_critic in range(self.n_critic):
                noise = torch.randn(batch_size,self.params["latent_space"],device=self.device)
                fake = self.gen(label,noise)
                self.optim_disc.zero_grad()
                fake_disc = self.disc(label,fake.detach())
                # assert fake_disc.device == torch.device("cuda")
                real_disc = self.disc(real)
                # assert real_disc.device == torch.device("cuda")
                
                gp = self.gradient_penalty(real,fake)
                
                loss_d = -torch.mean(real_disc) + torch.mean(fake_disc) + self.lam * gp
                loss_d.requires_grad_(True)
                
                loss_d.backward()
                self.optim_disc.step()
            self.optim_gen.zero_grad()
            noise = torch.randn(batch_size, self.params["latent_space"], device=self.device)
            fake_img = self.gen(label,noise)
            
            fake_disc = self.disc(label,fake_img)
            fake_loss = -torch.mean(fake_disc)            
            # fake_loss.requires_grad_(True)
            fake_loss.backward()
            self.optim_gen.step()
            if idx % 100 == 0:
                self.print_stats(epoch=self.epoch,batch_idx=idx,loss_d=loss_d, loss_g=fake_loss)
                self.predict(self.epoch)
                self.make_audio(self.epoch)
            #append loss to loss dictionary
            self.loss_values["loss_d"].append(loss_d)
            self.loss_values["loss_g"].append(fake_loss)
        self.save_models(self.gen,self.disc)
    
   
    
    
    def predict(self,epoch):
        #save path to the image folder
        image_path = os.path.join(self.name,"images")
        noise = torch.randn(self.params["batch_size"],self.params["latent_space"],device=self.device)
        label = torch.randint(0,9,noise.size(0))
        with torch.no_grad():
            fake = self.gen(noise).detach().cpu()
            vutils.save_image(vutils.make_grid(fake, padding=2, normalize=True),os.path.join(image_path,f"Spectrogramm_epoch_{epoch}.png"),normalize=True)
    
    def get_mean_nyquist(self):
        pass
    #soll letzten bin entferenen und am ende mitteln mit allen entfernen und wieder hinzufügen
 
            
    def make_audio(self,epoch):
        name_gen = repr(self.gen)
        self.load_models(name_gen=self.gen)
        noise = torch.randn(1,self.params["latent_space"],device=self.device)
        with torch.no_grad():
            fake = torch.squeeze(self.gen(noise).detach().cpu(),(0,1))
            gen_spec = fake.numpy()
            spec_gen = gen_spec[:-1,:]
            estimated_phase = librosa.griffinlim(S=np.vstack((gen_spec,np.zeros((1,128)))),n_iter=16,n_fft=256,hop_length=128)
            sf.write(f"{name_gen}_{self.name}_epoch_{epoch}.wav",estimated_phase,16000)
            