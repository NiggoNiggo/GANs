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
                conditional:bool=False,
                num_classes:int=0):
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
                         conditional=conditional,
                         num_classes=num_classes)
        #hier die condition mit einbringen
    
    def train_one_epoch(self,conditional):
        #printe output von data loader
        for idx, data in tqdm(enumerate(self.data_loader)):
            self.disc.to(self.device)
            self.gen.to(self.device)
            # save current batch idx as a class variable
            self.current_batch_idx = idx
            #set löabel and data to device
            if isinstance(data,list):
                data, label = data[0], data[1]
                real, label = data.to(self.device), label.to(self.device)
                batch_size = data.size(0)
            else:
                real = data.to(self.device)
                batch_size = data.size(0)
            for num_critic in range(self.n_critic):
                noise = torch.randn(batch_size,self.params["latent_space"],device=self.device)
                if self.conditional:
                    fake = self.gen(label,noise)
                else:
                    fake = self.gen(noise)
                self.optim_disc.zero_grad()
                if self.conditional:
                    fake_disc = self.disc(label,fake.detach())
                    real_disc = self.disc(label,real)
                else:
                    fake_disc = self.disc(label,fake.detach())
                    # assert fake_disc.device == torch.device("cuda")
                    real_disc = self.disc(real)
                    # assert real_disc.device == torch.device("cuda")
                if self.conditional:
                    gp = self.gradient_penalty(real,fake,label)
                else:
                    gp = self.gradient_penalty(real,fake)
                
                loss_d = -torch.mean(real_disc) + torch.mean(fake_disc) + self.lam * gp
                loss_d.requires_grad_(True)
                
                loss_d.backward()
                self.optim_disc.step()
            self.optim_gen.zero_grad()
            noise = torch.randn(batch_size, self.params["latent_space"], device=self.device)
            if self.conditional:
                fake_img = self.gen(label,noise)
            
                fake_disc = self.disc(label,fake_img)
            else:
                fake_img = self.gen(noise)
                fake_disc = self.disc(fake_img)
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
        with torch.no_grad():
            if self.conditional:
                label = torch.randint(0,self.num_classes,(self.params["batch_size"],),device=self.device)
                fake = self.gen(label,noise).detach().cpu()
            else:
                fake = self.gen(noise).detach().cpu()
            vutils.save_image(vutils.make_grid(fake, padding=2, normalize=True),os.path.join(image_path,f"Spectrogramm_epoch_{epoch}.png"),normalize=True)
    
    def get_mean_nyquist(self):
        pass
    #soll letzten bin entferenen und am ende mitteln mit allen entfernen und wieder hinzufügen
 
            
    def make_audio(self, epoch):
        name_gen = repr(self.gen)
        self.load_models(name_gen=self.gen)
        
        # Erzeuge Rauschvektoren für eine einzige Klasse (Batch-Größe von 1)
        
        
        with torch.no_grad():
            if self.conditional:
                noise = torch.randn(self.num_classes, self.params["latent_space"], device=self.device)
                # Erzeuge zufällige Labels für eine einzelne Klasse
                label = torch.randint(0, self.num_classes, (self.num_classes,), device=self.device)
                fake = self.gen(label, noise).detach().cpu()
            else:
                noise = torch.rands(4,self.params["latent_space"],device=self.device)
                fake = self.gen(noise).detach().cpu()
            for idx in range(fake.size(0)):
                gen_spec = np.squeeze(fake[idx],(0,1))

        
                # Rekonstruiere das Audiosignal
                estimated_phase = librosa.griffinlim(S=np.vstack((gen_spec,np.zeros((1,128)))), n_iter=16, n_fft=256, hop_length=128)
                
                # Speichere das Audiosignal mit einem eindeutigen Namen
                output_path = os.path.join(self.name, "images", f"{name_gen}_{self.name}_epoch_{epoch}_class_{idx}.wav")
                sf.write(output_path, estimated_phase, 16000)


        # for pred_id in range(fake.size(0)):
        #     spec_gen = fake[pred_id].squeeze(0).numpy()
            
        #     # Entferne den letzten Frequenz-Bin
        #     spec_gen = spec_gen[:-1, :]
        #     S = np.vstack((spec_gen, np.zeros((1, 128))))
        #     # Rekonstruiere das Audiosignal
        #     estimated_phase = librosa.griffinlim(S=S, n_iter=16, n_fft=256,win_length=256, hop_length=128)
            
        #     # Speichere das Audiosignal mit einem eindeutigen Namen
        #     sf.write(os.path.join(self.name, "images, "f"{name_gen}_{self.name}_epoch_{epoch}_id_{pred_id}.wav"), estimated_phase, 16000)
