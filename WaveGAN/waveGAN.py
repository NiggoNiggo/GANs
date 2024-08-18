from WGAN_GP.wgan_pg import WGAN
import soundfile as sf
import torch 
import torchvision.utils as vutils
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
class WaveGAN(WGAN):
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
        
            
    def train_one_epoch(self):
       return super().train_one_epoch()
    
    def _process_real_data(self, data: torch.tensor):
        return data.unsqueeze(1).to(self.params["device"])

    def _train_generator(self,batch_size):
        return super()._train_generator(batch_size)
    
    def _train_discriminator(self, real, fake):
        return super()._train_discriminator(real, fake)
    
   
    
    
    def predict(self,epoch):
        noise = self.make_noise(1)
        fake = self.gen(noise).detach().cpu()
        fig, ax = plt.subplots()
        output_path = os.path.join(self.save_path,self.name, "audio", f"{repr(self.gen)}_{self.name}_epoch_{epoch}.wav")
        plt.title(f" Epoch: {epoch}")
        data = fake.detach().squeeze(0,1).cpu().numpy()
        t = np.arange(0,len(data))
        ax.plot(t/16000,data)
        # ax.xlabel("time in s")
        # ax.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path,self.name,"images",f"result_epoch_{epoch}.png"))
        plt.close()
        sf.write(output_path, data, 16000)
    
    # def make_audio(self,epoch):
    #     self.load_models(self.gen, self.disc)
    #     noise = torch.randn(0,self.params["latent_space"],device=self.params["device"])
    #     fake = self.gen(noise).detach().cpu()
    #     sf.write(file=os.path.join(self.save_path,self.name,"audio",f"wave_gan_{self.name}_epoch_{epoch}.wav"),data=fake)
            
  