from WGAN_GP.wgan_pg import WGAN
from SpecGAN.spec_discriminator import SpecDiscriminator
from SpecGAN.spec_generator import SpecGenerator
import torch
from torch import optim
import os
import torchvision.utils as vutils
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from Base_Models.audio_data_loader import AudioDataset
from Utils.utils import init_weights
from Utils.parameters import parse_gan_args


class SpecGAN(WGAN):
    def __init__(self,
                 device:str,
                 params,
                name:str):
        super().__init__(device=device,
                         name=name,
                         params=params
                         )
        self.params = params 
        #hier die condition mit einbringen
    
    def init_models(self):
        # self.params = parse_specgan_arguments()
        self.gen = SpecGenerator(self.params.num_layers,
                            self.params.c,
                            self.params.d).to(self.device)
        self.disc = SpecDiscriminator(self.params.num_layers,
                                 self.params.c,
                                 self.params.d).to(self.device)

        #apply weights
        self.gen.apply(init_weights)
        self.disc.apply(init_weights)

        
        self.optim_gen= optim.Adam(self.gen.parameters(),
                               lr=self.params.lr,
                               betas=self.params.betas)
        #init optimizers Discriminator
        self.optim_disc = optim.Adam(self.disc.parameters(),
                               lr=self.params.lr,
                               betas=self.params.betas)
        
        dataset = AudioDataset(self.params.data_path)

        self.dataloader = torch.utils.data.DataLoader(
                                            dataset=dataset,
                                            batch_size=self.params.batchsize,
                                            shuffle=True) 

    def train_one_epoch(self):
        return super().train_one_epoch()
    
        

    def _train_discriminator(self, real, fake):
        return super()._train_discriminator(real,fake)
    
    
    def _train_generator(self,batch_size):
        return super()._train_generator(batch_size)
      
    

    def predict(self,epoch):
        #save path to the image folder
        image_path = os.path.join(self.save_path,self.name,"images")
        noise = self.make_noise(self.params.batchsize)
        with torch.no_grad():
            fake = self.gen(noise).detach().cpu()
            vutils.save_image(vutils.make_grid(fake, padding=2, normalize=True),os.path.join(image_path,f"Spectrogramm_epoch_{epoch}.png"),normalize=True)

 
            
    def make_audio(self, epoch):
        name_gen = repr(self.gen)
        self.load_models(name_gen=self.gen)
        
        
        
        with torch.no_grad():
            noise = self.make_noise(1)
            fake = self.gen(noise).detach().cpu()
            for idx in range(fake.size(0)):
                gen_spec = np.squeeze(fake[idx],(0,1))

                gen_spec = 10**(gen_spec / 20)

                estimated_phase = librosa.griffinlim(S=np.vstack((gen_spec,np.zeros((1,128)))), n_iter=16, n_fft=256, hop_length=128)
                
                output_path = os.path.join(self.save_path,self.name, "audio", f"{name_gen}_{self.name}_epoch_{epoch}_class_{idx}.wav")
                fig, ax = plt.subplots(1,2)
                plt.title(f" Epoch: {epoch}")
                ax[0].imshow(gen_spec)
                ax[0].axis("off")
                t = np.arange(0,len(estimated_phase))
                ax[1].plot(t/16000,estimated_phase)
                ax[1].set_xlabel("time in s")
                ax[1].set_ylabel("Amplitude")
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_path,self.name,"images",f"result_epoch_{epoch}.png"))
                plt.close()
                sf.write(output_path, estimated_phase, 16000)
