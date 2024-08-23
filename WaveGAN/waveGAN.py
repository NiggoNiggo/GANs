from WGAN_GP.wgan_pg import WGAN
import soundfile as sf
import torch 
from torch import optim
import torchvision.utils as vutils
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from Base_Models.audio_transformer import WaveNormalizer
from Utils.parameters import parse_wavegan_arguments
from WaveGAN.wavegan_dataset import WaveDataset

from WaveGAN.wave_discriminator import WaveDiscriminator
from WaveGAN.wave_generator import WaveGenerator
from Utils.utils import init_weights
import os
import numpy as np
class WaveGAN(WGAN):
    def __init__(self,
                
                device:str,
                name:str,
                ):
        super().__init__(
                         device=device,
                         name=name
                        )
    def init_models(self):
        self.params = parse_wavegan_arguments()
        
        self.gen = WaveGenerator(self.params.num_layers,self.params.c,self.params.d).to(self.device)
        self.disc = WaveDiscriminator(self.params.num_layers,self.params.c,self.params.d).to(self.device)

        #apply weights
        self.gen.apply(init_weights)
        self.disc.apply(init_weights)

        self.optim_gen = optim.Adam(params=self.gen.parameters(),
                                            lr=self.params.lr,
                                            betas=self.params.betas)
        #init optimizers Discriminator
        self.optim_disc = optim.Adam(params=self.disc.parameters(),
                                            lr=self.params.lr,
                                            betas=self.params.betas)
    
        dataset = WaveDataset(self.params.data_path,transform=WaveNormalizer())
        self.dataloader = torch.utils.data.DataLoader(
                                            dataset=dataset,
                                            batch_size=self.params.batchsize,
                                            shuffle=True)
        
            
    def train_one_epoch(self):
       return super().train_one_epoch()
    
    def _process_real_data(self, data: torch.tensor):
        return data.unsqueeze(1).to(self.device)

    def _train_generator(self,batch_size):
        return super()._train_generator(batch_size)
    
    def _train_discriminator(self, real, fake):
        return super()._train_discriminator(real, fake)
    
   
    
    
    def predict(self,epoch):
        noise = self.make_noise(1)
        fake = self.gen(noise).detach().cpu()
        # fake = WaveNormalizer().denormalize_waveform(fake)
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
            
  