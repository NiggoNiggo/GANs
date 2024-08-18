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

from Base_Models.audio_transformer import SpecGANTransformer

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
    
    
    def train_one_epoch(self):
        return super().train_one_epoch()
    
    # # def _process_real_data(self, data: torch.tensor):
        

    def _train_discriminator(self, real, fake):
        return super()._train_discriminator(real,fake)
    
    
    def _train_generator(self,batch_size):
        return super()._train_generator(batch_size)
      
    

    def predict(self,epoch):
        #save path to the image folder
        image_path = os.path.join(self.save_path,self.name,"images")
        noise = self.make_noise(self.params["batch_size"])
        with torch.no_grad():
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
            noise = self.make_noise(1)
            fake = self.gen(noise).detach().cpu()
            for idx in range(fake.size(0)):
                gen_spec = np.squeeze(fake[idx],(0,1))

                gen_spec = 10**(gen_spec / 20)



                # Rekonstruiere das Audiosignal
                estimated_phase = librosa.griffinlim(S=np.vstack((gen_spec,np.zeros((1,128)))), n_iter=16, n_fft=256, hop_length=128)
                
                # Speichere das Audiosignal mit einem eindeutigen Namen
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
