import os,re

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from torch import optim
import torch

from WGAN_GP.wgan_pg import WGAN
from WaveGAN.wave_generator import ConditionalWaveGenerator
from WaveGAN.wave_discriminator import ConditionalWavediscriminator
from WaveGAN.wavegan_dataset import ConditionalWaveDataset

from Base_Models.audio_transformer import WaveNormalizer

from WaveGAN.waveGAN import WaveGAN

from Utils.utils import init_weights


class ConditionalWaveGAN(WGAN):
    def __init__(self,
                 params,
                 device,
                 name
                #  num_classes
                 ):
        # self.num_classes = num_classes
        super().__init__(params=params,
                         device=device,
                         name=name)
        
    def init_models(self):
        """initalize the models and optimzer and the loss function. Additionally the dataset and dataloader are initalized
        the dataset is a custom datasez and can be changes or sdopted to your specific application (The dataset class is in the base models Folder)"""
        
        self.gen = ConditionalWaveGenerator(self.params.latent_space,self.params.audio_size,self.params.c,self.params.d,self.params.num_classes).to(self.device)
        self.disc = ConditionalWavediscriminator(self.params.audio_size,self.params.c,self.params.d,self.params.num_classes).to(self.device)
        gen_params = sum(p.numel() for p in self.gen.parameters())
        disc_params = sum(p.numel() for p in self.disc.parameters())
        print("gen params:",gen_params)
        print("disc params:",disc_params)
        # dummy_labels = torch.randint(0, self.params.num_classes, (self.params.batchsize,), device=self.device)
        # dummy_gen_input = (self.make_noise(self.params.batchsize),dummy_labels)

        # dummy_disc_input = (torch.randn((self.params.batchsize,self.params.audio_size,1),device=self.device),dummy_labels)

        # for k in dummy_disc_input:
        #     print(k.shape)
        # self.writer.add_graph(self.gen, dummy_gen_input)
        # self.writer.add_graph(self.disc, dummy_disc_input)
        #apply weights
        self.gen.apply(init_weights)
        self.disc.apply(init_weights)

        self.optim_gen = optim.Adam(params=self.gen.parameters(),
                                            lr=self.params.lr_g,
                                            betas=self.params.betas)
        #init optimizers Discriminator
        self.optim_disc = optim.Adam(params=self.disc.parameters(),
                                            lr=self.params.lr_d,
                                            betas=self.params.betas)
    
        self.dataset = ConditionalWaveDataset(self.params.data_path,transform=WaveNormalizer(self.params.audio_size))
        self.dataloader = torch.utils.data.DataLoader(
                                            dataset=self.dataset,
                                            batch_size=self.params.batchsize,
                                            shuffle=True)
        print(len(self.dataset),"len dataset")
        
            
    def train_one_epoch(self):
       return super().train_one_epoch()
    
    def _process_real_data(self, data:torch.tensor):
        """_process_real_data is a methode to process the real data from the dataloader to 
        the gan and make it trainable

        Parameters
        ----------
        data : torch.tensor
            output data from the dataloader

        Returns
        -------
        torch.tensor
            input data of the Discriminator
        """
        if self._conditional:
            data, label = data[0], data[1]
            return data.to(self.device), label.to(self.device)
        return data.to(self.device)

    def _train_generator(self,batch_size):
        return super()._train_generator(batch_size)
    
    def _train_discriminator(self,
                            real:torch.tensor,
                            fake:torch.tensor,
                            labels:torch.tensor=None)->float:
        return super()._train_discriminator(real,fake,labels)
    
    def validate_gan(self):#,epoch:int):
        """validate_gan validation routine for gans

        Parameters
        ----------
        epoch : int
            current epoch to validate the gan
        """
        
        self.make_audio(self.epoch,num_audios=len(self.dataset))
        real_path = self.dataset.path
        fake_path = os.path.join(self.params.save_path,self.name,"fakes")
        fid_score = self.fid_validation(real_path,fake_path,self.epoch)
        self.scores["fid"].append(float(fid_score.item()))
    
    


    def predict(self, epoch: int):
        """Predict makes an audio file and a plot for each class in a Conditional WaveGAN

        Parameters
        ----------
        epoch : int
            Current epoch
        """
        # Erstelle Rauschen und Labels für jede Klasse
        num_classes = self.params.num_classes
        noise = self.make_noise(num_classes)  # Erzeuge Rauschen für jede Klasse
        labels = torch.arange(num_classes).to(self.device)  # Erzeuge Labels von 0 bis num_classes-1
        
        # Erzeuge Samples für jede Klasse
        for i in range(num_classes):
            # Erzeuge Fake-Daten für die aktuelle Klasse
            sample_noise = noise[i:i+1]  # Wähle Rauschen für die aktuelle Klasse
            sample_label = labels[i:i+1]  # Wähle Label für die aktuelle Klasse
            fake = self.gen(sample_noise, sample_label).detach().cpu()  # Generiere Fake-Daten

            # Verarbeite die generierten Daten für die Visualisierung
            fig, ax = plt.subplots()
            output_path_audio = os.path.join(self.params.save_path, self.name, "audio", f"{self.name}_epoch_{epoch}_class_{i}.wav")
            output_path_image = os.path.join(self.params.save_path, self.name, "images", f"result_epoch_{epoch}_class_{i}.png")
            plt.title(f"Epoch: {epoch}, Class: {i}")
            
            # Nehme nur die erste Ausgabe und entferne überflüssige Dimensionen
            data = fake[0].detach().squeeze(0).cpu().numpy()  
            t = np.arange(0, len(data)) / 16000  # Zeitachse in Sekunden
            
            ax.plot(t, data)
            # string = f"Epoch {epoch} "
            # current_score = self.scores["fid"][epoch]
            # string += f"FID = {current_score} "
            # print(string)
            # ax.set_title(string)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            plt.tight_layout()
            
            # Speichere das Bild
            plt.savefig(output_path_image)
            plt.close()
            
            # Speichere die Audio-Datei
            sf.write(output_path_audio, data, 16000)
            
            # Logge die Audio-Ausgabe für TensorBoard
            self.writer.add_audio(f"Fake Audio Class {i}", fake[0], global_step=epoch, sample_rate=16000)

            # self.writer.add_image("Wave Form",fake.squeeze(0,1))
            #add spektrogramm
    
    def make_audio(self,
                   epoch:int,
                   num_audios:int):
        """make_audi makes a amount of audio files in a given epoch

        Parameters
        ----------
        epoch : int
            current epoch
        num_audios : int
            amount of audio samples to generate
        """
        name = repr(self.gen)
        # self.load_models(name=self.gen)
        for num in range(num_audios):
            noise = torch.randn(1,self.params.latent_space,device=self.device)
            if self._conditional:
                labels = torch.randint(0, self.params.num_classes, (1,), device=self.device)
                fake = self.gen(noise,labels).detach().cpu().numpy().squeeze()
            else:
                fake = self.gen(noise).detach().cpu().numpy().squeeze()
            sf.write(file=os.path.join(self.params.save_path,self.name,"fakes",f"wave_gan_{self.name}_epoch_{epoch}_num_{num}.wav"),data=fake,samplerate=16000)
            
  
    def valid_afterwards(self,
                         save_path:str,
                         mk_audio=False):
        info_dict = {}
        for file in os.listdir(save_path):
            if repr(self.disc) in file:
                continue
            path = os.path.join(save_path,file)
            self.gen.load_state_dict(torch.load(path,weights_only=True))
            epoch = re.search(r"\d+",path).group(0)
            print(epoch)
            self.validate_gan(epoch)
            if mk_audio:
                self.predict(epoch)