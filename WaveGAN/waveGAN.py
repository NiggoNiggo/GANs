from WGAN_GP.wgan_pg import WGAN
import soundfile as sf
import torch 
from torch import optim
import torchvision.utils as vutils
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from Base_Models.audio_transformer import WaveNormalizer
# from Utils.parameters import parse_wavegan_arguments
from WaveGAN.wavegan_dataset import WaveDataset

from WaveGAN.wave_discriminator import WaveDiscriminator
from WaveGAN.wave_generator import WaveGenerator
from Utils.utils import init_weights
import os
import numpy as np


from WaveGAN import args

class WaveGAN(WGAN):
    """
    A WaveGAN model that inherits from the WGAN class and is used for generating and processing audio signals.

    This class implements a WaveGAN for generating audio content. It includes the initialization of models and optimizers,
    the execution of training epochs, and saving generated audio outputs and their visualizations.

    :param params: An argparse.Namespace object that must contain the following arguments:
        - `num_layers` (int): The number of layers in the generator and discriminator (default: 5).
        - `d` (int): Model complexity (default: 64).
        - `c` (int): Number of channels (default: 1).
        - `epochs` (int): Number of epochs for training (default: 10).
        - `data_path` (str): Path to the training data (default: "C:\\Users\\analf\\Desktop\\Datasets_And_Results\\Datasets\\RS6").
        - `batchsize` (int): Batch size for training (default: 128).
        - `latent_space` (int): Dimension of the latent space (default: 100).
        - `lr` (float): Learning rate for the Adam optimizer (default: 2e-5).
        - `audio_size` (int): Length of the audio signal (default: 16384).
        - `lam` (float): Lambda for WGAN (default: 10).
        - `n_crit` (int): Number of critics in WGAN (default: 5).
        - `alpha` (float): Alpha for WGAN (default: 0.0001).
        - `betas` (float, nargs=2): Betas for the Adam optimizer (default: (0, 0.9)).
        - `dtype` (str): Data type (image or audio) (default: "audio").

    :param device: The device string (e.g., "cpu" or "cuda") on which the model should be trained.
    :param name: The name of the model or experiment, used for saving results.

    Methods:
    - `init_models()`: Initializes the generator and discriminator models, their optimizers, and the DataLoader.
    - `train_one_epoch()`: Executes a training epoch, utilizing the implementation from the base class.
    - `_process_real_data(data)`: Processes real data for training.
    - `_train_generator(batch_size)`: Trains the generator.
    - `_train_discriminator(real, fake)`: Trains the discriminator.
    - `predict(epoch)`: Generates and saves audio files and visualizations for the specified epoch.

    Note: To use this class, ensure that `params` is an `argparse.Namespace` object containing the arguments listed above. An example of creating such an object can be found in the `parse_gan_args` function.
    """
    def __init__(self,
                params,
                device:str,
                name:str,
                ):
        super().__init__(
                         device=device,
                         name=name,
                         params=params
                        )
        self.params = params 
    def init_models(self):
        # self.params = parse_wavegan_arguments()
        
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
            
  