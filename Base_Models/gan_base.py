import os
import re
from PIL import Image
import random


from Base_Models.audio_transformer import WaveNormalizer

import pandas as pd
import matplotlib.pyplot as plt
import torch 
import torchvision.utils as vutils
from torchsummary import summary
import torchaudio
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.tensorboard import SummaryWriter

class GanBase(object):
    def __init__(self,
                 device:str,
                 params,
                 name:str):
        """Base class for Gan training. train_one_epoch has to be implementet to adapt the 
        training to a various gan training
        This should be able to make training for DCGAN, WGAN and CycleGAN

        Parameters
        ----------
        params : argparser object
            contains all parameters to train the gan
        device : str
            if 'cuda' use gpu if 'cpu' use cpu
        name : str
            name to save the images and models a folder structer is created to this name
        """
        self.device = torch.device("cuda" if device == "cuda" else "cpu")
        self.name = name
        self.params = params 
        self.loss_values = {}#contains los values for variable nums of gens and disc
        self.init_models()
        self._create_directory()
        self.start_epoch = 0
        self.writer = SummaryWriter(os.path.join(self.params.save_path,self.name))
        self._conditional = self.check_conditional()
        self.scores = {"loss_d":[],
                       "loss_g":[],
                       "fid":[]}
        

    
    @property
    def conditional(self):
        return self._conditional
    
    def check_conditional(self):
        """check_conditional Checks if an instance of this base class is called in conditional or not 
        conditional constallation. If conditional is true the variable _conditional is changed for the 
        GAN and probably inheriated by the sub class

        Returns
        -------
        bool
            If True the GAN is conditional if False than not
        """
        #check if it is a instance of wavegan
        from WaveGAN.conditional_wavegan import ConditionalWaveGAN
        if isinstance(self, ConditionalWaveGAN):
            return True
        from WaveGAN.waveGAN import WaveGAN
        if isinstance(self,WaveGAN):
            return False
        #checks if is instance of wgan gp
        from WGAN_GP.wgan_pg import WGAN
        if isinstance(self, WGAN):
            return False
        
        from CGAN.cgan import CGAN
        if isinstance(self,CGAN):
            return True
        return False



    
    def init_models(self):
        """This methode has to be overwritten in order to initilize your gan models. Generator, Discriminator, loss Functions and optimzers.
        This function has to be done first and should be called in the very first begining
        """
        raise NotImplementedError
    
    def make_noise(self,batch_size:int)->torch.tensor:
        """makes a latent noise vector as input of the generator
        it decides the shape of the vector with respect to the 
        datatype specified in self.params["dtype"]

        Parameters
        ----------
        batch_size : int
            batch_size 

        Returns
        -------
        torch.tensor
            latent space vector
        """
        # if data is image
        if self.params.dtype == "image":
            return torch.randn(batch_size,self.params.latent_space,1,1,device=self.device)
        # if data is audio 
        elif self.params.dtype == "audio":
            return torch.randn(batch_size,self.params.latent_space,device=self.device)
            
    
    def make_entire_training(self):
        """Method to compute the full training routine for a GAN. To use this method self.train_one_epoch 
        has to be implementet because, this function just call self.train_one_epoch in the range of 
        self.params["epochs]. As well some statistic functions are called and saved during the training
        """
        #range from epochs to make the training
        epoch_range = range(self.start_epoch+1,self.params.epochs+self.start_epoch+1)
        for epoch in epoch_range:
            self.epoch = epoch
            # self.writer.add_scalar("Epoch",self.epoch)
            self.train_one_epoch()
            # self.validate_gan(epoch)
            if epoch == len(epoch_range)-1 or epoch % 2 == 0:
                self.save_models(self.gen,self.disc)
        """ Diese Clean models funktion muss erst nochmal vernünftig getestet werden"""
        # self.clean_models()
        self.writer.close()
    
    def tune_params(self,epochs):
        for epoch in range(epochs):
            self.epoch = epoch
            self.train_one_epoch()
            score = self.validate_gan(epoch)

    
    def train_one_epoch(self,conditional:bool):
        """Method has to be overwritten for every single Gan implementation
        The returns of this Method have to be the loss of one Epoch.
        This is the training routine for exactly one epoch.

        Returns
        ------
        Loss values of the specific GAN

        Raises
        ------
        NotImplementedError
            Has to be overwritten in order to train a Gan
        """
        raise NotImplementedError


    
    def _create_directory(self):
        """Create a folder system to organize all saved models and images ind different folders
        This Method runs automaticly in the init function"""
        #make dir for the given name
        
        if not os.path.exists(self.params.save_path):
            save_path = self.name
        else:
            save_path = os.path.join(self.params.save_path,self.name)

        os.makedirs(save_path,exist_ok=True)
        #list all needed folders
        dirs = ["models","images","fakes","optimization"]
        if self.params.dtype == "audio":
            dirs += ["audio"]
    
        #create folder for every listet folder
        for folder in dirs:
            os.makedirs(os.path.join(save_path,folder),exist_ok=True)
    
    def print_stats(self, **kwargs):
        """Method to print a informational String, after every Epoch. This function take just 
        additonal kwargs arguments some Examples are:
        
        optional params
        --------------
        epoch : int
            current Epoch
        loss_gen : float
            current loss of the Generator
        loss_disc : float
            current loss of the Discriminator
        batch_idx : int
            current Batch index
        **kwargs : optional all types
            prints all addiontal informations you pass this function
        """
        #start with empty string
        stats_string = ""
        #add every positional argument in the string
        for arg in kwargs:
            stats_string += f"{arg} = {kwargs[arg]} "
        print(stats_string)
    
    def save_models(self,*args):
        """This function takes just pytorch models in a various amount The model need to have a
        .__repr__() Methode implementet, that describe the type like "Discriminator_DCGAN_" to get 
        the maximum value out of the name.
        
        
        """
        #save the path to the models
        model_path = os.path.join(self.params.save_path,self.name,"models")
        for model in args:
            #hier noch ne Abfrage ob der Type richtig ist
            filename = f"{model.__repr__()}epoch_{self.epoch}.pth"
            torch.save(model.state_dict(),os.path.join(model_path,filename))
    
    def clean_models(self):
        """clean_models, cleans the models and deletes all but the highest (latest) models.
        """
        model_path = os.path.join(self.params.save_path, self.name, "models")
        pattern = re.compile(r"epoch_(\d+)")  # Regex to extract epoch numbers
        
        all_files = os.listdir(model_path)
        
        # Extract epoch numbers and pair them with filenames
        epoch_files = []
        for file in all_files:
            match = pattern.search(file)
            if match:
                epoch_num = int(match.group(1))  # Extract the epoch number
                epoch_files.append((epoch_num, file))
        
        if not epoch_files:
            return  # No valid models found
        
        # Sort by epoch number
        epoch_files.sort(reverse=True, key=lambda x: x[0])
        
        # The first two are the most recent models
        max_gen = epoch_files[0][1]
        max_disc = epoch_files[1][1] if len(epoch_files) > 1 else None

        # Delete all other files
        for _, file in epoch_files:
            if file != max_gen and file != max_disc:
                os.remove(os.path.join(model_path, file))


    def predict(self,epoch:int):
        """predict predict and make some images

        Parameters
        ----------
        epoch : int
            current epoch
        """
        #save path to the image folder
        image_path = os.path.join(self.params.save_path,self.name,"images")
        
        noise = torch.randn(self.params.batchsize,self.params.latent_space,1,1,device=self.device)
        with torch.no_grad():
            if self._conditional:
                labels = torch.randint(0, self.params.num_classes, (self.params.batchsize,1,1,1), device=self.device)
                fake = self.gen(noise,labels).detach().cpu()
            else:
                fake = self.gen(noise).detach().cpu()
            vutils.save_image(vutils.make_grid(fake, padding=2, normalize=True),os.path.join(image_path,f"result_epoch_{epoch}.png"),normalize=True)
    
    def print_summary(self,**kwargs):
        """print_summary show the torch summary of the given models to use this function do as follows
        repr(model)=model
        """
        for arg in kwargs:
            if arg == "disc":
                print(summary(kwargs[arg],(3,self.params.img_size,self.params.img_size)))
            elif arg == "gen":
                print(summary(kwargs[arg],(self.params.latent_space,1,1)))
    
 
        
    def make_gif(self,
                 output_path:str,
                 duration:int=500):
        """make_gif create  gif of images

        Parameters
        ----------
        output_path : str
            path where to save hif
        duration : int, optional
            time in ms fo the gif, by default 500
        """
        all_images = []
        for filename in sorted(os.listdir(os.path.join(self.params.save_path,self.name,"images"))):
            if filename.endswith(".png") or filename.endswith(".jpg") and filename != "Loss_Plot.png":
                image_path = os.path.join(r"C:\Users\analf\Desktop\Datasets_And_Results\Results\GANS",self.name, "images", filename)
                image = Image.open(image_path)
                fig, ax = plt.subplots()
                ax.imshow(image)
                ax.axis('off')  # Hide axes
                fig.canvas.draw()

                img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
                all_images.append(img)
                plt.close(fig)
        output_path = os.path.join(self.params.save_path,output_path)
        all_images[0].save(output_path, save_all=True, append_images=all_images[1:], duration=duration, loop=0)
        print(f"GIF saved as {output_path}")
        
    def load_models(self,**kwargs):
        """Methode to load torch models. You have to give the model.__repr__() (repr(model)) string
        with respect to the model itself.
        --------
        Example Usage : 
        -------------
        load_models(str(repr(gen))=gen,str(repr(disc))=disc)
        
        Common Arguments : **kwargs
        -------------
        repr(gen) = gen
        repr(disc) = disc
        .... 
        with this merthode it is possible to load any model for any gan that is implemented here
        with a __repr__ methode.
        
        """
        #get correct path
        path = os.path.join(self.params.save_path,self.name,"models")
        #list all models
        all_models = os.listdir(path)
        
        groups = {}
        #helper list to save all types of models
        types_models = []
        for model in all_models:
            types_models.append(model.split("_")[0])
            
        
        #eliminate duplicates
        types_models = set(types_models)
        for types in types_models:
            #sort the list to the fourth last element == epoch the first element
            groups[types] = sorted(all_models,key= lambda x: int(re.search(r"\d+",x).group()))[-1]
        

        
        for half_name, filename in groups.items():
            #arg name listed to give the model (gen or disc etc.) to this function
            for arg, model in kwargs.items():
                #arg == repr(mode) 
                # model == Model Class like Discriminator Generator or something similar
                if half_name in repr(model) and half_name in filename:
                    #load the model
                    model.load_state_dict(torch.load(os.path.join(self.params.save_path,self.name,"models",filename), weights_only=True))
        #return epoch from filename
        try:
            match = re.search(r"\d+", filename)
            if match:
                epoch = match.group()
                self.start_epoch = int(epoch)
        except UnboundLocalError:
            pass            
        


    def fid_validation(self,
                    real_path: str,
                    fake_path: str,
                    epoch: int,
                    num_files: int = 500)->torch.tensor:
        """fid_validation of GANS for audio and images. Return fid_score and print it

        Parameters
        ----------
        real_path : str
            path to the real data
        fake_path : str
            path to the fake data
        epoch : int
            current epoch
        num_files : int, optional
            amount of files to compute fid, by default 100

        Returns
        -------
        torch.tensor
            fid value 
        """
        #search fake files
        fake_files = []
        for r, d, f in os.walk(fake_path):
            fake_files.extend([os.path.join(r, file) for file in f if file.find(f"epoch_{epoch}")])

        #searcj real files
        real_files = []
        if real_path.endswith(".csv"):
            data = pd.read_csv(real_path,index_col=False)
            real_files = data["Filename"].to_list()
        else:
            for r, d, f in os.walk(real_path):
                real_files.extend([os.path.join(r, file) for file in f if file.find(f"epoch_{epoch}")])

        #break if no files found
        if len(fake_files) == 0 or len(real_files) == 0:
            print("No files found for the specified epoch.",len(fake_files),len(real_files))
            return None

        #shuffle real data
        random.shuffle(real_files)
        #just take some samples
        real_files = real_files[:num_files]
        fake_files = fake_files[:num_files]

        real_specs = []
        fake_specs = []
        #do audio processing
        if self.params.dtype == "audio":
            
            apply_spec = torchaudio.transforms.Spectrogram(n_fft=256, hop_length=128)
            apply_db = torchaudio.transforms.AmplitudeToDB()
            norm = WaveNormalizer(self.params.audio_size)

            for real_file, fake_file in zip(real_files, fake_files):
                # try:
                real_audio, _ = torchaudio.load(real_file)
                fake_audio, _ = torchaudio.load(fake_file)

                real_audio = norm(real_audio)

                # Compute spectrogram
                S_real = apply_spec(real_audio)
                S_fake = apply_spec(fake_audio)
                
                db_real = apply_db(S_real)
                db_fake = apply_db(S_fake)

                # Repeat channels to match 3-channel input
                db_real = db_real.repeat(3, 1, 1)
                db_fake = db_fake.repeat(3, 1, 1)

                real_specs.append(db_real)
                fake_specs.append(db_fake)
                # except Exception as e:
                #     print(f"Error processing files {real_file} and {fake_file}: {e}")
                #     continue
        # do image processing
        elif self.params.dtype == "image":
            pass

        # Concatenate all spectrograms along the batch dimension
        real_data = torch.stack(real_specs, dim=0).to(torch.uint8)
        fake_data = torch.stack(fake_specs, dim=0).to(torch.uint8)

        
        # Compute FID
        
        fid = FrechetInceptionDistance()
        fid.update(real_data, real=True)
        fid.update(fake_data, real=False)
        fid_score = fid.compute()

        #remove fake files
        for file in os.listdir(fake_path):
            os.remove(os.path.join(self.params.save_path,self.name,"fakes",file))

        print(f"The FID Score in epoch {epoch} is {fid_score}")
        # self.writer.add_scalar("FID",fid_score,global_step=self.epoch)
        self.scores["fid"].append(fid_score)
        return fid_score

    
            
            
            
        


                

    
                
                
                
