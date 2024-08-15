# from dcgan_discriminator import Discriminator
# from dcgan_generator import Generator
# from loss_functions import GANLoss
import os
import re
from PIL import Image

import matplotlib.pyplot as plt
import torch 
import torchvision.utils as vutils
from torchsummary import summary

class GanBase(object):
    def __init__(self,
                 gen,
                 disc,
                 optim_gen,
                 optim_disc,
                 loss_fn,
                 dataloader,
                 params:dict,
                 device:str,
                 name:str,
                 conditional:bool=False,
                 num_classes:int=0):
        """Base class for Gan training. train_one_epoch has to be implementet to adapt the 
        training to a various gan training
        This should be able to make training for DCGAN, WGAN and CycleGAN

        Parameters
        ----------
        gen : Generator 
            Generator for GAN
        disc : Discriminator
            Discriminator for GAN
        optim_gen : Optimzer 
            Optimizer for Generato
        optim_disc : Optimzer 
            Optimizer for Discriminator
        loss_fn : Loss funktion
            Loss function for GAN
        dataloader : Data Loader
            pytorch data loader that contains all data
        params : dict
            contains all parameters to train the gan
        device : str
            if 'cuda' use gpu if 'cpu' use cpu
        name : str
            name to save the images and models a folder structer is created to this name
        conditional : bool default False
            if True the system is trained with conditonal networks, You only have to update the 
            dataset the models are changes dynamaiclly
        num_classes : int
            amount of classes if the GAN is conditional
        """
        self.gen = gen
        self.disc = disc
        self.optim_gen = optim_gen
        self.optim_disc = optim_disc
        self.loss_fn = loss_fn
        self.data_loader = dataloader
        self.params = params
        self.device = torch.device("cuda" if device == "cuda" else "cpu")
        self.name = name
        self.loss_values = {}#contains los values for variable nums of gens and disc
        self.save_path = r"C:\Users\analf\Desktop\Datasets_And_Results\Results\GANS"
        self._create_directory()
        self.start_epoch = 0
        self.conditional = conditional
        self.num_classes = num_classes
        
        

    
    def make_entire_training(self):
        """Method to compute the full training routine for a GAN. To use this method self.train_one_epoch 
        has to be implementet because, this function just call self.train_one_epoch in the range of 
        self.params["epochs]. As well some statistic functions are called and saved during the training
        """
        #range from epochs to make the training
        epoch_range = range(self.start_epoch+1,self.params["epochs"]+self.start_epoch+1)
        #device how often a model should be saved
        if len(epoch_range) <20:
            self.save_step = 5
        elif len(epoch_range) > 20 and len(epoch_range)< 50:
            self.save_step = 10
        elif len(epoch_range) > 50:
            self.save_step = 20
        self.last_epoch = epoch_range[-1]
        for epoch in epoch_range:
            self.epoch = epoch
            self.train_one_epoch()
        self.plot_loss()
    
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



    def validate_model(self):
        """Method has to be overwritten for every single Gan implementation
        This Methode Validate the GAN but is individual for every GAN.

        Raises
        ------
        NotImplementedError
            Has to be overwritten in order to validate a GAN
        """
        raise NotImplementedError
    
    def _create_directory(self):
        """Create a folder system to organize all saved models and images ind different folders
        This Method runs automaticly in the init function"""
        #make dir for the given name
        os.makedirs(os.path.join(self.save_path,self.name),exist_ok=True)
        #list all needed folders
        if self.params["dtype"] == "audio":
            dirs = ["models","images","audio"]
        else:
            dirs = ["models","images"]
        #create folder for every listet folder
        for folder in dirs:
            os.makedirs(os.path.join(self.save_path,self.name,folder),exist_ok=True)
    
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
        model_path = os.path.join(self.save_path,self.name,"models")
        for model in args:
            #hier noch ne Abfrage ob der Type richtig ist
            filename = f"{model.__repr__()}epoch_{self.epoch}.pth"
            print(filename)
            torch.save(model.state_dict(),os.path.join(model_path,filename))
    
    def predict(self,epoch):
        #save path to the image folder
        image_path = os.path.join(self.save_path,self.name,"images")
        noise = torch.randn(self.params["batch_size"],self.params["latent_space"],1,1,device=self.device)
        with torch.no_grad():
            fake = self.gen(noise).detach().cpu()
            vutils.save_image(vutils.make_grid(fake, padding=2, normalize=True),os.path.join(image_path,f"result_epoch_{epoch}.png"),normalize=True)
    
    def print_summary(self,**kwargs):
        """Muss umgebaut werden, sobald nmehr gens oder disc dazu kommen
        print the summar of a  torch model 
        """
        for arg in kwargs:
            if arg == "disc":
                print(summary(kwargs[arg],(3,self.params["img_size"],self.params["img_size"])))
            elif arg == "gen":
                print(summary(kwargs[arg],(self.params["latent_space"],1,1)))
    
    def plot_loss(self):
        fig, ax = plt.subplots(figsize=(10, 6))  

        # Farben und Linienstile für bessere Unterscheidung
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        linestyles = ['-', '--', '-.', ':']
        
        for i, (loss, values) in enumerate(self.loss_values.items()):
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            ax.plot(range(len(values)), values, label=loss, color=color, linestyle=linestyle, linewidth=2)

        # Titel und Achsenbeschriftungen
        ax.set_title(f"Loss Values for {self.name}", fontsize=16, fontweight='bold')
        ax.set_xlabel("Iterations", fontsize=14)
        ax.set_ylabel("Loss Value", fontsize=14)
        
        # Legende und Grid hinzufügen
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Formatierungen für die Achsen
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()  # Verbessert die Layout-Automatisierung
        plt.savefig(os.path.join("images","Loss_Plot.png"))
        plt.show()
        
    def make_gif(self,output_path:str,duration=500):
        all_images = []
        for filename in sorted(os.listdir(os.path.join(self.save_path,self.name,"images"))):
            if filename.endswith(".png") or filename.endswith(".jpg") and filename != "Loss_Plot.png":
                image_path = os.path.join(self.name, "images", filename)
                image = Image.open(image_path)
                fig, ax = plt.subplots()
                ax.imshow(image)
                ax.axis('off')  # Hide axes
                fig.canvas.draw()

                # Convert canvas to image and append
                img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
                all_images.append(img)
                plt.close(fig)
        output_path = os.path.join(self.save_path,output_path)
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
        path = os.path.join(self.save_path,self.name,"models")
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
        

        
        # halfname == groups[type] == Type of the model (Critiker, Generator, Discriminator)
        # filename == the name of the model to load
        for half_name, filename in groups.items():
            #arg name listed to give the model (gen or disc etc.) to this function
            for arg, model in kwargs.items():
                #arg == repr(mode) 
                # model == Model Class like Discriminator Generator or something similar
                if half_name in repr(model) and half_name in filename:
                    #load the model
                    model.load_state_dict(torch.load(os.path.join(self.save_path,self.name,"models",filename), weights_only=True))
        #return epoch from filename
        try:
            match = re.search(r"\d+", filename)
            if match:
                epoch = match.group()
                self.start_epoch = int(epoch)
        except UnboundLocalError:
            pass            
        

            
            
            