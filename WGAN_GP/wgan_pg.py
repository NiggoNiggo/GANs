import torch 
from tqdm.auto import tqdm

from WGAN_GP.wgan_gp_critic import Critiker
from DCGAN.dcgan_generator import Generator
from Base_Models.gan_base import GanBase


class WGAN(GanBase):
    def __init__(self,
                 gen,
                 disc,
                 optim_gen,
                 optim_disc,
                 dataloader,
                 params:dict,
                 device:str,
                 name:str,
                 lam:int=10,
                 n_critic:int=5,
                 alpha=0.0001,
                 betas:tuple=(0,0.9),
                 loss_fn=None,
                 conditional=False,
                 num_classes:int=0
                 ):
        super().__init__(gen,disc,optim_gen,optim_disc,loss_fn,dataloader,params,device,name,conditional=conditional,num_classes=num_classes)
        self.lam = lam
        self.n_critic = n_critic
        self.alpha = alpha
        self.betas = betas
        #init the loss values for discriminator
        self.loss_values["loss_d"] = []
        #init the loss values for Generator
        self.loss_values["loss_g"] = []
    
    def gradient_penalty(self,
                         real_sample:torch.tensor,
                         fake_sample:torch.tensor)->float:
        """gradient_penalty calcualte the gradient penalty term for the wasserstein distance

        Parameters
        ----------
        real_sample : torch.tensor
            real images
        fake_sample : torch.tensor
            fake images

        Returns
        -------
        gradient_penalty : float
            gradient penalty to add on the wasserstein distance 
        """
        if self.params["dtype"] == "image":
            alpha = torch.rand(real_sample.shape[0],1,1,1,device=self.device)
        elif self.params["dtype"] == "audio":
            alpha = torch.rand(real_sample.shape[0],1,1,device=self.device)
        #interpolates the images with th egiven formular of the paper
        interpolates = (alpha * real_sample ) + ((1 - alpha)*fake_sample).requires_grad_(True)
        # interpolates.requires_grad_(True)
        d_interpolates = self.disc(interpolates)
        #calculate the gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        #resize the gradients
        gradients = gradients.view(gradients.size(0), -1)
        #calculate the gradients
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def train_one_epoch(self):
        """This Methode implements the main application of wgan. This methode can be inheriated from any GAN that is 
        trained with the wasserstein GAN. To fully use this you have to adapt the following methods:
        self._train_generator()
        self._train_discriminator()
        self._process_real()
        These funcrion makes this basisclass adaptable to any GAN that is trained with WGAN
        """
        #printe output von data loader
        for idx, data in tqdm(enumerate(self.data_loader)):
            batch_size = data.size(0)
            #process the real data in a proper formation
            real = self._process_real_data(data)
            loss_d = 0
            #train crit til it is perfect trained
            for num_critic in range(self.n_critic):
                #make a noise to pass through the generator
                noise = self.make_noise(batch_size)
                #gen a fake image
                fake = self.gen(noise)
                #get the discriminator loss from the train discriminator methode
                loss_d += self._train_discriminator(real, fake).item()
            #mean of the loss 
            loss_d = loss_d/self.params["n_crit"]
            #get the generator loss by the 
            fake_loss = self._train_generator(batch_size)
            #print some stats 
            if idx % 100 == 0:
                self.print_stats(epoch=self.epoch,batch_idx=idx,loss_d=loss_d, loss_g=fake_loss)
                self.predict(self.epoch)
            #append loss to loss dictionary
            self.loss_values["loss_d"].append(loss_d)
            self.loss_values["loss_g"].append(fake_loss)

      

        if self.epoch % 15 == 0:
            self.save_models(self.gen,self.disc)
    
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
        if isinstance(data, list):
            data, label = data[0], data[1]
            return data.to(self.params["device"]), label.to(self.params["device"])
        return data.to(self.params["device"])
    
    def _train_discriminator(self, real:torch.tensor, fake:torch.tensor)->float:
        """_train_discriminator train the discriminator in the wgan process pipeline

        Parameters
        ----------
        real : torch.tensor
            real data 
        fake : torch.tensor
            fake data

        Returns
        -------
        float
            Discriminator loss 
        """
        #zeros gradients
        self.optim_disc.zero_grad()
        #discriminate real and fake images
        real_disc = self.disc(real)
        fake_disc = self.disc(fake.detach())
        #calculate the gradient penalty
        gp = self.gradient_penalty(real, fake)
        #calculate loss d
        loss_d = -torch.mean(real_disc) + torch.mean(fake_disc) + self.lam * gp
        #backpropagation
        loss_d.backward()
        #optimzer step
        self.optim_disc.step()
        return loss_d

    def _train_generator(self,batch_size:int)->float:
        """_train_generator in the wgan processing pipeline

        Parameters
        ----------
        batch_size : int
            batch size of the dataloader 

        Returns
        -------
        float
            Generator loss 
        """
        #zero gradients
        self.optim_gen.zero_grad()
        #generate a noise for fake images
        noise = self.make_noise(batch_size)
        #gen a fake image
        fake_img = self.gen(noise)
        #discriminate fake images
        fake_disc = self.disc(fake_img)
        fake_loss = -torch.mean(fake_disc)
        #backpropagation
        fake_loss.backward()
        #optimzer step
        self.optim_gen.step()
        return fake_loss
        