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
                 ):
        super().__init__(gen,disc,optim_gen,optim_disc,loss_fn,dataloader,params,device,name)
        self.lam = lam
        self.n_critic = n_critic
        self.alpha = alpha
        self.betas = betas
        #init the loss values for discriminator
        self.loss_values["loss_d"] = []
        #init the loss values for Generator
        self.loss_values["loss_g"] = []
    
    def gradient_penalty(self,
                         real_sample,
                         fake_sample):
        alpha = torch.rand(real_sample.size(0),1,1,1,device=self.device)
        interpolates = (alpha * real_sample ) + ((1 - alpha)*fake_sample).requires_grad_(True)
        # interpolates.requires_grad_(True)
        d_interpolates = self.disc(interpolates)
        # fake = torch.ones(real_sample.size(0), 1, device=self.device, requires_grad=False)
        # print(interpolates.shape,d_interpolates.shape,fake.shape)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def train_one_epoch(self):
        #printe output von data loader
        for idx, data in tqdm(enumerate(self.data_loader)):
            batch_size = data.size(0)
            # save current batch idx as a class variable
            self.current_batch_idx = idx
            real = data.to(self.device)
            for num_critic in range(self.n_critic):
                noise = torch.randn(real.size(0),self.params["latent_space"],1,1,device=self.device)
                fake = self.gen(noise)
                self.optim_disc.zero_grad()
                
                real_disc = self.disc(real)
                fake_disc = self.disc(fake.detach())
                
                gp = self.gradient_penalty(real,fake)
                
                loss_d = -torch.mean(real_disc) + torch.mean(fake_disc) + self.lam * gp
                loss_d.requires_grad_(True)
                
                loss_d.backward()
                self.optim_disc.step()
            
            self.optim_gen.zero_grad()
            noise = torch.randn(batch_size, self.params["latent_space"], 1, 1, device=self.device)
            fake_img = self.gen(noise)
            
            fake_disc = self.disc(fake_img)
            fake_loss = -torch.mean(fake_disc)            
            # fake_loss.requires_grad_(True)
            fake_loss.backward()
            self.optim_gen.step()
            if idx % 100 == 0:
                self.print_stats(epoch=self.epoch,batch_idx=idx,loss_d=loss_d, loss_g=fake_loss)
                self.predict(self.epoch)
            #append loss to loss dictionary
            self.loss_values["loss_d"].append(loss_d)
            self.loss_values["loss_g"].append(fake_loss)
        self.save_models(self.gen,self.disc)
        
                
    