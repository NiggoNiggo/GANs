# from training import TrainingBase
import os, sys, json, torch


from WGAN_GP.wgan_pg import WGAN
from DCGAN.dcgan import DCGAN
from CGAN.cgan import CGAN
from Base_Models.gan_base import GanBase
from Base_Models.audio_transformer import SpecGANTransformer
from Utils.parameters import load_parameters 
from SpecGAN.specGAN_parameters import load_parameters as load_spec_params

from SpecGAN.spec_generator import SpecGenerator
from SpecGAN.specGAN import SpecGAN
from SpecGAN.spec_discriminator import SpecDiscriminator
from SpecGAN.spec_conditional_generator import ConditionalSpecGenerator
from SpecGAN.spec_conditional_discriminator import ConditionalSpecDiscriminator

# spec = SpecGenerator(64,1)
# latent = torch.randn((1,100))
# out = spec(latent)
# print(out.shape)

# disc = SpecDiscriminator(64,1)
# out = disc(out)
# print(out.shape)

import subprocess

subprocess.run([sys.executable,"Utils\parameters.py"])
subprocess.run([sys.executable,"SpecGAN\specGAN_parameters.py"])
    
current_gan = "wgan"
if __name__ == "__main__":
    if current_gan in ["dcgan","wgan","cgan"]:
        params = load_parameters("dcgan_parameters.json")
        gen = params["Generator"](*params["Generator_params"].values()).to(params["device"])
        
        
        data_loader = torch.utils.data.DataLoader(
                                            dataset=params["Dataset"](params["data_path"],
                                                                        params["transforms"]),
                                            batch_size=params["batch_size"],
                                            shuffle=True)
        
        
        if current_gan == "dcgan":
            disc = params["Discriminator"](*params["Discriminator_params"].values()).to(params["device"])
            gen_optim = params["gen_optimizer"](params=gen.parameters(),
                                                lr=params["lr_dcgan"],
                                                betas=params["betas_dcgan"])
            #init optimizers Discriminator
            disc_optim = params["disc_optimizer"](params=disc.parameters(),
                                                lr=params["lr_dcgan"],
                                                betas=params["betas_dcgan"])
            loss_fn = params["loss_fn"]
            dcgan = DCGAN(gen=gen,
                        disc=disc,
                        optim_gen=gen_optim,
                        optim_disc=disc_optim,
                        loss_fn=loss_fn,
                        dataloader=data_loader,
                        params=params,
                        device="cuda",
                        name="test")
            dcgan.print_summary(gen=dcgan.gen,disc=dcgan.disc)
            dcgan.make_entire_training()
    
        elif current_gan == "wgan":
            crit = params["Critiker"](*params["Discriminator_params"].values()).to(params["device"])
            # for layer in crit.model:
            #     print(layer)
            gen_optim = params["gen_optimizer"](params=gen.parameters(),
                                            lr=params["lr_wgan"],
                                            betas=params["betas_wgan"])
            #init optimizers Discriminator
            disc_optim = params["disc_optimizer"](params=crit.parameters(),
                                                lr=params["lr_wgan"],
                                                betas=params["betas_wgan"])

            wgan = WGAN(gen=gen,
                        disc=crit,
                        optim_gen=gen_optim,
                        optim_disc=disc_optim,
                        dataloader=data_loader,
                        params=params,
                        device="cuda",
                        name="wgan_128_jellyfish",
                        lam=params["lam"],
                        n_critic=params["n_crit"],
                        alpha=params["alpha"],
                        betas=params["betas_wgan"])
            wgan.print_summary(gen=wgan.gen,disc=wgan.disc)
            name_gen, name_disc = repr(gen), repr(crit)
            wgan.load_models(name_gen=gen,name_disc=crit)
          
            
            # wgan.make_gif("cat_wgan.gif")
            wgan.make_entire_training()
        
        if current_gan == "cgan":
            disc = params["Discriminator"](*params["Discriminator_params"].values()).to(params["device"])
            gen_optim = params["gen_optimizer"](params=gen.parameters(),
                                                lr=params["lr_dcgan"],
                                                betas=params["betas_dcgan"])
            #init optimizers Discriminator
            disc_optim = params["disc_optimizer"](params=disc.parameters(),
                                                lr=params["lr_dcgan"],
                                                betas=params["betas_dcgan"])
            loss_fn = params["loss_fn"]
            cgan = CGAN(gen=gen,
                        disc=disc,
                        optim_gen=gen_optim,
                        optim_disc=disc_optim,
                        loss_fn=loss_fn,
                        dataloader=data_loader,
                        params=params,
                        device="cuda",
                        name="yellys_cond")
            dcgan.print_summary(gen=cgan.gen,disc=cgan.disc)
            dcgan.make_entire_training()
        
    elif current_gan == "specgan":
        params = load_spec_params("specgan_parameters.json")
        print(params["device"])
        gen = params["Generator"](*params["Generator_params"].values()).to(params["device"])
        disc = params["Discriminator"](*params["Discriminator_params"].values()).to(params["device"])

            
        gen_optim = params["gen_optimizer"](params=gen.parameters(),
                                            lr=params["lr"],
                                            betas=params["betas"])
        #init optimizers Discriminator
        disc_optim = params["disc_optimizer"](params=disc.parameters(),
                                            lr=params["lr"],
                                            betas=params["betas"])
        print(*params["Dataset_params"].values())
        data_loader = torch.utils.data.DataLoader(
                                            dataset=params["Dataset"](*params["Dataset_params"].values()),
                                            batch_size=params["batch_size"],
                                            shuffle=True)
        x = next(iter(data_loader))
        
    
        # print(x.shape)
        specgan = SpecGAN(gen=gen,
                    disc=disc,
                    optim_gen=gen_optim,
                    optim_disc=disc_optim,
                    dataloader=data_loader,
                    params=params,
                    device="cuda",
                    name="specgan_minst",
                    lam=params["lam"],
                    n_critic=params["n_crit"],
                    alpha=params["alpha"],
                    betas=params["betas"])
        # specgan.print_summary(gen=specgan.gen,disc=specgan.disc)
        name_gen, name_disc = repr(gen), repr(disc)
        specgan.load_models(name_gen=gen,name_disc=disc)
        
        specgan.make_entire_training()
        # specgan.make_audio(8)



    elif current_gan == "spec_cgan":
        params = load_spec_params("specgan_conditional_parameters.json")
        print(*params["Generator_params"].values())
        gen = ConditionalSpecGenerator(*params["Generator_params"].values(),num_labels=10)
        disc = ConditionalSpecDiscriminator(*params["Discriminator_params"].values(),num_labels=10)
        print(gen,disc)

        gen_optim = params["gen_optimizer"](params=gen.parameters(),
                                            lr=params["lr"],
                                            betas=params["betas"])
        #init optimizers Discriminator
        disc_optim = params["disc_optimizer"](params=disc.parameters(),
                                            lr=params["lr"],
                                            betas=params["betas"])
    
        data_loader = torch.utils.data.DataLoader(
                                            dataset=params["Dataset"](*params["Dataset_conditional_params"].values()),
                                            batch_size=params["batch_size"],
                                            shuffle=True)
        conditional_specgan = SpecGAN(gen=gen,
                                disc=disc,
                                optim_gen=gen_optim,
                                optim_disc=disc_optim,
                                dataloader=data_loader,
                                params=params,
                                device="cuda",
                                name="mins_specgan_final",
                                lam=params["lam"],
                                n_critic=params["n_crit"],
                                alpha=params["alpha"],
                                betas=params["betas"],
                                conditional=False,
                                num_classes=10)
        conditional_specgan.make_entire_training()
        #muss mir die samples mit meinem preprocessing nnochmal anhören bevor ich die in das system schicke


# lr niedrige n_critic geriner und batchsize geringer
#specgan und condtional specgan muss ich zum laufen bekommen

    # FID validation für images
    





# lustige befehle in der tastaur mit anderen Lichteffekten machen 


    
    
    # 1 euro von krombacher auszahlen lassen
    
    #Drum GAN
    #FACE GAN
    #SRGAchat
    #SRGAchaML
    
    
    
    # dann wavegan
    
    

    #batchsize erhöhen
