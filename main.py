# from training import TrainingBase
import os, sys, json, torch, argparse


from WGAN_GP.wgan_pg import WGAN
from DCGAN.dcgan import DCGAN
from CGAN.cgan import CGAN
from Base_Models.gan_base import GanBase
from Base_Models.audio_transformer import SpecGANTransformer
from Utils.parameters import load_parameters 
from SpecGAN.specGAN_parameters import load_parameters as load_spec_params
from Utils.utils import init_weights


from SpecGAN.spec_generator import SpecGenerator
from SpecGAN.specGAN import SpecGAN
from SpecGAN.spec_discriminator import SpecDiscriminator
from SpecGAN.spec_conditional_generator import ConditionalSpecGenerator
from SpecGAN.spec_conditional_discriminator import ConditionalSpecDiscriminator


from WaveGAN.wavegan_parameters import load_parameters as load_wave_params
from WaveGAN.wave_discriminator import WaveDiscriminator
from WaveGAN.wave_generator import WaveGenerator
from WaveGAN.wavegan_dataset import WaveDataset
from WaveGAN.waveGAN import WaveGAN

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
# subprocess.run([sys.executable,"WaveGAN\waveGAN_parameters.py"])
    
current_gan = "wavegan"
if __name__ == "__main__":
    if current_gan in ["dcgan","wgan","cgan"]:
        params = load_parameters("dcgan_parameters.json")
        gen = params["Generator"](*params["Generator_params"].values()).to(params["device"])
        gen.apply(init_weights)
        
        
        data_loader = torch.utils.data.DataLoader(
                                            dataset=params["Dataset"](params["data_path"],
                                                                        params["transforms"]),
                                            batch_size=params["batch_size"],
                                            shuffle=True)
        
        
        if current_gan == "dcgan":
            disc = params["Discriminator"](*params["Discriminator_params"].values()).to(params["device"])
            disc.apply(init_weights)
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
            crit.apply(init_weights)
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
            disc.apply(init_weights)
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
        gen = params["Generator"](*params["Generator_params"].values()).to(params["device"])
        disc = params["Discriminator"](*params["Discriminator_params"].values()).to(params["device"])

        #apply weights
        gen.apply(init_weights)
        disc.apply(init_weights)

            
        gen_optim = params["gen_optimizer"](params=gen.parameters(),
                                            lr=params["lr"],
                                            betas=params["betas"])
        #init optimizers Discriminator
        disc_optim = params["disc_optimizer"](params=disc.parameters(),
                                            lr=params["lr"],
                                            betas=params["betas"])
        data_loader = torch.utils.data.DataLoader(
                                            dataset=params["Dataset"](*params["Dataset_params"].values()),
                                            batch_size=params["batch_size"],
                                            shuffle=True) 
        # for idx in range(120):
        #     x = next(iter(data_loader))
        #     print(x.shape)
        #     import matplotlib.pyplot as plt 
        #     plt.figure(figsize=(10,4))
        #     plt.imshow(x[0].squeeze(0).numpy())
        #     plt.show()

        
    
        # print(x.shape)
        specgan = SpecGAN(gen=gen,
                    disc=disc,
                    optim_gen=gen_optim,
                    optim_disc=disc_optim,
                    dataloader=data_loader,
                    params=params,
                    device="cuda",
                    name="specgan_drums_II",
                    lam=params["lam"],
                    n_critic=params["n_crit"],
                    alpha=params["alpha"],
                    betas=params["betas"])
        # specgan.print_summary(gen=specgan.gen,disc=specgan.disc)
        name_gen, name_disc = repr(gen), repr(disc)
        specgan.load_models(name_gen=gen,name_disc=disc)
        # specgan.print_summary(gen=gen,disc=disc)
        
        specgan.make_entire_training()
        # specgan.make_audio(8)




    elif current_gan == "spec_cgan":
        params = load_spec_params("specgan_conditional_parameters.json")
        gen = ConditionalSpecGenerator(*params["Generator_params"].values(),num_labels=10)
        disc = ConditionalSpecDiscriminator(*params["Discriminator_params"].values(),num_labels=10)

        #apply weights
        gen.apply(init_weights)
        disc.apply(init_weights)

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
                                name="Rs6",
                                lam=params["lam"],
                                n_critic=params["n_crit"],
                                alpha=params["alpha"],
                                betas=params["betas"],
                                conditional=False,
                                num_classes=10)
        conditional_specgan.make_entire_training()
    


    elif current_gan == "wavegan":
        params = load_wave_params("wavegan_parameters.json")
        gen = WaveGenerator(*params["Generator_params"].values()).to(params["device"])
        disc = WaveDiscriminator(*params["Discriminator_params"].values()).to(params["device"])

        #apply weights
        gen.apply(init_weights)
        disc.apply(init_weights)

        gen_optim = params["gen_optimizer"](params=gen.parameters(),
                                            lr=params["lr"],
                                            betas=params["betas"])
        #init optimizers Discriminator
        disc_optim = params["disc_optimizer"](params=disc.parameters(),
                                            lr=params["lr"],
                                            betas=params["betas"])
    
        data_loader = torch.utils.data.DataLoader(
                                            dataset=params["Dataset"](*params["Dataset_params"].values()),
                                            batch_size=params["batch_size"],
                                            shuffle=True)
        wavegan = WaveGAN(gen=gen,
                        disc=disc,
                        optim_gen=gen_optim,
                        optim_disc=disc_optim,
                        dataloader=data_loader,
                        params=params,
                        device=params["device"],
                        name="wavegan_RS6",
                        lam=params["lam"],
                        n_critic=params["n_crit"],
                        alpha=params["alpha"],
                        betas=params["betas"],
                        conditional=False,
                        num_classes=0)
        # wavegan.make_gif("wave_gan_to_epoch_75.gif")
        name_gen, name_disc = repr(gen), repr(disc)
        wavegan.load_models(name_gen=gen,name_disc=disc)
        # wavegan.make_gif("wavegan.gif")
        wavegan.make_entire_training()



        # får rs6 muss ich wohl doch nochmal bisschen mehr preprocessing machen und die yeit optimal nutyen aus dem datensatyt, dass nicht verschwendet wird 
    # für rs6 conditional einbauen für beschleunigen und nicht bechleunigen
    #und auf 4s erweitern 
        
        # soundcard downloaden und mal qualität von mic beurteilen


# drum dataset aus samplelib zusammenstellen?

#mal bisschen die parameter bei den funktionsaufrufen überprüfen, dass dort etwas eingespart wird und nicht jedes mal der aufruf befehl so langfe ist 
# beispiel devices ist nämlich in params mit drin


#base klassen besser definiern um mehr code zu sparen 
#insgesamt mal bisschen aufräumen


# Audio preprocessing genauer für specgan anschauen und mal aufschreiben


#specgan und condtional specgan muss ich zum laufen bekommen


#damm muss ich auuch mal das GAN für 128x128 fertig trainieren und ein richtiges CGAN machen

    
    # 1 euro von krombacher auszahlen lassen
    
    #Drum GAN
        #-> dafür brauche ich nochmal paar Samples aus Samplelib.db dafür müsste ich nochmnal eine pre processing pipline schreiben
    #SRGAchat
    #SRGAchaML
    
    
    
    # dann wavegan wird auf Arbeit gemacht 
    #dann auch ein conditional wavegan implementieren 
    


    # hifigan und voice cloning mit tacotron machen. das das soll aber vorgeladen sein und auf eigene Aufnahmen trainiert werden
    

