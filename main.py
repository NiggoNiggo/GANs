
from WGAN_GP.wgan_pg import WGAN
from DCGAN.dcgan import DCGAN


from SpecGAN.specGAN import SpecGAN


from WaveGAN.waveGAN import WaveGAN

from Utils.parameters import parse_gan_type,parse_gan_args


try:
    parser = parse_gan_type()
    current_gan, remaining_args = parse_gan_type()
    print(f"Run {current_gan}")
    print(remaining_args)
    args = parse_gan_args(current_gan,remaining_args)
    print(f"The following Arguments are loaded for the GAN:{args}")
except:
    current_gan = "wavegan"
    args = parse_gan_args(current_gan,[])
    print(f"No Gan type selected. Start Training {current_gan}")

    
# current_gan = "wavegan"
if __name__ == "__main__":
    
    if current_gan in ["cgan","wgan","dcgan"]:
        #load the necessary arguments to cgan, dcgan and wgan
        if current_gan == "dcgan":
            #init optimizers Discriminator
            dcgan = DCGAN(
                        device="cuda",
                        params=args,
                        name="test_dcgan")
            dcgan.make_entire_training()

        elif current_gan == "wgan":

            wgan = WGAN(
                        device="cuda",
                        params=args,
                        name="wgan_128_jellyfish")
            
            wgan.print_summary(gen=wgan.gen,disc=wgan.disc)
            name_gen, name_disc = repr(wgan.gen), repr(wgan.disc)
            wgan.load_models(name_gen=wgan.gen,name_disc=wgan.disc)
        
            wgan.make_entire_training()
    
    
    elif current_gan == "specgan":
        specgan = SpecGAN(device="cuda",
                        name="specgan_drums_II",
                        params=args)
        name_gen, name_disc = repr(specgan.gen), repr(specgan.disc)
        specgan.load_models(name_gen=specgan.gen,name_disc=specgan.disc)
        
        specgan.make_entire_training()




    


    elif current_gan == "wavegan":
        
        wavegan = WaveGAN(
                        device="cuda",
                        name="wavegan_",
                        params=args
                        )
        # wavegan.make_gif("wave_gan_to_epoch_75.gif")
        name_gen, name_disc = repr(wavegan.gen), repr(wavegan.disc)
        wavegan.load_models(name_gen=wavegan.gen,name_disc=wavegan.disc)
        # wavegan.make_gif("rs6.gif")
        wavegan.make_entire_training()

 