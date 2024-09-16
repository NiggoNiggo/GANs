from WGAN_GP.wgan_pg import WGAN
from DCGAN.dcgan import DCGAN
from SpecGAN.specGAN import SpecGAN
from WaveGAN.waveGAN import WaveGAN
from WaveGAN.conditional_wavegan import ConditionalWaveGAN

from Utils.parameters import parse_gan_type,parse_gan_args
from Utils.hyperparameter_tuning import RandomSearch


from torchsummary import summary


try:
    parser = parse_gan_type()
    current_gan, remaining_args = parse_gan_type()
    print(f"Run {current_gan}")
    print(remaining_args)
    args = parse_gan_args(current_gan,remaining_args)
    print(f"The following Arguments are loaded for the GAN:{args}")
except:
    current_gan = "conditional_wavegan"
    args = parse_gan_args(current_gan,[])
    print(f"No Gan type selected. Start Training {current_gan}")

    
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
                        name="conditional_1s",
                        params=args
                        )
        name_gen, name_disc = repr(wavegan.gen), repr(wavegan.disc)
        wavegan.load_models(name_gen=wavegan.gen,name_disc=wavegan.disc)
        wavegan.make_entire_training()
    
    elif current_gan == "conditional_wavegan":
        
        # wavegan = ConditionalWaveGAN(params=args,
        #                              device="cuda",
        #                              name="test_new_4s")
        
        data = {
            "lr_g":[1e-5,1e-4,3e-4],
            "lr_d":[1e-5,1e-4,5e-3],
            "batchsize":[32,64,128],
            "n_crit":[1,5,10],
            "lam":[5,10,15]
            }    

        tuner = RandomSearch(2,data,args,2,ConditionalWaveGAN)
        tuner.tune_params(name="randomSearchTest",
                          device="cuda")


        # name_gen, name_disc = repr(wavegan.gen), repr(wavegan.disc)
        # wavegan.load_models(name_gen=wavegan.gen,name_disc=wavegan.disc)
        # wavegan.make_entire_training()
        # wavegan.valid_afterwards(r"F:\new_models",mk_audio=True)
        
    #fehlt nur noch loss function plotten und fid plotten mit kwargs
    #early stop bedingung
    #delete unnessesary files