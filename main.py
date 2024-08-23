
def parse_main_arguments():
    parser = argparse.ArgumentParser(description="Select GAN type and handle training arguments.")
    parser.add_argument('--gan_type', type=str, choices=["cgan", "wgan", "dcgan", "specgan", "wavegan"], required=True, help="Specify the GAN type to use.")
    return parser.parse_args()
try:
    parser = parse_main_arguments()
    current_gan = parser.gan_type
    print(f"Run {current_gan}")
except:
    print("Wavegan is standard modell so now run WaveGAN")
    current_gan = "wavegan"
# from training import TrainingBase
import argparse

from WGAN_GP.wgan_pg import WGAN
from DCGAN.dcgan import DCGAN


from SpecGAN.specGAN import SpecGAN


from WaveGAN.waveGAN import WaveGAN









    
# current_gan = "wavegan"
if __name__ == "__main__":
    
    if current_gan in ["cgan","wgan","dcgan"]:
        #load the necessary arguments to cgan, dcgan and wgan
        if current_gan == "dcgan":
            #init optimizers Discriminator
            dcgan = DCGAN(
                        device="cuda",
                        name="test_dcgan")
            dcgan.make_entire_training()

        elif current_gan == "wgan":

            wgan = WGAN(
                        device="cuda",
                        name="wgan_128_jellyfish")
            
            wgan.print_summary(gen=wgan.gen,disc=wgan.disc)
            name_gen, name_disc = repr(wgan.gen), repr(wgan.disc)
            wgan.load_models(name_gen=wgan.gen,name_disc=wgan.disc)
        
            wgan.make_entire_training()
    
        #hier muss später dann noch für cgan das gemacht werden
    
    elif current_gan == "specgan":
        specgan = SpecGAN(device="cuda",
                        name="specgan_drums_II")
        # specgan.print_summary(gen=specgan.gen,disc=specgan.disc)
        name_gen, name_disc = repr(specgan.gen), repr(specgan.disc)
        specgan.load_models(name_gen=specgan.gen,name_disc=specgan.disc)
        # specgan.print_summary(gen=gen,disc=disc)
        
        specgan.make_entire_training()




    


    elif current_gan == "wavegan":
        
        wavegan = WaveGAN(
                        device="cuda",
                        name="wavegan_RS6_4s",
                        )
        # wavegan.make_gif("wave_gan_to_epoch_75.gif")
        name_gen, name_disc = repr(wavegan.gen), repr(wavegan.disc)
        wavegan.load_models(name_gen=wavegan.gen,name_disc=wavegan.disc)
        # wavegan.make_gif("rs6.gif")
        wavegan.make_entire_training()

    # ziel auch nochmal jedes testen mit jeder funktino also laden, speichern validieren usw

    # får rs6 muss ich wohl doch nochmal bisschen mehr preprocessing machen und die yeit optimal nutyen aus dem datensatyt, dass nicht verschwendet wird 
    # für rs6 conditional einbauen für beschleunigen und nicht bechleunigen
    #und auf 4s erweitern 
        



    #argparser benutzen anstatt dict dann mit json datei arbeiten, wobei daas eigentlich far nicht so viel sinn mache

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
        