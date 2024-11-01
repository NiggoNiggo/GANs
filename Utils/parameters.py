import argparse

def parse_gan_type():
    parser = argparse.ArgumentParser(description="Select GAN type and handle training arguments.")
    parser.add_argument('--gan_type', type=str, choices=["cgan", "wgan", "dcgan", "specgan", "wavegan"], required=True, help="Specify the GAN type to use.")
    
    args, remaining_args = parser.parse_known_args()
    
    return args.gan_type, remaining_args


def parse_gan_args(selcted_gan, remaining_args):
    parser = argparse.ArgumentParser(description="Training von GANS")

    if selcted_gan in ["wgan","cgan","dcgan"]:
        parser.add_argument('--num_layers', type=int, default=6, help='Num layers')
        parser.add_argument('--epochs', type=int, default=5, help='Anzahl der Epochen')
        parser.add_argument('--betas_dcgan', type=float, nargs=2, default=(0.5, 0.999), help='Betas für den Adam-Optimizer bei DCGAN')
        parser.add_argument('--data_path', type=str,default=r"F:\DataSets\Images\Zebra", help='Pfad zu den Trainingsdaten')
        parser.add_argument('--batchsize', type=int, default=64, help='Batch-Größe für das Training')
        parser.add_argument('--latent_space', type=int, default=100, help='Dimension des latenten Raums')
        parser.add_argument('--lr_dcgan', type=float, default=2e-5, help='Lernrate für DCGAN')
        parser.add_argument('--lr_wgan', type=float, default=1e-4, help='Lernrate für WGAN')
        parser.add_argument('--img_size', type=int, default=128, help='Bildgröße')
        parser.add_argument('--lam', type=float, default=10, help='Lambda für WGAN')
        parser.add_argument('--n_crit', type=int, default=5, help='Anzahl der Kritiker bei WGAN')
        parser.add_argument('--alpha', type=float, default=0.0001, help='Alpha für WGAN')
        parser.add_argument('--dtype', type=str, default='image', help='Datentyp (Bild oder Audio)')
        
    elif selcted_gan == "wavegan" or selcted_gan == "conditional_wavegan":
        parser.add_argument('--d', type=int, default=64, help='Model complexity')
        parser.add_argument('--c', type=int, default=1, help='Num channels')
        parser.add_argument('--epochs', type=int, default=100, help='Anzahl der Epochen')
        parser.add_argument('--data_path', type=str,default=r"H:\Datasets\RS6", help='Pfad zu den Trainingsdaten')
        parser.add_argument('--batchsize', type=int, default=64, help='Batch-Größe für das Training')
        parser.add_argument('--latent_space', type=int, default=100, help='Dimension des latenten Raums')
        parser.add_argument('--lr_g', type=float, default=1e-4, help='Lernrate Generator in Wavegan')
        parser.add_argument('--lr_d', type=float, default=1e-4, help='Lernrate für Criticer in Wavegan')
        parser.add_argument('--audio_size', type=int, default=16384,choices=[16384,65536], help='Length of the audio signal')
        parser.add_argument('--lam', type=float, default=10, help='Lambda für WGAN')
        parser.add_argument('--n_crit', type=int, default=5, help='Anzahl der Kritiker bei WGAN')
        parser.add_argument('--alpha', type=float, default=0.0001, help='Alpha für WGAN')
        parser.add_argument('--betas', type=float, nargs=2, default=(0.5, 0.9), help='Betas für den Adam-Optimizer bei WGAN')
        parser.add_argument('--dtype', type=str, default='audio', help='Datentyp (Bild oder Audio)')
        if selcted_gan == "conditional_wavegan":
            parser.add_argument("--num_classes",type=int,default=2, help="Amount of classes if conditional")
    elif selcted_gan == "specgan":
        parser.add_argument('--num_layers', type=int, default=5, help='Num layers')
        parser.add_argument('--d', type=int, default=64, help='Model complexity')
        parser.add_argument('--c', type=int, default=1, help='Num channels')
        parser.add_argument('--epochs', type=int, default=10, help='Anzahl der Epochen')
        parser.add_argument('--data_path', type=str,default=r"C:\Users\analf\Desktop\Datasets_And_Results\Datasets\Chords", help='Pfad zu den Trainingsdaten')
        parser.add_argument('--batchsize', type=int, default=128, help='Batch-Größe für das Training')
        parser.add_argument('--latent_space', type=int, default=100, help='Dimension des latenten Raums')
        parser.add_argument('--lr', type=float, default=2e-5, help='Lernrate für SpecGAN')
        parser.add_argument('--img_size', type=int, default=128, help='Bildgröße für das Spektrum')
        parser.add_argument('--lam', type=float, default=10, help='Lambda für WGAN')
        parser.add_argument('--n_crit', type=int, default=1, help='Anzahl der Kritiker bei WGAN')
        parser.add_argument('--alpha', type=float, default=0.0001, help='Alpha für WGAN')
        parser.add_argument('--betas', type=float, nargs=2, default=(0.5, 0.9), help='Betas für den Adam-Optimizer bei WGAN')
        parser.add_argument('--dtype', type=str, default='audio', help='Datentyp (Bild oder Audio)')
    parser.add_argument('--save_path', type=str, default=r'H:\Results\GANS', help='path where to save the results')
    args = parser.parse_args(remaining_args)
    return args


