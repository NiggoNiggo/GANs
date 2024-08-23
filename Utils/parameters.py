import argparse


#das hier kann man bestimmt auch noch besser und kleverer löscen

def parse_normal_gans_arguments():
    parser = argparse.ArgumentParser(description="Training von DCGAN, CGAN and GAN")

    # Allgemeine Argumente
    parser.add_argument('--num_layers', type=int, default=6, help='Num layers')
    parser.add_argument('--epochs', type=int, default=10, help='Anzahl der Epochen')
    parser.add_argument('--betas_dcgan', type=float, nargs=2, default=(0.5, 0.999), help='Betas für den Adam-Optimizer bei DCGAN')
    parser.add_argument('--data_path', type=str,default=r"F:\DataSets\Images\Zebra", help='Pfad zu den Trainingsdaten')
    parser.add_argument('--batchsize', type=int, default=128, help='Batch-Größe für das Training')
    parser.add_argument('--latent_space', type=int, default=100, help='Dimension des latenten Raums')
    parser.add_argument('--lr_dcgan', type=float, default=2e-5, help='Lernrate für DCGAN')
    parser.add_argument('--lr_wgan', type=float, default=1e-4, help='Lernrate für WGAN')
    parser.add_argument('--img_size', type=int, default=128, help='Bildgröße')
    parser.add_argument('--lam', type=float, default=10, help='Lambda für WGAN')
    parser.add_argument('--n_crit', type=int, default=5, help='Anzahl der Kritiker bei WGAN')
    parser.add_argument('--alpha', type=float, default=0.0001, help='Alpha für WGAN')
    parser.add_argument('--dtype', type=str, default='image', help='Datentyp (Bild oder Audio)')

    return parser.parse_args()



def parse_specgan_arguments():
    parser = argparse.ArgumentParser(description="Training von SpecGAN")

    # Allgemeine Argumente
    parser.add_argument('--num_layers', type=int, default=5, help='Num layers')
    parser.add_argument('--d', type=int, default=64, help='Model complexity')
    parser.add_argument('--c', type=int, default=1, help='Num channels')
    parser.add_argument('--epochs', type=int, default=10, help='Anzahl der Epochen')
    parser.add_argument('--data_path', type=str,default=r"C:\Users\analf\Desktop\Datasets_And_Results\Datasets\RS6", help='Pfad zu den Trainingsdaten')
    parser.add_argument('--batchsize', type=int, default=128, help='Batch-Größe für das Training')
    parser.add_argument('--latent_space', type=int, default=100, help='Dimension des latenten Raums')
    parser.add_argument('--lr', type=float, default=2e-5, help='Lernrate für SpecGAN')
    parser.add_argument('--img_size', type=int, default=128, help='Bildgröße für das Spektrum')
    parser.add_argument('--lam', type=float, default=10, help='Lambda für WGAN')
    parser.add_argument('--n_crit', type=int, default=5, help='Anzahl der Kritiker bei WGAN')
    parser.add_argument('--alpha', type=float, default=0.0001, help='Alpha für WGAN')
    parser.add_argument('--betas', type=float, nargs=2, default=(0, 0.9), help='Betas für den Adam-Optimizer bei WGAN')
    parser.add_argument('--dtype', type=str, default='audio', help='Datentyp (Bild oder Audio)')

    return parser.parse_args()


def parse_wavegan_arguments():
    parser = argparse.ArgumentParser(description="Training von SpecGAN")

    # Allgemeine Argumente
    parser.add_argument('--num_layers', type=int, default=5, help='Num layers')
    parser.add_argument('--d', type=int, default=64, help='Model complexity')
    parser.add_argument('--c', type=int, default=1, help='Num channels')
    parser.add_argument('--epochs', type=int, default=10, help='Anzahl der Epochen')
    parser.add_argument('--data_path', type=str,default=r"C:\Users\analf\Desktop\Datasets_And_Results\Datasets\RS6", help='Pfad zu den Trainingsdaten')
    parser.add_argument('--batchsize', type=int, default=128, help='Batch-Größe für das Training')
    parser.add_argument('--latent_space', type=int, default=100, help='Dimension des latenten Raums')
    parser.add_argument('--lr', type=float, default=2e-5, help='Lernrate für SpecGAN')
    parser.add_argument('--audio_size', type=int, default=16384, help='Length of the audio signal')
    parser.add_argument('--lam', type=float, default=10, help='Lambda für WGAN')
    parser.add_argument('--n_crit', type=int, default=5, help='Anzahl der Kritiker bei WGAN')
    parser.add_argument('--alpha', type=float, default=0.0001, help='Alpha für WGAN')
    parser.add_argument('--betas', type=float, nargs=2, default=(0, 0.9), help='Betas für den Adam-Optimizer bei WGAN')
    parser.add_argument('--dtype', type=str, default='audio', help='Datentyp (Bild oder Audio)')

    return parser.parse_args()
