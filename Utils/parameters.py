import argparse




def parse_arguments():
    parser = argparse.ArgumentParser(description="Training von DCGAN")

    # Allgemeine Argumente
    parser.add_argument('--num_layers', type=int, default=6, help='Num layers')
    parser.add_argument('--epochs', type=int, default=10, help='Anzahl der Epochen')
    parser.add_argument('--device', type=str, default='cuda', help='Gerät zum Training (cuda oder cpu)')
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
    parser.add_argument('--betas_wgan', type=float, nargs=2, default=(0, 0.9), help='Betas für den Adam-Optimizer bei WGAN')
    parser.add_argument('--dtype', type=str, default='image', help='Datentyp (Bild oder Audio)')

    return parser.parse_args()



