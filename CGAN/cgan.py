from WGAN_GP.wgan_pg import WGAN

class CGAN(WGAN):
    def __init__(self,
                 gen,
                 disc,
                 optim_gen,
                 optim_disc,
                 dataloader,
                 params,
                 device,
                 name,
                 lam,
                 n_critic,
                 alpha,
                 betas):
        super().__init__(gen=gen,
                         disc=disc,
                         optim_disc=optim_disc,
                         optim_gen=optim_gen,
                         dataloader=dataloader,
                         params=params,
                         device=device,
                         name=name,
                         lam=lam,
                         n_critic=n_critic,
                         alpha=alpha,
                         betas=betas)
    