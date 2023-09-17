import torch
import torch.nn as nn
import torch.nn.functional as F
from u_net import Unet

class DIFFUSION(nn.Module):
    '''
        DENOISING DIFFUSION IMPLICIT MODELS, Jiaming Songら, 2020
    '''
    def __init__(self, image_size=32, channels=3, start_beta=1e-4, end_beta=0.02, step=1000):
        super(DIFFUSION, self).__init__()
        beta = torch.linspace(start_beta, end_beta, step)
        alpha = 1-beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.step = step

        self.epsilon_net = Unet(
            dim=image_size,
            dim_mults=(1, 2, 4, 8),
            channels=channels,
            with_time_emb=True,
            resnet_block_groups=2,
        )
    
    def forward(self, x, t, epsilon):
        alpha0 = torch.sqrt(self.alpha_bar[t]).reshape(-1,1,1,1)
        alpha1 = torch.sqrt(1-self.alpha_bar[t]).reshape(-1,1,1,1)
        return self.epsilon_net(x*alpha0+alpha1*epsilon, t)

    def sample(self, x_t):
        for t in range(self.step-1,-1,-1):
            step = torch.tensor([t], device=x_t.device, dtype=torch.float)
            with torch.no_grad():
                pred_noise = self.epsilon_net(x_t, step)
            alpha_bar = self.alpha_bar[t]
            alpha = self.alpha[t]
            beta = self.beta[t]
            x_t = 1/torch.sqrt(alpha) * (x_t - ((1-alpha)/torch.sqrt(1-alpha_bar)) * pred_noise)
            if t > 0:
                z = torch.sqrt(beta)*torch.randn_like(x_t)
            else:
                z = 0
            x_t += z
        return x_t