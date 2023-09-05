import torch
import torch.nn as nn
import torch.nn.functional as F
from u_net import Unet

class DIFFUSION(nn.Module):
    '''
        DENOISING DIFFUSION IMPLICIT MODELS, Jiaming Songら, 2020
    '''
    def __init__(self, image_size=28, channels=1, start_beta=1e-4, end_beta=0.02, step=1000):
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
        alpha0 = torch.sqrt(self.alpha_bar[t-1]).reshape(-1,1,1,1)
        alpha1 = torch.sqrt(1-self.alpha_bar[t-1]).reshape(-1,1,1,1)
        return self.epsilon_net(x*alpha0+alpha1*epsilon, t)

    def sample(self, x_t, diffusion_steps=1000, mu=0.):
        '''
            Eq.(12)
            x[t-1] = √a[t-1] * (x[t] - √1-a[t] * e(x[t],t)) / √a[t] + √1-a[t-1]-sigma[t]**2 * e(x[t],t) + sigma[t]*n
            n ~ N(0, I)
            sigma = √(1-a[t-1])/(1-a[t]) * √1-a[t]/a[t-1]
        '''
        assert self.step >= diffusion_steps
        interval = self.step//diffusion_steps
        steps = [i*interval for i in range(diffusion_steps)]
        steps.append(self.step-1)
        steps.append(1)
        steps = list(set(steps))
        steps.sort()
        for d_t, t in zip(reversed(steps[:-1]),reversed(steps[1:])):
            step = torch.tensor([t], device=x_t.device, dtype=torch.float)
            epsilon = self.epsilon_net(x_t, step)
            d_alpha = self.alpha_bar[d_t]
            alpha = self.alpha_bar[t]
            sigma = torch.sqrt((1-d_alpha)/(1-alpha))*torch.sqrt(1-alpha/d_alpha)
            pred_x0 = torch.sqrt(d_alpha)*(x_t-torch.sqrt(1.-alpha)*epsilon)/torch.sqrt(alpha)
            to_xt = torch.sqrt(1.-d_alpha-sigma**2)*epsilon
            noise = mu*sigma*torch.randn_like(x_t)
            x_t = pred_x0+to_xt+noise
            x_t = x_t.detach()
        return x_t