import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim.adam import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from model import DIFFUSION
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((32, 32))])
dataset = torchvision.datasets.MNIST(root='./data', 
                                        train=True,
                                        download=True,
                                        transform=transform)
dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=600,
                                            shuffle=True,
                                            num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_size = 32
channels = 1
beta = 1e-3
step = 1000
epoch = 100
model = DIFFUSION(image_size, channels).to(device)
optim = Adam(model.parameters(), lr=1e-4)

def lr_lambda(epoch):
    if epoch < 4:
        return 10**(-2+0.5*epoch)
    elif epoch < 30:
        return 1.
    elif epoch < 50:
        return 1e-1
    else:
        return 1e-2
    
lr_scheduler = LambdaLR(optim, lr_lambda)
if __name__ == "__main__":
    writer = SummaryWriter()
    for j in range(epoch):
        for i, (x, _) in enumerate(tqdm(dataloader)):
            optim.zero_grad()
            x = x.to(device)
            x = x-0.5
            epsilon = torch.randn_like(x).to(device)
            t = torch.randint(low=1, high=step, size=(x.size(0),)).to(device)
            epsilon_hat = model(x, t, epsilon)
            loss = F.mse_loss(epsilon, epsilon_hat)
            loss.backward()
            optim.step()
            writer.add_scalar(f"loss/{j}", float(loss), i)
        lr_scheduler.step()
        epsilon = torch.randn(size=(16, 1, image_size, image_size)).to(device)
        x = model.sample(epsilon, diffusion_steps=1000, mu=1.)
        x = (x+0.5).clamp(0., 1.)
        grid = torchvision.utils.make_grid(x, 4)
        writer.add_image("ddpm", grid, j)
        x = model.sample(epsilon, diffusion_steps=100, mu=0.)
        x = (x+0.5).clamp(0., 1.)
        grid = torchvision.utils.make_grid(x, 4)
        writer.add_image("ddim", grid, j)