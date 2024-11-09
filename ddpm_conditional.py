import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        coefficient_epsilon = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + coefficient_epsilon * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x



def train(start_epoch = 1,run_name = "DDPM_conditional",epochs = 400,batch_size = 10,image_size = 64,num_classes = 10,dataset_path = "img",device = "cuda",lr = 3e-4):

    setup_logging(run_name)

    start_epoch = start_epoch

    if torch.cuda.is_available() :
        device = "cuda"
    else :
        device = "cpu"
    #device = 'cpu'
    print(device)

    dataloader = get_data(image_size,dataset_path,batch_size)
    model = UNet_conditional(size = image_size,num_classes=num_classes,device=device).to(device)
    checkpoint_model = torch.load("models/DDPM_conditional/ckpt.pt", map_location=device)
    model.load_state_dict(checkpoint_model)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    optimizer.load_state_dict(
        torch.load("models/DDPM_conditional/optim.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    checkpoint_ema = torch.load("models/DDPM_conditional/ema_ckpt.pt", map_location=device)
    ema_model.load_state_dict(checkpoint_ema)

    for epoch in range(start_epoch,epochs):
        logging.info(f"Starting epoch {epoch}:")
        print(epoch)
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))




if __name__ == '__main__':
    train(start_epoch=1)

