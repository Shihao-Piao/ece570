from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
import torch
from torchvision.utils import save_image
from ddpm_conditional import Diffusion
from utils import get_data

dataloader = get_data(batch_size = 1,image_size = 64,dataset_path = "img")

diff = Diffusion(device="cpu")

image, label = next(iter(dataloader))
print(label)
t = torch.Tensor([50, 100, 150, 200, 250, 300, 600, 999]).long()

noised_image, _ = diff.noise_images(image, t)
save_image(noised_image.add(1).mul(0.5), "noise.jpg")

#airplane:0, auto:1, bird:2, cat:3, deer:4, dog:5, frog:6, horse:7, ship:8, truck:9

#add process image
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        process_images = []
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

                if i % 200 == 0:
                    tmp = x
                    tmp = (tmp.clamp(-1, 1) + 1) / 2
                    tmp = (tmp * 255).type(torch.uint8)
                    process_images.append(tmp)
                if i == 1:
                    tmp = x
                    tmp = (tmp.clamp(-1, 1) + 1) / 2
                    tmp = (tmp * 255).type(torch.uint8)
                    process_images.append(tmp)

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x,process_images

def test(image_size = 64,num_classes = 10,device = "cpu",lr = 3e-4):
    model = UNet_conditional(size=image_size, num_classes=num_classes, device=device).to(device)
    checkpoint_model = torch.load(
        "models/DDPM_conditional/ckpt.pt",
        map_location=device)
    model.load_state_dict(checkpoint_model)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    optimizer.load_state_dict(
        torch.load("models/DDPM_conditional/optim.pt",
                   map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

    diffusion = Diffusion(img_size=image_size, device=device)

    labels = torch.tensor([9]).long().to(device)
    sampled_images, process_images= diffusion.sample(model, n=len(labels), labels=labels)
    process_plot(process_images)
    process_img = torch.cat(process_images,dim=0)
    save_images(process_img, os.path.join("results", "process.jpg"))

test()