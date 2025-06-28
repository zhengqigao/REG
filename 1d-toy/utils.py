import torch
from torch import nn
from torch.nn import functional as F
from embedding import PositionalEmbedding
from tqdm.auto import tqdm
import numpy as np

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, hidden_size: int, hidden_layers: int, emb_size: int, num_class,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal", ):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.class_emb = nn.Embedding(num_class, emb_size)

        concat_size = len(self.time_mlp.layer) + \
                      len(self.input_mlp.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 1))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t, label):
        x_emb = self.input_mlp(x[:, 0])
        t_emb = self.time_mlp(t)

        ind = label == -1

        class_embd = self.class_emb(torch.clamp(label, min=0))
        class_embd = torch.where(ind.unsqueeze(-1), torch.zeros_like(class_embd), class_embd)

        x = torch.cat((x_emb, t_emb + class_embd), dim=-1)
        x = self.joint_mlp(x)
        return x


class Diffusion():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear",
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                 ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = num_timesteps
        self.device = device
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2
        self.betas = self.betas.to(self.device)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                    1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample, no_variance = False):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0 and no_variance == False:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps

    def train(self, model, dataloader, lr, num_epochs, dropout_prob):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
        )
        model.train()
        for epoch in tqdm(range(num_epochs), desc="Training"):
            for step, data in enumerate(dataloader):
                batch, labels = data
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                noise = torch.randn(batch.shape, device=self.device)
                timesteps = torch.randint(
                    0, self.num_timesteps, (batch.shape[0],), device=self.device
                ).long()

                noisy = self.add_noise(batch, noise, timesteps)
                mask = torch.rand(labels.shape) < dropout_prob
                labels[mask] = -1

                noise_pred = model(noisy, timesteps, labels)
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        return model

    def sample(self, model, sample, label, cfg_scale = 0.0, guided_model = None):
        model.eval()
        if guided_model is None:
            guided_model = model
        else:
            guided_model.eval()

        timesteps = list(range(self.num_timesteps))[::-1]
        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            t = torch.from_numpy(np.repeat(t, sample.shape[0])).long().to(self.device)
            with torch.no_grad():
                    residual_uncond = guided_model(sample, t, (-1 * torch.ones(sample.shape[0])).to(torch.int).to(self.device))
                    residual_cond = model(sample, t, (label * torch.ones(sample.shape[0])).to(torch.int).to(self.device))
                    residual = residual_cond + cfg_scale * (residual_cond - residual_uncond)
                    sample = self.step(residual, t[0], sample)
        return sample


