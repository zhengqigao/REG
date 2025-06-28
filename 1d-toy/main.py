import argparse
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from data import generate_gaussian_mixture
from utils import MLP, Diffusion
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=2000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=20)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal",
                        choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal",
                        choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument('--num-sample', type=int, default=8000)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--plot-save', type=str, default='save')
    parser.add_argument('--seed', type=int, default=0)
    config = parser.parse_args()

    device = torch.device(f"cuda:{config.device}" if (torch.cuda.is_available() and config.gpu>=0) else "cpu")


    name = f'diffusion_model_e{config.num_epochs}_numstep{config.num_timesteps}_seed{config.seed}_s{config.beta_start}_e{config.beta_end}_dim{config.hidden_size}_layer{config.hidden_layers}'

    cond_mean, cond_std = [0.5, 1.5], [0.25, 0.25]
    uncond_mean, uncond_std = [-1, 1], [0.5, 0.5]

    cond_data = generate_gaussian_mixture(config.num_sample, len(cond_mean), cond_mean, cond_std, device=device)
    uncond_data = generate_gaussian_mixture(config.num_sample, len(uncond_mean), uncond_mean, uncond_std, device=device)

    cond_dataloader = DataLoader(TensorDataset(cond_data.view(-1,1), torch.zeros_like(cond_data, dtype = torch.int)),
                                 batch_size=config.train_batch_size, shuffle=True, drop_last=True)
    uncond_dataloader = DataLoader(TensorDataset(uncond_data.view(-1,1), -1 * torch.ones_like(uncond_data, dtype = torch.int)),
                                   batch_size=config.train_batch_size, shuffle=True, drop_last=True)

    diffusion = Diffusion(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device = device)

    cond_model = MLP(
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        num_class=1,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding,
    ).to(device)

    uncond_model = MLP(
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        num_class=1,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding,
    ).to(device)

    cond_model = diffusion.train(cond_model, cond_dataloader, config.learning_rate, config.num_epochs, dropout_prob = 0.0)
    uncond_model = diffusion.train(uncond_model, uncond_dataloader, config.learning_rate, config.num_epochs, dropout_prob=1.0)

    samples = torch.randn(config.eval_batch_size, 1).to(device)
    cond_sample= diffusion.sample(cond_model, samples, label = 0, cfg_scale = 0.0, guided_model=uncond_model)
    uncond_sample = diffusion.sample(uncond_model, samples, label = -1, cfg_scale=0.0, guided_model=uncond_model)

    plt.figure()
    plt.hist(cond_sample.view(-1).cpu().numpy(), bins=100, density=True)
    plt.title('conditional distribution')
    plt.savefig('./figures/cond_hist.png')

    plt.figure()
    plt.hist(uncond_sample.view(-1).cpu().numpy(), bins=100, density=True)
    plt.title('unconditional distribution')
    plt.savefig('./figures/uncond_hist.png')
    if config.plot_save == 'plot':
        plt.show()

    torch.save(cond_model.state_dict(), f'./models/{name}_cond.pth')
    torch.save(uncond_model.state_dict(), f'./models/{name}_uncond.pth')
    dict = {
        'hidden_size': config.hidden_size,
        'hidden_layers': config.hidden_layers,
        'emb_size': config.embedding_size,
        'num_class': 1,
        'time_emb': config.time_embedding,
        'input_emb': config.input_embedding,
        'num_timesteps': config.num_timesteps,
        'beta_schedule': config.beta_schedule,
        'beta_start': config.beta_start,
        'beta_end': config.beta_end,
        'cond_mean': cond_mean,
        'cond_std': cond_std,
        'uncond_mean': uncond_mean,
        'uncond_std': uncond_std,
        'num_sample': config.num_sample,
    }

    with open(f'./models/{name}.pkl', 'wb') as pickle_file:
        pickle.dump(dict, pickle_file)
