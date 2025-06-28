import argparse
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from data import log_density_gmm
from utils import MLP, Diffusion

plt.rcParams["figure.dpi"] = 300
cmap = 'OrRd'
colors =['#4A90E2', '#76B041', '#FF4D4D']
# colors =['#FF4D4D', 'orange',  '#4A90E2', ] # yellow: , green: '#76B041', red: '#FF4D4D', blue: #4A90E2
linewidth = 3
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/diffusion_model_e200_numstep20_seed0_s0.0001_e0.02_dim128_layer3_cond.pth",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--every-t-step", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=1000)
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--include-deterministic", action="store_true", default=False)
    
    args, unknown = parser.parse_known_args()
    grid_size = args.grid_size
    torch.manual_seed(args.seed)  # For CPU
    torch.cuda.manual_seed(args.seed)  # For current GPU
    torch.cuda.manual_seed_all(args.seed)  # For all GPUs (if applicable)
    np.random.seed(args.seed)  # For NumPy

    name = args.model_path.split("/")[-1].split("_cond.pth")[0]

    with open(f"./models/{name}.pkl", "rb") as pickle_file:
        config = pickle.load(pickle_file)

    device = torch.device(
        f"cuda:{config.gpu}" if torch.cuda.is_available() and config.gpu >= 0 else "cpu"
    )

    cond_model = MLP(
        hidden_size=config["hidden_size"],
        hidden_layers=config["hidden_layers"],
        emb_size=config["emb_size"],
        num_class=config["num_class"],
        time_emb=config["time_emb"],
        input_emb=config["input_emb"],
    ).to(device)
    uncond_model = MLP(
        hidden_size=config["hidden_size"],
        hidden_layers=config["hidden_layers"],
        emb_size=config["emb_size"],
        num_class=config["num_class"],
        time_emb=config["time_emb"],
        input_emb=config["input_emb"],
    ).to(device)

    cond_model.load_state_dict(torch.load(f"./models/{name}_cond.pth"))
    uncond_model.load_state_dict(torch.load(f"./models/{name}_uncond.pth"))

    diffusion = Diffusion(
        num_timesteps=config["num_timesteps"],
        beta_schedule=config["beta_schedule"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
    )

    uncond_model.eval()
    cond_model.eval()
    xmin, xmax = -1, 2

    timesteps = list(range(len(diffusion)))[::-1]

    all_nabla_logct = []
    all_nabla_logourt = []
    all_nabla_logcgolden = []
    all_nabla_logcgolden2 = []
    all_t = []
    for i, t in enumerate(tqdm(timesteps, desc="Sampling")):  # num_step -1 ,..., 0
        if t % args.every_t_step == 0:

            grid_points = torch.linspace(xmin, xmax, grid_size).view(-1, 1)
            t = torch.from_numpy(np.repeat(t, grid_points.shape[0])).long()
            with torch.no_grad():

                grid_points1 = grid_points.clone().detach().requires_grad_(True)
                grid_points2_initial = grid_points.clone().detach().requires_grad_(True)

                # corresponds to -sqrt(1-alpha_bar) * nabla_logct
                wrk_uncond = uncond_model(
                    grid_points,
                    t,
                    (-1 * torch.ones(grid_points.shape[0])).to(torch.int).to(device),
                )
                wrk_cond = cond_model(
                    grid_points,
                    t,
                    (torch.zeros(grid_points.shape[0])).to(torch.int).to(device),
                )
                nabla_logct = args.guidance_scale * (wrk_cond - wrk_uncond)
                # step_by_logct = diffusion.step(wrk_cond + nabla_logct, t[0], grid_points)

                # corresponds to -sqrt(1-alpha_bar) * nabla_logct * correction

                with torch.enable_grad():
                    wrk_cond = cond_model(
                        grid_points1,
                        t,
                        (torch.zeros(grid_points.shape[0])).to(torch.int).to(device),
                    )
                    tmp = torch.autograd.grad(
                        wrk_cond.sum(), grid_points1, create_graph=False
                    )[0].detach()
                grid_points1.requires_grad_(False)
                nabla_logourt = (
                    args.guidance_scale
                    * (1 - diffusion.sqrt_one_minus_alphas_cumprod[t[0]] * tmp)
                    * (wrk_cond - wrk_uncond)
                )

                grad_multiplier = (
                    1 - diffusion.sqrt_one_minus_alphas_cumprod[t[0]] * tmp
                )

                # step_by_logourt = diffusion.step(wrk_cond + nabla_logourt, t[0], grid_points1)

                # golden nabla_logc with probabilistic sampler
                nabla_logcgolden = 0

                repeat = 100

                with torch.enable_grad():
                    logpx0 = torch.zeros(args.grid_size, repeat)
                    logpx0_y = torch.zeros(args.grid_size, repeat)
                    for j in range(repeat):
                        grid_points2 = grid_points2_initial
                        for tt in timesteps[i:]:
                            wrk_cond = cond_model(
                                grid_points2,
                                torch.from_numpy(np.repeat(tt, grid_points.shape[0]))
                                .long()
                                .to(grid_points2.device),
                                (torch.zeros(grid_points.shape[0]))
                                .to(torch.int)
                                .to(device),
                            )
                            grid_points2 = diffusion.step(wrk_cond, tt, grid_points2)
                        # if j == 0:
                        #     plt.figure()
                        #     plt.hist(grid_points2.detach().numpy(), bins=100)
                        #     plt.title(f"histogram at t={t[0]}")
                        #     plt.show()

                        logpx0_y[:, j] = log_density_gmm(
                            grid_points2, config["cond_mean"], config["cond_std"]
                        )
                        logpx0[:, j] = log_density_gmm(
                            grid_points2, config["uncond_mean"], config["uncond_std"]
                        )
                    # off by torch.log(torch.tensor(1.0/repeat)), but it has zero grad w.r.t. xt
                    wrk = args.guidance_scale * torch.logsumexp(
                        logpx0_y - logpx0, dim=1
                    )
                    nabla_logcgolden = (
                        -diffusion.sqrt_one_minus_alphas_cumprod[t[0]]
                        * torch.autograd.grad(
                            wrk.sum(), grid_points2_initial, retain_graph=False
                        )[0].detach()
                    )
                
                # golden with deterministic sampler
                repeat = 1
                with torch.enable_grad():
                    logpx0 = torch.zeros(args.grid_size, repeat)
                    logpx0_y = torch.zeros(args.grid_size, repeat)
                    for j in range(repeat):
                        grid_points2 = grid_points2_initial
                        for tt in timesteps[i:]:
                            wrk_cond = cond_model(
                                grid_points2,
                                torch.from_numpy(np.repeat(tt, grid_points.shape[0]))
                                .long()
                                .to(grid_points2.device),
                                (torch.zeros(grid_points.shape[0]))
                                .to(torch.int)
                                .to(device),
                            )
                            grid_points2 = diffusion.step(wrk_cond, tt, grid_points2, no_variance=True)
                        # if j == 0:
                        #     plt.figure()
                        #     plt.hist(grid_points2.detach().numpy(), bins=100)
                        #     plt.title(f"histogram at t={t[0]}")
                        #     plt.show()

                        logpx0_y[:, j] = log_density_gmm(
                            grid_points2, config["cond_mean"], config["cond_std"]
                        )
                        logpx0[:, j] = log_density_gmm(
                            grid_points2, config["uncond_mean"], config["uncond_std"]
                        )
                    # off by torch.log(torch.tensor(1.0/repeat)), but it has zero grad w.r.t. xt
                    wrk = args.guidance_scale * torch.logsumexp(
                        logpx0_y - logpx0, dim=1
                    )
                    nabla_logcgolden2 = (
                        -diffusion.sqrt_one_minus_alphas_cumprod[t[0]]
                        * torch.autograd.grad(
                            wrk.sum(), grid_points2_initial, retain_graph=False
                        )[0].detach()
                    )

                # wrk_cond = cond_model(grid_points2_initial, t, (torch.zeros(grid_points.shape[0])).to(torch.int).to(device))
                # step_by_logcgolden = diffusion.step(wrk_cond + nabla_logcgolden, t[0], grid_points2_initial)

            all_nabla_logcgolden.append(nabla_logcgolden.view(-1))
            all_nabla_logct.append(nabla_logct.view(-1))
            all_nabla_logourt.append(nabla_logourt.view(-1))
            all_nabla_logcgolden2.append(nabla_logcgolden2.view(-1))
            all_t.append(t[0])

            plt.figure(figsize=(8, 6))
            plt.plot(
                grid_points1,
                nabla_logcgolden,
                color=colors[0],
                linestyle="-",
                linewidth=linewidth,
                label=r"$\nabla \log E_t(x_t,y)$ (p)".format(t[0] + 1),
            )
            if args.include_deterministic:
                plt.plot(
                    grid_points1,
                    nabla_logcgolden2,
                    color=colors[0],
                    linestyle="-.",
                    linewidth=linewidth,
                    label=r"$\nabla \log E_t(x_t,y)$ (d)".format(t[0] + 1),
                )
            plt.plot(
                grid_points1,
                nabla_logct,
                color=colors[1],
                linestyle="-",
                linewidth=linewidth,
                label=r"$\nabla \log R_t(x_t,y)$".format(t[0] + 1),
            )
            plt.plot(
                grid_points1,
                nabla_logourt,
                color=colors[2],
                linestyle="-",
                linewidth=linewidth,
                label=r"$\nabla \log R^{{our}}_t(x_t,y)$".format(
                    t[0] + 1
                ),
            )

            plt.xlabel("Grid Points", fontsize=22)
            plt.ylabel("Gradient Value", fontsize=22)
            plt.gca().spines["top"].set_linewidth(linewidth)
            plt.gca().spines["right"].set_linewidth(linewidth)
            plt.gca().spines["bottom"].set_linewidth(linewidth)
            plt.gca().spines["left"].set_linewidth(linewidth)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.title(f"(t = {t[0]+1})", fontsize=16)
            # plt.legend(fontsize=12)
            plt.tight_layout()

            plt.savefig(f"./figures/{name}_compare_{t[0]+1}.png")


    all_nabla_logcgolden = torch.stack(all_nabla_logcgolden)
    all_nabla_logct = torch.stack(all_nabla_logct)
    all_nabla_logourt = torch.stack(all_nabla_logourt)
    all_t = torch.stack(all_t)

    # Expand grid_points1 and all_t to compatible shapes
    grid_points1_broadcast = (
        grid_points1.squeeze(1).unsqueeze(0).repeat(all_t.shape[0], 1)
    )  # Shape: (10, 200)
    all_t_broadcast = all_t.unsqueeze(1).repeat(
        1, grid_points1.shape[0]
    )  # Shape: (10, 200)

    # Flatten tensors for scatter plotting
    x_coords = grid_points1_broadcast.flatten()  # Shape: (10 * 200,)
    y_coords = all_t_broadcast.flatten() + 1  # Shape: (10 * 200,)
    c_cgolden = all_nabla_logcgolden.flatten()  # Shape: (10 * 200,)
    c_court = all_nabla_logourt.flatten()  # Shape: (10 * 200,)
    c_ct = all_nabla_logct.flatten()  # Shape: (10 * 200,)

    v_min = torch.min(torch.cat((c_cgolden, c_court, c_ct))).item()
    v_max = torch.max(torch.cat((c_cgolden, c_court, c_ct))).item()
    # Create figure and subplots
    fig, axes = plt.subplots(3, 1, figsize=(18, 4))

    # Scatter plot for all_nabla_logcgolden
    scatter_0 = axes[0].scatter(
        x_coords,
        y_coords,
        c=c_cgolden,
        cmap=cmap,
        s=60,
        alpha=0.7,
        vmin=v_min,
        vmax=v_max,
        marker="s",
    )
    axes[0].set_title(r"$\nabla \log R(x_t, y)$")
    axes[0].set_xlabel("Grid Points")
    axes[0].set_ylabel("Time Step")
    fig.colorbar(scatter_0, ax=axes[0], orientation="vertical")

    # Scatter plot for all_nabla_logourt
    scatter_1 = axes[1].scatter(
        x_coords,
        y_coords,
        c=c_court,
        cmap=cmap,
        s=60,
        alpha=0.7,
        vmin=v_min,
        vmax=v_max,
        marker="s",
    )
    axes[1].set_title(r"$\nabla \log C^{our}_t(x_t, y)$")
    axes[1].set_xlabel("Grid Points")
    axes[1].set_ylabel("Time Step")
    fig.colorbar(scatter_1, ax=axes[1], orientation="vertical")

    # Scatter plot for all_nabla_logct
    scatter_2 = axes[2].scatter(
        x_coords,
        y_coords,
        c=c_ct,
        cmap=cmap,
        s=60,
        alpha=0.7,
        vmin=v_min,
        vmax=v_max,
        marker="s",
    )
    axes[2].set_title(r"$\nabla \log C_t(x_t, y)$")
    axes[2].set_xlabel("Grid Points")
    axes[2].set_ylabel("Time Step")
    fig.colorbar(scatter_2, ax=axes[2], orientation="vertical")

    plt.tight_layout()

    # save figures
    plt.savefig(f"./figures/{name}_scatter.png")

    # Create figure and subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    max_color = max(
        (c_cgolden - c_court).abs().max().item(), (c_cgolden - c_ct).abs().max().item()
    )

    # Scatter plot for |nabla_logcgolden - nabla_logourt|
    scatter_0 = axes[0].scatter(
        x_coords,
        y_coords,
        c=(c_cgolden - c_court).abs(),
        cmap=cmap,
        s=160,
        alpha=0.7,
        vmin=0,
        vmax=max_color,
        marker="s",
    )
    axes[0].set_title(r"$|\nabla \log R(x_t, y) - \log C^{our}_t(x_t, y)|$", fontsize=20)
    axes[0].set_xlabel("Grid Points", fontsize=20)
    axes[0].set_ylabel("Time Step", fontsize=20)
    axes[0].set_yticks(range(0, 20, 4))
    axes[0].tick_params(axis='y', labelsize=20)
    axes[0].tick_params(axis='x', labelsize=20)
    
    # Scatter plot for |nabla_logcgolden - nabla_logct|
    scatter_1 = axes[1].scatter(
        x_coords,
        y_coords,
        c=(c_cgolden - c_ct).abs(),
        cmap=cmap,
        s=160,
        alpha=0.7,
        vmin=0,
        vmax=max_color,
        marker="s",
    )
    axes[1].set_title(r"$|\nabla \log R(x_t, y) - \log C_t(x_t, y)|$", fontsize=20)
    axes[1].set_xlabel("Grid Points", fontsize=20)
    axes[1].set_ylabel("Time Step", fontsize=20)
    axes[1].set_yticks(range(0, 20, 4))
    axes[1].tick_params(axis='y', labelsize=20)
    axes[1].tick_params(axis='x', labelsize=20)

    # Add colorbars for each subplot
    cbar_0 = fig.colorbar(scatter_0, ax=axes[0], orientation="vertical")
    cbar_0.ax.tick_params(labelsize=10)
    cbar_1 = fig.colorbar(scatter_1, ax=axes[1], orientation="vertical")
    cbar_1.ax.tick_params(labelsize=10)

    # Adjust layout
    plt.tight_layout()

    # save figures
    plt.savefig(f"./figures/{name}_scatter_diff.png")

    # Create a 2D heatmap
    plt.figure(figsize=(18,6))
    heatmap, xedges, yedges = np.histogram2d(x_coords.numpy(), y_coords.numpy(), bins=(grid_size, len(torch.unique(y_coords))), weights=(c_cgolden - c_ct).abs().numpy())
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=max_color)
    plt.colorbar(label='Gradient Value')
    plt.xlabel('Grid Points', fontsize=20)
    plt.ylabel('Time Step', fontsize=20)
    plt.gca().spines["top"].set_linewidth(linewidth)
    plt.gca().spines["right"].set_linewidth(linewidth)
    plt.gca().spines["bottom"].set_linewidth(linewidth)
    plt.gca().spines["left"].set_linewidth(linewidth)
    plt.title(r"$|\nabla \log R(x_t, y) - \log C_t(x_t, y)|$")
    plt.savefig(f"./figures/{name}_heatmap_ct.png")

    plt.figure(figsize=(18,6))
    heatmap, xedges, yedges = np.histogram2d(x_coords.numpy(), y_coords.numpy(), bins=(grid_size, len(torch.unique(y_coords))), weights=(c_cgolden - c_court).abs().numpy())
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=max_color)
    plt.colorbar(label='Gradient Value')
    plt.xlabel('Grid Points', fontsize=20)
    plt.ylabel('Time Step', fontsize=20)
    plt.title(r"$|\nabla \log R(x_t, y) - \log C_our(x_t, y)|$")
    plt.gca().spines["top"].set_linewidth(linewidth)
    plt.gca().spines["right"].set_linewidth(linewidth)
    plt.gca().spines["bottom"].set_linewidth(linewidth)
    plt.gca().spines["left"].set_linewidth(linewidth)
            
    plt.savefig(f"./figures/{name}_heatmap_cour.png")

    if args.show:
        plt.show()
