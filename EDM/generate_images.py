# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the given model."""

import os
import re
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from typing import Optional

def construct_cfg(cfg_type: str, cfg_scale: Optional[float] = 0.0, *args) -> dict:
    cfg_param = {'cfg_type': cfg_type}
    if cfg_type == 'original' or cfg_type == 'reg_original':
        pass
    elif cfg_type == 'intervalt' or cfg_type == 'reg_intervalt':
        assert len(args) == 2
        vals = [float(a) for a in args]
        cfg_param['intervalt'] = (min(vals), max(vals))
    elif cfg_type == 'linear' or cfg_type == 'reg_linear':
        assert len(args) == 1
        cfg_param['end_scale'] = float(args[0])
    elif cfg_type == 'cosine' or cfg_type == 'reg_cosine':
        assert len(args) == 1
        cfg_param['s'] = float(args[0])
    else:
        raise NotImplementedError(f"{cfg_type} is not implemented in constructing cfg_param dict")
    return cfg_param

warnings.filterwarnings('ignore', '`resume_download` is deprecated')

#----------------------------------------------------------------------------
# Configuration presets.

model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions'

config_presets = {
    'edm2-img512-xs-fid':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.135.pkl'),      # fid = 3.53
    'edm2-img512-xs-dino':             dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.200.pkl'),      # fd_dinov2 = 103.39
    'edm2-img512-s-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.130.pkl'),       # fid = 2.56
    'edm2-img512-s-dino':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.190.pkl'),       # fd_dinov2 = 68.64
    'edm2-img512-m-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.100.pkl'),       # fid = 2.25
    'edm2-img512-m-dino':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.155.pkl'),       # fd_dinov2 = 58.44
    'edm2-img512-l-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.085.pkl'),       # fid = 2.06
    'edm2-img512-l-dino':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.155.pkl'),       # fd_dinov2 = 52.25
    'edm2-img512-xl-fid':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.085.pkl'),      # fid = 1.96
    'edm2-img512-xl-dino':             dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.155.pkl'),      # fd_dinov2 = 45.96
    'edm2-img512-xxl-fid':             dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.070.pkl'),     # fid = 1.91
    'edm2-img512-xxl-dino':            dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.150.pkl'),     # fd_dinov2 = 42.84
    'edm2-img64-s-fid':                dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.075.pkl'),        # fid = 1.58
    'edm2-img64-m-fid':                dnnlib.EasyDict(net=f'{model_root}/edm2-img64-m-2147483-0.060.pkl'),        # fid = 1.43
    'edm2-img64-l-fid':                dnnlib.EasyDict(net=f'{model_root}/edm2-img64-l-1073741-0.040.pkl'),        # fid = 1.33
    'edm2-img64-xl-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img64-xl-0671088-0.040.pkl'),       # fid = 1.33
    'edm2-img512-xs-guid-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.045.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.045.pkl', guidance=1.40, guid_network_labels = 'False'), # fid = 2.91
    'edm2-img512-xs-guid-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.150.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.150.pkl', guidance=1.70, guid_network_labels = 'False'), # fd_dinov2 = 79.94
    'edm2-img512-s-guid-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.025.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.025.pkl', guidance=1.40, guid_network_labels = 'False'), # fid = 2.23
    'edm2-img512-s-guid-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.085.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.085.pkl', guidance=1.90, guid_network_labels = 'False'), # fd_dinov2 = 52.32
    'edm2-img512-m-guid-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.030.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.20, guid_network_labels = 'False'), # fid = 2.01
    'edm2-img512-m-guid-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.015.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=2.00, guid_network_labels = 'False'), # fd_dinov2 = 41.98
    'edm2-img512-l-guid-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.015.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.20, guid_network_labels = 'False'), # fid = 1.88
    'edm2-img512-l-guid-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.035.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.035.pkl', guidance=1.70, guid_network_labels = 'False'), # fd_dinov2 = 38.20
    'edm2-img512-xl-guid-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.020.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.020.pkl', guidance=1.20, guid_network_labels = 'False'), # fid = 1.85
    'edm2-img512-xl-guid-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.030.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.70, guid_network_labels = 'False'), # fd_dinov2 = 35.67
    'edm2-img512-xxl-guid-fid':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',      gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.20, guid_network_labels = 'False'), # fid = 1.81
    'edm2-img512-xxl-guid-dino':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',      gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.70, guid_network_labels = 'False'), # fd_dinov2 = 33.09
    'edm2-img512-s-autog-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.070.pkl',        gnet=f'{model_root}/edm2-img512-xs-0134217-0.125.pkl',        guidance=2.10, guid_network_labels = 'True'), # fid = 1.34
    'edm2-img512-s-autog-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.120.pkl',        gnet=f'{model_root}/edm2-img512-xs-0134217-0.165.pkl',        guidance=2.45, guid_network_labels = 'True'), # fd_dinov2 = 36.67
    'edm2-img512-xxl-autog-fid':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.075.pkl',      gnet=f'{model_root}/edm2-img512-m-0268435-0.155.pkl',         guidance=2.05, guid_network_labels = 'True'), # fid = 1.25
    'edm2-img512-xxl-autog-dino':      dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.130.pkl',      gnet=f'{model_root}/edm2-img512-m-0268435-0.205.pkl',         guidance=2.30, guid_network_labels = 'True'), # fd_dinov2 = 24.18
    'edm2-img512-s-uncond-autog-fid':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-uncond-2147483-0.070.pkl', gnet=f'{model_root}/edm2-img512-xs-uncond-0134217-0.110.pkl', guidance=2.85, guid_network_labels = 'True'), # fid = 3.86
    'edm2-img512-s-uncond-autog-dino': dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-uncond-2147483-0.090.pkl', gnet=f'{model_root}/edm2-img512-xs-uncond-0134217-0.125.pkl', guidance=2.90, guid_network_labels = 'True'), # fd_dinov2 = 90.39
    'edm2-img64-s-autog-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.045.pkl',         gnet=f'{model_root}/edm2-img64-xs-0134217-0.110.pkl',         guidance=1.70, guid_network_labels = 'True'), # fid = 1.01
    'edm2-img64-s-autog-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.105.pkl',         gnet=f'{model_root}/edm2-img64-xs-0134217-0.175.pkl',         guidance=2.20, guid_network_labels = 'True'), # fd_dinov2 = 31.85
}


#----------------------------------------------------------------------------
# EDM sampler from the paper
# "Elucidating the Design Space of Diffusion-Based Generative Models",
# extended to support classifier-free guidance.

def edm_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
    cfg = None,
):
    assert isinstance(cfg, dict), 'cfg must be a dict'

    def denoise(x, t, cfg):
        guid_network_labels = labels if cfg['guid_network_labels'] else None
        if cfg['cfg_type'] == 'original':
            Dx = net(x, t, labels).to(dtype)
            if guidance == 1:
                return Dx
            ref_Dx = gnet(x, t, guid_network_labels).to(dtype)
            res = ref_Dx.lerp(Dx, guidance) # original in edm, but essentially is the same as: ref_Dx + guidance * (Dx - ref_Dx)
            peak_mem = torch.cuda.max_memory_allocated()
            print(f"Peak memory during computation: {peak_mem/1024**3:.4f} GB GB")
            return res
        elif cfg['cfg_type'] == 'linear':
            Dx = net(x, t, labels).to(dtype)
            k = (guidance - cfg['end_scale']) / (sigma_max - sigma_min)
            cur_guidance = k * (t - sigma_max) + guidance
            ref_Dx = gnet(x, t, guid_network_labels).to(dtype)
            return Dx + cur_guidance * (Dx - ref_Dx)
        elif cfg['cfg_type'] == 'intervalt':
            Dx = net(x, t, labels).to(dtype)
            if cfg['intervalt'][0] <= t <= cfg['intervalt'][1]:
                ref_Dx = gnet(x, t, guid_network_labels).to(dtype)
                return ref_Dx.lerp(Dx, guidance)
            else:
                return Dx
        elif cfg['cfg_type'] == 'cosine':
            Dx = net(x, t, labels).to(dtype)
            progress = 1.0 - t / sigma_max
            # print(f"t={t}, progress= {progress}")
            coeff = (1 - torch.cos(torch.pi * torch.pow(progress, cfg['s']))) * 0.5 * guidance
            ref_Dx = gnet(x, t, guid_network_labels).to(dtype)
            res = Dx + coeff * (Dx - ref_Dx)
            return res
        elif cfg['cfg_type'] == 'reg_original':
            
            with torch.enable_grad():
                x = x.detach()
                x.requires_grad_(True)
                Dx = net(x, t, labels)

                wrk = torch.autograd.grad(Dx.sum(), x, create_graph=False, retain_graph=False)[0].detach()

                peak_mem = torch.cuda.max_memory_allocated()

                x.requires_grad_(False)
                
            Dx = Dx.detach()
            ref_Dx = gnet(x, t, guid_network_labels).to(dtype)
            res = Dx + guidance * wrk * (Dx - ref_Dx)
            del Dx, wrk, ref_Dx
            torch.cuda.empty_cache()
            return res
        elif cfg['cfg_type'] == 'reg_intervalt':
            if cfg['intervalt'][0] <= t <= cfg['intervalt'][1]:
                with torch.enable_grad():
                    x = x.detach()
                    x.requires_grad_(True)
                    Dx = net(x, t, labels)
                    wrk = torch.autograd.grad(Dx.sum(), x, create_graph=False, retain_graph=False)[0].detach()
                    x.requires_grad_(False)
                Dx = Dx.detach()
                ref_Dx = gnet(x, t, guid_network_labels).to(dtype)
                res = Dx + guidance * wrk * (Dx - ref_Dx)
                del Dx, wrk, ref_Dx
                torch.cuda.empty_cache()
                return res
            else:
                res = net(x, t, labels)
                return res
        elif cfg['cfg_type'] == 'reg_linear':
            with torch.enable_grad():
                x = x.detach()
                x.requires_grad_(True)
                Dx = net(x, t, labels)
                wrk = torch.autograd.grad(Dx.sum(), x, create_graph=False, retain_graph=False)[0].detach()
                x.requires_grad_(False)
            Dx = Dx.detach()
            k = (guidance - cfg['end_scale']) / (sigma_max - sigma_min)
            cur_guidance = k * (t - sigma_max) + guidance
            ref_Dx = gnet(x, t, guid_network_labels).to(dtype)
            res = Dx + cur_guidance * wrk * (Dx - ref_Dx)
            del Dx, wrk, ref_Dx
            torch.cuda.empty_cache()
            return res
        elif cfg['cfg_type'] == 'reg_cosine':
            with torch.enable_grad():
                x = x.detach()
                x.requires_grad_(True)
                Dx = net(x, t, labels)
                wrk = torch.autograd.grad(Dx.sum(), x, create_graph=False, retain_graph=False)[0].detach()
                x.requires_grad_(False)
            Dx = Dx.detach()
            progress = 1.0 - t / sigma_max
            coeff = (1 - torch.cos(torch.pi * torch.pow(progress, cfg['s']))) * 0.5 * guidance
            ref_Dx = gnet(x, t, guid_network_labels).to(dtype)
            res = Dx + coeff * wrk * (Dx - ref_Dx)
            return res
        else:
            raise NotImplementedError(f"{cfg['cfg_type']} is not implemented.")

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        d_cur = (x_hat - denoise(x_hat, t_hat, cfg)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next, cfg)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Generate images for the given seeds in a distributed fashion.
# Returns an iterable that yields
# dnnlib.EasyDict(images, labels, noise, batch_idx, num_batches, indices, seeds)

def generate_images(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    gnet                = None,                 # Reference network for guidance. None = same as main network.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    class_idx           = None,                 # Class label. None = select randomly.
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    sampler_fn          = edm_sampler,          # Which sampler function to use.
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load main network.
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading network from {net} ...')
        with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        net = data['ema'].to(device)
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guidance network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=(verbose and dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
    if gnet is None:
        gnet = net

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]

    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                r = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                r.seeds = [seeds[idx] for idx in indices]
                if len(r.seeds) > 0:

                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, r.seeds)
                    r.noise = rnd.randn([len(r.seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
                    r.labels = None
                    if net.label_dim > 0:
                        r.labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[len(r.seeds)], device=device)]
                        if class_idx is not None:
                            r.labels[:, :] = 0
                            r.labels[:, class_idx] = 1

                    # Generate images.
                    latents = dnnlib.util.call_func_by_name(func_name=sampler_fn, net=net, noise=r.noise,
                        labels=r.labels, gnet=gnet, randn_like=rnd.randn_like, **sampler_kwargs)
                    r.images = encoder.decode(latents)

                    # Save images.
                    if outdir is not None:
                        for seed, image in zip(r.seeds, r.images.permute(0, 2, 3, 1).cpu().numpy()):
                            image_dir = os.path.join(outdir, f'{seed//1000*1000:06d}') if subdirs else outdir
                            os.makedirs(image_dir, exist_ok=True)
                            PIL.Image.fromarray(image, 'RGB').save(os.path.join(image_dir, f'{seed:06d}.png'))

                # Yield results.
                torch.distributed.barrier() # keep the ranks in sync
                yield r

    return ImageIterable()

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--preset',                   help='Configuration preset', metavar='STR',                             type=str, default=None)
@click.option('--net',                      help='Network pickle filename', metavar='PATH|URL',                     type=str, default=None)
@click.option('--gnet',                     help='Reference network for guidance', metavar='PATH|URL',              type=str, default=None)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, default='./figures')
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True, default=False)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)

@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--sigma_min',                help='Lowest noise level', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',                help='Highest noise level', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--guidance',                 help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT',   type=float, default=None)
@click.option('--S_churn', 'S_churn',       help='Stochasticity strength', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',           help='Stoch. min noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',           help='Stoch. max noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',       help='Stoch. noise inflation', metavar='FLOAT',                         type=float, default=1, show_default=True)
@click.option('--cfg_type', help='Configuration type',metavar='STR', type=str, default='original',show_default=True)
@click.option('--cfg_param', help='Configuration parameters (list of values)', metavar='STR',multiple=True, default=(), show_default=True
)
@click.option('--guid_network_labels', 'guid_network_labels',
              type=click.Choice(['True', 'False'], case_sensitive=False),
              help='Enable (True) or disable (False) guidance network labels',
              default=None, show_default=True)

def cmdline(preset, **opts):
    """Generate random images using the given model.

    Examples:

    \b
    # Generate a couple of images and save them as out/*.png
    python generate_images.py --preset=edm2-img512-s-guid-dino --outdir=out

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img64-s-fid --outdir=out --subdirs --seeds=0-49999
    """
    opts = dnnlib.EasyDict(opts)

    # Apply preset.
    if preset is not None:
        if preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{preset}"')
        for key, value in config_presets[preset].items():
            if opts[key] is None:
                opts[key] = value

    # Validate options.
    if opts.net is None:
        raise click.ClickException('Please specify either --preset or --net')
    if opts.guidance is None:
        opts.gnet = None
    elif opts.gnet is None:
        raise click.ClickException('Please specify --gnet when using guidance')

    opts['cfg'] = construct_cfg(opts.cfg_type, opts.guidance, *opts.cfg_param)
    opts['cfg']['guid_network_labels'] = True if opts['guid_network_labels'].lower() == 'true' else False
    del opts['cfg_type']
    del opts['cfg_param']
    del opts['guid_network_labels']
    # Generate.
    dist.init()
    image_iter = generate_images(**opts)
    for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()


#----------------------------------------------------------------------------