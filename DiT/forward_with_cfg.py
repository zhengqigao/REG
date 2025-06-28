import torch
import torch.nn as nn
import numpy as np
import math
import warnings
from typing import Optional
from torch.cuda.amp import autocast

def forward_with_cfg(self, x, t, y, cfg_param: Optional[dict]= None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]

        if not self.training:
            assert torch.all(t[:len(t) // 2] == t[len(t) // 2:]), "First half of t does not equal the second half in sampling"
        else:
            warnings.warn(f"REG only modifies code for sampling, haven't considered training.")

        if cfg_param['cfg_type'] == 'original':
            with autocast():
                combined = torch.cat([half, half], dim=0)
                model_out = self.forward(combined, t, y)
                eps, rest = model_out[:, :3], model_out[:, 3:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                # notice that in original DiT codebase, they use: half_eps = uncond_eps + a * (cond_eps - uncond_eps)
                # but we use: half_eps = cond_eps + b * (cond_eps - uncond_eps)
                # essentially they are identical by the relationship a - 1 = b.
                # We use the second one for implementing our approach.
                # So the value of cfg_scale in our case should differ from the original DiT by 1.0
                half_eps = cond_eps + cfg_param['cfg_scale'] * (cond_eps - uncond_eps)
        elif cfg_param['cfg_type'] == 'intervalt':
            with autocast():
                interval_min, interval_max = cfg_param['intervalt']
                in_interval = (interval_min <= t[0]) & (t[0] <= interval_max)
                if in_interval:
                    combined = torch.cat([half, half], dim=0)
                    model_out = self.forward(combined, t, y)
                    eps, rest = model_out[:, :3], model_out[:, 3:]
                    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                    half_eps = cond_eps + cfg_param['cfg_scale'] * (cond_eps - uncond_eps)
                else:
                    model_out = self.forward(half, t[:half.shape[0]], y[:half.shape[0]])
                    half_eps, rest = model_out[:, :3], model_out[:, 3:]
                    rest = torch.cat([rest, rest], dim=0)
        elif cfg_param['cfg_type'] == 'linear':
            with autocast():
                start_scale, end_scale = cfg_param['cfg_scale'], cfg_param['end_scale']
                k = (end_scale - start_scale) / (0-1000) # in sampling, t from 1000 to 0.
                combined = torch.cat([half, half], dim=0)
                model_out = self.forward(combined, t, y)
                eps, rest = model_out[:, :3], model_out[:, 3:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                shape = tuple([-1] + [1] * (cond_eps.ndimension() - 1))
                half_eps = cond_eps + (t[:len(t)//2] * k + start_scale).view(*shape) * (cond_eps - uncond_eps)
        elif cfg_param['cfg_type'] == 'cosine':
            with autocast():
                combined = torch.cat([half, half], dim=0)
                model_out = self.forward(combined, t, y)
                eps, rest = model_out[:, :3], model_out[:, 3:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                progress = 1 - t[:len(t)//2]/1000
                coeff = (1 - torch.cos(torch.pi * torch.pow(progress, cfg_param['s']))) * 0.5 * cfg_param['cfg_scale']
                shape = tuple([-1] + [1] * (cond_eps.ndimension() - 1))
                half_eps = uncond_eps + coeff.view(*shape) * (cond_eps - uncond_eps)
                # The above is the original used in MDT, and it is equivalent to:
                # half_eps = cond_eps + (coeff.view(*shape)-1) * (cond_eps - uncond_eps)
        elif cfg_param['cfg_type'] == 'reg_original':
            alpha_bar = cfg_param['alpha_bar']
            with autocast():
                with torch.enable_grad():
                    half.requires_grad_(True)
                    combined = torch.cat([half, half], dim=0)
                    model_out = self.forward(combined, t, y)
                    eps  = model_out[:, :self.in_channels]
                    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                    shape = tuple([-1] + [1] * (cond_eps.ndimension() - 1))
                    tmp = torch.autograd.grad(cond_eps.sum(), half, create_graph=False)[0].detach()
                    half.requires_grad_(False)
                cond_eps, uncond_eps = cond_eps.detach(), uncond_eps.detach()
                rest = model_out[:, 3:].detach()
                coeff = cfg_param['cfg_scale'] * (1 - torch.sqrt(1 - alpha_bar[:len(alpha_bar)//2]).view(*shape) * tmp)
                half_eps = (cond_eps + coeff * (cond_eps - uncond_eps))[:,:3]
        elif cfg_param['cfg_type'] == 'reg_intervalt':
            alpha_bar = cfg_param['alpha_bar']
            interval_min, interval_max = cfg_param['intervalt']
            assert torch.max(t) == torch.min(t), f"the max and min of t is identical"
            with autocast():
                in_interval = (interval_min <= t[0]) & (t[0] <= interval_max)
                if in_interval:
                    with torch.enable_grad():
                        half.requires_grad_(True)
                        combined = torch.cat([half, half], dim=0)
                        model_out = self.forward(combined, t, y)
                        eps  = model_out[:, :self.in_channels]
                        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                        shape = tuple([-1] + [1] * (cond_eps.ndimension() - 1))
                        tmp = torch.autograd.grad(cond_eps.sum(), half, create_graph=False)[0].detach()
                        half.requires_grad_(False)
                    cond_eps, uncond_eps = cond_eps.detach(), uncond_eps.detach()
                    rest = model_out[:, 3:].detach()
                    coeff = cfg_param['cfg_scale'] * (1 - torch.sqrt(1 - alpha_bar[:len(alpha_bar)//2]).view(*shape) * tmp)
                    half_eps = (cond_eps + coeff * (cond_eps - uncond_eps))[:,:3]
                else:
                    model_out = self.forward(half, t[:half.shape[0]], y[:half.shape[0]])
                    half_eps, rest = model_out[:, :3], model_out[:, 3:]
                    rest = torch.cat([rest, rest], dim=0)
        elif cfg_param['cfg_type'] == 'reg_cosine':
            alpha_bar = cfg_param['alpha_bar']
            with autocast():
                with torch.enable_grad():
                    half.requires_grad_(True)
                    combined = torch.cat([half, half], dim=0)
                    model_out = self.forward(combined, t, y)
                    eps  = model_out[:, :self.in_channels]
                    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                    shape = tuple([-1] + [1] * (cond_eps.ndimension() - 1))
                    tmp = torch.autograd.grad(cond_eps.sum(), half, create_graph=False)[0].detach()
                    half.requires_grad_(False)
                cond_eps, uncond_eps = cond_eps.detach(), uncond_eps.detach()
                rest = model_out[:, 3:].detach()
                progress = 1 - t[:len(t)//2]/1000 # in sampling, t from 1000 to 0.
                coeff = (1 - torch.cos(torch.pi * torch.pow(progress, cfg_param['s']))) * 0.5 * cfg_param['cfg_scale']
                coeff = coeff.view(*shape) * (1 - torch.sqrt(1 - alpha_bar[:len(alpha_bar)//2]).view(*shape) * tmp)
                half_eps = (cond_eps + coeff * (cond_eps - uncond_eps))[:,:3]
        elif cfg_param['cfg_type'] == 'reg_linear':
            alpha_bar = cfg_param['alpha_bar']
            start_scale, end_scale = cfg_param['cfg_scale'], cfg_param['end_scale']
            k = (end_scale - start_scale) / (0 - 1000)  # in sampling, t from 1000 to 0.
            with autocast():
                with torch.enable_grad():
                    half.requires_grad_(True)
                    combined = torch.cat([half, half], dim=0)
                    model_out = self.forward(combined, t, y)
                    eps  = model_out[:, :self.in_channels]
                    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                    shape = tuple([-1] + [1] * (cond_eps.ndimension() - 1))
                    tmp = torch.autograd.grad(cond_eps.sum(), half, create_graph=False)[0].detach()
                    half.requires_grad_(False)
                cond_eps, uncond_eps = cond_eps.detach(), uncond_eps.detach()
                rest = model_out[:, 3:].detach()
                coeff = (t[:len(t) // 2] * k + start_scale).view(*shape) * (1 - torch.sqrt(1 - alpha_bar[:len(alpha_bar)//2]).view(*shape) * tmp)
                half_eps = (cond_eps + coeff * (cond_eps - uncond_eps))[:,:3]
        else:
            raise NotImplementedError(f"Unsupported cfg_type={cfg_param['cfg_type']}")
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

