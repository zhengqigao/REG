from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusion3Pipeline
import torch
import argparse
from tqdm import tqdm
import os
from PIL import Image
from torch.distributed import init_process_group, barrier
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

# Model dictionary for supported models
model_dict = {'sd-v1-4': 'CompVis/stable-diffusion-v1-4',
              'sd-v1-5': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
              'sd-v2-1': "stabilityai/stable-diffusion-2-1",
              'sd-xl': "stabilityai/stable-diffusion-xl-base-1.0",
              'sd-v3': "stabilityai/stable-diffusion-3-medium-diffusers",
              'sd-v3l':"stabilityai/stable-diffusion-3.5-large"}

class PromptDataset(Dataset):
    def __init__(self, prompts, max_prompts= float('inf')):
        self.prompts = prompts
        self.max_prompts = min(max_prompts, len(prompts))

    def __len__(self):
        return self.max_prompts

    def __getitem__(self, idx):
        return idx, self.prompts[idx]

def construct_cfg(cfg_type: str, args) -> dict:
    cfg_param = {}
    if cfg_type == 'original' or cfg_type == 'gradoriginal':
        pass
    elif cfg_type == 'intervalt' or cfg_type == 'gradintervalt':
        assert len(args) == 2
        vals = [float(a) for a in args]
        cfg_param['intervalt'] = (min(vals), max(vals))
    elif cfg_type == 'linear' or cfg_type == 'gradlinear':
        assert len(args) == 1
        cfg_param['end_scale'] = float(args[0])
    elif cfg_type == 'cosine' or cfg_type == 'gradcosine':
        assert len(args) == 1
        cfg_param['s'] = float(args[0])
    else:
        raise NotImplementedError(f"{cfg_type} is not implemented in constructing cfg_param dict")
    return cfg_param


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-name', type=str, default='sd-v1-4', help='The name of the model')
    parser.add_argument('--batch-size', type=int, default=5, help='Input batch size for testing')
    parser.add_argument('--prompt-dir', type=str, default='./data/prompt_5000_random.txt', help='Path to the prompt file')
    parser.add_argument('--save-dir', type=str, default='./samples/', help='Directory to save generated images')
    parser.add_argument('--cfg-type', type=str, default='original', help='Config type for Stable Diffusion')
    parser.add_argument('--cfg-param', nargs='+', type=str, default=[], help='Config parameters for Stable Diffusion')
    parser.add_argument('--guidance', type=float, default=7.5, help='Guidance scale for image generation')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()

    # Initialize distributed processing
    torch.set_grad_enabled(False)
    dist.init_process_group("nccl")
    world_size = torch.cuda.device_count()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    seed = args.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)

    with open(args.prompt_dir, 'r') as f:
        prompts = [line.strip() for line in f.readlines()]


    save_dir = os.path.join(args.save_dir, f"{args.model_name}-scale-{args.guidance}-seed-{args.seed}-sample-{len(prompts)}-type-{args.cfg_type}-param-{'-'.join(args.cfg_param)}")
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    # Divide prompts among processes
    dataset = PromptDataset(prompts)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    assert len(dataset) % world_size == 0, f"the total number of samples should be divided by world_size"
    assert (len(dataset) // world_size) % args.batch_size == 0, f"the samples requires on each gpu should be divided by batch_size"


    barrier()

    # Load Stable Diffusion pipeline
    cfg_param = construct_cfg(args.cfg_type, args.cfg_param)
    if args.model_name in ['sd-v3', 'sd-v3l']:
        sd_pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_dict[args.model_name],
            torch_dtype=torch.float16
        ).to(rank) 
        sd_pipeline.scheduler.set_timesteps(num_inference_steps=50)
    elif args.model_name != "sd-xl":
        sd_pipeline = StableDiffusionPipeline.from_pretrained(
            model_dict[args.model_name],
            torch_dtype=torch.float16
        ).to(rank)
    else:
        sd_pipeline = StableDiffusionXLPipeline.from_pretrained(model_dict[args.model_name], torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(rank)
        sd_pipeline.cfg_type = args.cfg_type
        sd_pipeline.cfg_param = cfg_param

    # Generate images
    print(f"Rank {rank}: Loaded {len(prompts)} prompts from {args.prompt_dir}, cfg_type: {args.cfg_type}, cfg_param:{'-'.join(cfg_param)}.")
    print(f"save-dir: {save_dir}")

    with tqdm(total=len(dataset) // world_size, desc=f"Rank {rank} Generating Images") as pbar:
        sd_pipeline.set_progress_bar_config(disable=True)
        for batch_idx, (prompt_idx, batch_prompts) in enumerate(dataloader):
            if args.model_name in ['sd-v3', 'sd-v3l']:
                images = sd_pipeline(
                    list(batch_prompts),
                    num_images_per_prompt=1,
                    guidance_scale=args.guidance,
                    cfg_param = cfg_param,
                    cfg_type = args.cfg_type,
                ).images
            elif args.model_name != "sd-xl":
                images = sd_pipeline(
                    list(batch_prompts),
                    num_images_per_prompt=1,
                    guidance_scale=args.guidance,
                    cfg_param= cfg_param,
                    cfg_type = args.cfg_type,
                ).images
            else:
                images = sd_pipeline(
                    list(batch_prompts),
                    num_images_per_prompt=1,
                    guidance_scale=args.guidance,
                ).images

            # Save images
            for i, image in enumerate(images):
                global_idx = prompt_idx[i]
                save_path = os.path.join(save_dir, f"{global_idx}.png")
                image.save(save_path)

            pbar.update(len(batch_prompts))

    dist.barrier()
    dist.destroy_process_group()
