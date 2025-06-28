import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from tqdm import tqdm
import numpy as np
from math import ceil
from torchmetrics.image.fid import FrechetInceptionDistance
import re
from matplotlib import pyplot as plt
from collections import defaultdict
import heapq

def capture(file_str):
    pattern = r"scale-([\d\.eE+-]+)-seed-(\d+)-(.*)"

    # Search for matches
    match = re.search(pattern, file_str)

    if match:
        scale = match.group(1)
        seed = match.group(2)
        str_after_seed = match.group(3)
    else:
        raise RuntimeError(f"Could not match {file_str}")
    return float(scale), int(seed), str_after_seed

def get_fromdict(data_dict, identifier, scale, seed):
    if identifier in data_dict.keys():
        item = data_dict[identifier]
        for i, cur in enumerate(item):
            if cur[0] == scale and cur[1] == seed:
                return i, int(cur[2])
    return None, 0

def load_images(image_folder, transform=None) -> torch.tensor:
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if
                   f.endswith(('.png', '.jpg', '.jpeg'))]
    try:
        # this line is for generated image folder which include images with name purely integer.png
        image_files = sorted(image_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except:
        # this line is for COCO, ref image batch.
        image_files = image_files

    images = []
    for image_file in tqdm(image_files, desc=f"Loading images from {image_folder}"):
        try:
            img = Image.open(image_file).convert("RGB")
            if transform is not None:
                img = transform(img)
            else:
                img = torch.from_numpy(np.array(img).astype("uint8"))
            images.append(img)
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
    return torch.stack(images)


def calculate_clip_scores(image_folder, prompts, num_batch=1000):
    """Calculate Clip score for paired images (in image_folder) and prompts"""
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    images = load_images(image_folder).permute(0,3,1,2)  # (B, C, H, W)
    # print(f"in calcualte_clip_scores, images.shape={images.shape}")
    assert images.shape[0] == len(prompts), f"The number of images {images.shape[0]} doesn't match with the prompts: {len(prompts)}"

    # See https://huggingface.co/docs/diffusers/v0.21.0/conceptual/evaluation for the expectation of the input shape and pixel value range of clip_score_fn
    assert images.shape[1] == 3 and images.max() == 255 and images.min() == 0, "Check the image argument in calulating clip score"

    # Batch processing for CLIP score calculation
    num_image = images.shape[0]
    batch_size = max(1, ceil(num_image / num_batch))
    scores = []

    for i in tqdm(range(num_batch), desc="Calculating CLIP scores"):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, num_image)
        cur_score = clip_score_fn(images[start_index:end_index], prompts[start_index:end_index]).mean().item()
        scores.append(cur_score * (end_index - start_index))
        # print(f"Start index: {start_index}, End index: {end_index}, Batch score: {cur_score}")
    return sum(scores) / num_image


def calculate_fid(image_folder, ref_folder, num_batch=1000, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
    """Calculate FID between two image folders."""
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Inception expects 299x299 images
        transforms.ToTensor(), # Normalize to [0,1]
    ])

    # Load images from both folders
    gen_image = load_images(image_folder, transform).to(device) # (B, C, H, W)
    ref_image = load_images(ref_folder, transform).to(device) # (B, C, H, W)

    # See https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html for the expectation of the input shape and pixel value range of FrechetInceptionDistance
    assert gen_image.shape[1:] == (3,299, 299) and gen_image.max() <= 1.0 and gen_image.min() == 0 \
           and ref_image.shape[1:] == (3,299, 299) and ref_image.max() <= 1.0 and ref_image.min() == 0, "Check the image argument in calulating fid"

    fid = FrechetInceptionDistance(feature=2048, normalize=True, input_img_size=(3, 299, 299)).to(device)

    def update_fid(image_tensor, num_batch, is_real):
        num_image = image_tensor.shape[0]
        batch_size = max(1, ceil(num_image / num_batch))
        for i in tqdm(range(num_batch), desc=f"Update Fid for {['Gen', 'Real'][int(is_real)]}"):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, num_image)
            cur_image = image_tensor[start_index:end_index]
            # print(f"start index: {start_index}, end index: {end_index}, cur_image.shape = {cur_image.shape}")
            fid.update(cur_image, real=is_real)

    update_fid(ref_image, num_batch, True)
    update_fid(gen_image, num_batch, False)

    fid_score = fid.compute().item()

    return fid_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-dirs', type=str, nargs='+', default=['./samples/'], help='Directory to save generated images')
    parser.add_argument('--prompt-dir', type=str, default='./data/prompt_5000_random.txt',
                        help='Path to the prompt file')
    parser.add_argument('--ref-dir', type=str, default='/vision/vision_data_2/COCO_2017/val2017', help='Directory to COCO reference image')
    parser.add_argument('--num-batch', type=int, default = 1000, help = 'number of batches')
    parser.add_argument('--data-path', type=str, default = './data_dict_rebuttal_test.npz')
    parser.add_argument('--best-k', type=int, default=10)
    parser.add_argument('--only-plot', action='store_true', default=False)
    parser.add_argument('--prefix', type=str, default='')
    args = parser.parse_args()

    # Load prompts
    with open(args.prompt_dir, "r") as f:
        prompts = [line.strip() for line in f.readlines()]

    if args.data_path and os.path.exists(args.data_path):
        npzfile = np.load(args.data_path)
        data_dict = {key: npzfile[key] for key in
                     npzfile}  # format: {cfg-identifier: [[scale1, seed1, number_pngs1, fid1, clip1],
        # [scale2, seed2, number_pngs2, fid2, clip2]# ]}
    else:
        data_dict = {}

    if not args.only_plot:
        for cur_dir in args.sample_dirs:
            for d in os.listdir(cur_dir):
                file_path = os.path.join(cur_dir, d)
                num_image_in_path = sum([1 for f in os.listdir(file_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

                if num_image_in_path < len(prompts):
                    print(f"there are {num_image_in_path} in paths, fewer than numer of prompts {len(prompts)}. Skip process {file_path}")
                    continue
                elif num_image_in_path > len(prompts):
                    assert f"there are {num_image_in_path} in {file_path}, more than numer of prompts {len(prompts)}"
                else:
                    scale, seed, identifier = capture(file_path)
                    index, num_image_in_npz = get_fromdict(data_dict, identifier, scale, seed)
                    if num_image_in_npz  == len(prompts):
                        print(f"npz file alreay stores resutls corresponding to len(prompts)={len(prompts)}, skip processing {file_path}")
                        continue
                    print(f"process for {file_path}")
                    assert num_image_in_npz == 0 or num_image_in_npz == len(prompts), f"Shouldn't trigger this assertion, otherwise bug exists."

                    clip = calculate_clip_scores(file_path, prompts, num_batch=args.num_batch)
                    fid = calculate_fid(file_path, args.ref_dir, num_batch=args.num_batch)
                    print(f"Clip score: {clip}\nFID: {fid}")

                    if index is not None:
                        data_dict[identifier][index] = [scale, seed, len(prompts), fid, clip]
                    else:
                        ## TODO: double check, a trick for distributed updating, if only one process,
                        # then the following line is useless, because data_dict is not changed.
                        # however, it we have multiple process updating the npz at the same time...
                        if args.data_path and os.path.exists(args.data_path):
                            npzfile = np.load(args.data_path)
                            data_dict = {key: npzfile[key] for key in
                                         npzfile}

                        cur = data_dict.get(identifier, np.empty((0, 5)))
                        new_entry = np.array([[scale, seed, len(prompts), fid, clip]])
                        data_dict[identifier] = np.concatenate((cur, new_entry), axis=0)
                    np.savez(args.data_path, **data_dict)

    plt.figure()
    min_fid, min_config = float('inf'), None
    heap, min_k = [], args.best_k
    for k, v in data_dict.items():
        res = defaultdict(list)
        cur_fid = float('inf')
        for i in range(len(v)):
            cur_v = v[i]
            cur_seed = cur_v[1]
            cur_fid = min(cur_fid, cur_v[3])
            if cur_v[3] < 80:
                res[cur_seed].append(cur_v)  # cur_v[4], cur_v[3]: clip, FID

        heapq.heappush(heap, (-cur_fid, k))
        if len(heap) > min_k:
            heapq.heappop(heap)

        for seed, res_seed in res.items():
            res_seed = np.array(res_seed)
            ind = np.argsort(res_seed[:, 4])
            res_seed = res_seed[ind]
            # Plot the curve with markers
            plt.plot(res_seed[:, 4], res_seed[:, 3], '-', marker='x' if 'grad' in k else 'o', label=f'{k}-seed{seed}')
            for x, y, annotation in zip(res_seed[:, 4], res_seed[:, 3], res_seed[:, 0]):
                plt.annotate(f'{annotation:.2f}', (x, y), textcoords="offset points", xytext=(5, 5), ha='center',fontsize=6)

            header = ' '.join([f'c{i}' for i in range(len(res_seed))])
            np.savetxt(f"./figure_data/{args.prefix}_{k.replace('-', '_')}_seed{seed}.dat", res_seed, header=header,
                       comments="", fmt="%.6f")

        plt.xlabel('CLip')
        plt.ylabel('FID')
        plt.legend()

    plt.savefig('./figures/pf.png')
    plt.show()

    for i, (metric, k) in enumerate(sorted((-metric, x) for metric, x in heap)):
        print(f"Rank {i}, FID: {metric}, config: {k}")
