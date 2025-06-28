## Instructions

Our code builds upon EDM and introduces modifications to support various types of guidance during denoising. Follow the steps below to reproduce our experiments:

1. Clone the [EDMv2 GitHub repository](https://github.com/NVlabs/edm2) onto your server and navigate to the cloned directory.

2. Replace the original `generate_images.py` file in the EDMv2 repository with the version provided in our REG repository located at `REG/EDM/generate_images.py`. 

3. Run the experiment using the following command:

    ```shell
    # N is the number of GPUs available on one node.
    # Vanilla CFG
    torchrun --standalone --nproc_per_node=N generate_images.py --batch=8 --preset=edm2-img512-s-guid-fid --outdir=./test --seeds=0-49999 --guidance=2.0 --cfg_type=original --guid_network_labels=False 

    # Vanilla CFG + REG
    torchrun --standalone --nproc_per_node=N generate_images.py --batch=8 --preset=edm2-img512-s-guid-fid --outdir=./test --seeds=0-49999 --guidance=2.0 --cfg_type=reg_original --guid_network_labels=False 

    # Interval CFG 
    torchrun --standalone --nproc_per_node=N generate_images.py --batch=8 --preset=edm2-img512-s-guid-fid --outdir=./test --seeds=0-49999 --guidance=2.0 --cfg_type=intervalt --guid_network_labels=False --cfg-param 0.28 --cfg-param 2.90 

    # Interval CFG + REG 
    torchrun --standalone --nproc_per_node=N generate_images.py --batch=8 --preset=edm2-img512-s-guid-fid --outdir=./test --seeds=0-49999 --guidance=2.0 --cfg_type=reg_intervalt --guid_network_labels=False --cfg-param 0.28 --cfg-param 2.90 
    ```

5. The above command will generate 50K images stored in your local folder `test`. After the command finishes, FID and IS evaluation can be performed using `calculate_metrics.py`. Detailed instructions are available in the [EDMv2 repository](https://github.com/NVlabs/edm2).

## Additional Notes

### Guidance Weight 

The definition of guidance weight may vary across repositories:

```
noise_pred = cond +  w1 * (cond - uncond)
noise_pred = uncond + w2 * (cond - uncond)
```

These two definitions are essentially equivalent, related by `w1 + 1 = w2`. In our context, this distinction is not critical since we sweep the guidance weight across a broad range and plot the FID versus IS scores.

### FID and IS Evaluation

For conditional ImageNet generation, evaluating metrics on 50K images is standard practice. Note that EDM and DiT use different backends for FID and IS evaluation, which may result in slight discrepancies in FID and IS values for the same set of 50K generated images.


### Autoguidance

[Autoguidance](https://arxiv.org/abs/2406.02507) needs to find a valid bad version to provide the guidance signal, so we only evalulate Autoguidance when the 'bad' checkpoints are provided. 