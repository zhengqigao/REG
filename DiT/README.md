## Instructions

Our code builds upon DiT and introduces modifications to support various types of guidance during denoising. Follow the steps below to reproduce our experiments:

1. Clone the [DiT GitHub repository](https://github.com/facebookresearch/DiT) onto your server and navigate to the cloned directory.

2. Replace the original `sample_ddp.py` file in the DiT repository with the version provided in our REG repository located at `REG/DiT/sample_ddp.py`. This modification primarily introduces two additional command-line arguments: `cfg-type` and `cfg-param`.

3. Update the `forward_with_cfg` method in the `DiT` class within `models.py` by replacing it with the version provided in `REG/DiT/forward_with_cfg.py`.

4. Run the experiment using the following command:

    ```shell
    # Vanilla CFG with cfg-weight=1.5
    torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py --model DiT-XL/2 --num-fid-samples 50000 --cfg-scale 1.5 --cfg-type original --image-size 256 --sample-dir test --per-proc-batch-size 8

    # Vanilla CFG + REG with cfg-weight=1.5
    torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py --model DiT-XL/2 --num-fid-samples 50000 --cfg-scale 1.5 --cfg-type reg --image-size 256 --sample-dir test --per-proc-batch-size 8
    ```

5. The above command will generate 50K images stored in your local folder `test`. After the command finishes, FID and IS evaluation can be performed using ADM's evaluation suite. Detailed instructions are available in both the [DiT repository](https://github.com/facebookresearch/DiT) and the original [ADM repository](https://github.com/openai/guided-diffusion/tree/main/evaluations).

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

### Channels for Guidance Application

DiT performs diffusion in latent space. However, in the original repository, guidance is applied only to the [first three channels](https://github.com/facebookresearch/DiT/blob/main/models.py#L262), despite the model having four channels in total. Further discussion on this topic can be found in this [GitHub issue](https://github.com/facebookresearch/DiT/pull/12). For our experiments, we adhere to the three-channel guidance approach, as this aligns with the model checkpoint provided and the results reported in the original paper.