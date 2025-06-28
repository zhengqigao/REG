## Instructions

This repository builds upon the `diffusers` library and introduces modifications to support various types of guidance during denoising. Follow the steps below to reproduce our experiments:

1. Create a new Python environment and install the `diffusers` library. This step is crucial because we will modify the source code of `diffusers`, ensuring that your existing environment remains unaffected.

2. Replace the original `__call__` method in `pipeline_stable_diffusion.py` with the version provided in our repository located at `REG/diffusers/call_method.py`.

3. Experiment with different guidance types and parameters using the example provided in `test.py`.

After completing the steps above, you can use `sample_ddp.py` to generate samples and `calculate_metric.py` to evaluate the CLIP and IS scores. We randomly select 5,000 captions from the COCO 2017 validation set as text prompts.