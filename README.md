# REG  
[ICML'25] [REG: Rectified Gradient Guidance for Conditional Diffusion Models](https://arxiv.org/pdf/2501.18865)

TL;DR: Previous studies have identified a discrepancy between guidance theory and its practical implementation. In this work, we propose a refined explanation for guidance theory, inspired by the observation that the original theory focuses on scaling marginal distributions, whereas the correct formulation should aim to scale the joint distribution.

## Instruction

REG requires only minimal modifications—just a one-line change in the reverse denoising process—so it should be straightforward to implement.

- For the ImageNet experiments in the paper, we forked [DiT](https://github.com/facebookresearch/DiT) and [EDMv2](https://github.com/NVlabs/edm2) and applied our modifications.
- For the text-to-image experiments, we modify the source code of Hugging Face's `diffusers` library.

Please refer to each subfolders for detailed instructions.
