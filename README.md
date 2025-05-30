# REG  
[ICML'25] [REG: Rectified Gradient Guidance for Conditional Diffusion Models](https://arxiv.org/pdf/2501.18865)

## Instruction

The code will be released in the fall. I am currently doing a summer internship, which consumes all my bandwidth... If you need early access, feel free to contact me at [zhengqi@mit.edu](mailto:zhengqi@mit.edu), and I can share a preliminary version (note: it may not be fully cleaned up). On the other hand, REG requires only minimal modifications—just a one-line change in the reverse denoising process—so it should be straightforward to implement.

- For the ImageNet experiments in the paper, we forked [DiT](https://github.com/facebookresearch/DiT) and [EDM2](https://github.com/NVlabs/edm2) and applied our modifications.
- For the text-to-image experiments, we modified the source code of Hugging Face's `diffusers` library.
