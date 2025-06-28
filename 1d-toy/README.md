## Instruction

This folder contains all the necessary code to reproduce the results presented in the 1D synthetic experiments.

To generate figures illustrating the performance of CFG with REG and without REG, compared to the optimal solution, run the command ``python visual1.py``. These results correspond to Figures 1 and 5 in our manuscript.

If desired, we also provide a training script, ``main.py``, which can be used to train diffusion models under various parameters. Please note that due to differences in computing environments, the trained models you obtain may slightly differ from the ones we provide. However, they should still produce similar visualizations when using ``visual1.py``.

Additionally, we intentionally limit the size of the diffusion model in this experiment. This work is heavily inspired by [Interval Guidance](https://arxiv.org/abs/2404.07724) and [AutoGuidance](https://arxiv.org/abs/2406.02507).