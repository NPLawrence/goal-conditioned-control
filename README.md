# goal-conditioned-control

Companion code to our [paper](https://arxiv.org/abs/2512.06471) on connections between goal-conditioned reinforcement learning and optimal control.

## Set-up
Create a virtual enviroment and install dependencies:
```
uv init
uv sync
```
I couldn't install Neuromancer (for the double inverted pendulum example) via uv at this time. For that example, simply run the notebook, which will handle installation via pip.

## Citation
To reference this work, please use the following bib entry:
```
@article{lawrence2025goalconditioned,
  title={Why Goal-Conditioned Reinforcement Learning Works: Relation to Dual Control},
  author={Lawrence, Nathan P and Mesbah, Ali},
  journal={arXiv preprint arXiv:2512.06471},
  year={2025}
}
```
