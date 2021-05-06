# Soft Actor-Critic (SAC) implementation in PyTorch

This is PyTorch implementation of Soft Actor-Critic (SAC) [[ArXiv]](https://arxiv.org/abs/1812.05905).

If you use this code in your research project please cite us as:
```
@misc{pytorch_sac,
  author = {Yarats, Denis and Kostrikov, Ilya},
  title = {Soft Actor-Critic (SAC) implementation in PyTorch},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/denisyarats/pytorch_sac}},
}
```

## Requirements
We assume you have access to a gpu that can run CUDA 9.2. Then, the simplest way to install all required dependencies is to create an anaconda environment and activate it:
```
conda env create -f conda_env.yml
source activate pytorch_sac
```

## Instructions
To train an SAC agent on the `cheetah run` task run:
```
python train.py env=cheetah_run
```
This will produce `exp` folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. One can attacha tensorboard to monitor training by running:
```
tensorboard --logdir exp
```

## Results
An extensive benchmarking of SAC on the DM Control Suite against D4PG. We plot an average performance of SAC over 5 seeds together with p95 confidence intervals. Importantly, we keep the hyperparameters fixed across all the tasks. Note that results for D4PG are reported after 10^8 steps and taken from the original paper.
![Results](pytorch_sac/figures/dm_control.png)
