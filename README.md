# Mixed Traffic Control and Coordination from Pixels

- Paper in [ArXiV](https://arxiv.org/abs/2302.09167v4)
- ICRA 2024 [Presentation Video](https://youtu.be/g-aBGhdFHXU)
- ICRA 2024 [Poster](https://github.com/tmvllrrl/mtc-pixels/blob/master/ICRA_Poster_medium.pdf)

## Installation

Instructions for installing and setting up everything for this repo are located [here.](https://docs.google.com/document/d/1Niz2ysr3W74fNFhhazhC540pEqdYxzkTznYa0d7TiiU/edit?usp=sharing)

## Training

Example commands for training each of the five environments from the paper are provided in run.sh. An example command for training on the ring environment is the following:

```
python configs/train.py ring
```

More details about training and running the environments are provided in the instructions.

## Evaluation

Example commands for evaluating a trained policy are provided in eval.sh. More details for evaluating a trained policy are provided for in the instructions.

## Citing

Please cite our work using the following bibtex:
```
@inproceedings{villarreal2023mixed,
  title={Mixed Traffic Control and Coordination from Pixels},
  author={Villarreal, Michael and Poudel, Bibek and Pan, Jia and Li, Weizi},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
```

This code base originally comes from the following paper:

C. Wu, A. Kreidieh, K. Parvate, E. Vinitsky, A. Bayen, "Flow: Architecture and Benchmarking for Reinforcement Learning in Traffic Control," CoRR, vol. abs/1710.05465, 2017. [Online]. Available: https://arxiv.org/abs/1710.05465.

Please also consider cite this work.

