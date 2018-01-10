# Action Conditioned GANS
This is an implementation of conditional video prediction based on various generative adversarial networks. The technical details of this project are outlined [here](https://github.com/yidingjiang/action_conditioned_GANs/blob/master/report/report.pdf).

**Some results** (many gifs!): [here](https://github.com/yidingjiang/Action_Conditioned_GAN_demo)

## Dependencies
* tensorflow1.0 (tested on 1.0)
* numpy
* matplotlib
* opencv-python
* pillow
* imageio
* h5py

## Installation

`pip install requirements.txt`

## Data

This project is done on the **Push Dataset** from Google Brain.

Download from https://sites.google.com/site/brainrobotdata/home/push-dataset

## Train

`python train.py PATH/TO/INPUT/DATA OUTPUT/PATH`

## Training details
* `--adv`: boolean, whether to use adversarial loss
  * True (default)
* `--loss`: string, what loss to use
  * `bce` for cross entropy (default)
  * `wass` for wasserstein loss
* `--opt`: string, what optimizer to use
  * `adam` for ADAM Optimizer (default)
  * `rmsprop` for RMSProp
* `--dna`: boolean, whether to use dynamic neural advection
  * True (default)

## Test

`python test.py MODEL/PATH INPUT/FRAME/PATH INPUT/ACTION/PATH RESULT/SAVE/PATH`

Note: Input frames and actions need to be numpy files for flexibility.

## Training details
* `--dna`: boolean, whether to use dynamic neural advection
  * True (default)
