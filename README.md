# CycleGAN-PyTorch

### Overview
This repository contains an op-for-op PyTorch reimplementation of [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593).

### Table of contents
1. [About Cycle Generative Adversarial Networks](#about-cycle-generative-adversarial-networks)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights)
    * [Download dataset](#download-dataset)
4. [Test](#test)
4. [Train](#train)
    * [Example (horse2zebra)](#example-horse2zebra)
5. [Contributing](#contributing) 
6. [Credit](#credit)

### About Cycle Generative Adversarial Networks

If you're new to DCGAN, here's an abstract straight from the paper:

Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G:X→Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F:Y→X and introduce a cycle consistency loss to push F(G(X))≈X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives a random noise z and generates images from this noise, which is called G(z).Discriminator is a discriminant network that discriminates whether an image is real. The input is x, x is a picture, and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

#### Clone and install requirements

```bash
$ git clone https://github.com/Lornatang/CycleGAN_PyTorch
$ cd CycleGAN_PyTorch/
$ pip3 install -r requirements.txt
```

#### Download pretrained weights

```bash
$ cd weights/
$ bash download_weights.sh <datasets-name>
```

#### Download dataset

```bash
$ cd data/
$ bash get_dataset.sh <datasets-name>
```

### Test (e.g horse2zebra)

Using pre training model to generate pictures.

```bash
$ python3 test.py --netG_A2B weights/horse2zebra/netG_A2B.pth --netG_B2A weights/horse2zebra/netG_B2A.pth
```

<span align="left"><img src="assets/real_A.jpg" width="256" alt=""></span>
<span align="right"><img src="assets/fake_B.png" width="256" alt=""></span><br>
<span align="left"><img src="assets/real_B.jpg" width="256" alt=""></span>
<span align="right"><img src="assets/fake_A.png" width="256" alt=""></span>

### Train

```text
usage: train.py [-h] [--dataroot DATAROOT] [-j N] [--epochs N]
                [--start-epoch N] [-b N] [--lr LR] [--beta1 BETA1]
                [--beta2 BETA2] [-p N] [--world-size WORLD_SIZE] [--rank RANK]
                [--dist-url DIST_URL] [--dist-backend DIST_BACKEND]
                [--netG_A2B NETG_A2B] [--netG_B2A NETG_B2A] [--netD_A NETD_A]
                [--netD_B NETD_B] [--outf OUTF] [--image-size IMAGE_SIZE]
                [--seed SEED] [--gpu GPU] [--multiprocessing-distributed]
                name
```

#### Example (horse2zebra)

```bash
$ python3 train.py horse2zebra --gpu 0
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python3 train.py horse2zebra --gpu 0 --netG_A2B weights/horse2zebra/netG_A2B_epoch_*.pth --netG_B2A weights/horse2zebra/netG_B2A_epoch_*.pth --netD_A weights/horse2zebra/netD_A_epoch_*.pth --netD_B weights/horse2zebra/netD_B_epoch_*.pth
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 

### Credit

#### Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
_Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros_ <br>

**Abstract** <br>
Image-to-image translation is a class of vision and graphics problems where the goal 
is to learn the mapping between an input image and an output image using a training 
set of aligned image pairs. However, for many tasks, paired training data will not be 
available. We present an approach for learning to translate an image from a source 
domain X to a target domain Y in the absence of paired examples. Our goal is to learn 
a mapping G:X→Y such that the distribution of images from G(X) is indistinguishable
from the distribution Y using an adversarial loss. Because this mapping is highly
under-constrained, we couple it with an inverse mapping F:Y→X and introduce a cycle 
consistency loss to push F(G(X))≈X (and vice versa). Qualitative results are presented 
on several tasks where paired training data does not exist, including collection 
style transfer, object transfiguration, season transfer, photo enhancement, etc. 
Quantitative comparisons against several prior methods demonstrate the superiority
of our approach.

[[Paper]](https://arxiv.org/pdf/1703.10593)) [[Authors' Implementation]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}
```