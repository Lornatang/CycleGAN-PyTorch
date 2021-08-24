# CycleGAN-PyTorch

### Overview
This repository contains an op-for-op PyTorch reimplementation of [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593).

[Demo and Docker image on Replicate](https://replicate.ai/lornatang/cyclegan-pytorch)

### Table of contents
1. [About Cycle Generative Adversarial Networks](#about-cycle-generative-adversarial-networks)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights)
    * [Download dataset](#download-dataset)
4. [Test](#test)
4. [Train](#train)
    * [Example](#example)
    * [Resume training](#resume-training)
5. [Contributing](#contributing) 
6. [Credit](#credit)

### About Cycle Generative Adversarial Networks

If you're new to CycleGAN, here's an abstract straight from the paper:

Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G:X→Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F:Y→X and introduce a cycle consistency loss to push F(G(X))≈X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives a random noise z and generates images from this noise, which is called G(z).Discriminator is a discriminant network that discriminates whether an image is real. The input is x, x is a picture, and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

#### Clone and install requirements

```bash
$ git clone https://github.com/Lornatang/CycleGAN-PyTorch
$ cd CycleGAN-PyTorch/
$ pip3 install -r requirements.txt
```

#### Download pretrained weights

```bash
# example: horse2zebra
$ cd weights/
$ bash download_weights.sh horse2zebra
```

#### Download dataset

```bash
# example: horse2zebra
$ cd data/
$ bash get_dataset.sh horse2zebra
```

### Test

The following commands can be used to test the whole test.

```bash
# Example: horse2zebra
$ python3 test.py --dataset horse2zebra --cuda
```

For single image processing, use the following command.

```bash
$ python3 test_image.py --file assets/horse.png --model-name weights/horse2zebra/netG_A2B.pth --cuda
```

InputA --> StyleB  --> RecoveryA

<img src="assets/apple.png" title="InputA"/><img src="assets/fake_orange.png" title="StyleB"><img src="assets/fake_apple.png" title="RecoveryA">
<img src="assets/cezanne.png" title="InputA"/><img src="assets/fake_photo.png" title="StyleB"><img src="assets/fake_cezanne.png" title="RecoveryA">
<img src="assets/facades.png" title="InputA"/><img src="assets/fake_facades1.png" title="StyleB"><img src="assets/fake_facades2.png" title="RecoveryA">
<img src="assets/horse.png" title="InputA"/><img src="assets/fake_zebra.png" title="StyleB"><img src="assets/fake_horse.png" title="RecoveryA">

### Train

```text
usage: train.py [-h] [--dataroot DATAROOT] [--dataset DATASET] [--epochs N]
                [--decay_epochs DECAY_EPOCHS] [-b N] [--lr LR] [-p N] [--cuda]
                [--netG_A2B NETG_A2B] [--netG_B2A NETG_B2A] [--netD_A NETD_A]
                [--netD_B NETD_B] [--image-size IMAGE_SIZE] [--outf OUTF]
                [--manualSeed MANUALSEED]

PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent
Adversarial Networks`

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   path to datasets. (default:./data)
  --dataset DATASET     dataset name. (default:`horse2zebra`)Option:
                        [apple2orange, summer2winter_yosemite, horse2zebra,
                        monet2photo, cezanne2photo, ukiyoe2photo,
                        vangogh2photo, maps, facades, selfie2anime,
                        iphone2dslr_flower, ae_photos, ]
  --epochs N            number of total epochs to run
  --decay_epochs DECAY_EPOCHS
                        epoch to start linearly decaying the learning rate to
                        0. (default:100)
  -b N, --batch-size N  mini-batch size (default: 1), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel
  --lr LR               learning rate. (default:0.0002)
  -p N, --print-freq N  print frequency. (default:100)
  --cuda                Enables cuda
  --netG_A2B NETG_A2B   path to netG_A2B (to continue training)
  --netG_B2A NETG_B2A   path to netG_B2A (to continue training)
  --netD_A NETD_A       path to netD_A (to continue training)
  --netD_B NETD_B       path to netD_B (to continue training)
  --image-size IMAGE_SIZE
                        size of the data crop (squared assumed). (default:256)
  --outf OUTF           folder to output images. (default:`./outputs`).
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:none)

```

#### Example

```bash
# Example: horse2zebra
$ python3 train.py --dataset horse2zebra --cuda
```

#### Resume training

If you want to load weights that you've trained before, run the following command.

```bash
# Example: horse2zebra for epoch 100
$ python3 train.py --dataset horse2zebra \
    --netG_A2B weights/horse2zebra/netG_A2B_epoch_100.pth \
    --netG_B2A weights/horse2zebra/netG_B2A_epoch_100.pth \
    --netD_A weights/horse2zebra/netD_A_epoch_100.pth \
    --netD_B weights/horse2zebra/netD_B_epoch_100.pth --cuda
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
