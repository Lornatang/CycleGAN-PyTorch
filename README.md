# CycleGAN-PyTorch

### Overview
This repository contains an op-for-op PyTorch reimplementation of [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://xxx.itp.ac.cn/pdf/1511.06434).

The goal of this implementation is to be simple, highly extensible, and easy to integrate into your own projects. This implementation is a work in progress -- new features are currently being implemented.  

At the moment, you can easily:  
 * Load pretrained Generate models 
 * Use Generate models for extended dataset

_Upcoming features_: In the next few days, you will be able to:
 * Quickly finetune an Generate on your own dataset
 * Export Generate models for production

### Table of contents
1. [About Deep Convolutional Generative Adversarial Networks](#about-deep-convolutional-generative-adversarial-networks)
2. [Model Description](#model-description)
3. [Installation](#installation)
4. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Extended dataset](#example-extended-dataset)
    * [Example: Visual](#example-visual)
5. [Contributing](#contributing) 

### About Deep Convolutional Generative Adversarial Networks

If you're new to DCGAN, here's an abstract straight from the paper:

In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives a random noise z and generates images from this noise, which is called G(z).Discriminator is a discriminant network that discriminates whether an image is real. The input is x, x is a picture, and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

Install from pypi:
```bash
$ pip3 install cycle_pytorch
```

Install from source:
```bash
$ git clone https://github.com/Lornatang/CycleGAN-PyTorch.git
$ cd CycleGAN-PyTorch
$ pip3 install -e .
``` 

### Usage

#### Loading pretrained models

Load an Deep-Convolutional-Generative-Adversarial-Networks:
```python
from dcgan_pytorch import Generator
model = Generator.from_name("g-mnist")
```

Load a pretrained Deep-Convolutional-Generative-Adversarial-Networks:
```python
from dcgan_pytorch import Generator
model = Generator.from_pretrained("g-mnist")
```

#### Example: Extended dataset

As mentioned in the example, if you load the pre-trained weights of the MNIST dataset, it will create a new `imgs` directory and generate 64 random images in the `imgs` directory.

```python
import os
import torch
import torchvision.utils as vutils
from dcgan_pytorch import Generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Generator.from_pretrained("g-mnist")
model.to(device)
# switch to evaluate mode
model.eval()

try:
    os.makedirs("./imgs")
except OSError:
    pass

with torch.no_grad():
    for i in range(64):
        noise = torch.randn(64, 100, 1, 1, device=device)
        fake = model(noise)
        vutils.save_image(fake.detach(), f"./imgs/fake_{i:04d}.png", normalize=True)
    print("The fake image has been generated!")
```

#### Example: Visual

```text
cd $REPO$/framework
sh start.sh
```

Then open the browser and type in the browser address [http://127.0.0.1:10001/](http://127.0.0.1:10001/).
Enjoy it.

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 