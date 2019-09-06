# PyTorch-CycleGAN
A Pytorch implementation of CycleGAN (https://arxiv.org/abs/1703.10593)

## Prerequisites
Code is intended to work with ```Python 3.7.x```, it hasn't been tested with previous versions

### [PyTorch & torchvision](http://pytorch.org/)
Follow the instructions in [pytorch.org](http://pytorch.org) for your current setup

## Training
### 1. Setup the dataset
First, you will need to download and setup a dataset. The easiest way is to use one of the already existing datasets on UC Berkeley's repository:
```
python3 download.py <dataset_name>
```
Valid <dataset_name> are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos

Alternatively you can build your own dataset by setting up the following directory structure:

    .
    ├── datasets                   
    |   ├── <dataset_name>         # i.e. brucewayne2batman
    |   |   ├── train              # Training
    |   |   |   ├── A              # Contains domain A images (i.e. Bruce Wayne)
    |   |   |   └── B              # Contains domain B images (i.e. Batman)
    |   |   └── test               # Testing
    |   |   |   ├── A              # Contains domain A images (i.e. Bruce Wayne)
    |   |   |   └── B              # Contains domain B images (i.e. Batman)
    
### 2. Train!
```
python3 train.py --dataroot datasets/<dataset_name>/ --cuda
```
This command will start a training session using the images under the *dataroot/train* directory with the hyperparameters that showed best results according to CycleGAN authors. You are free to change those hyperparameters, see ```python3 train.py --help``` for a description of those.

Both generators and discriminators weights will be saved under the output directory.

If you don't own a GPU remove the --cuda option, although I advise you to get one!


![Generator loss](https://github.com/Lornatang/PyTorch-CycleGAN/raw/master/outputs/loss_G.png)
![Discriminator loss](https://github.com/Lornatang/PyTorch-CycleGAN/raw/master/outputs/loss_D.png)
![Generator GAN loss](https://github.com/Lornatang/PyTorch-CycleGAN/raw/master/outputs/loss_G_GAN.png)
![Generator identity loss](https://github.com/Lornatang/PyTorch-CycleGAN/raw/master/outputs/loss_G_identity.png)
![Generator cycle loss](https://github.com/Lornatang/PyTorch-CycleGAN/raw/master/output/sloss_G_cycle.png)

## Testing
```
python3 test.py --dataroot datasets/<dataset_name>/ --cuda
```
This command will take the images under the *dataroot/test* directory, run them through the generators and save the output under the *outputs/A* and *outputs/B* directories. As with train, some parameters like the weights to load, can be tweaked, see ```python3 test.py --help``` for more information.

Examples of the generated outputs (default params, horse2zebra dataset):

![Real horse](https://github.com/Lornatang/PyTorch-CycleGAN/raw/master/outputs/real_A.jpg)
![Fake zebra](https://github.com/Lornatang/PyTorch-CycleGAN/raw/master/outputs/fake_B.png)
![Real zebra](https://github.com/Lornatang/PyTorch-CycleGAN/raw/master/outputs/real_B.jpg)
![Fake horse](https://github.com/Lornatang/PyTorch-CycleGAN/raw/master/outputs/fake_A.png)

## License
This project is licensed under the Apache License - see the [LICENSE](https://github.com/Lornatang/PyTorch-CycleGAN/raw/master/LICENSE) file for details

## Acknowledgments
Code is basically implementation of [PyTorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN). All credit goes to the authors of [CycleGAN](https://arxiv.org/abs/1703.10593), Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A.
