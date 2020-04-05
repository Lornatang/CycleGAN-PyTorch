# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import itertools
import os
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image

from cyclegan_pytorch import Generator
from cyclegan_pytorch import ImageDataset

parser = argparse.ArgumentParser(description="PyTorch CycleGAN")
parser.add_argument("--dataroot", type=str, default="./data/horse2zebra/",
                    help="path to datasets")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="number of data loading workers ``default:4``")
parser.add_argument("--netG_A2B", default="./weights/netG_A2B.pth", type=str, metavar="PATH",
                    help="path to latest generator checkpoint "
                         "``default:'./weights/netG_A2B.pth'``.")
parser.add_argument("--netG_B2A", default="./weights/netG_B2A.pth", type=str, metavar="PATH",
                    help="path to latest discriminator checkpoint. "
                         "``default:'./weights/netG_B2A.pth'``.")
parser.add_argument("--dist-backend", default="nccl", type=str,
                    help="distributed backend")
parser.add_argument("--outf", default="./gen",
                    help="folder to output images. ``default:'./gen'``.")
parser.add_argument("--image-size", type=int, default=256,
                    help="size of the data crop (squared assumed)")
parser.add_argument("--gpu", default=0, type=int,
                    help="GPU id to use.")
parser.add_argument("--seed", default=None, type=int,
                    help="seed for initializing training.")


def test():
    try:
        os.makedirs(args.outf)
        os.makedirs(os.path.join(args.outf, "A"))
        os.makedirs(os.path.join(args.outf, "B"))
    except OSError:
        pass

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed training. "
                      "This will turn on the CUDNN deterministic setting, "
                      "which can slow down your training considerably! "
                      "You may see unexpected behavior when restarting "
                      "from checkpoints.")

    if args.gpu is not None:
        warnings.warn("You have chosen a specific GPU. This will completely "
                      "disable data parallelism.")

    # create model
    netG_A2B = Generator(3, 3)
    netG_B2A = Generator(3, 3)

    # move to GPU
    torch.cuda.set_device(args.gpu)
    netG_A2B = netG_A2B.cuda(args.gpu)
    netG_B2A = netG_B2A.cuda(args.gpu)

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(args.netG_A2B))
    netG_B2A.load_state_dict(torch.load(args.netG_B2A))

    # Set model mode
    netG_A2B.eval()
    netG_B2A.eval()

    dataset = ImageDataset(args.dataroot,
                           transform=transforms.Compose(
                               [transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]),
                           mode="test")

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=int(args.workers))

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for i, data in progress_bar:
        # get batch size data
        real_images_A = data["A"]
        real_images_B = data["B"]
        if args.gpu is not None:
            real_images_A = real_images_A.cuda(args.gpu, non_blocking=True)
            real_images_B = real_images_B.cuda(args.gpu, non_blocking=True)

        # Generate output
        fake_A = 0.5 * (netG_B2A(real_images_B).data + 1.0)
        fake_B = 0.5 * (netG_A2B(real_images_A).data + 1.0)

        # Save image files
        save_image(fake_A, f"gen/A/{i + 1:04d}.png", normalize=True)
        save_image(fake_B, f"gen/B/{i + 1:04d}.png", normalize=True)

        progress_bar.set_description(f"Generated images {i + 1:04d} of {len(dataloader):04d}")
