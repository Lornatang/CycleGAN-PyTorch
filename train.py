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
import torch.nn.parallel
import torch.nn.parallel
import torch.optim
import torch.optim
import torch.utils.data
import torch.utils.data
import torch.utils.data.distributed
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from cyclegan_pytorch import DecayLR
from cyclegan_pytorch import Discriminator
from cyclegan_pytorch import Generator
from cyclegan_pytorch import ImageDataset
from cyclegan_pytorch import weights_init

parser = argparse.ArgumentParser(description="PyTorch CycleGAN")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="path to datasets. (default:`./data`)")
parser.add_argument("name", type=str,
                    help="dataset name. "
                         "Option: [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, "
                         "cezanne2photo, ukiyoe2photo, vangogh2photo, maps, facades, "
                         "iphone2dslr_flower, ae_photos]")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="number of data loading workers. (default:4)")
parser.add_argument("--epochs", default=200, type=int, metavar="N",
                    help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N",
                    help="manual epoch number (useful on restarts)")
parser.add_argument("--decay_epochs", type=int, default=100,
                    help="epoch to start linearly decaying the learning rate to 0. (default:100)")
parser.add_argument("-b", "--batch-size", default=1, type=int,
                    metavar="N",
                    help="mini-batch size (default: 1), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="learning rate. (default:0.0002)")
parser.add_argument("--beta1", type=float, default=0.5,
                    help="beta1 for adam. (default:0.5)")
parser.add_argument("--beta2", type=float, default=0.999,
                    help="beta2 for adam. (default:0.999)")
parser.add_argument("-p", "--print-freq", default=100, type=int,
                    metavar="N", help="print frequency. (default:100)")
parser.add_argument("--world-size", default=-1, type=int,
                    help="number of nodes for distributed training")
parser.add_argument("--rank", default=-1, type=int,
                    help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://224.66.41.62:23456", type=str,
                    help="url used to set up distributed training")
parser.add_argument("--dist-backend", default="nccl", type=str,
                    help="distributed backend")
parser.add_argument("--netG_A2B", default="", help="path to netG_A2B (to continue training)")
parser.add_argument("--netG_B2A", default="", help="path to netG_B2A (to continue training)")
parser.add_argument("--netD_A", default="", help="path to netD_A (to continue training)")
parser.add_argument("--netD_B", default="", help="path to netD_B (to continue training)")
parser.add_argument("--outf", default="./outputs",
                    help="folder to output images. (default:`./outputs`).")
parser.add_argument("--image-size", type=int, default=256,
                    help="size of the data crop (squared assumed). (default:256)")
parser.add_argument("--seed", default=None, type=int,
                    help="seed for initializing training. (default:none)")
parser.add_argument("--gpu", default=None, type=int,
                    help="GPU id to use. (default:none)")
parser.add_argument("--multiprocessing-distributed", action="store_true",
                    help="Use multi-processing distributed training to launch "
                         "N processes per node, which has N GPUs. This is the "
                         "fastest way to use PyTorch for either single node or "
                         "multi node data parallel training")

valid_dataset_name = ["apple2orange", "summer2winter_yosemite", "horse2zebra",
                      "monet2photo", "cezanne2photo", "ukiyoe2photo", "vangogh2photo",
                      "maps, facades", "iphone2dslr_flower"]


def main():
    args = parser.parse_args()

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
        warnings.warn(
            "You have chosen a specific GPU. This will completely disable data parallelism.")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training!")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    netG_A2B = Generator(3, 3)
    netG_B2A = Generator(3, 3)
    netD_A = Discriminator(3)
    netD_B = Discriminator(3)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            netG_A2B.cuda(args.gpu)
            netG_B2A.cuda(args.gpu)
            netD_A.cuda(args.gpu)
            netD_B.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            netG_A2B = torch.nn.parallel.DistributedDataParallel(netG_A2B, device_ids=[args.gpu])
            netG_B2A = torch.nn.parallel.DistributedDataParallel(netG_B2A, device_ids=[args.gpu])
            netD_A = torch.nn.parallel.DistributedDataParallel(netD_A, device_ids=[args.gpu])
            netD_B = torch.nn.parallel.DistributedDataParallel(netD_B, device_ids=[args.gpu])
        else:
            netG_A2B.cuda()
            netG_B2A.cuda()
            netD_A.cuda()
            netD_B.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            netG_A2B = torch.nn.parallel.DistributedDataParallel(netG_A2B)
            netG_B2A = torch.nn.parallel.DistributedDataParallel(netG_B2A)
            netD_A = torch.nn.parallel.DistributedDataParallel(netD_A)
            netD_B = torch.nn.parallel.DistributedDataParallel(netD_B)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        netG_A2B = netG_A2B.cuda(args.gpu)
        netG_B2A = netG_B2A.cuda(args.gpu)
        netD_A = netD_A.cuda(args.gpu)
        netD_B = netD_B.cuda(args.gpu)

    else:
        # DataParallel will divide and allocate batch_size to all available
        # GPUs
        netG_A2B = torch.nn.DataParallel(netG_A2B).cuda()
        netG_B2A = torch.nn.DataParallel(netG_B2A).cuda()
        netD_A = torch.nn.DataParallel(netD_A).cuda()
        netD_B = torch.nn.DataParallel(netD_B).cuda()

    # apply weight init
    netG_A2B.apply(weights_init)
    netG_B2A.apply(weights_init)
    netD_A.apply(weights_init)
    netD_B.apply(weights_init)

    # resume trainning
    if args.netG_A2B != "":
        netG_A2B.load_state_dict(torch.load(opt.netG_A2B))
    if args.netG_B2A != "":
        netG_B2A.load_state_dict(torch.load(opt.netG_B2A))
    if args.netD_A != "":
        netD_A.load_state_dict(torch.load(opt.netD_A))
    if args.netD_B != "":
        netD_B.load_state_dict(torch.load(opt.netD_B))

    # define loss function (adversarial_loss) and optimizer
    adversarial_loss = torch.nn.MSELoss().cuda(args.gpu)
    cycle_loss = torch.nn.L1Loss().cuda(args.gpu)
    identity_loss = torch.nn.L1Loss().cuda(args.gpu)

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.lr,
                                     betas=(args.beta1, args.beta2))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.lr,
                                     betas=(args.beta1, args.beta2))

    lr_lambda = DecayLR(args.epochs, args.start_epoch, args.decay_epochs).step
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lr_lambda)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lr_lambda)

    cudnn.benchmark = True

    dataroot = os.path.join(args.dataroot, args.name)
    assert os.path.exists(dataroot), f"Please check that your dataset is exist."

    # Dataset
    dataset = ImageDataset(dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(int(args.image_size * 1.12), Image.BICUBIC),
                               transforms.RandomCrop(args.image_size),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                           unaligned=True)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=int(args.workers))

    assert len(dataloader) > 0, f"Please check that your dataset name. Option: {valid_dataset_name}"

    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    try:
        os.makedirs(os.path.join(args.outf, str(args.name), "A"))
        os.makedirs(os.path.join(args.outf, str(args.name), "B"))
    except OSError:
        pass

    try:
        os.makedirs(os.path.join("weights", str(args.name)))
    except OSError:
        pass

    for epoch in range(args.start_epoch, args.epochs):

        # switch to train mode
        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

        for i, data in progress_bar:
            # get batch size data
            real_images_A = data["A"]
            real_images_B = data["B"]
            if args.gpu is not None:
                real_images_A = real_images_A.cuda(args.gpu, non_blocking=True)
                real_images_B = real_images_B.cuda(args.gpu, non_blocking=True)

            # real data label is 1, fake data label is 0.
            real_label = torch.full((real_images_A.size(0), 1), 1, requires_grad=False)
            fake_label = torch.full((real_images_B.size(0), 1), 0, requires_grad=False)

            if args.gpu is not None:
                real_label = real_label.cuda(args.gpu, non_blocking=True)
                fake_label = fake_label.cuda(args.gpu, non_blocking=True)

            ##############################################
            # (1) Update G network: Generators A2B and B2A
            ##############################################
            optimizer_G.zero_grad()

            # Identity loss
            # G_B2A(A) should equal A if real A is fed
            sample_A = netG_B2A(real_images_A)
            loss_identity_A = identity_loss(sample_A, real_images_A) * 5.0
            # G_A2B(B) should equal B if real B is fed
            sample_B = netG_A2B(real_images_B)
            loss_identity_B = identity_loss(sample_B, real_images_B) * 5.0

            # GAN loss
            fake_B = netG_A2B(real_images_A)
            fake_output_B = netD_B(fake_B)
            loss_GAN_A2B = adversarial_loss(fake_output_B, real_label)

            fake_A = netG_B2A(real_images_B)
            fake_output_A = netD_A(fake_A)
            loss_GAN_B2A = adversarial_loss(fake_output_A, real_label)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = cycle_loss(recovered_A, real_images_A) * 10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = cycle_loss(recovered_B, real_images_B) * 10.0

            # Total loss
            errG = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            errG.backward()
            # Update G
            optimizer_G.step()

            ##############################################
            # (2) Update D network: Discriminator A
            ##############################################
            optimizer_D_A.zero_grad()

            # Real A image loss
            real_output_A = netD_A(real_images_A)
            D_x_A = adversarial_loss(real_output_A, real_label)

            # Fake A image loss
            fake_output_A = netD_A(fake_A.detach())
            errD_fake_A = adversarial_loss(fake_output_A, fake_label)

            # Total A image loss
            loss_D_A = (D_x_A + errD_fake_A) / 2
            loss_D_A.backward()
            # Update D for A
            optimizer_D_A.step()

            ##############################################
            # (3) Update D network: Discriminator B
            ##############################################
            optimizer_D_B.zero_grad()

            # Real B image loss
            real_output_B = netD_B(real_images_B)
            D_x_B = adversarial_loss(real_output_B, real_label)

            # Fake B image loss
            fake_output_B = netD_B(fake_B.detach())
            errD_fake_B = adversarial_loss(fake_output_B, fake_label)

            # Total B image loss
            loss_D_B = (D_x_B + errD_fake_B) / 2
            loss_D_B.backward()
            # Update D for B
            optimizer_D_B.step()

            progress_bar.set_description(
                f"[{epoch}/{args.epochs - 1}][{i}/{len(dataloader) - 1}] "
                f"Loss_D: {(loss_D_A + loss_D_B).item():.4f} "
                f"Loss_G: {errG.item():.4f} "
                f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.4f} "
                f"loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.4f} "
                f"loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB).item():.4f}")

            if i % args.print_freq == 0:
                vutils.save_image(real_images_A,
                                  f"{args.outf}/{args.name}/A/real_samples.png",
                                  normalize=True)
                vutils.save_image(real_images_B,
                                  f"{args.outf}/{args.name}/B/real_samples.png",
                                  normalize=True)

                fake_A = 0.5 * (netG_B2A(real_images_B).data + 1.0)
                fake_B = 0.5 * (netG_A2B(real_images_A).data + 1.0)

                vutils.save_image(fake_A.detach(),
                                  f"{args.outf}/{args.name}/A/fake_samples_epoch_{epoch}.png",
                                  normalize=True)
                vutils.save_image(fake_B.detach(),
                                  f"{args.outf}/{args.name}/B/fake_samples_epoch_{epoch}.png",
                                  normalize=True)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            # do checkpointing
            torch.save(netG_A2B.state_dict(), f"weights/{args.name}/netG_A2B_epoch_{epoch}.pth")
            torch.save(netG_B2A.state_dict(), f"weights/{args.name}/netG_B2A_epoch_{epoch}.pth")
            torch.save(netD_A.state_dict(), f"weights/{args.name}/netD_A_epoch_{epoch}.pth")
            torch.save(netD_B.state_dict(), f"weights/{args.name}/netD_B_epoch_{epoch}.pth")

            # save last checkpoint
        if epoch == args.epochs - 1:
            torch.save(netG_A2B.state_dict(), f"weights/{args.name}/netG_A2B.pth")
            torch.save(netG_B2A.state_dict(), f"weights/{args.name}/netG_B2A.pth")
            torch.save(netD_A.state_dict(), f"weights/{args.name}/netD_A.pth")
            torch.save(netD_B.state_dict(), f"weights/{args.name}/netD_B.pth")

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()


if __name__ == "__main__":
    main()
