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

import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from cyclegan_pytorch import DecayLR
from cyclegan_pytorch import Discriminator
from cyclegan_pytorch import Generator
from cyclegan_pytorch import ImageDataset
from cyclegan_pytorch import weights_init

parser = argparse.ArgumentParser(description="PyTorch CycleGAN")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="path to datasets. (default:./data)")
parser.add_argument("name", type=str,
                    help="dataset name. "
                         "Option: [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, "
                         "cezanne2photo, ukiyoe2photo, vangogh2photo, maps, facades, "
                         "iphone2dslr_flower, ae_photos]")
parser.add_argument("-j", "--workers", default=8, type=int, metavar="N",
                    help="number of data loading workers. (default:8)")
parser.add_argument("--epochs", default=200, type=int, metavar="N",
                    help="number of total epochs to run")
parser.add_argument("--image-size", type=int, default=256,
                    help="size of the data crop (squared assumed). (default:256)")
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
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--netG_A2B", default="", help="path to netG_A2B (to continue training)")
parser.add_argument("--netG_B2A", default="", help="path to netG_B2A (to continue training)")
parser.add_argument("--netD_A", default="", help="path to netD_A (to continue training)")
parser.add_argument("--netD_B", default="", help="path to netD_B (to continue training)")
parser.add_argument("--outf", default="./outputs",
                    help="folder to output images. (default:`./outputs`).")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")
parser.add_argument("--ngpu", default=1, type=int,
                    help="GPU id to use. (default:None)")
parser.add_argument("--multiprocessing-distributed", action="store_true",
                    help="Use multi-processing distributed training to launch "
                         "N processes per node, which has N GPUs. This is the "
                         "fastest way to use PyTorch for either single node or "
                         "multi node data parallel training")

valid_dataset_name = ["apple2orange", "summer2winter_yosemite", "horse2zebra",
                      "monet2photo", "cezanne2photo", "ukiyoe2photo", "vangogh2photo",
                      "maps, facades", "iphone2dslr_flower"]

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs("weights")
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

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
    os.makedirs(os.path.join(args.outf, str(args.name), "A"))
    os.makedirs(os.path.join(args.outf, str(args.name), "B"))
except OSError:
    pass

try:
    os.makedirs(os.path.join("weights", str(args.name)))
except OSError:
    pass

device = torch.device("cuda:0" if args.cuda else "cpu")
ngpu = int(args.ngpu)

assert (args.batch_size > 2 and ngpu < 1), "You used multi GPU training, you should probably run with --batch-size 2"

# create model
netG_A2B = Generator(3, 3).to(device)
netG_B2A = Generator(3, 3).to(device)
netD_A = Discriminator(3).to(device)
netD_B = Discriminator(3).to(device)

if args.cuda and ngpu > 1 and args.batch_size > 1:
    netG_A2B = torch.nn.DataParallel(netG_A2B).to(device)
    netG_B2A = torch.nn.DataParallel(netG_B2A).to(device)
    netD_A = torch.nn.DataParallel(netD_A).to(device)
    netD_B = torch.nn.DataParallel(netD_B).to(device)

netG_A2B.apply(weights_init)
netG_B2A.apply(weights_init)
netD_A.apply(weights_init)
netD_B.apply(weights_init)

if args.netG_A2B != "":
    netG_A2B.load_state_dict(torch.load(args.netG_A2B))
if args.netG_B2A != "":
    netG_B2A.load_state_dict(torch.load(args.netG_B2A))
if args.netD_A != "":
    netG_A2B.load_state_dict(torch.load(args.netD_A))
if args.netD_B != "":
    netG_B2A.load_state_dict(torch.load(args.netD_B))

# define loss function (adversarial_loss) and optimizer
adversarial_loss = torch.nn.MSELoss().to(device)
cycle_loss = torch.nn.L1Loss().to(device)
identity_loss = torch.nn.L1Loss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.lr,
                                 betas=(args.beta1, args.beta2))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.lr,
                                 betas=(args.beta1, args.beta2))

lr_lambda = DecayLR(args.epochs, 0, args.decay_epochs).step
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lr_lambda)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lr_lambda)

g_losses = []
d_losses = []

identity_losses = []
gan_losses = []
cycle_losses = []

for epoch in range(0, args.epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
        # get batch size data
        real_image_A = data["A"].to(device)
        real_image_B = data["B"].to(device)
        batch_size = real_image_A.size(0)

        # real data label is 1, fake data label is 0.
        real_label = torch.full((batch_size, 1), 1, requires_grad=False, device=device)
        fake_label = torch.full((batch_size, 1), 0, requires_grad=False, device=device)

        ##############################################
        # (1) Update G network: Generators A2B and B2A
        ##############################################

        # Set G_A and G_B's gradients to zero
        optimizer_G.zero_grad()

        # Identity loss
        # G_B2A(A) should equal A if real A is fed
        identity_image_A = netG_B2A(real_image_A)
        loss_identity_A = identity_loss(identity_image_A, real_image_A) * 5.0
        # G_A2B(B) should equal B if real B is fed
        identity_image_B = netG_A2B(real_image_B)
        loss_identity_B = identity_loss(identity_image_B, real_image_B) * 5.0

        # GAN loss
        # GAN loss D_A(G_A(A))
        fake_image_A = netG_B2A(real_image_B)
        fake_output_A = netD_A(fake_image_A)
        loss_GAN_B2A = adversarial_loss(fake_output_A, real_label)
        # GAN loss D_B(G_B(B))
        fake_image_B = netG_A2B(real_image_A)
        fake_output_B = netD_B(fake_image_B)
        loss_GAN_A2B = adversarial_loss(fake_output_B, real_label)

        # Cycle loss
        recovered_image_A = netG_B2A(fake_image_B)
        loss_cycle_ABA = cycle_loss(recovered_image_A, real_image_A) * 10.0

        recovered_image_B = netG_A2B(fake_image_A)
        loss_cycle_BAB = cycle_loss(recovered_image_B, real_image_B) * 10.0

        # Combined loss and calculate gradients
        errG = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

        # Calculate gradients for G_A and G_B
        errG.backward()
        # Update G_A and G_B's weights
        optimizer_G.step()

        ##############################################
        # (2) Update D network: Discriminator A
        ##############################################

        # Set D_A gradients to zero
        optimizer_D_A.zero_grad()

        # Real A image loss
        real_output_A = netD_A(real_image_A)
        D_x_A = adversarial_loss(real_output_A, real_label)

        # Fake A image loss
        fake_output_A = netD_A(fake_image_A.detach())
        errD_fake_A = adversarial_loss(fake_output_A, fake_label)

        # Combined loss and calculate gradients
        loss_D_A = (D_x_A + errD_fake_A) / 2

        # Calculate gradients for D_A
        loss_D_A.backward()
        # Update D_A weights
        optimizer_D_A.step()

        ##############################################
        # (3) Update D network: Discriminator B
        ##############################################

        # Set D_B gradients to zero
        optimizer_D_B.zero_grad()

        # Real B image loss
        real_output_B = netD_B(real_image_B)
        D_x_B = adversarial_loss(real_output_B, real_label)

        # Fake B image loss
        fake_output_B = netD_B(fake_image_B.detach())
        errD_fake_B = adversarial_loss(fake_output_B, fake_label)

        # Combined loss and calculate gradients
        loss_D_B = (D_x_B + errD_fake_B) / 2

        # Calculate gradients for D_B
        loss_D_B.backward()
        # Update D_B weights
        optimizer_D_B.step()

        progress_bar.set_description(
            f"[{epoch}/{args.epochs - 1}][{i}/{len(dataloader) - 1}] "
            f"Loss_D: {(loss_D_A + loss_D_B).item():.4f} "
            f"Loss_G: {errG.item():.4f} "
            f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.4f} "
            f"loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.4f} "
            f"loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB).item():.4f}")

        if i % args.print_freq == 0:
            # Save Losses for plotting later
            g_losses.append(errG.item())
            d_losses.append((loss_D_A + loss_D_B).item())

            identity_losses.append((loss_identity_A + loss_identity_B).item())
            gan_losses.append((loss_GAN_A2B + loss_GAN_B2A).item())
            cycle_losses.append((loss_cycle_ABA + loss_cycle_BAB).item())

            vutils.save_image(real_image_A,
                              f"{args.outf}/{args.name}/A/real_samples.png",
                              normalize=True)
            vutils.save_image(real_image_B,
                              f"{args.outf}/{args.name}/B/real_samples.png",
                              normalize=True)

            fake_image_A = 0.5 * (netG_B2A(real_image_B).data + 1.0)
            fake_image_B = 0.5 * (netG_A2B(real_image_A).data + 1.0)

            vutils.save_image(fake_image_A.detach(),
                              f"{args.outf}/{args.name}/A/fake_samples_epoch_{epoch}.png",
                              normalize=True)
            vutils.save_image(fake_image_B.detach(),
                              f"{args.outf}/{args.name}/B/fake_samples_epoch_{epoch}.png",
                              normalize=True)

        # do check pointing
        torch.save(netG_A2B.state_dict(), f"weights/{args.name}/netG_A2B_epoch_{epoch}.pth")
        torch.save(netG_B2A.state_dict(), f"weights/{args.name}/netG_B2A_epoch_{epoch}.pth")
        torch.save(netD_A.state_dict(), f"weights/{args.name}/netD_A_epoch_{epoch}.pth")
        torch.save(netD_B.state_dict(), f"weights/{args.name}/netD_B_epoch_{epoch}.pth")

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

    plt.figure(figsize=(20, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G_Loss")
    plt.plot(d_losses, label="D_Loss")
    plt.plot(identity_losses, label="Identity_Loss")
    plt.plot(gan_losses, label="Gan_Loss")
    plt.plot(cycle_losses, label="Cycle_Loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("result.png")
