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
import os
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from cyclegan_pytorch import Generator
from cyclegan_pytorch import ImageDataset

parser = argparse.ArgumentParser(description="PyTorch CycleGAN")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="path to datasets. (default:./data)")
parser.add_argument("name", type=str,
                    help="dataset name. "
                         "Option: [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, "
                         "cezanne2photo, ukiyoe2photo, vangogh2photo, maps, facades, "
                         "iphone2dslr_flower]")
parser.add_argument("-j", "--workers", default=8, type=int, metavar="N",
                    help="number of data loading workers (default:8)")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--outf", default="./gen",
                    help="folder to output images. (default: `./gen`).")
parser.add_argument("--image-size", type=int, default=256,
                    help="size of the data crop (squared assumed). (default:256)")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

valid_dataset_name = ["apple2orange", "summer2winter_yosemite", "horse2zebra",
                      "monet2photo", "cezanne2photo", "ukiyoe2photo", "vangogh2photo",
                      "maps, facades", "iphone2dslr_flower"]

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
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

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
netG_A2B = Generator(3, 3).to(device)
netG_B2A = Generator(3, 3).to(device)

# Load state dicts
netG_A2B.load_state_dict(torch.load(os.path.join("weights", str(args.name), "netG_A2B.pth")))
netG_B2A.load_state_dict(torch.load(os.path.join("weights", str(args.name), "netG_B2A.pth")))

# Set model mode
netG_A2B.eval()
netG_B2A.eval()

progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

for i, data in progress_bar:
    # get batch size data
    real_images_A = data["A"].to(device)
    real_images_B = data["B"].to(device)

    # Generate output
    fake_A = 0.5 * (netG_B2A(real_images_B).data + 1.0)
    fake_B = 0.5 * (netG_A2B(real_images_A).data + 1.0)

    # Save image files
    vutils.save_image(fake_A, f"gen/{args.name}/A/{i + 1:04d}.png", normalize=True)
    vutils.save_image(fake_B, f"gen/{args.name}/B/{i + 1:04d}.png", normalize=True)

    progress_bar.set_description(f"Generated images {i + 1} of {len(dataloader)}")
