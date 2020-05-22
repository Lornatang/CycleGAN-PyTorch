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

import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from cyclegan_pytorch import Generator

parser = argparse.ArgumentParser(
    description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
parser.add_argument("--file", type=str, help="Video name.")
parser.add_argument("--model-name", type=str, default="horse2zebra",
                    help="dataset name.  (default:`horse2zebra`)"
                         "Option: [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, "
                         "cezanne2photo, ukiyoe2photo, vangogh2photo, selfie2anime]")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--image-size", type=int, default=256,
                    help="size of the data crop (squared assumed). (default:256)")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

args = parser.parse_args()
print(args)

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
model = Generator().to(device)

# Load state dicts
model.load_state_dict(torch.load(os.path.join("weights", str(args.model_name), "netG_A2B.pth")))

# Set model mode
model.eval()

# Load video
videoCapture = cv2.VideoCapture(args.file)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
frame_numbers = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
video_size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
compared_video_size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * 2 + 10),
                       int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) + 10 + int(
                           int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * 2 + 10) / int(
                               10 * int(int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)) // 5 + 1)) * int(
                               int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)) // 5 - 9)))
output_video_name = "out_" + args.file.split(".")[0] + ".mp4"
output_compared_name = "compare_" + args.file.split(".")[0] + ".mp4"
sr_video_writer = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc("M", "P", "E", "G"), fps, video_size)
compared_video_writer = cv2.VideoWriter(output_compared_name, cv2.VideoWriter_fourcc("M", "P", "E", "G"), fps,
                                        compared_video_size)

pre_process = transforms.Compose([transforms.Resize(args.image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# read frame
success, frame = videoCapture.read()
test_bar = tqdm(range(int(frame_numbers)), desc="[processing video and saving result videos]")
for index in test_bar:
    if success:
        image = pre_process(frame).unsqueeze(0)

        image = image.to(device)

        out = model(image)
        out = out.cpu()
        out_image = out.data[0].numpy()
        out_image *= 255.0
        out_image = (np.uint8(out_image)).transpose((1, 2, 0))
        # save sr video
        sr_video_writer.write(out_image)

        # make compared video and crop shot of left top\right top\center\left bottom\right bottom
        out_image = ToPILImage()(out_image)
        crop_out_images = transforms.FiveCrop(size=out_image.width // 5 - 9)(out_image)
        crop_out_images = [np.asarray(transforms.Pad(padding=(10, 5, 0, 0))(img)) for img in crop_out_images]
        out_image = transforms.Pad(padding=(5, 0, 0, 5))(out_image)
        compared_image = transforms.Resize(size=(video_size[1], video_size[0]), interpolation=Image.BICUBIC)(
            ToPILImage()(frame))
        crop_compared_images = transforms.FiveCrop(size=compared_image.width // 5 - 9)(compared_image)
        crop_compared_images = [np.asarray(transforms.Pad(padding=(0, 5, 10, 0))(img)) for img in crop_compared_images]
        compared_image = transforms.Pad(padding=(0, 0, 5, 5))(compared_image)
        # concatenate all the pictures to one single picture
        top_image = np.concatenate((np.asarray(compared_image), np.asarray(out_image)), axis=1)
        bottom_image = np.concatenate(crop_compared_images + crop_out_images, axis=1)
        bottom_image = np.asarray(transforms.Resize(
            size=(int(top_image.shape[1] / bottom_image.shape[1] * bottom_image.shape[0]), top_image.shape[1]))(
            ToPILImage()(bottom_image)))
        final_image = np.concatenate((top_image, bottom_image))
        # save compared video
        compared_video_writer.write(final_image)
        # next frame
        success, frame = videoCapture.read()
