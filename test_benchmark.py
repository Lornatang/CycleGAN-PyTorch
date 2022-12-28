# Copyright 2022 Lorna Authors. All Rights Reserved.
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
from glob import glob

import torch
from torchvision.utils import save_image
from natsort import natsorted
import model
from imgproc import preprocess_one_image
from utils import make_directory, load_pretrained_state_dict


def main(args):
    # Create result image dir
    benchmark_dir = f"./results/benchmark/{os.path.basename(args.model_weights_dir)}/{args.model_type}"
    make_directory(benchmark_dir)

    device = choice_device(args.device)
    g_model = model.__dict__[args.model_arch_name]()
    g_model = g_model.to(device)

    # Load image
    image = preprocess_one_image(args.image_path, True, False, device)

    # Load model weights
    model_weights_list = natsorted(glob(f"{args.model_weights_dir}/{args.model_type}*"))
    for model_weights in model_weights_list:
        print(f"Process `{model_weights}`...")
        g_model = load_pretrained_state_dict(g_model, model_weights)
        g_model.eval()

        with torch.no_grad():
            gen_image = g_model(image)
            save_image(gen_image.detach(),
                       f"{benchmark_dir}/{os.path.basename(model_weights)[:-8]}_{os.path.basename(args.image_path)}",
                       normalize=True)


def choice_device(device: str = "cpu") -> torch.device:
    # Select model processing equipment type
    if device == "cuda":
        device = torch.device("cuda", 0)
    elif device[:4] == "cuda":
        device = torch.device(device)
    else:
        device = torch.device("cpu")

    return device


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="./figure/apple.jpg",
                        help="Input image path. Default: ``./figure/apple.jpg``")
    parser.add_argument("--model_arch_name", type=str, default="cyclenet",
                        help="Generator arch model name.  Default: ``cyclenet``")
    parser.add_argument("--model_weights_dir", type=str, default="./samples/CycleGAN-apple2orange",
                        help="Generator model weights dir path.  Default: ``./samples/CycleGAN-apple2orange``")
    parser.add_argument("--model_type", type=str, default="g_A2B", choices=["g_A2B", "g_B2A"],
                        help="Generator model dir path.  Default: ``g_A2B``")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device. Default: ``cuda``.")
    args = parser.parse_args()

    main(args)
