import os
import cog
import tempfile
import argparse
import random
import timeit
from pathlib import Path
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from cyclegan_pytorch import Generator

class CycleganPredictor(cog.Predictor):
    def setup(self):
        """Load the CycleGan pre-trained model"""
        model_name = "weights/horse2zebra/netG_A2B.pth"
        cudnn.benchmark = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # create model
        self.model = Generator().to(self.device)
        # Load state dicts
        self.model.load_state_dict(torch.load(model_name, map_location=self.device))
        # Set model mode
        self.model.eval()

    @cog.input("input", type=Path, help="Content image")
    @cog.input("image_size", type=int, default=256,
                help="size of the data crop (squared assumed)")

    def predict(self, input, image_size):
        """Separate the vocal track from an audio mixture"""
        #compute prediction

        # Load image and pre-process
        output_path = Path(tempfile.mkdtemp()) / "output.png"
        image = Image.open(str(input))
        pre_process = transforms.Compose([transforms.Resize(image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                          ])
        image = pre_process(image).unsqueeze(0)
        image = image.to(self.device)

        #compute prediction
        start = timeit.default_timer()
        fake_image = self.model(image)
        elapsed = (timeit.default_timer() - start)
        print(f"cost {elapsed:.4f}s")
        #save results
        vutils.save_image(fake_image.detach(), str(output_path), normalize=True)

        return output_path
