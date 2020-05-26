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
import glob
import os
import random
import time
from threading import Thread

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, unaligned=False, mode="train"):
        self.transform = transform
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}/A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}/B") + "/*.*"))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class VideoDataset:
    """ For reading camera or network data

    Load data types from data flow.

    Args:
        dataroot (str): Data flow file name.
        image_size (int): Image size in default data flow. (default:``416``).
    """

    def __init__(self, dataroot, image_size=416):

        self.mode = "images"
        self.image_size = image_size

        sources = [dataroot]

        n = len(sources)
        self.images = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f"{i + 1}/{n}: {s}... ", end="")

            capture = cv2.VideoCapture(0 if s == "0" else s)
            assert capture.isOpened(), f"Failed to open {s}"

            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS) % 100
            _, self.images[i] = capture.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, capture]), daemon=True)
            print(f"Success ({width}*{height} at {fps:.2f}FPS).")
            thread.start()
        print("")

    def update(self, index, capture):
        # Read next stream frame in a daemon thread
        num = 0
        while capture.isOpened():
            num += 1
            # Grabs the next frame from video file or capturing device.
            capture.grab()
            # read every 4th frame
            if num == 4:
                _, self.images[index] = capture.retrieve()
                num = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        raw_image = self.images.copy()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        image = [x for x in raw_image]

        # Stack
        image = np.stack(image, 0)

        # BGR convert to RGB (batch_size 3 x 416 x 416)
        image = image[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        # Return a contiguous array
        image = np.ascontiguousarray(image)

        return image, raw_image

    def __len__(self):
        return 0
