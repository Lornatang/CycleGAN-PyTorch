# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import os
import queue
import random
import threading

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from imgproc import image_to_tensor, random_rotate, random_horizontally_flip, random_vertically_flip

__all__ = [
    "ImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]


class ImageDataset(Dataset):
    def __init__(
            self,
            src_images_dir: str,
            dst_images_dir: str,
            unpaired: bool,
            resized_image_size: int,
    ) -> None:
        super(ImageDataset, self).__init__()
        self.src_image_file_names = [os.path.join(src_images_dir, image_file_name) for image_file_name in
                                     os.listdir(src_images_dir)]
        self.dst_image_file_names = [os.path.join(dst_images_dir, image_file_name) for image_file_name in
                                     os.listdir(dst_images_dir)]
        self.unpaired = unpaired
        self.resized_image_size = resized_image_size

    def __getitem__(self, batch_index: int) -> [dict[str, Tensor], dict[str, Tensor]]:
        # Read a batch of image data
        src_image = cv2.imread(self.src_image_file_names[batch_index])
        if self.unpaired:
            dst_image = cv2.imread(self.dst_image_file_names[random.randint(0, len(self.src_image_file_names) - 1)])
        else:
            dst_image = cv2.imread(self.dst_image_file_names[batch_index])

        # Normalize the image data
        src_image = src_image.astype(np.float32) / 255.
        dst_image = dst_image.astype(np.float32) / 255.

        # Resized image
        src_image = cv2.resize(src_image, (self.resized_image_size, self.resized_image_size), interpolation=cv2.INTER_CUBIC)
        src_image = cv2.resize(src_image, (self.resized_image_size, self.resized_image_size), interpolation=cv2.INTER_CUBIC)

        # Image processing operations
        src_image = random_rotate(src_image, [90, 180, 270])
        src_image = random_horizontally_flip(src_image, 0.5)
        src_image = random_vertically_flip(src_image, 0.5)
        dst_image = random_rotate(dst_image, [90, 180, 270])
        dst_image = random_horizontally_flip(dst_image, 0.5)
        dst_image = random_vertically_flip(dst_image, 0.5)

        # BGR convert RGB
        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        dst_image = cv2.cvtColor(dst_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        src_tensor = image_to_tensor(src_image, False, False)
        dst_tensor = image_to_tensor(dst_image, False, False)

        return {"src": src_tensor, "dst": dst_tensor}

    def __len__(self) -> int:
        return len(self.src_image_file_names)


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
