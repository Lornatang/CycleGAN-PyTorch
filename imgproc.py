# Copyright 2023 Dakewe Biotech Corporation. All Rights Reserved.
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
import random
from typing import Any

import cv2
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torchvision.transforms import functional as F_vision

__all__ = [
    "image_to_tensor", "tensor_to_image",
    "preprocess_one_image",
    "center_crop_torch", "random_crop_torch", "random_rotate_torch", "random_vertically_flip_torch",
    "random_horizontally_flip_torch",
]


def image_to_tensor(image: ndarray, range_norm: bool, half: bool) -> Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("dst_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, range_norm=True, half=False)

    """
    # Convert image data type to Tensor data type
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor


def tensor_to_image(tensor: Tensor, range_norm: bool, half: bool) -> Any:
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_image = cv2.imread("dst_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, range_norm=False, half=False)

    """
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image


def preprocess_one_image(image_path: str, range_norm: bool, half: bool, device: torch.device) -> Tensor:
    # read an image using OpenCV
    image = cv2.imread(image_path).astype(np.float32) / 255.0

    # BGR image channel data to RGB image channel data
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert RGB image channel data to image formats supported by PyTorch
    tensor = image_to_tensor(image, range_norm, half).unsqueeze_(0)

    # Data transfer to the specified device
    tensor = tensor.to(device, non_blocking=True)

    return tensor


def center_crop_torch(
        src_images: ndarray | Tensor | list[ndarray] | list[Tensor],
        dst_images: ndarray | Tensor | list[ndarray] | list[Tensor],
        patch_size: int,
) -> [ndarray, ndarray] or [Tensor, Tensor] or [list[ndarray], list[ndarray]] or [list[Tensor], list[Tensor]]:
    """Intercept two images to specify the center area

    Args:
        src_images (ndarray | Tensor | list[ndarray] | list[Tensor]): Source image read by PyTorch
        dst_images (ndarray | Tensor | list[ndarray] | list[Tensor]): Destination image read by PyTorch
        patch_size (int): The size of the intercepted image

    Returns:
        src_images (ndarray or Tensor or): the intercepted ground truth image
        dst_images (ndarray or Tensor or): low-resolution intercepted images
    """

    if src_images.shape[2] != dst_images.shape[2]:
        raise ValueError("The height of the source image and the destination image must be the same")
    if src_images.shape[3] != dst_images.shape[3]:
        raise ValueError("The width of the source image and the destination image must be the same")

    if not isinstance(src_images, list):
        src_images = [src_images]
    if not isinstance(dst_images, list):
        dst_images = [dst_images]

    # detect input image type
    input_type = "Tensor" if torch.is_tensor(src_images[0]) else "Numpy"

    if input_type == "Tensor":
        image_height, image_width = src_images[0].size()[-2:]
    else:
        image_height, image_width = src_images[0].shape[0:2]

    # Just need to find the top and left coordinates of the image
    top = (image_height - patch_size) // 2
    left = (image_width - patch_size) // 2

    # Capture low-resolution images
    if input_type == "Tensor":
        src_images = [src_image[
                      :,
                      :,
                      top: top + patch_size,
                      left: left + patch_size] for src_image in src_images]
        dst_images = [dst_image[
                     :,
                     :,
                     top: top + patch_size,
                     left: left + patch_size] for dst_image in dst_images]
    else:
        src_images = [src_image[
                      :,
                      :,
                      top: top + patch_size,
                      left: left + patch_size] for src_image in src_images]
        dst_images = [dst_image[
                     top: top + patch_size,
                     left: left + patch_size,
                     ...] for dst_image in dst_images]

    # When the input has only one image
    if len(src_images) == 1:
        src_images = src_images[0]
    if len(dst_images) == 1:
        dst_images = dst_images[0]

    return src_images, dst_images


def random_crop_torch(
        src_images: ndarray | Tensor | list[ndarray] | list[Tensor],
        dst_images: ndarray | Tensor | list[ndarray] | list[Tensor],
        patch_size: int,
) -> [ndarray, ndarray] or [Tensor, Tensor] or [list[ndarray], list[ndarray]] or [list[Tensor], list[Tensor]]:
    """Randomly intercept two images in the specified area

    Args:
        src_images (ndarray | Tensor | list[ndarray] | list[Tensor]): Source image read by PyTorch
        dst_images (ndarray | Tensor | list[ndarray] | list[Tensor]): Destination image read by PyTorch
        patch_size (int): The size of the intercepted image

    Returns:
        src_images (ndarray or Tensor or): the intercepted ground truth image
        dst_images (ndarray or Tensor or): low-resolution intercepted images

    """

    if src_images.shape[2] != dst_images.shape[2]:
        raise ValueError("The height of the source image and the destination image must be the same")
    if src_images.shape[3] != dst_images.shape[3]:
        raise ValueError("The width of the source image and the destination image must be the same")

    if not isinstance(src_images, list):
        src_images = [src_images]
    if not isinstance(dst_images, list):
        dst_images = [dst_images]

    # detect input image type
    input_type = "Tensor" if torch.is_tensor(src_images[0]) else "Numpy"

    if input_type == "Tensor":
        image_height, image_width = src_images[0].size()[-2:]
    else:
        image_height, image_width = src_images[0].shape[0:2]

    # Just need to find the top and left coordinates of the image
    top = random.randint(0, image_height - patch_size)
    left = random.randint(0, image_width - patch_size)

    # Capture low-resolution images
    if input_type == "Tensor":
        src_images = [src_image[
                     :,
                     :,
                     top: top + patch_size,
                     left: left + patch_size] for src_image in src_images]
        dst_images = [dst_image[
                     :,
                     :,
                     top: top + patch_size,
                     left: left + patch_size] for dst_image in dst_images]
    else:
        src_images = [src_image[
                     :,
                     :,
                     top: top + patch_size,
                     left: left + patch_size] for src_image in src_images]
        dst_images = [dst_image[
                     :,
                     :,
                     top: top + patch_size,
                     left: left + patch_size] for dst_image in dst_images]

    # When the input has only one image
    if len(src_images) == 1:
        src_images = src_images[0]
    if len(dst_images) == 1:
        dst_images = dst_images[0]

    return src_images, dst_images


def random_rotate_torch(
        src_images: ndarray | Tensor | list[ndarray] | list[Tensor],
        dst_images: ndarray | Tensor | list[ndarray] | list[Tensor],
        angles: list,
        center: tuple = None,
        rotate_scale_factor: float = 1.0
) -> [ndarray, ndarray] or [Tensor, Tensor] or [list[ndarray], list[ndarray]] or [list[Tensor], list[Tensor]]:
    """Randomly rotate the image

    Args:
        src_images (ndarray | Tensor | list[ndarray] | list[Tensor]): ground truth images read by the PyTorch library
        dst_images (ndarray | Tensor | list[ndarray] | list[Tensor]): low-resolution images read by the PyTorch library
        angles (list): List of random rotation angles
        center (optional, tuple[int, int]): Rotation center. Default: None
        rotate_scale_factor (optional, float): Rotation scaling factor. Default: 1.0

    Returns:
        src_images (ndarray or Tensor or): ground truth image after rotation
        dst_images (ndarray or Tensor or): Rotated low-resolution images
    """

    if src_images.shape[2] != dst_images.shape[2]:
        raise ValueError("The height of the source image and the destination image must be the same")
    if src_images.shape[3] != dst_images.shape[3]:
        raise ValueError("The width of the source image and the destination image must be the same")

    # Randomly choose the rotation angle
    angle = random.choice(angles)

    if not isinstance(src_images, list):
        src_images = [src_images]
    if not isinstance(dst_images, list):
        dst_images = [dst_images]

    # detect input image type
    input_type = "Tensor" if torch.is_tensor(src_images[0]) else "Numpy"

    if input_type == "Tensor":
        image_height, image_width = src_images[0].size()[-2:]
    else:
        image_height, image_width = src_images[0].shape[0:2]

    # Rotate all images
    if center is None:
        center = [image_width // 2, image_height // 2]

    matrix = cv2.getRotationMatrix2D(center, angle, rotate_scale_factor)

    if input_type == "Tensor":
        src_images = [F_vision.rotate(src_image, angle, center=center) for src_image in src_images]
        dst_images = [F_vision.rotate(dst_image, angle, center=center) for dst_image in dst_images]
    else:
        src_images = [cv2.warpAffine(src_image, matrix, (image_width, image_height)) for src_image in src_images]
        dst_images = [cv2.warpAffine(dst_image, matrix, (image_width, image_height)) for dst_image in dst_images]

    # When the input has only one image
    if len(src_images) == 1:
        src_images = src_images[0]
    if len(dst_images) == 1:
        dst_images = dst_images[0]

    return src_images, dst_images


def random_horizontally_flip_torch(
        src_images: ndarray | Tensor | list[ndarray] | list[Tensor],
        dst_images: ndarray | Tensor | list[ndarray] | list[Tensor],
        p: float = 0.5
) -> [ndarray, ndarray] or [Tensor, Tensor] or [list[ndarray], list[ndarray]] or [list[Tensor], list[Tensor]]:
    """Randomly flip the image up and down

    Args:
        src_images (ndarray): ground truth images read by the PyTorch library
        dst_images (ndarray): low resolution images read by the PyTorch library
        p (optional, float): flip probability. Default: 0.5

    Returns:
        src_images (ndarray or Tensor or): flipped ground truth images
        dst_images (ndarray or Tensor or): flipped low-resolution images
    """

    if src_images.shape[2] != dst_images.shape[2]:
        raise ValueError("The height of the source image and the destination image must be the same")
    if src_images.shape[3] != dst_images.shape[3]:
        raise ValueError("The width of the source image and the destination image must be the same")

    # Randomly generate flip probability
    flip_prob = random.random()

    if not isinstance(src_images, list):
        src_images = [src_images]
    if not isinstance(dst_images, list):
        dst_images = [dst_images]

    # detect input image type
    input_type = "Tensor" if torch.is_tensor(src_images[0]) else "Numpy"

    if flip_prob > p:
        if input_type == "Tensor":
            src_images = [F_vision.hflip(src_image) for src_image in src_images]
            dst_images = [F_vision.hflip(dst_image) for dst_image in dst_images]
        else:
            src_images = [cv2.flip(src_image, 1) for src_image in src_images]
            dst_images = [cv2.flip(dst_image, 1) for dst_image in dst_images]

    # When the input has only one image
    if len(src_images) == 1:
        src_images = src_images[0]
    if len(dst_images) == 1:
        dst_images = dst_images[0]

    return src_images, dst_images


def random_vertically_flip_torch(
        src_images: ndarray | Tensor | list[ndarray] | list[Tensor],
        dst_images: ndarray | Tensor | list[ndarray] | list[Tensor],
        p: float = 0.5
) -> [ndarray, ndarray] or [Tensor, Tensor] or [list[ndarray], list[ndarray]] or [list[Tensor], list[Tensor]]:
    """Randomly flip the image left and right

    Args:
        src_images (ndarray): ground truth images read by the PyTorch library
        dst_images (ndarray): low resolution images read by the PyTorch library
        p (optional, float): flip probability. Default: 0.5

    Returns:
        src_images (ndarray or Tensor or): flipped ground truth images
        dst_images (ndarray or Tensor or): flipped low-resolution images
    """

    if src_images.shape[2] != dst_images.shape[2]:
        raise ValueError("The height of the source image and the destination image must be the same")
    if src_images.shape[3] != dst_images.shape[3]:
        raise ValueError("The width of the source image and the destination image must be the same")

    # Randomly generate flip probability
    flip_prob = random.random()

    if not isinstance(src_images, list):
        src_images = [src_images]
    if not isinstance(dst_images, list):
        dst_images = [dst_images]

    # detect input image type
    input_type = "Tensor" if torch.is_tensor(src_images[0]) else "Numpy"

    if flip_prob > p:
        if input_type == "Tensor":
            src_images = [F_vision.vflip(src_image) for src_image in src_images]
            dst_images = [F_vision.vflip(dst_image) for dst_image in dst_images]
        else:
            src_images = [cv2.flip(src_image, 0) for src_image in src_images]
            dst_images = [cv2.flip(dst_image, 0) for dst_image in dst_images]

    # When the input has only one image
    if len(src_images) == 1:
        src_images = src_images[0]
    if len(dst_images) == 1:
        dst_images = dst_images[0]

    return src_images, dst_images
