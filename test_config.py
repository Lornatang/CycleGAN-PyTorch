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
# Set program device
device = "cuda:0"
seed = 0
# Model arch name
model_arch_name = "yolov3_tiny_voc"
# Set to True for faster training/testing
cache_images = False
# Set to True if the label is rectangular
test_rect_label = False
# Set to True if there is only 1 detection classes
single_classes = False
# If use grayscale image
gray = False
# Export ONNX model
onnx_export = False
# For test
conf_threshold = 0.001
iou_threshold = 0.5
save_json = False
test_augment = False
verbose = True

test_dataset_config_path = f"./data/voc.data"
test_image_size = 416
batch_size = 64
num_workers = 4
model_weights_path = f"./results/pretrained_models/YOLOv3_tiny-VOC0712-d24f2c25.pth.tar"
