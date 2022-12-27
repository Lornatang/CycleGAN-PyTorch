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
d_model_arch_name = "path_discriminator"
g_model_arch_name = "cyclenet"
# Experiment name, easy to save weights and log files
exp_name = "CycleGAN-apple2orange"

# Dataset config for training
src_image_path = f"./data/apple2orange/trainA"
dst_image_path = f"./data/apple2orange/trainB"
unpaired = True

# Crop image for PathDiscriminator
resized_image_size = 256
crop_image_size = 70
batch_size = 1
num_workers = 4

# Load the address of the pre-trained model
load_pretrained = False
pretrained_d_A_model_weights_path = f""
pretrained_d_B_model_weights_path = f""
pretrained_g_A2B_model_weights_path = f""
pretrained_g_B2A_model_weights_path = f""

# Define this parameter when training is interrupted or migrated
load_resume = False
resume_d_A_model_weights_path = f""
resume_d_B_model_weights_path = f""
resume_g_A2B_model_weights_path = f""
resume_g_B2A_model_weights_path = f""

# Total num epochs
epochs = 200

# Loss weight
identity_weight = [5.0]
cycle_weight = [10.0]
adversarial_weight = [1.0]

# Optimizer parameter
optim_lr = 2e-4
optim_betas = (0.5, 0.999)
optim_eps = 1e-4
optim_weight_decay = 0.0

# Dynamic learning rate
decay_epochs = epochs // 2

# EMA moving average model parameters
model_ema_decay = 0.999

# How many iterations to print the training result
print_frequency = 100
