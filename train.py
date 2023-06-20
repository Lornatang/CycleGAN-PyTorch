# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
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
import time
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import model
from dataset import CUDAPrefetcher, ImageDataset
from imgproc import random_crop_torch, random_rotate_torch, random_vertically_flip_torch, random_horizontally_flip_torch
from utils import load_pretrained_state_dict, load_resume_state_dict, make_directory, save_checkpoint, DecayLR, \
    ReplayBuffer, Summary, AverageMeter, ProgressMeter


def main():
    # Read parameters from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./configs/CYCLEGAN.yaml",
                        help="Path to train config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    # Fixed random number seed
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed_all(config["SEED"])

    # Because the size of the input image is fixed, the fixed CUDNN convolution method can greatly increase the running speed
    cudnn.benchmark = True

    # Initialize the mixed precision method
    scaler = amp.GradScaler()

    # Default to start training from scratch
    start_epoch = 0

    # Define the running device number
    device = torch.device("cuda", config["DEVICE_ID"])

    train_data_prefetcher = load_datasets(config, device)
    g_A_model, g_B_model, ema_g_A_model, ema_g_B_model, d_A_model, d_B_model = build_model(config, device)
    identity_criterion, adversarial_criterion, cycle_criterion = define_loss(config, device)
    g_optimizer, d_optimizer = define_optimizer(g_A_model,
                                                g_B_model,
                                                d_A_model,
                                                d_B_model,
                                                config)
    g_scheduler, d_scheduler = define_scheduler(g_optimizer,
                                                d_optimizer,
                                                config)

    # Load the pre-trained model weights and fine-tune the model
    print("Check whether to load pretrained model weights...")
    if config["TRAIN"]["CHECKPOINT"]["LOAD_PRETRAINED"]:
        g_A_model = load_pretrained_state_dict(g_A_model,
                                               False,
                                               config["TRAIN"]["CHECKPOINT"]["PRETRAINED_G_A_MODEL_WEIGHTS_PATH"])
        g_B_model = load_pretrained_state_dict(g_B_model,
                                               False,
                                               config["TRAIN"]["CHECKPOINT"]["PRETRAINED_G_B_MODEL_WEIGHTS_PATH"])
        d_A_model = load_pretrained_state_dict(d_A_model,
                                               False,
                                               config["TRAIN"]["CHECKPOINT"]["PRETRAINED_D_A_MODEL_WEIGHTS_PATH"])
        d_B_model = load_pretrained_state_dict(d_B_model,
                                               False,
                                               config["TRAIN"]["CHECKPOINT"]["PRETRAINED_D_B_MODEL_WEIGHTS_PATH"])
        print(f"Loaded pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    # Load the last training interruption node
    print("Check whether the resume model is restored...")
    if config["TRAIN"]["CHECKPOINT"]["LOAD_RESUME"]:
        g_A_model, ema_g_A_model, start_epoch, g_optimizer, g_scheduler = load_resume_state_dict(
            g_A_model,
            ema_g_A_model,
            g_optimizer,
            g_scheduler,
            config["MODEL"]["G"]["COMPILED"],
            config["TRAIN"]["CHECKPOINT"]["RESUME_G_A_MODEL_WEIGHTS_PATH"],
        )
        g_B_model, ema_g_B_model, start_epoch, g_optimizer, g_scheduler = load_resume_state_dict(
            g_B_model,
            ema_g_B_model,
            g_optimizer,
            g_scheduler,
            config["MODEL"]["G"]["COMPILED"],
            config["TRAIN"]["CHECKPOINT"]["RESUME_G_B_MODEL_WEIGHTS_PATH"],
        )
        d_A_model, _, start_epoch, d_A_optimizer, d_A_scheduler = load_resume_state_dict(
            d_A_model,
            None,
            d_optimizer,
            d_scheduler,
            config["MODEL"]["D"]["COMPILED"],
            config["TRAIN"]["CHECKPOINT"]["RESUME_D_A_MODEL_WEIGHTS_PATH"],
        )
        d_B_model, _, start_epoch, d_B_optimizer, d_B_scheduler = load_resume_state_dict(
            d_B_model,
            None,
            d_optimizer,
            d_scheduler,
            config["MODEL"]["D"]["COMPILED"],
            config["TRAIN"]["CHECKPOINT"]["RESUME_D_B_MODEL_WEIGHTS_PATH"],
        )
        print(f"Loaded resume model weights successfully.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", config["EXP_NAME"])
    results_dir = os.path.join("results", config["EXP_NAME"])
    make_directory(samples_dir)
    make_directory(results_dir)
    make_directory(os.path.join(samples_dir, "A"))
    make_directory(os.path.join(samples_dir, "B"))

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config["EXP_NAME"]))

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    for epoch in range(start_epoch, config["TRAIN"]["HYP"]["EPOCHS"]):
        train(g_A_model,
              g_B_model,
              ema_g_A_model,
              ema_g_B_model,
              d_A_model,
              d_B_model,
              train_data_prefetcher,
              identity_criterion,
              adversarial_criterion,
              cycle_criterion,
              g_optimizer,
              d_optimizer,
              fake_A_buffer,
              fake_B_buffer,
              epoch,
              scaler,
              writer,
              device,
              config)
        print("\n")

        # Update LR
        g_scheduler.step()
        d_scheduler.step()

        is_last = (epoch + 1) == config["TRAIN"]["HYP"]["EPOCHS"]
        save_checkpoint({"epoch": epoch + 1,
                         "state_dict": g_A_model.state_dict(),
                         "ema_state_dict": ema_g_A_model.state_dict(),
                         "optimizer": g_optimizer.state_dict(),
                         "scheduler": g_scheduler.state_dict()},
                        f"g_A_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "g_A_best.pth.tar",
                        "g_A_last.pth.tar",
                        True,
                        is_last)
        save_checkpoint({"epoch": epoch + 1,
                         "state_dict": g_B_model.state_dict(),
                         "ema_state_dict": ema_g_B_model.state_dict(),
                         "optimizer": g_optimizer.state_dict(),
                         "scheduler": g_scheduler.state_dict()},
                        f"g_B_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "g_B_best.pth.tar",
                        "g_B_last.pth.tar",
                        True,
                        is_last)
        save_checkpoint({"epoch": epoch + 1,
                         "state_dict": d_A_model.state_dict(),
                         "optimizer": d_optimizer.state_dict(),
                         "scheduler": d_scheduler.state_dict()},
                        f"d_A_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "d_A_best.pth.tar",
                        "d_A_last.pth.tar",
                        True,
                        is_last)
        save_checkpoint({"epoch": epoch + 1,
                         "state_dict": d_B_model.state_dict(),
                         "optimizer": d_optimizer.state_dict(),
                         "scheduler": d_scheduler.state_dict()},
                        f"d_B_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "d_B_best.pth.tar",
                        "d_B_last.pth.tar",
                        True,
                        is_last)


def load_datasets(
        config: Any,
        device: torch.device,
) -> CUDAPrefetcher:
    # Load dataset
    train_datasets = ImageDataset(
        config["TRAIN"]["DATASET"]["SRC_IMAGE_PATH"],
        config["TRAIN"]["DATASET"]["DST_IMAGE_PATH"],
        config["TRAIN"]["DATASET"]["UNPAIRED"],
        config["TRAIN"]["DATASET"]["IMAGE_SIZE"],
    )
    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
                                  shuffle=config["TRAIN"]["HYP"]["SHUFFLE"],
                                  num_workers=config["TRAIN"]["HYP"]["NUM_WORKERS"],
                                  pin_memory=config["TRAIN"]["HYP"]["PIN_MEMORY"],
                                  drop_last=True,
                                  persistent_workers=config["TRAIN"]["HYP"]["PERSISTENT_WORKERS"])

    # Place all data on the preprocessing data loader
    train_data_prefetcher = CUDAPrefetcher(train_dataloader, device)

    return train_data_prefetcher


def build_model(
        config: Any,
        device: torch.device,
) -> [nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    g_A_model = model.__dict__[config["MODEL"]["G"]["NAME"]](in_channels=config["MODEL"]["G"]["IN_CHANNELS"],
                                                             out_channels=config["MODEL"]["G"]["OUT_CHANNELS"],
                                                             channels=config["MODEL"]["G"]["CHANNELS"])
    g_B_model = model.__dict__[config["MODEL"]["G"]["NAME"]](in_channels=config["MODEL"]["G"]["IN_CHANNELS"],
                                                             out_channels=config["MODEL"]["G"]["OUT_CHANNELS"],
                                                             channels=config["MODEL"]["G"]["CHANNELS"])
    d_A_model = model.__dict__[config["MODEL"]["D"]["NAME"]](in_channels=config["MODEL"]["D"]["IN_CHANNELS"],
                                                             out_channels=config["MODEL"]["D"]["OUT_CHANNELS"],
                                                             channels=config["MODEL"]["D"]["CHANNELS"])
    d_B_model = model.__dict__[config["MODEL"]["D"]["NAME"]](in_channels=config["MODEL"]["D"]["IN_CHANNELS"],
                                                             out_channels=config["MODEL"]["D"]["OUT_CHANNELS"],
                                                             channels=config["MODEL"]["D"]["CHANNELS"])
    # Create an Exponential Moving Average Model
    if config["MODEL"]["EMA"]["ENABLE"]:
        # Generate an exponential average model based on a generator to stabilize model training
        ema_decay = config["MODEL"]["EMA"]["DECAY"]
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
            (1 - ema_decay) * averaged_model_parameter + ema_decay * model_parameter
        ema_g_A_model = AveragedModel(g_A_model, device=device, avg_fn=ema_avg_fn)
        ema_g_B_model = AveragedModel(g_B_model, device=device, avg_fn=ema_avg_fn)
    else:
        ema_g_A_model = None
        ema_g_B_model = None

    g_A_model = g_A_model.to(device)
    g_B_model = g_B_model.to(device)
    ema_g_A_model = ema_g_A_model.to(device)
    ema_g_B_model = ema_g_B_model.to(device)
    d_A_model = d_A_model.to(device)
    d_B_model = d_B_model.to(device)

    return g_A_model, g_B_model, ema_g_A_model, ema_g_B_model, d_A_model, d_B_model


def define_loss(config: Any, device: torch.device) -> [nn.L1Loss, nn.MSELoss, nn.L1Loss]:
    if config["TRAIN"]["LOSSES"]["IDENTITY_LOSS"]["NAME"] == "l1":
        identity_criterion = nn.L1Loss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['IDENTITY_LOSS']['NAME']} is not implemented.")

    if config["TRAIN"]["LOSSES"]["ADVERSARIAL_LOSS"]["NAME"] == "lsgan":
        adversarial_criterion = nn.MSELoss()
    elif config["TRAIN"]["LOSSES"]["ADVERSARIAL_LOSS"]["NAME"] == "vanilla":
        adversarial_criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['ADVERSARIAL_LOSS']['NAME']} is not implemented.")

    if config["TRAIN"]["LOSSES"]["CYCLE_LOSS"]["NAME"] == "l1":
        cycle_criterion = nn.L1Loss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['CYCLE_LOSS']['NAME']} is not implemented.")

    identity_criterion = identity_criterion.to(device)
    adversarial_criterion = adversarial_criterion.to(device)
    cycle_criterion = cycle_criterion.to(device)

    return identity_criterion, cycle_criterion, adversarial_criterion


def define_optimizer(
        g_A_model: nn.Module,
        g_B_model: nn.Module,
        d_A_model: nn.Module,
        d_B_model: nn.Module,
        config: Any,
) -> [optim, optim]:
    if config["TRAIN"]["OPTIM"]["NAME"] == "Adam":
        g_optimizer = optim.Adam(itertools.chain(g_A_model.parameters(), g_B_model.parameters()),
                                 config["TRAIN"]["OPTIM"]["LR"],
                                 config["TRAIN"]["OPTIM"]["BETAS"],
                                 config["TRAIN"]["OPTIM"]["EPS"],
                                 config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"])
        d_optimizer = optim.Adam(itertools.chain(d_A_model.parameters(), d_B_model.parameters()),
                                 config["TRAIN"]["OPTIM"]["LR"],
                                 config["TRAIN"]["OPTIM"]["BETAS"],
                                 config["TRAIN"]["OPTIM"]["EPS"],
                                 config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"])
    else:
        raise NotImplementedError(f"Optimizer {config['TRAIN']['OPTIM']['NAME']} is not implemented.")

    return g_optimizer, d_optimizer


def define_scheduler(
        g_optimizer: optim.Adam,
        d_optimizer: optim.Adam,
        config: Any,
) -> [lr_scheduler, lr_scheduler]:
    if config["TRAIN"]["LR_SCHEDULER"]["NAME"] == "LambdaLR":
        lr_lambda = DecayLR(config["TRAIN"]["HYP"]["EPOCHS"], 0, config["TRAIN"]["LR_SCHEDULER"]["DECAY_EPOCHS"]).step
        g_scheduler = lr_scheduler.LambdaLR(g_optimizer, lr_lambda)
        d_scheduler = lr_scheduler.LambdaLR(d_optimizer, lr_lambda)
    else:
        raise NotImplementedError(f"Scheduler {config['TRAIN']['LR_SCHEDULER']['NAME']} is not implemented.")

    return g_scheduler, d_scheduler


def train(
        g_A_model: nn.Module,
        g_B_model: nn.Module,
        ema_g_A_model: nn.Module,
        ema_g_B_model: nn.Module,
        d_A_model: nn.Module,
        d_B_model: nn.Module,
        train_data_prefetcher: CUDAPrefetcher,
        identity_criterion: nn.L1Loss,
        adversarial_criterion: nn.MSELoss,
        cycle_criterion: nn.L1Loss,
        g_optimizer: optim.Adam,
        d_optimizer: optim.Adam,
        fake_A_buffer: ReplayBuffer,
        fake_B_buffer: ReplayBuffer,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device,
        config: Any,
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_data_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    d_losses = AverageMeter("D loss", ":6.6f", Summary.NONE)
    g_losses = AverageMeter("G loss", ":6.6f", Summary.NONE)

    progress = ProgressMeter(batches,
                             [batch_time, data_time, d_losses, g_losses],
                             f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    d_A_model.train()
    d_B_model.train()
    g_A_model.train()
    g_B_model.train()

    identity_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["IDENTITY_LOSS"]["WEIGHT"]).to(device)
    adversarial_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["ADVERSARIAL_LOSS"]["WEIGHT"]).to(device)
    cycle_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["CYCLE_LOSS"]["WEIGHT"]).to(device)

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_data_prefetcher.reset()
    batch_data = train_data_prefetcher.next()

    batch_size = batch_data["src"].size(0)

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        real_image_A = batch_data["src"].to(device, non_blocking=True)
        real_image_B = batch_data["dst"].to(device, non_blocking=True)

        # image data augmentation
        real_image_A, real_image_B = random_crop_torch(real_image_A,
                                                       real_image_B,
                                                       config["TRAIN"]["DATASET"]["IMAGE_SIZE"])
        real_image_A, real_image_B = random_rotate_torch(real_image_A, real_image_B, [0, 90, 180, 270])
        real_image_A, real_image_B = random_vertically_flip_torch(real_image_A, real_image_B)
        real_image_A, real_image_B = random_horizontally_flip_torch(real_image_A, real_image_B)

        ##############################################
        # (1) Update G network: Generators A2B and B2A
        ##############################################

        # Initialize generator gradients
        g_optimizer.zero_grad(set_to_none=True)

        # During generator model training, disable discriminator model backpropagation
        for d_parameters in d_A_model.parameters():
            d_parameters.requires_grad = False
        for d_parameters in d_B_model.parameters():
            d_parameters.requires_grad = False

        # Mixed precision training
        with amp.autocast():
            # Generator fake and cycle image
            fake_image_B = g_A_model(real_image_A)
            recovered_image_A = g_B_model(fake_image_B)
            fake_image_A = g_B_model(real_image_B)
            recovered_image_B = g_A_model(fake_image_A)

            # Identity loss
            identity_image_A = g_A_model(real_image_B)
            loss_identity_A = torch.sum(torch.mul(identity_weight, identity_criterion(identity_image_A, real_image_B)))
            identity_image_B = g_B_model(real_image_A)
            loss_identity_B = torch.sum(torch.mul(identity_weight, identity_criterion(identity_image_B, real_image_A)))

            # GAN loss
            fake_output_A = d_A_model(fake_image_B)
            real_label = torch.tensor(1).expand_as(fake_output_A).to(device, non_blocking=True)
            loss_adversarial_A = torch.sum(torch.mul(adversarial_weight, adversarial_criterion(fake_output_A, real_label)))
            fake_output_B = d_B_model(fake_image_A)
            real_label = torch.tensor(1).expand_as(fake_output_B).to(device, non_blocking=True)
            loss_adversarial_B = torch.sum(torch.mul(adversarial_weight, adversarial_criterion(fake_output_B, real_label)))

            # Cycle loss
            loss_cycle_A = torch.sum(torch.mul(cycle_weight, cycle_criterion(recovered_image_A, real_image_A)))
            loss_cycle_B = torch.sum(torch.mul(cycle_weight, cycle_criterion(recovered_image_B, real_image_B)))

            # Combined loss and calculate gradients
            g_loss = loss_identity_A + loss_identity_B + loss_adversarial_A + loss_adversarial_B + loss_cycle_A + loss_cycle_B

        # Backpropagation
        scaler.scale(g_loss).backward()
        # update generator weights
        scaler.step(g_optimizer)
        scaler.update()

        # Update EMA
        ema_g_A_model.update_parameters(g_A_model)
        ema_g_B_model.update_parameters(g_B_model)

        fake_image_A = fake_A_buffer.push_and_pop(fake_image_A)
        fake_image_B = fake_B_buffer.push_and_pop(fake_image_B)

        ##############################################
        # (2) Update D network: Discriminator A
        ##############################################

        # Initialize discriminator gradients
        d_optimizer.zero_grad(set_to_none=True)

        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in d_A_model.parameters():
            d_parameters.requires_grad = True

        # Mixed precision training
        with amp.autocast():
            # Real A image loss
            real_output_A = d_A_model(real_image_B)
            real_label = torch.tensor(1).expand_as(real_output_A).to(device, non_blocking=True)
            loss_real_A = adversarial_criterion(real_output_A, real_label)

            # Fake A image loss
            fake_output_A = d_A_model(fake_image_B.detach())
            fake_label = torch.tensor(0).expand_as(fake_output_A).to(device, non_blocking=True)
            loss_fake_A = adversarial_criterion(fake_output_A, fake_label)

            # Combined loss and calculate gradients
            loss_d_A = torch.div(torch.add(loss_real_A, loss_fake_A), 2)

        # Backpropagation
        scaler.scale(loss_d_A).backward()

        ##############################################
        # (3) Update D network: Discriminator B
        ##############################################

        # Initialize discriminator gradients
        d_optimizer.zero_grad(set_to_none=True)

        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in d_B_model.parameters():
            d_parameters.requires_grad = True

        # Mixed precision training
        with amp.autocast():
            # Real B image loss
            real_output_B = d_B_model(real_image_A)
            real_label = torch.tensor(1).expand_as(real_output_B).to(device, non_blocking=True)
            loss_real_B = adversarial_criterion(real_output_B, real_label)

            # Fake B image loss
            fake_output_B = d_B_model(fake_image_A.detach())
            fake_label = torch.tensor(0).expand_as(fake_output_B).to(device, non_blocking=True)
            loss_fake_B = adversarial_criterion(fake_output_B, fake_label)

            # Combined loss and calculate gradients
            loss_d_B = torch.div(torch.add(loss_real_B, loss_fake_B), 2)

        # Backpropagation
        scaler.scale(loss_d_B).backward()
        # update generator weights
        scaler.step(d_optimizer)
        scaler.update()

        # Statistical loss value for terminal data output
        d_losses.update((loss_d_A + loss_d_B).item(), batch_size)
        g_losses.update(g_loss.item(), batch_size)

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config["TRAIN"]["PRINT_FREQ"] == 0:
            total_batch_index = batch_index + epoch * batches

            # Record loss during training and output to file
            writer.add_scalar("Train/D(A)_Loss", loss_d_A.item(), total_batch_index)
            writer.add_scalar("Train/D(B)_Loss", loss_d_B.item(), total_batch_index)
            writer.add_scalar("Train/D_Loss", (loss_d_A + loss_d_B).item(), total_batch_index)
            writer.add_scalar("Train/Identity_Loss", (loss_identity_A + loss_identity_B).item(), total_batch_index)
            writer.add_scalar("Train/Adversarial_Loss", (loss_adversarial_B + loss_adversarial_A).item(), total_batch_index)
            writer.add_scalar("Train/Cycle_Loss", (loss_cycle_A + loss_cycle_B).item(), total_batch_index)
            writer.add_scalar("Train/G_Loss", g_loss.item(), total_batch_index)
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_data_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1

        # Save training image
        if batch_index == batches:
            save_image(real_image_A,
                       f"./samples/{config['EXP_NAME']}/A/real_image_A_epoch_{epoch:04d}.jpg",
                       normalize=True)
            save_image(real_image_B,
                       f"./samples/{config['EXP_NAME']}/B/real_image_B_epoch_{epoch:04d}.jpg",
                       normalize=True)

            # Normalize [-1, 1] to [0, 1]
            fake_image_A = 0.5 * (g_B_model(real_image_B).data + 1.0)
            fake_image_B = 0.5 * (g_A_model(real_image_A).data + 1.0)

            save_image(fake_image_A.detach(),
                       f"./samples/{config['EXP_NAME']}/A/fake_image_A_epoch_{epoch:04d}.jpg",
                       normalize=True)
            save_image(fake_image_B.detach(),
                       f"./samples/{config['EXP_NAME']}/B/fake_image_B_epoch_{epoch:04d}.jpg",
                       normalize=True)


if __name__ == "__main__":
    main()
