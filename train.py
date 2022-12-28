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
import itertools
import os
import random
import time

import numpy as np
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import config
import model
from dataset import CUDAPrefetcher, ImageDataset
from utils import load_pretrained_state_dict, load_resume_state_dict, make_directory, save_state_dict, DecayLR, \
    ReplayBuffer, Summary, AverageMeter, ProgressMeter


def main():
    device = torch.device(config.device)
    # Fixed random number seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Because the size of the input image is fixed, the fixed CUDNN convolution method can greatly increase the running speed
    cudnn.benchmark = True

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Initialize the number of training epochs
    start_epoch = 0

    train_prefetcher = load_dataset(config.src_image_path,
                                    config.dst_image_path,
                                    config.unpaired,
                                    config.resized_image_size,
                                    config.batch_size,
                                    config.num_workers,
                                    device)
    d_A_model, d_B_model, g_A2B_model, g_B2A_model, ema_g_A2B_model, ema_g_B2A_model = build_model(
        config.d_model_arch_name,
        config.g_model_arch_name,
        config.model_ema_decay,
        device)

    cycle_criterion, identity_criterion, adversarial_criterion = define_loss(device)
    d_A_optimizer, d_B_optimizer, g_optimizer = define_optimizer(d_A_model,
                                                                 d_B_model,
                                                                 g_A2B_model,
                                                                 g_B2A_model,
                                                                 config.optim_lr,
                                                                 config.optim_betas,
                                                                 config.optim_eps,
                                                                 config.optim_weight_decay)
    d_A_scheduler, d_B_scheduler, g_scheduler = define_scheduler(d_A_optimizer,
                                                                 d_B_optimizer,
                                                                 g_optimizer,
                                                                 config.decay_epochs,
                                                                 config.epochs)

    # Load the pre-trained model weights and fine-tune the model
    print("Check whether to load pretrained model weights...")
    if config.load_pretrained:
        d_A_model = load_pretrained_state_dict(d_A_model, config.pretrained_d_A_model_weights_path)
        d_B_model = load_pretrained_state_dict(d_B_model, config.pretrained_d_B_model_weights_path)
        g_A2B_model = load_pretrained_state_dict(g_A2B_model, config.pretrained_g_A2B_model_weights_path)
        g_B2A_model = load_pretrained_state_dict(g_B2A_model, config.pretrained_g_B2A_model_weights_path)
        print(f"Loaded pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    # Load the last training interruption node
    print("Check whether the resume model is restored...")
    if config.load_resume:
        d_A_model, _, start_epoch, d_A_optimizer, d_A_scheduler = load_resume_state_dict(
            d_A_model,
            config.resume_d_A_model_weights_path,
            None,
            d_A_optimizer,
            d_A_scheduler,
        )
        d_B_model, _, start_epoch, d_B_optimizer, d_B_scheduler = load_resume_state_dict(
            d_B_model,
            config.resume_d_B_model_weights_path,
            None,
            d_B_optimizer,
            d_B_scheduler,
        )
        g_A2B_model, ema_g_A2B_model, start_epoch, g_optimizer, g_scheduler = load_resume_state_dict(
            g_A2B_model,
            config.resume_g_A2B_model_weights_path,
            ema_g_A2B_model,
            g_optimizer,
            g_scheduler,
        )
        g_B2A_model, ema_g_B2A_model, start_epoch, g_optimizer, g_scheduler = load_resume_state_dict(
            g_B2A_model,
            config.resume_g_B2A_model_weights_path,
            ema_g_B2A_model,
            g_optimizer,
            g_scheduler,
        )
        print(f"Loaded resume model weights successfully.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)
    make_directory(os.path.join(samples_dir, "A"))
    make_directory(os.path.join(samples_dir, "B"))

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    for epoch in range(start_epoch, config.epochs):
        train(d_A_model,
              d_B_model,
              g_A2B_model,
              g_B2A_model,
              ema_g_A2B_model,
              ema_g_B2A_model,
              train_prefetcher,
              identity_criterion,
              adversarial_criterion,
              cycle_criterion,
              d_A_optimizer,
              d_B_optimizer,
              g_optimizer,
              fake_A_buffer,
              fake_B_buffer,
              epoch,
              scaler,
              writer,
              device,
              config.print_frequency,
              samples_dir)
        print("\n")

        # Update LR
        d_A_scheduler.step()
        d_B_scheduler.step()
        g_scheduler.step()

        is_last = (epoch + 1) == config.epochs
        save_state_dict({"epoch": epoch + 1,
                         "state_dict": d_A_model.state_dict(),
                         "optimizer": d_A_optimizer.state_dict(),
                         "scheduler": d_A_scheduler.state_dict()},
                        f"d_A_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "d_A_best.pth.tar",
                        "d_A_last.pth.tar",
                        True,
                        is_last)
        save_state_dict({"epoch": epoch + 1,
                         "state_dict": d_B_model.state_dict(),
                         "optimizer": d_B_optimizer.state_dict(),
                         "scheduler": d_B_scheduler.state_dict()},
                        f"d_B_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "d_B_best.pth.tar",
                        "d_B_last.pth.tar",
                        True,
                        is_last)
        save_state_dict({"epoch": epoch + 1,
                         "state_dict": g_A2B_model.state_dict(),
                         "ema_state_dict": ema_g_A2B_model.state_dict(),
                         "optimizer": g_optimizer.state_dict(),
                         "scheduler": g_scheduler.state_dict()},
                        f"g_A2B_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "g_A2B_best.pth.tar",
                        "g_A2B_last.pth.tar",
                        True,
                        is_last)
        save_state_dict({"epoch": epoch + 1,
                         "state_dict": g_B2A_model.state_dict(),
                         "ema_state_dict": ema_g_B2A_model.state_dict(),
                         "optimizer": g_optimizer.state_dict(),
                         "scheduler": g_scheduler.state_dict()},
                        f"g_B2A_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "g_B2A_best.pth.tar",
                        "g_B2A_last.pth.tar",
                        True,
                        is_last)


def load_dataset(
        src_image_path: str,
        dst_image_path: str,
        unpaired: bool,
        resized_image_size: int,
        batch_size: int,
        num_workers: int,
        device: torch.device) -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = ImageDataset(src_image_path, dst_image_path, unpaired, resized_image_size)
    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, device)

    return train_prefetcher


def build_model(
        d_model_arch_name: str,
        g_model_arch_name: str,
        model_ema_decay: float,
        device: torch.device,
) -> [nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    d_src_model = model.__dict__[d_model_arch_name]()
    d_dst_model = model.__dict__[d_model_arch_name]()
    g_src_to_dst_model = model.__dict__[g_model_arch_name]()
    g_dst_to_src_model = model.__dict__[g_model_arch_name]()

    # Create an Exponential Moving Average Model
    ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
        (1 - model_ema_decay) * averaged_model_parameter + model_ema_decay * model_parameter
    ema_g_src_to_dst_model = AveragedModel(g_src_to_dst_model, device=device, avg_fn=ema_avg_fn)
    ema_g_dst_to_src_model = AveragedModel(g_dst_to_src_model, device=device, avg_fn=ema_avg_fn)

    d_src_model = d_src_model.to(device)
    d_dst_model = d_dst_model.to(device)
    g_src_to_dst_model = g_src_to_dst_model.to(device)
    g_dst_to_src_model = g_dst_to_src_model.to(device)
    ema_g_src_to_dst_model = ema_g_src_to_dst_model.to(device)
    ema_g_dst_to_src_model = ema_g_dst_to_src_model.to(device)

    return d_src_model, d_dst_model, g_src_to_dst_model, g_dst_to_src_model, ema_g_src_to_dst_model, ema_g_dst_to_src_model


def define_loss(device) -> [nn.L1Loss, nn.MSELoss, nn.L1Loss]:
    identity_criterion = nn.L1Loss()
    adversarial_criterion = nn.MSELoss()
    cycle_criterion = nn.L1Loss()

    identity_criterion = identity_criterion.to(device)
    adversarial_criterion = adversarial_criterion.to(device)
    cycle_criterion = cycle_criterion.to(device)

    return identity_criterion, cycle_criterion, adversarial_criterion


def define_optimizer(
        d_src_model: nn.Module,
        d_dst_model: nn.Module,
        g_src_to_dst_model: nn.Module,
        g_dst_to_src_model: nn.Module,
        optim_lr: float,
        optim_betas: tuple,
        optim_eps: float,
        optim_weight_decay: float,
) -> [optim.Adam, optim.Adam, optim.Adam]:
    d_src_optimizer = torch.optim.Adam(d_src_model.parameters(),
                                       optim_lr,
                                       optim_betas,
                                       optim_eps,
                                       optim_weight_decay)
    d_dst_optimizer = torch.optim.Adam(d_dst_model.parameters(),
                                       optim_lr,
                                       optim_betas,
                                       optim_eps,
                                       optim_weight_decay)
    g_optimizer = torch.optim.Adam(itertools.chain(g_src_to_dst_model.parameters(), g_dst_to_src_model.parameters()),
                                   optim_lr,
                                   optim_betas,
                                   optim_eps,
                                   optim_weight_decay)

    return d_src_optimizer, d_dst_optimizer, g_optimizer


def define_scheduler(
        d_src_optimizer: optim.Adam,
        d_dst_optimizer: optim.Adam,
        g_optimizer: optim.Adam,
        decay_epoch: int,
        epochs: int,
) -> [lr_scheduler.LambdaLR, lr_scheduler.LambdaLR, lr_scheduler.LambdaLR, ]:
    lr_lambda = DecayLR(epochs, 0, decay_epoch).step
    d_src_scheduler = lr_scheduler.LambdaLR(d_src_optimizer, lr_lambda)
    d_dst_scheduler = lr_scheduler.LambdaLR(d_dst_optimizer, lr_lambda)
    g_scheduler = lr_scheduler.LambdaLR(g_optimizer, lr_lambda)

    return d_src_scheduler, d_dst_scheduler, g_scheduler


def train(
        d_A_model: nn.Module,
        d_B_model: nn.Module,
        g_A2B_model: nn.Module,
        g_B2A_model: nn.Module,
        ema_g_A2B_model: nn.Module,
        ema_g_B2A_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        identity_criterion: nn.L1Loss,
        adversarial_criterion: nn.MSELoss,
        cycle_criterion: nn.L1Loss,
        d_A_optimizer: optim.Adam,
        d_B_optimizer: optim.Adam,
        g_optimizer: optim.Adam,
        fake_A_buffer: ReplayBuffer,
        fake_B_buffer: ReplayBuffer,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device,
        print_frequency: int,
        samples_dir: str,
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
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
    g_A2B_model.train()
    g_B2A_model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        real_image_A = batch_data["src"].to(device, non_blocking=True)
        real_image_B = batch_data["dst"].to(device, non_blocking=True)
        identity_weight = torch.Tensor(config.identity_weight).to(device)
        adversarial_weight = torch.Tensor(config.adversarial_weight).to(device)
        cycle_weight = torch.Tensor(config.cycle_weight).to(device)

        batch_size = real_image_A.size(0)
        real_label = torch.full((batch_size, 3), 1, device=device, dtype=torch.float32)
        fake_label = torch.full((batch_size, 3), 0, device=device, dtype=torch.float32)

        ##############################################
        # (1) Update G network: Generators A2B and B2A
        ##############################################

        # Initialize generator gradients
        g_optimizer.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            # Identity loss
            # G_B2A(A) should equal A if real A is fed
            identity_image_A = g_B2A_model(real_image_A)
            loss_identity_A = torch.sum(torch.mul(identity_weight, identity_criterion(identity_image_A, real_image_A)))
            # G_A2B(B) should equal B if real B is fed
            identity_image_B = g_B2A_model(real_image_B)
            loss_identity_B = torch.sum(torch.mul(identity_weight, identity_criterion(identity_image_B, real_image_B)))

            # GAN loss
            # GAN loss D_A(G_A(A))
            fake_image_A = g_B2A_model(real_image_B)
            fake_output_A = d_A_model(fake_image_A)
            loss_adversarial_B2A = torch.sum(torch.mul(adversarial_weight, adversarial_criterion(fake_output_A, real_label)))
            # GAN loss D_B(G_B(B))
            fake_image_B = g_A2B_model(real_image_A)
            fake_output_B = d_B_model(fake_image_B)
            loss_adversarial_A2B = torch.sum(torch.mul(adversarial_weight, adversarial_criterion(fake_output_B, real_label)))

            # Cycle loss
            recovered_image_A = g_B2A_model(fake_image_B)
            loss_cycle_ABA = torch.sum(torch.mul(cycle_weight, cycle_criterion(recovered_image_A, real_image_A)))
            recovered_image_B = g_A2B_model(fake_image_A)
            loss_cycle_BAB = torch.sum(torch.mul(cycle_weight, cycle_criterion(recovered_image_B, real_image_B)))

            # Combined loss and calculate gradients
            g_loss = loss_identity_A + loss_identity_B + loss_adversarial_A2B + loss_adversarial_B2A + loss_cycle_ABA + loss_cycle_BAB

        # Backpropagation
        scaler.scale(g_loss).backward()
        # update generator weights
        scaler.step(g_optimizer)
        scaler.update()

        # Update EMA
        ema_g_A2B_model.update_parameters(ema_g_A2B_model)
        ema_g_B2A_model.update_parameters(ema_g_B2A_model)

        fake_image_A = fake_A_buffer.push_and_pop(fake_image_A)
        fake_image_B = fake_B_buffer.push_and_pop(fake_image_B)

        ##############################################
        # (2) Update D network: Discriminator A
        ##############################################

        # Initialize discriminator gradients
        d_A_optimizer.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            # Real A image loss
            real_output_A = d_A_model(real_image_A)
            loss_real_A = adversarial_criterion(real_output_A, real_label)

            # Fake A image loss
            fake_output_A = d_A_model(fake_image_A.detach())
            loss_fake_A = adversarial_criterion(fake_output_A, fake_label)

            # Combined loss and calculate gradients
            loss_d_A = torch.div(torch.add(loss_real_A, loss_fake_A), 2)

        # Backpropagation
        scaler.scale(loss_d_A).backward()
        # update generator weights
        scaler.step(d_A_optimizer)
        scaler.update()

        ##############################################
        # (3) Update D network: Discriminator B
        ##############################################

        # Initialize discriminator gradients
        d_B_optimizer.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            # Real B image loss
            real_output_B = d_B_model(real_image_B)
            loss_real_B = adversarial_criterion(real_output_B, real_label)

            # Fake B image loss
            fake_output_B = d_B_model(fake_image_B.detach())
            loss_fake_B = adversarial_criterion(fake_output_B, fake_label)

            # Combined loss and calculate gradients
            loss_d_B = torch.div(torch.add(loss_real_B, loss_fake_B), 2)

        # Backpropagation
        scaler.scale(loss_d_B).backward()
        # update generator weights
        scaler.step(d_B_optimizer)
        scaler.update()

        # Statistical loss value for terminal data output
        d_losses.update((loss_d_A + loss_d_B).item(), batch_size)
        g_losses.update(g_loss.item(), batch_size)

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % print_frequency == 0:
            total_batch_index = batch_index + epoch * batches + 1

            # Record loss during training and output to file
            writer.add_scalar("Train/D(A)_Loss", loss_d_A.item(), total_batch_index)
            writer.add_scalar("Train/D(B)_Loss", loss_d_B.item(), total_batch_index)
            writer.add_scalar("Train/D_Loss", (loss_d_A + loss_d_B).item(), total_batch_index)
            writer.add_scalar("Train/G_Identity_Loss", (loss_identity_A + loss_identity_B).item(), total_batch_index)
            writer.add_scalar("Train/G_Adversarial_Loss", (loss_adversarial_A2B + loss_adversarial_B2A).item(), total_batch_index)
            writer.add_scalar("Train/G_Cycle_Loss", (loss_cycle_ABA + loss_cycle_BAB).item(), total_batch_index)
            writer.add_scalar("Train/G_Loss", g_loss.item(), total_batch_index)
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1

        # Save training image
        if batch_index == batches:
            save_image(real_image_A,
                       f"{samples_dir}/A/real_image_A_epoch_{epoch:04d}.jpg",
                       normalize=True)
            save_image(real_image_B,
                       f"{samples_dir}/B/real_image_B_epoch_{epoch:04d}.jpg",
                       normalize=True)

            # Normalize [-1, 1] to [0, 1]
            fake_image_A = 0.5 * (g_B2A_model(real_image_B).data + 1.0)
            fake_image_B = 0.5 * (g_A2B_model(real_image_A).data + 1.0)

            save_image(fake_image_A.detach(),
                       f"{samples_dir}/A/fake_image_A_epoch_{epoch:04d}.jpg",
                       normalize=True)
            save_image(fake_image_B.detach(),
                       f"{samples_dir}/B/fake_image_B_epoch_{epoch:04d}.jpg",
                       normalize=True)


if __name__ == "__main__":
    main()
