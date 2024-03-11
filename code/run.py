#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Custom Diffusion authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# Downloaded from 
# https://github.com/huggingface/diffusers/blob/main/examples/custom_diffusion/train_custom_diffusion.py

# Portions removed by Dmitri Chouchkov 
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path

import diffusers

import numpy as np
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, io
from tqdm.auto import tqdm

from dataset import ImageSegmentationDataset, labels 


from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

from unet.unet import ImprovedBottleNeck, Unet, AttentionBottleNeck, BCELoss_class_weighted

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__)

# take mask of dimension # B, H, W
def getBoundaryMask(x: torch.Tensor) -> torch.Tensor:
    top = x[:, 1 : -1, 1: - 1] - x[:, 0 : -2, 1: - 1]
    bot = x[:, 1 : -1, 1: - 1] - x[:, 2 : , 1: - 1]
    left = x[:, 1 : -1, 1: - 1] - x[:, 1 : -1, 0: - 2]
    right = x[:, 1 : -1, 1: - 1] - x[:, 1 : -1, 2: ]
    return torch.logical_or(torch.logical_or(top, bot), torch.logical_or(left, right))


def parse_args(input_args=None):
    ## Parameters to configure training and testing schedules
    parser = argparse.ArgumentParser(description="Custom Segmentation Training Script")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Path to last batch of parameters",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        default='output.param',
        help="Name of output model weight file",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        required=True,
        default=None,
        help="A folder containing raw images",
    )
    parser.add_argument(
        "--label_data_dir",
        type=str,
        required=True,
        default=None,
        help="A folder containing label information for some images",
    )
    parser.add_argument(
        "--testing_proportion",
        type=float,
        default=0.1,
        help="Proportion of samples to be withheld for testing",
    )
    parser.add_argument(
        "--epochs_before_validate",
        type=int,
        default=1,
        help=(
            "Number of training epochs before running on test set"
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=-1,
        help=(
            "Stop training early if training score does not improve after this many itterations, off if negative (default)"
        ),
    )
    parser.add_argument(
        "--freeze_testing_rng",
        type=int,
        default= 42,
        help=("Use this rng seed each time to generate consistent test samples",
              "   set to -1 to generate new test samples each validation run")
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/",
        required=True,
        help="Output Directory",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="A seed for reproducible training."
    )
    ## parameters to determine dataset transformations ##
    parser.add_argument(
        "--fit_width",
        action="store_true",
        default=False,
        help=(
            "Always include maximum width in the crop"
        ),
    )
    parser.add_argument(
        "--max_scale_factor",
        type=float,
        default=1.0,
        help=(
            "Maximum amount source image is scaled in either direction before cropping",
            "Ignored if --fit_width is on"
        ),
    )
    parser.add_argument(
        "--horizontal_flip",
        action="store_true",
        default=False,
        help=(
            "Randomly flip images horizontally"
        ),
    )
    parser.add_argument(
        "--vertical_flip",
        action="store_true",
        default=False,
        help=(
            "Randomly flip images vertically, not recommended"
        ),
    )

    ## Training parameters ##
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=4, 
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default= 500
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=250,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=10,
        help="Maximum number of saved checkpoints",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--checkpoint_to_output",
        action='store_true',
        default=False,
        help=(
            "Immediately store checkpoint parameters in output and terminate"
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=False,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'Only Supported Platform is Tensorboard'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        default=False,
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )
    parser.add_argument(
        "--model_features",
        type=int,
        default=48,
        help="base model feature size"
    )
    parser.add_argument(
        "--model_width",
        type=int,
        default=512,
        help="expected input width in pixels for model"
    )
    parser.add_argument(
        "--model_height",
        type=int,
        default=512,
        help="expected input height in pixels for model"
    )
    parser.add_argument(
        "--model_bottleneck",
        type=str,
        default="default",
        help="specify model bottleneck, either default, improved, attention, attentionXL"
    )
    parser.add_argument(
        "--upgrade_bottleneck",
        type = str,
        default = "none",
        help="Load a pretrained model, freezing everything but the chosen bottleneck (improved, attention, attentionXL), train and save new model" 
    )
    parser.add_argument(
        "--model_norm",
        type = str,
        default="GroupVectorNorm",
        help="GroupVectorNorm or GroupNorm"
    )
    parser.add_argument(
        "--model_use_local_attention",
        action="store_true",
        default=False,
        help="up blocks will use local attention with convolution"
    )
    parser.add_argument(
        "--upgrade_local_attention",
        action="store_true",
        default=False,
        help="Model will upgrade to local attention" 
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args



def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    BASE_FEATURES = args.model_features
    WIDTH = args.model_width
    HEIGHT = args.model_height
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        cpu=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision= args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("seg512", config=vars(args))

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # load model with specified bottleneck
    unet = Unet(3, 24, HEIGHT, WIDTH, base_features=BASE_FEATURES, bottleneck = args.model_bottleneck, useAttention=args.model_use_local_attention, norm= args.model_norm)
    if args.pretrained_model_path is not None:
        unet.load_state_dict(torch.load(args.pretrained_model_path)) 
    else:
        unet.init_weights()

    # For mixed precision training 
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # unet = unet.to(accelerator.device, dtype=weight_dtype)
    unet = unet.to(accelerator.device)          # try without cast

    # if we are upgrading the bottleneck, only the bottleneck gets trained
    # that way we can bring the model to where it was before (roughly) and then train it from there
    # this doesn't really matter for the convolution model which trains very fast but it does matter 
    # for the attention model that trains very slowly

    upgrades = (1 if args.upgrade_bottleneck != "none" else 0) + (1 if args.upgrade_local_attention else 0) 
    if upgrades > 1:
        raise NotImplementedError("cannot perform multiple model structure changes simultaneously, please do one upgrade at a time")

    # process bottleneck upgrades
    if(args.upgrade_bottleneck == 'improved'):
        unet.bottleneck = ImprovedBottleNeck(BASE_FEATURES * 16,BASE_FEATURES * 32 ,24,HEIGHT//32,WIDTH//32)
        unet.bottleneck.init_weights()
        target = unet.bottleneck
    elif(args.upgrade_bottleneck =='attention'):
        unet.bottleneck = AttentionBottleNeck(BASE_FEATURES * 16,BASE_FEATURES * 32,4,HEIGHT//32,WIDTH//32, n_blocks=3, norm=args.model_norm)
        unet.bottleneck.init_weights()
        target = unet.bottleneck
    elif(args.upgrade_bottleneck == 'attentionXL'):
        if args.model_bottleneck == 'attention':
            unet.bottleneck.grow(3)
            target = unet.bottleneck
        else:
            unet.bottleneck = AttentionBottleNeck(BASE_FEATURES * 16,BASE_FEATURES * 32,4, HEIGHT//32, WIDTH//32, n_blocks=6, norm=args.model_norm)
            unet.bottleneck.init_weights()
            target = unet.bottleneck
    else:
        target = unet

    # params = unet.bottleneck.parameters()
    accelerator.register_for_checkpointing(unet)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params = list(target.named_parameters())
    
    def isLocalAttention(layer: str):
        # bit of testing just to make sure
        return (layer[0].find('.attention') >= 0) and (layer[0].find('upBlocks.') >= 0)
    
    # handle local attention upgrade
    if args.upgrade_local_attention:
        unet.enable_attention() # in case we weight decayed to zero during pretraining

    paramList = [{"params": [x[1] for x in list(filter(isLocalAttention, params))], "lr": args.learning_rate if args.model_use_local_attention else 0},
                 {"params": [x[1] for x in list(filter(lambda label: not isLocalAttention(label), params))], "lr": args.learning_rate if not args.upgrade_local_attention else 0}
                 ]

    # Optimizer creation
    optimizer = optimizer_class(
        paramList,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = ImageSegmentationDataset(args.raw_data_dir, args.label_data_dir, transform= 'FitWidth' if args.fit_width else 'unsupported', seed = args.seed)
    if args.testing_proportion > 0:
        test_dataset = train_dataset.split(args.testing_proportion)
    else:
        test_dataset = None
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        drop_last=args.mixed_precision != 'all'
    )

    # We turn off shuffling so that items in the test dataset are accessed in a consistent manner
    # That way the RNG should be applied consistently 

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.train_batch_size,
        shuffle = args.freeze_testing_rng == -1,
        num_workers=args.dataloader_num_workers,
        drop_last=args.mixed_precision != 'all'
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # apparently constant is good enough? 
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # prepare everything with accelerator
    if test_dataloader is not None: 
        test_dataloader = accelerator.prepare(test_dataloader)
    
    # hopefully this works right
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
    )


    # unet.load_checkpoint('./output_double/checkpoint-300/','model')

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        torch.cuda.empty_cache() 
        if args.resume_from_checkpoint == "latest": 
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        elif args.resume_from_checkpoint == "best":
            path = 'best'
        else:
            path = os.path.basename(args.resume_from_checkpoint)

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            if args.resume_from_checkpoint != "best":
                global_step = int(path.split("-")[1])
            else:
                global_step = 0

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

            # adjust lr schedule if need be
            for index, lr in enumerate(lr_scheduler.scheduler.base_lrs):
                lr_scheduler.scheduler.base_lrs[index] = args.learning_rate
        # model crashing after restoring from state due to memory
        torch.cuda.empty_cache() 
        # exit early
        if args.checkpoint_to_output:
            unet = accelerator.unwrap_model(unet).to(torch.float32)
            torch.save(unet.state_dict(), os.path.join(args.output_dir, args.output_file_name ))
            return

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


    # best score achieved for training set 
    best_test_score = torch.inf
    patience = 0
    bdry_loss_fn = BCELoss_class_weighted.apply
    cross_entropy_weight = torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 2.0, 2.0, 5.0, 1.0, 5.0, 5.0]).to(accelerator.device)

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        seg_losses = []
        bdry_losses = []
        for step, (input, target) in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # normalize
                input = input.to(weight_dtype)/255.0
                logits, bdry = unet(input)
                target = torch.squeeze(target,1).to(torch.long)
                seg_loss = torch.nn.functional.cross_entropy(logits,target, weight = cross_entropy_weight, reduction='mean',)
                target_bdry = getBoundaryMask(target).to(dtype=weight_dtype)
                bdry = bdry[:,:, 1:-1, 1:-1].squeeze(1)
                bdry_loss: torch.Tensor = bdry_loss_fn(bdry,target_bdry,[1.0, 5.0]).mean()
                loss = seg_loss + bdry_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                # store losses
                seg_losses.append(seg_loss.item())
                bdry_losses.append(bdry_loss.item())

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {"seg loss": np.average(seg_losses[-args.gradient_accumulation_steps: ]),
                        "bdry loss": np.average(bdry_losses[-args.gradient_accumulation_steps: ]),
                         "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            

            if global_step >= args.max_train_steps:
                break
        #print("average epoch loss: ", np.average(losses))
        if (epoch % args.epochs_before_validate == 0 or epoch == args.num_train_epochs - 1) and accelerator.is_local_main_process and test_dataloader is not None:

            # TODO: Make sure that this works with accelerate, it might not 
            if args.freeze_testing_rng != -1:
                test_dataset.set_seed(args.freeze_testing_rng)
            
            seg_losses = []
            bdry_losses = []
            seg_images = []
            bdry_images = []
            unet.train()
            with torch.no_grad():
                for step, (input, target) in enumerate(test_dataloader):
                    input = input.to(weight_dtype)/255
                    logits, bdry = unet(input)
                    target = torch.squeeze(target,1).to(torch.long)
                    seg_loss = torch.nn.functional.cross_entropy(logits,target, weight = cross_entropy_weight, reduction='mean',)
                    target_bdry = getBoundaryMask(target).to(dtype = weight_dtype)
                    bdry = bdry[:,:, 1:-1, 1:-1].squeeze(1)
                    bdry_loss: torch.Tensor = bdry_loss_fn(bdry,target_bdry,[1.0, 5.0]).mean()
                
                    # log segmentation result
                    seg_images.append(torch.argmax(logits,1).to(torch.uint8).unsqueeze(1).cpu().numpy())
                    # log boundary result
                    bdry_images.append(torch.mul(bdry,255).unsqueeze(dim=1).to(torch.uint8).cpu().numpy())

                    seg_losses.append(seg_loss.item())
                    bdry_losses.append(bdry_loss.item())
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_seg_images = np.concatenate([np.asarray(img) for img in seg_images], axis=0)*10
                    np_bdry_images = np.concatenate([np.asarray(img) for img in bdry_images], axis=0)
                    tracker.writer.add_images("segmentation", np_seg_images[:100,:,:,:], epoch, dataformats="NCHW")
                    tracker.writer.add_images("bondary", np_bdry_images[:100,:,:,:], epoch, dataformats="NCHW")
            loss = np.average(seg_losses) + np.average(bdry_losses)

            print("average segmentation loss: ", np.average(seg_losses))
            print("average boundary loss: ", np.average(bdry_losses))

            # when using accelerate we can only store checkpoints
            # it's too much work to unhook the model                 
            if loss < best_test_score:
                best_test_score = loss
                # save the milestone
                save_path = os.path.join(args.output_dir, f"best")
                if os.path.exists(save_path):
                    shutil.rmtree(save_path)
                # accelerator.save_state(save_path)
                patience = 0
            else:
                patience += 1
                if patience >= args.patience and args.patience > 0:
                    logger.exception("patience exceeded, training terminated prematurely") 
                    break
            
    # Save the weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet).to(torch.float32)
        torch.save(unet.state_dict(), os.path.join(args.output_dir, args.output_file_name ))

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)