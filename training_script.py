#!/usr/bin/env python3
import argparse
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
import sys
import os
import gc
import numpy as np
import torch
import torch.optim as optim
from colorama import Fore, Back, Style
from tensorboardX import SummaryWriter
from tqdm import tqdm
import math 
import json

# Assume these are imported from your modules
from model import get_diffusion_model
from style_encoder import get_style_encoder
from datasets import get_dataset
from utils.flame import FLAME, FLAMEConfig
from utils.model_common import load_pretrained_model, save_args
from utils.common import compute_loss_no_vert, compute_loss
from utils.common import compute_KL_loss
import utils 

def infinite_data_loader(data_loader):
    while True:
        for data in data_loader:
            yield data

def print_GPU_usage():
    """Print GPU memory usage for debugging"""
    device_id = torch.cuda.current_device()
    free_memory, total_memory = torch.cuda.mem_get_info(device_id)
    used_memory = total_memory - free_memory
    print(f"Total GPU memory: {total_memory / 1024**2:.2f} MB")
    print(f"Free GPU memory: {free_memory / 1024**2:.2f} MB")
    print(f"Used GPU memory: {used_memory / 1024**2:.2f} MB")


def clear_cuda_cache():
    """Helper function to clear CUDA cache and delete tensors"""
    gc.collect()
    torch.cuda.empty_cache()


def train(args, model, style_enc, train_loader, val_loader, optimizer,
          save_dir, scheduler=None, writer=None, flame=None, out_abc_dir=None, start_iter=0):
    """Main training function"""
    loss_weights = load_loss_weights(args, model.device)
    device = model.device
    save_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    data_loader = infinite_data_loader(train_loader)
    
    # Get dataset and coefficient statistics
    if len(args.dataset_type.split("+")) <= 1:
        dataset = train_loader.dataset
    else:
        dataset = train_loader.dataset.datasets[0]
    coef_stats = dataset.coef_stats
    if coef_stats is not None:
        coef_stats = {x: coef_stats[x].to(device) for x in coef_stats}
    
    audio_unit = dataset.audio_unit
    predict_head_pose = not args.no_head_pose
    loss_log = defaultdict(lambda: deque(maxlen=args.log_smooth_win))
    pbar = tqdm(range(start_iter, args.max_iter + 1), initial=start_iter, total=args.max_iter + 1, dynamic_ncols=True)
    
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    
    for it in pbar:
        audio_pair, coef_pair, audio_stats = next(data_loader)   
        clear_cuda_cache()
        audio_pair = [audio.to(device) for audio in audio_pair]
        coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]

        if args.dataset_type[:9] == "HDTF_TFHP" or args.dataset_type == 'flame_mead_ravdess':
            motion_coef_pair = [
                utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose) for i in range(2)
            ]
        else:
            motion_coef_pair = [coef_pair[0]["motion"], coef_pair[1]["motion"]]
            
        # Use shape coefficients from the first frame as condition
        if coef_pair[0]['shape'].ndim == 2:  # (N, 100)
            shape_coef = coef_pair[0]['shape'].clone().to(device)
        else:  # (N, L, 100)
            shape_coef = coef_pair[0]['shape'][:, 0].clone().to(device)

        # Extract style features
        if args.style_enc_model_style[:3] == 'vae':
            style_mu_logvar_pair = [style_enc(motion_coef_pair[i]) for i in range(2)]
            style_pair = [x[0] for x in style_mu_logvar_pair]
            mu_pair = [x[1] for x in style_mu_logvar_pair]
            logvar_pair = [x[2] for x in style_mu_logvar_pair]
        else:
            raise ValueError(f"Style Encoder Model style {args.style_enc_model_style} not recognized")

        # Initialize losses
        losses = {}
        for key in loss_weights:
            losses[key] = torch.tensor(0.0, device=device)

        # Process each of the two clips
        for i in range(2):
            audio = audio_pair[i]  # (N, L_a)
            motion_coef = motion_coef_pair[i]  # (N, L, dims)
            style = style_pair[i] if style_enc is not None else None
            
            # Implement cross-style if enabled
            if args.use_cross_style:
                if np.random.rand() < args.prob_cross_style:
                    # Use cross style
                    style = style_pair[1 - i]
                    
            batch_size = audio.shape[0]
            
            # Truncate input audio and motion according to trunc_prob
            if (i == 0 and np.random.rand() < args.trunc_prob1) or (i != 0 and np.random.rand() < args.trunc_prob2):
                audio_in, motion_coef_in, end_idx = utils.truncate_motion_coef_and_audio(
                    audio, motion_coef, args.n_motions, audio_unit, args.pad_mode, expression_code_size=64)
            else:
                audio_in = audio
                motion_coef_in, end_idx = motion_coef, None

            # Set up indicator for padded frames if needed
            if args.use_indicator:
                if end_idx is not None:
                    indicator = torch.arange(args.n_motions, device=device).expand(batch_size, -1) < end_idx.unsqueeze(1)
                else:
                    indicator = torch.ones(batch_size, args.n_motions, device=device)
            else:
                indicator = None

            # Process input shape coefficients
            if args.do_ignore_shape:
                input_shape_coef = torch.zeros_like(shape_coef)
            else:
                input_shape_coef = shape_coef
            
            use_CFG_during_training = not args.do_ignore_cfg

            # Run model for the current clip
            if i == 0:
                noise, target, prev_motion_coef, prev_audio_feat = model(
                    motion_coef_in, audio_in, input_shape_coef, style, indicator=indicator, train_with_CFG=use_CFG_during_training)
                
                if end_idx is not None:  # was truncated, needs to use the complete feature
                    prev_motion_coef = motion_coef[:, -args.n_prev_motions:]
                    with torch.no_grad():
                        prev_audio_feat = model.extract_audio_feature(audio)[:, -args.n_prev_motions:]
                else:
                    prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                    prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
            else:
                noise, target, _, _ = model(
                    motion_coef_in, audio_in, input_shape_coef, style,
                    prev_motion_coef, prev_audio_feat, indicator=indicator, train_with_CFG=use_CFG_during_training)
            
            torch.cuda.empty_cache()

            # Compute losses based on the model output
            if args.training_loss_style == 'MSMD':
                if args.use_vertex_space and (args.dataset_type[:9] == "HDTF_TFHP" or args.dataset_type == 'flame_mead_ravdess'):
                    loss_dict = compute_loss(args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, 
                                            coef_stats, flame, end_idx, return_dict=True)
                else:
                    loss_dict = compute_loss_no_vert(
                        args, i == 0, shape_coef, motion_coef_in, noise, target, prev_motion_coef, 
                        coef_stats, flame, end_idx, return_dict=True)
            else:
                raise ValueError(f"Training loss style {args.training_loss_style} not recognized")

            # Add KL divergence loss for VAE
            if args.style_enc_model_style[:3] == 'vae':
                kl_loss = compute_KL_loss(mu_pair[i], logvar_pair[i])
                loss_dict['kl_div'] = kl_loss
                
            # Aggregate all losses
            for key in loss_dict:
                if loss_weights[key] > 0 and loss_dict[key] is not None:
                    losses[key] += loss_dict[key]

        # Calculate total loss and backpropagate
        loss = 0
        for key in losses:
            if loss_weights[key] > 0:
                loss_log[key].append(losses[key].item())
                loss += losses[key] * loss_weights[key]
        
        loss.backward()
        loss_log['loss'].append(loss.item())
        
        # Apply gradient accumulation if specified
        if it % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Logging in the progress bar
        description = 'Train loss: ['
        for key in loss_log:
            if key == "loss":
                description += f'{key}: {np.mean(loss_log[key]):.3e}, '
            if key != 'loss' and loss_weights[key] > 0:
                description += f'{key}: {np.mean(loss_log[key]):.3e}, '
        description += ']'
        pbar.set_description(description)

        # Log to tensorboard
        if it % args.log_iter == 0 and writer is not None:
            writer.add_scalar('train/loss', np.mean(loss_log['loss']), it)
            for key in loss_log:
                if key != 'loss' and loss_weights[key] > 0:
                    writer.add_scalar(f'train/{key}', np.mean(loss_log[key]), it)
            writer.add_scalar('opt/lr', optimizer.param_groups[0]['lr'], it)
                       
        # Update learning rate
        if scheduler is not None:
            if args.scheduler != 'WarmupThenDecay' or (args.scheduler == 'WarmupThenDecay' and it < args.cos_max_iter):
                scheduler.step()

        # Save model checkpoints
        if (it % args.save_iter == 0 and it != 0 and it != start_iter) or it == args.max_iter:
            torch.save({
                'args': args,
                'model': model.state_dict(),
                'style_enc': style_enc.state_dict(),
                'iter': it,
            }, save_dir / f'iter_{it:07}.pt')
        
        # Clean up memory
        del audio_pair, coef_pair, motion_coef_pair, losses, style_pair
        clear_cuda_cache()

        # Run validation
        if (it % args.val_iter == 0 and it != 0 and it != start_iter) or it == args.max_iter:
            test(args, loss_weights, model, style_enc, val_loader, it, 1, 'val', writer, flame, out_abc_dir=out_abc_dir)


@torch.no_grad()
def test(args, loss_weights, model, style_enc, test_loader, current_iter, n_rounds=10,
         mode='val', writer=None, flame=None, out_abc_dir=None, do_save=False, do_save_path=None, 
         do_ignore_style=False):
    """Validation and testing function"""
    is_training = model.training
    device = model.device
    model.eval()
    
    # Get dataset and coefficient statistics
    if len(args.dataset_type.split("+")) <= 1:
        dataset = test_loader.dataset
    else:
        dataset = test_loader.dataset.datasets[0]
    coef_stats = dataset.coef_stats
    if coef_stats is not None:
        coef_stats = {x: coef_stats[x].to(device) for x in coef_stats}
    
    audio_unit = dataset.audio_unit
    predict_head_pose = not args.no_head_pose
    
    loss_log = defaultdict(list)
    
    for test_round in range(n_rounds):
        for audio_pair, coef_pair, audio_stats in test_loader:
            audio_pair = [audio.to(device) for audio in audio_pair]
            coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]
            
            if args.dataset_type[:9] == "HDTF_TFHP" or args.dataset_type == 'flame_mead_ravdess':
                motion_coef_pair = [
                    utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose) for i in range(2)
                ]
            else:
                motion_coef_pair = [coef_pair[0]["motion"], coef_pair[1]["motion"]]
                
            # Use shape coefficients from the first frame
            if coef_pair[0]['shape'].ndim == 2:  # (N, 100)
                shape_coef = coef_pair[0]['shape'].clone().to(device)
            else:  # (N, L, 100)
                shape_coef = coef_pair[0]['shape'][:, 0].clone().to(device)

            # Extract style features
            if args.style_enc_model_style[:3] == 'vae':
                if do_ignore_style:  # Use mean style instead
                    mean_motion_coef_pair = [torch.zeros_like(motion_coef) for motion_coef in motion_coef_pair]
                    style_mu_logvar_pair = [style_enc(mean_motion_coef_pair[i]) for i in range(2)]
                else:
                    style_mu_logvar_pair = [style_enc(motion_coef_pair[i]) for i in range(2)]
                
                style_pair = [x[0] for x in style_mu_logvar_pair]
                mu_pair = [x[1] for x in style_mu_logvar_pair]
                logvar_pair = [x[2] for x in style_mu_logvar_pair]
            else:
                raise ValueError(f"Style Encoder Model style {args.style_enc_model_style} not recognized")

            # Initialize losses
            losses = {}
            for key in loss_weights:
                losses[key] = torch.tensor(0.0, device=device)
            
                
            # Process each clip
            for i in range(2):
                audio = audio_pair[i]
                motion_coef = motion_coef_pair[i]
                style = style_pair[i] if style_enc is not None else None
                
                if args.use_cross_style:
                    style = style_pair[1 - i]
                    
                batch_size = audio.shape[0]

                # Prepare inputs
                audio_in = audio
                motion_coef_in, end_idx = motion_coef, None

                if args.use_indicator:
                    indicator = torch.ones(batch_size, args.n_motions, device=device)
                else:
                    indicator = None

                input_shape_coef = shape_coef if not args.do_ignore_shape else torch.zeros_like(shape_coef)
                use_CFG_during_training = not args.do_ignore_cfg

                # Run model for inference
                if i == 0:
                    noise, target, prev_motion_coef, prev_audio_feat = model(
                        motion_coef_in, audio_in, input_shape_coef, style, indicator=indicator, 
                        train_with_CFG=use_CFG_during_training)
                    
                    prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                    prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
                else:
                    noise, target, _, _ = model(
                        motion_coef_in, audio_in, input_shape_coef, style,
                        prev_motion_coef, prev_audio_feat, indicator=indicator, 
                        train_with_CFG=use_CFG_during_training)
                
                # Compute losses
                if args.training_loss_style == 'MSMD':
                    if args.use_vertex_space and (args.dataset_type[:9] == "HDTF_TFHP" or args.dataset_type == 'flame_mead_ravdess'):
                        loss_dict = compute_loss(args, i == 0, shape_coef, motion_coef_in, noise, target, 
                                                prev_motion_coef, coef_stats, flame, end_idx, return_dict=True)
                    else:
                        loss_dict = compute_loss_no_vert(args, i == 0, shape_coef, motion_coef_in, noise, target, 
                                                       prev_motion_coef, coef_stats, flame, end_idx, return_dict=True)
                else:
                    raise ValueError(f"Training loss style {args.training_loss_style} not recognized")

                # Add KL loss for VAE
                if args.style_enc_model_style[:3] == 'vae':
                    kl_loss = compute_KL_loss(mu_pair[i], logvar_pair[i])
                    loss_dict['kl_div'] = kl_loss
            
                
                # Aggregate losses
                for key in loss_dict:
                    if loss_weights[key] > 0 and loss_dict[key] is not None:
                        losses[key] += loss_dict[key]
        
            # Calculate total loss
            loss = 0
            for key in losses:
                if loss_weights[key] > 0:
                    loss_log[key].append(losses[key].item())
                    loss += losses[key] * loss_weights[key]
            
            loss_log['loss'].append(loss.item())
    
    # Log results
    if writer is not None:
        writer.add_scalar(f'{mode}/loss', np.mean(loss_log['loss']), current_iter)
        for key in loss_log:
            if key != 'loss' and loss_weights[key] > 0:
                writer.add_scalar(f'{mode}/{key}', np.mean(loss_log[key]), current_iter)
    
    # Save metrics if requested
    if do_save:
        save_path = do_save_path
        for key in loss_log:
            mean_val = np.mean(loss_log[key])
            std_val = np.std(loss_log[key])
            loss_log[key] = {
                "mean": float(mean_val),
                "std": float(std_val),
                "n_samples": len(loss_log[key])
            }

        with open(save_path, 'w') as f:
            json.dump(loss_log, f)
    
    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Restore model state
    if is_training:
        model.train()
    
    return loss_log


def load_loss_weights(args, device):
    """Load and configure loss weights"""
    loss_weights = {
        "noise": torch.tensor(1).float().to(device),
        "vert": torch.tensor(args.l_vert).float().to(device),
        "vel": torch.tensor(args.l_vel).float().to(device),
        "smooth": torch.tensor(args.l_smooth).float().to(device),
        "head_angle": torch.tensor(args.l_head_angle).float().to(device),
        "head_vel": torch.tensor(args.l_head_vel).float().to(device),
        "head_smooth": torch.tensor(args.l_head_smooth).float().to(device),
        "head_trans": torch.tensor(args.l_head_trans).float().to(device),
    }
    
    # Adjust weights based on vertex space and dataset type
    if not args.use_vertex_space:
        print("Not using vertex space loss")
        loss_weights["vel"] *= 4.5E-8
        loss_weights["smooth"] *= 4E-7
        
    if not (args.dataset_type[:9] == "HDTF_TFHP" or args.dataset_type == 'flame_mead_ravdess') and args.use_vertex_space:
        print("Using vertex space loss on non-HDTF dataset")
        loss_weights["vert"] *= 1E-7
        loss_weights["vel"] *= 1E-7
        loss_weights["smooth"] *= 2E-8
        
    # Add KL divergence loss for VAE
    if args.training_loss_style == 'MSMD':
        print("Using VAE loss")
        loss_weights["kl_div"] = torch.tensor(args.l_kl_div).float().to(device)
        
    # Add style adherence loss if enabled
        
    return loss_weights


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Main entry point for training and testing"""
    parser = argparse.ArgumentParser(description='MSMD training script')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    
    # Base configuration
    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--data_root', type=str, required=True, help='path to dataset')
    parser.add_argument('--max_iter', type=int, default=2000000, help='maximum iterations')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loading')
    
    # Model architecture settings
    parser.add_argument('--generator_model_style', type=str, default='MSMD',
                        help='style of the generator model')
    parser.add_argument('--style_enc_model_style', type=str, default='vae2', 
                        help='style of the style encoder model')
    parser.add_argument('--training_loss_style', type=str, default='MSMD', 
                        help='style of the training loss')
    parser.add_argument('--dataset_type', type=str, default='ravdess+celebv-text-medium',
                        help='dataset type')
    parser.add_argument('--audio_model', type=str, default='hubert', help='audio feature extractor model')
    parser.add_argument('--d_style', type=int, default=256, help='dimension of the style embedding')
    
    # Feature options
    parser.add_argument('--use_indicator', action='store_true', help='use indicator for padded frames')
    parser.add_argument('--use_cross_style', action='store_true', help='enable cross-style transfer')
    parser.add_argument('--use_vertex_space', action='store_true', help='use vertex space for loss computation')
    parser.add_argument('--num_of_basis', type=int, default=4, help='number of basis for the style encoder')
    parser.add_argument('--prob_cross_style', type=float, default=0.5, help='probability of using cross style')
    
    # Loss weights
    parser.add_argument('--l_vert', type=float, default=1.0, help='weight for vertex loss')
    parser.add_argument('--l_vel', type=float, default=0.5, help='weight for velocity loss')
    parser.add_argument('--l_smooth', type=float, default=10.0, help='weight for smoothness loss')
    parser.add_argument('--l_kl_div', type=float, default=1e-7, help='weight for KL divergence loss')
    parser.add_argument('--l_head_angle', type=float, default=1.0, help='weight for head angle loss')
    parser.add_argument('--l_head_vel', type=float, default=0.5, help='weight for head velocity loss')
    parser.add_argument('--l_head_smooth', type=float, default=0.5, help='weight for head smoothness loss')
    parser.add_argument('--l_head_trans', type=float, default=0.5, help='weight for head translation loss')
    
    # Training parameters
    parser.add_argument('--scheduler', type=str, default='Warmup', choices=['Warmup', 'WarmupThenDecay'], 
                        help='learning rate scheduler')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--warm_iter', type=int, default=5000, help='warmup iterations')
    parser.add_argument('--cos_max_iter', type=int, default=1000000, help='cosine annealing max iterations')
    parser.add_argument('--min_lr_ratio', type=float, default=0.1, help='minimum learning rate ratio')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    
    # Sequence parameters
    parser.add_argument('--n_motions', type=int, default=750, help='number of motion frames per window')
    parser.add_argument('--n_prev_motions', type=int, default=100, help='number of previous motion frames to condition on')
    parser.add_argument('--fps', type=int, default=30, help='frames per second')
    parser.add_argument('--trunc_prob1', type=float, default=0.5, help='probability of truncating the first window')
    parser.add_argument('--trunc_prob2', type=float, default=0.5, help='probability of truncating the second window')
    parser.add_argument('--pad_mode', type=str, default='zero', help='padding mode for truncated sequences')
    parser.add_argument('--rot_repr', type=str, default='euler', help='rotation representation')
    
    # Other settings
    parser.add_argument('--no_head_pose', action='store_true', help='disable head pose prediction')
    parser.add_argument('--do_ignore_shape', action='store_true', help='ignore shape parameters')
    parser.add_argument('--do_ignore_cfg', action='store_true', help='ignore classifier-free guidance during training')
    parser.add_argument('--log_iter', type=int, default=100, help='log interval')
    parser.add_argument('--save_iter', type=int, default=10000, help='save interval')
    parser.add_argument('--val_iter', type=int, default=10000, help='validation interval')
    parser.add_argument('--log_smooth_win', type=int, default=50, help='smoothing window for logging')
    parser.add_argument('--continue_from', type=str, default=None, help='continue from checkpoint')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load FLAME model if using vertex space
    flame = None
    if (args.l_vert > 0 or args.l_vel > 0) and args.use_vertex_space:
        if args.dataset_type[:9] == "HDTF_TFHP" or args.dataset_type == 'flame_mead_ravdess':
            flame = FLAME(FLAMEConfig).to(device)
    
    # Create style encoder and diffusion model
    style_enc = get_style_encoder(args, args.style_enc_model_style).to(device)
    model = get_diffusion_model(args, device=device)
    
    # Load pretrained model if continuing from checkpoint
    start_iter = 0
    if args.continue_from is not None:
        exp_dir = Path(args.continue_from)
        print(f"Starting from pre-trained, continuing from {exp_dir}")
        args, model, style_enc, start_iter = load_pretrained_model(args, model, style_enc, parser=parser)
        print("Pretrained model loaded")
    else:
        # Create experiment directory
        exp_dir = Path('/experiments/DPT') / f'{args.exp_name}-{datetime.now().strftime("%y%m%d_%H%M%S")}'
        exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Move models to device
    model.to(device)
    style_enc.to(device)
    
    if args.mode == 'train':
        # Set up optimizer
        optimizer = torch.optim.Adam([
            {'params': filter(lambda p: p.requires_grad, style_enc.parameters()), 'lr': args.lr},
            {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}
        ])
        
        # Save args
        save_args(args, exp_dir)
        
        # Set up logging directories
        log_dir = exp_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        out_abc_dir = exp_dir / 'out_abc'
        out_abc_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard writer
        writer = SummaryWriter(str(log_dir))
        print(Back.RED + Fore.YELLOW + Style.BRIGHT + exp_dir.name + Style.RESET_ALL)
        print('Model parameters: ', count_parameters(model))
        
        # Load datasets
        print(f"Loading dataset {args.dataset_type}")
        train_dataset, val_dataset, train_loader, val_loader = get_dataset(args, device)
        
        # Set up learning rate scheduler
        if args.scheduler == 'Warmup':
            from utils.scheduler import GradualWarmupScheduler
            scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter)
        elif args.scheduler == 'WarmupThenDecay':
            from utils.scheduler import GradualWarmupScheduler
            after_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cos_max_iter - args.warm_iter,
                                                                 args.lr * args.min_lr_ratio)
            scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter, after_scheduler)
        else:
            scheduler = None
        # Start training
        train(args, model, style_enc, train_loader, val_loader, optimizer, exp_dir / 'checkpoints', 
              scheduler, writer, flame, out_abc_dir=out_abc_dir, start_iter=start_iter)
    else:
        # Testing mode
        print(f"Loading dataset {args.dataset_type} for testing")
        args.batch_size = min(args.batch_size, 2)  # Use smaller batch size for testing
        train_dataset, val_dataset, train_loader, val_loader = get_dataset(args, device)
        print("Dataset loaded")
        
        # Run test with provided loss weights
        with torch.no_grad():
            test_results = test(
                args, 
                load_loss_weights(args, device), 
                model, 
                style_enc, 
                val_loader, 
                args.max_iter, 
                n_rounds=5, 
                mode='test', 
                writer=None, 
                flame=flame
            )
            
            print("Test results:")
            for key in test_results:
                if isinstance(test_results[key], list) and len(test_results[key]) > 0:
                    print(f"{key}: {np.mean(test_results[key]):.4f}")


if __name__ == '__main__':
    main()