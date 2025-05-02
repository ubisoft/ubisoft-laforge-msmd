#!/usr/bin/env python
import argparse
import os
import sys
import gc
import math
from pathlib import Path
import pickle as pkl
from subprocess import check_call

import cv2
from scipy.interpolate import interp1d

import numpy as np
import torch
import torch.nn.functional as F
import librosa
from scipy.io import wavfile
import json
# Import custom modules (make sure these paths are correct for your installation)
from model import get_diffusion_model
from style_encoder import get_style_encoder
from datasets import get_dataset
from utils.flame import FLAME, FLAMEConfig
from utils.model_common import load_pretrained_model, save_args
from utils.common import compute_loss_no_vert, compute_loss
from utils.common import compute_KL_loss
import utils 
from models import get_diffusion_model

# =============================================================================
# Helper functions (as in your original code)
# =============================================================================
@torch.no_grad()
def infer_coeffs(model, args, audio, shape_coef, audio_unit, style_feats=None,
                 n_repetitions: int = 1, cfg_mode=None, cfg_cond=None, cfg_scale: float = 1.15,
                 include_shape: bool = False, dynamic_threshold=(0, 1, 4)):
    clip_len = int(len(audio) / 16000 * args.fps)
    stride = args.n_motions
    n_audio_samples = round(audio_unit * args.n_motions)
    n_subdivision = 1 if clip_len <= args.n_motions else math.ceil(clip_len / stride)
    n_padding_audio_samples = n_audio_samples * n_subdivision - len(audio)
    n_padding_frames = math.ceil(n_padding_audio_samples / audio_unit)
    if n_padding_audio_samples > 0:
        audio = F.pad(audio, (0, n_padding_audio_samples), value=0)
    audio_feat = model.extract_audio_feature(audio.unsqueeze(0), args.n_motions * n_subdivision)
    coef_list = []
    for i in range(n_subdivision):
        start_idx = i * stride
        end_idx = start_idx + args.n_motions
        indicator = torch.ones((n_repetitions, args.n_motions)).to(model.device) if args.use_indicator else None
        if indicator is not None and i == n_subdivision - 1 and n_padding_frames > 0:
            indicator[:, -n_padding_frames:] = 0
        audio_in = audio_feat[:, start_idx:end_idx].expand(n_repetitions, -1, -1)
        style_feat = style_feats[i] if isinstance(style_feats, list) else style_feats
        if i == 0:
            motion_feat, noise, prev_audio_feat = model.sample(
                audio_in, shape_coef, style_feat, indicator=indicator,
                cfg_mode=cfg_mode, cfg_cond=cfg_cond, cfg_scale=cfg_scale,
                dynamic_threshold=dynamic_threshold
            )
        else:
            motion_feat, noise, prev_audio_feat = model.sample(
                audio_in, shape_coef, style_feat, prev_motion_feat, prev_audio_feat, noise,
                indicator=indicator, cfg_mode=cfg_mode, cfg_cond=cfg_cond, cfg_scale=cfg_scale,
                dynamic_threshold=dynamic_threshold
            )
        prev_motion_feat = motion_feat[:, -args.n_prev_motions:].clone()
        prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
        motion_coef = motion_feat
        if i == n_subdivision - 1 and n_padding_frames > 0:
            motion_coef = motion_coef[:, :-n_padding_frames]
        coef_list.append(motion_coef)
    motion_coef = torch.cat(coef_list, dim=1)
    return motion_coef

# =============================================================================
# Model-loading function
# =============================================================================
def load_args(save_dir):
    with open(save_dir / 'args.json', 'r') as f:
        args_dict = json.load(f)
    args = argparse.Namespace(**args_dict)
    return args
def load_model(model_root: str, model_name: str, iter_num: str, device: torch.device):
    """
    Loads the diffusion talking-head model and the style encoder.
    """
    # Load the training arguments
    model_args = load_args(Path(os.path.join(model_root, "DPT", model_name)))
    # (Optionally adjust dataset paths here as needed)
    # Create the main model
    model = get_diffusion_model(model_args)
    model_ckpt_path = Path(model_root) / "DPT" / model_name / "checkpoints" / f"iter_{iter_num}.pt"
    model_data = torch.load(model_ckpt_path, map_location=device)
    enc_style = model_args.style_enc_model_style
    enc_model = get_style_encoder(model_args, enc_style)
    enc_model.load_state_dict(model_data['style_enc'])
    enc_model.eval()
    style_enc = enc_model
    model.load_state_dict(model_data['model'])
    model.eval()
    return model, style_enc, model_args

# =============================================================================
# loading expression code function
# =============================================================================

def query_for_motion_coeff(args: argparse.Namespace,
                            expression_code_full_path: str,
                           head_rot_full_path: str,
                           device: str = "cuda",
                           original_fps: float = 30,
                           target_fps: float = 25):
    """
    Loads expression code and head rotation from the given full file paths,
    normalizes them using coefficient statistics, optionally resamples them to a target FPS,
    and returns the normalized motion coefficients and a dummy shape coefficient tensor.
    
    Parameters:
        expression_code_full_path (str): Full path to the expression code pkl file.
        head_rot_full_path (str): Full path to the head rotation pkl file.
        device (str): The device to load the tensors onto (e.g. "cuda" or "cpu").
        original_fps (float, optional): The original frames per second of the data.
            If provided and different from target_fps, the data will be resampled.
        target_fps (float): The desired frames per second after resampling (default: 25).
    
    Returns:
        motion_coeff (torch.Tensor): A tensor (with a batch dimension) containing the normalized and,
                                     if needed, resampled motion coefficients (expression + head rotation).
        shape_coef (torch.Tensor): A dummy shape coefficient tensor of shape (1, 100).
    """
    # Load coefficient statistics (assumes they are stored as a tensor in a pkl file)
    coef_stats_path = args.coef_dict_path
    with open(coef_stats_path, "rb") as f:
        coef_stats = pkl.load(f)

    # Load expression code and head rotation using pkl
    expression_coef = pkl.load(open(expression_code_full_path, "rb"))
    head_rot = pkl.load(open(head_rot_full_path, "rb"))
    
    # If the loaded expression code is a tensor, detach and convert to numpy.
    expression_coef = expression_coef.detach().cpu().numpy()
    # If head_rot is a tensor, convert it similarly.
    if isinstance(head_rot, torch.Tensor):
        head_rot = head_rot.detach().cpu().numpy()
    
    # Normalize using coefficient statistics (adding a small epsilon to avoid division by zero)
    exp_mean = coef_stats['exp_mean'].detach().cpu().numpy()
    exp_std  = coef_stats['exp_std'].detach().cpu().numpy() + 1e-9
    pose_mean = coef_stats['pose_mean'].detach().cpu().numpy()
    pose_std  = coef_stats['pose_std'].detach().cpu().numpy() + 1e-9

    expression_coef = (expression_coef - exp_mean) / exp_std
    head_rot = (head_rot - pose_mean) / pose_std
    
    # Optionally resample to target_fps if original_fps is provided and is different
    if original_fps is not None and original_fps != target_fps:
        num_frames = expression_coef.shape[0]
        # Create a normalized time axis for the current frames
        x = np.linspace(0, 1, num=num_frames)
        # Determine the new number of frames based on the desired target FPS
        new_num_frames = int(round(num_frames / original_fps * target_fps))
        xnew = np.linspace(0, 1, num=new_num_frames)
        
        # Resample the expression coefficients and head rotation along the time axis
        f_exp = interp1d(x, expression_coef, axis=0)
        expression_coef = f_exp(xnew)
        
        f_head = interp1d(x, head_rot, axis=0)
        head_rot = f_head(xnew)
    
    # Convert the arrays to torch tensors and add a batch dimension
    expression_tensor = torch.from_numpy(expression_coef).to(device).unsqueeze(0).float()
    head_rot_tensor = torch.from_numpy(head_rot).to(device).unsqueeze(0).float()
    
    # Create a dummy shape coefficient tensor of zeros (shape: [1, 100])
    shape_coef = torch.zeros((1, 100), device=device).float()
    
    # Concatenate expression and head rotation along the last dimension to form motion coefficients
    motion_coeff = torch.cat([expression_tensor, head_rot_tensor], dim=2).float().to(device)
    
    return motion_coeff, shape_coef

# =============================================================================
# Main function: parse arguments and run inference on a single style+audio pair.
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Single inference for MSMD.")
    parser.add_argument("--model_root", type=str, required=True, help="Root directory for models.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model.")
    parser.add_argument("--model_iter", type=str, required=True, help="Checkpoint iteration (as string).")
    parser.add_argument("--style_clip_exp_code_path", type=str, required=True, help="Name of the style video clip.")
    parser.add_argument("--style_clip_head_rot_path", type=str, required=True, help="Name of the style video clip.")
    parser.add_argument("--audio_clip", type=str, required=True, help="Name of the audio clip (without extension).")
    parser.add_argument("--coef_dict_path", type=str, default="PATH-TO-COEF-STATS", help="Path to the coefficient statistics.")
    # Flags (set to default values as specified)
    parser.add_argument("--cfg_level", type=float, default=1.4, help="Configuration level (e.g., CFG scale).")
    parser.add_argument("--output_dir", type=str, default="/experiments/refactor", help="Directory to save outputs.")
    parser.add_argument("--versions_of_render", type=int, default=1, help="the number of times to render the video")
    
    # (Any additional arguments such as n_motions, n_prev_motions, fps, etc., should be in your model args.)
    
    Example_argument_list = [
        "--model_root", "/experiments",
        "--model_name", "MSMD", 
        "--model_iter", "0470000",
        "--style_clip_exp_code_path", "/data/expression_code_ver2/video_name.pkl", # <===================== path the video
        "--style_clip_head_rot_path", "/data/head_orientations/video_name.pkl",
        "--audio_clip", "/data/evan_iconic_speech/full_audios/video_name_full_audio.wav",
        "--versions_of_render", "1",
    ]

    # args = parser.parse_args(TEST_argument_list)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and style encoder
    model, style_enc, model_args = load_model(args.model_root, args.model_name, args.model_iter, device)
    model.to(device)
    style_enc.to(device)

    # Query the dataset for the style clip; here we ignore the returned audio.
    motion_coeff, shape_coef = query_for_motion_coeff(args.style_clip_exp_code_path, args.style_clip_head_rot_path, device=device)
    motion_coeff = motion_coeff.to(device)
    shape_coef = shape_coef.unsqueeze(1).to(device)

    # Load the audio clip (assumed to be stored in a known root)
    audio_source_path = args.audio_clip
    audio_data = librosa.load(audio_source_path, sr=16000)[0]
    # Normalize audio
    audio_data = (audio_data - audio_data.mean()) / (audio_data.std() + 1e-5)
    audio_tensor = torch.tensor(audio_data).float().to(device)

    # Compute the style code.
    if model_args.style_enc_model_style.startswith("vae"):
        style_coeff = style_enc.sample(motion_coeff[:, :100, :])
    else:
        style_coeff = style_enc(motion_coeff[:, :100, :]).to(device)

    # ========================= this should store the mean and std of the dataset, used to normalize and un-normalize the expression code =========================
    coef_stats = pkl.load(open(args.coef_dict_path, "rb"))
    # Get coefficient statistics from the dataset and send to device.
    coef_stats = {k: v.to(device) for k, v in coef_stats.items()}

    # Prepare output directories.
    style_clip_name = os.path.splitext(os.path.basename(args.style_clip_exp_code_path))[0]
    audio_clip_name = os.path.splitext(os.path.basename(args.audio_clip))[0]
    output_clip_name = f"style=_{style_clip_name}_audio={audio_clip_name}"
    
    folder_name = f"{args.model_name}_iter_{args.model_iter}"
    save_dir = os.path.join(args.output_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    temp_subfolder = os.path.join(save_dir, "temp")
    os.makedirs(temp_subfolder, exist_ok=True)
    video_subfolder = os.path.join(save_dir, output_clip_name)
    os.makedirs(video_subfolder, exist_ok=True)

    # Save the normalized audio as a .wav file.
    audio_path = os.path.join(temp_subfolder, output_clip_name)
    wavfile.write(audio_path, 16000, audio_tensor.cpu().numpy())
    # -------------------------------------------------------------------------
    for count_i in range(0, args.versions_of_render):
        # Inference
        np.random.seed(count_i)
        torch.manual_seed(count_i)
        with torch.no_grad():
            overall_coef = infer_coeffs(
                model, model_args, audio_tensor, shape_coef, 640.0, style_coeff,
                cfg_scale=args.cfg_level, dynamic_threshold=None
            )
        overall_expression_code = overall_coef[0, :, :-3] * coef_stats['exp_std'] + coef_stats['exp_mean']
        overall_head_rot = overall_coef[0, :, -3:] * coef_stats['pose_std'] + coef_stats['pose_mean']
        overall_exp_code_path = os.path.join(temp_subfolder, f"overall_exp_code_{output_clip_name}_seed_{count_i}.pkl")
        overall_head_rot_path = os.path.join(temp_subfolder, f"overall_head_rot_{output_clip_name}_seed_{count_i}.pkl")
        pkl.dump(overall_expression_code.cpu().numpy(), open(overall_exp_code_path, "wb"))
        pkl.dump(overall_head_rot.cpu().numpy(), open(overall_head_rot_path, "wb"))
        
        # =========================================================================       
        # Use SEREP/FLAME decoder to generate mesh from the expression coefficients now 
        # =========================================================================


if __name__ == "__main__":
    main()
