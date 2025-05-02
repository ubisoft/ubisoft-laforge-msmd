import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import argparse
def save_args(args, save_dir):
    if type(save_dir) == str:
        save_dir = Path(save_dir)
    args_dict = vars(args)
    # convert all path objects to strings
    for key, value in args_dict.items():
        if isinstance(value, Path):
            args_dict[key] = str(value)
    # if any is none ignore it
    to_be_del = []
    for key, value in args_dict.items():
        # print(value)
        if value is None or value == 'None':
            to_be_del.append(key)
            print("Ignoring None value for key:", key)
    for key in to_be_del:
        del args_dict[key]
    with open(save_dir / 'args.json', 'w') as f:
        json.dump(args_dict, f)

def load_args_with_defaults(save_dir, parser):
    """
    Load arguments from a saved file, falling back to default values for new arguments
    
    Args:
        save_dir: Directory containing args.json
        parser: ArgumentParser object containing current argument definitions
    """
    # Load saved arguments
    with open(save_dir / 'args.json', 'r') as f:
        saved_args_dict = json.load(f)
    
    # Get default values from parser
    default_args = vars(parser.parse_args([]))
    
    # Update default args with saved args
    default_args.update(saved_args_dict)
    
    # Create namespace with combined arguments
    args = argparse.Namespace(**default_args)
    return args

def load_args(save_dir):
    with open(save_dir / 'args.json', 'r') as f:
        args_dict = json.load(f)
    args = argparse.Namespace(**args_dict)
    return args

def load_pretrained_model(args, model, style_encoder, device='cuda', parser=None):
    exp_dir = Path(args.continue_from)
    try:
        if parser is None:
            saved_args = load_args(exp_dir)
            saved_args.continue_from = str(exp_dir)
            saved_args.max_iter = args.max_iter
        else:
            saved_args = load_args_with_defaults(exp_dir, parser)
            saved_args.continue_from = str(exp_dir)
            saved_args.max_iter = args.max_iter
        # args = saved_args
    except:
        raise ValueError("Could not load the args from the experiment directory")
    
    checkpoints_dir = exp_dir / 'checkpoints'
    checkpoint_files = sorted(checkpoints_dir.glob('iter_*.pt'))
    if len(checkpoint_files) == 0:
        raise ValueError(f'No checkpoints found in {checkpoints_dir}')
    latest_checkpoint = checkpoint_files[-1]
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    style_encoder.load_state_dict(checkpoint['style_enc'])
    model.load_state_dict(checkpoint['model'])
    start_iter = checkpoint.get('iter', 0)
    return args, model, style_encoder, start_iter
    # else:
    #     raise ValueError(f"Style Encoder Model style {args.style_enc_model_style} and Generator Model style {args.generator_model_style} not recognized")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=600):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # vanilla sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, x.shape[1], :]
        return self.dropout(x)

def enc_dec_mask(T, S, frame_width=2, expansion=0, device='cuda'):
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, max(0, (i - expansion) * frame_width):(i + expansion + 1) * frame_width] = 0
    return (mask == 1).to(device=device)


def pad_audio(audio, audio_unit=320, pad_threshold=80):
    batch_size, audio_len = audio.shape
    n_units = audio_len // audio_unit
    side_len = math.ceil((audio_unit * n_units + pad_threshold - audio_len) / 2)
    if side_len >= 0:
        reflect_len = side_len // 2
        replicate_len = side_len % 2
        if reflect_len > 0:
            audio = F.pad(audio, (reflect_len, reflect_len), mode='reflect')
            audio = F.pad(audio, (reflect_len, reflect_len), mode='reflect')
        if replicate_len > 0:
            audio = F.pad(audio, (1, 1), mode='replicate')

    return audio
