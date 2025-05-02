import torch.nn as nn
import sys
import torch
from utils.model_common import PositionalEncoding
from utils.flame import FLAME, FLAMEConfig

def get_style_encoder(args, style_encoder_model_style="diffposetalk"):
    print(args.dataset_type)

    if style_encoder_model_style == "vae2":
        print("training model: StyleEncoder_VAE2")
        return StyleEncoder_VAE2(args)
    
class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims  # Tuple of dimensions to permute

    def forward(self, x):
        return x.permute(*self.dims)

class StyleEncoder_VAE(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.input_dim = 67
        if args.dataset_type[:9] == 'HDTF_TFHP' or args.dataset_type == "flame_mead_ravdess":
            self.input_dim = 54

        self.motion_coef_dim = self.input_dim
        self.conv_feature_dim = 512
        self.output_size = args.d_style * 2 * 2

        self.pre_conv_permute = Permute((0, 2, 1))
        self.post_conv_permute = Permute((0, 2, 1))
        # these are the input layers
        self.input_layers = [
            # conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.motion_coef_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),            
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
        ]
        self.input_layers = nn.Sequential(*self.input_layers)

        # apply positional encoding
        self.PE = PositionalEncoding(self.conv_feature_dim)

        # one transformer decoder
        self.encoder = nn.TransformerEncoderLayer(
            d_model=self.conv_feature_dim, nhead=8, dim_feedforward=self.conv_feature_dim, activation='gelu', batch_first=True
        )

        # end up with two more 1D conv layers
        self.output_layers = [
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.1),
            nn.ReLU(),
            # apply layer norm
            nn.LayerNorm(self.output_size),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.output_size, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.ReLU(),
        ]
        self.output_layers = nn.Sequential(*self.output_layers)
    
    def forward(self, motion_coef, do_sample=False):
        """
        :param motion_coef: (batch_size, seq_len, motion_coef_dim)
        :param audio: (batch_size, seq_len)
        :return: (batch_size, feature_dim)
        """
        batch_size, seq_len, _ = motion_coef.shape
        # Motion
        motion_feat = self.input_layers(motion_coef)
        motion_feat = self.PE(motion_feat)
        
        feat = self.encoder(motion_feat)

        out = self.output_layers(feat)

        # average pooling
        out = out.mean(dim=1) # dim 1 is the seq_len

        mu = out[:, :self.output_size//2]
        logvar = out[:, self.output_size//2:]
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        # determine if we should sample or not
        if do_sample:
            return mu + eps * std
        else:
            out = mu + eps * std
            return out, mu, logvar
    
    def sample(self, motion_coef):
        out, mu, logvar = self.forward(motion_coef)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class StyleEncoder_VAE2(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.input_dim = 67
        if args.dataset_type[:9] == 'HDTF_TFHP' or args.dataset_type == "flame_mead_ravdess":
            self.input_dim = 54

        self.motion_coef_dim = self.input_dim
        self.conv_feature_dim = 512
        self.output_size = args.d_style * 2

        self.pre_conv_permute = Permute((0, 2, 1))
        self.post_conv_permute = Permute((0, 2, 1))
        # these are the input layers
        self.input_layers = [
            # conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.motion_coef_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),            
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
        ]
        self.input_layers = nn.Sequential(*self.input_layers)

        # apply positional encoding
        self.PE = PositionalEncoding(self.conv_feature_dim)

        # one transformer decoder
        self.encoder = nn.TransformerEncoderLayer(
            d_model=self.conv_feature_dim, nhead=8, dim_feedforward=self.conv_feature_dim, activation='gelu', batch_first=True
        )

        # end up with two more 1D conv layers
        self.output_layers = [
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.1),
            nn.ELU(),
            # apply layer norm
            nn.LayerNorm(self.output_size),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.output_size, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
        ]
        self.output_layers = nn.Sequential(*self.output_layers)
    
    def forward(self, motion_coef, do_sample=False):
        """
        :param motion_coef: (batch_size, seq_len, motion_coef_dim)
        :param audio: (batch_size, seq_len)
        :return: (batch_size, feature_dim)
        """
        batch_size, seq_len, _ = motion_coef.shape
        # Motion
        motion_feat = self.input_layers(motion_coef)
        motion_feat = self.PE(motion_feat)
        
        feat = self.encoder(motion_feat)

        out = self.output_layers(feat)

        # average pooling
        out = out.mean(dim=1) # dim 1 is the seq_len

        mu = out[:, :self.output_size//2]
        logvar = out[:, self.output_size//2:] # this cannot have a relu since we need negative values!!!!!
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        # determine if we should sample or not
        if do_sample:
            return mu + eps * std
        else:
            out = mu + eps * std
            return out, mu, logvar
    
    def sample(self, motion_coef):
        out, mu, logvar = self.forward(motion_coef)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
