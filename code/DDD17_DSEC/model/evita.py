import math
import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from torchvision.ops import deform_conv2d

from model.decoder.segformer_decoder import SegFormerHead

# ==============================================================================
# Base Components
# ==============================================================================

class LayerNorm(nn.Module):
    """
    Custom LayerNorm supporting channels_last and channels_first formats.
    
    Inputs:
        - x (Tensor): Input feature map. Dimensions depend on data_format.
    Outputs:
        - out (Tensor): Normalized feature map. Dimensions match input.
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MLP(nn.Module):
    """
    Multilayer Perceptron for feature transformation with depthwise convolution.
    
    Inputs:
        - x (Tensor): Input features [B, H, W, C]
    Outputs:
        - out (Tensor): Transformed features [B, H, W, C]
    """
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pos(x) + x
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        x = self.fc2(x)
        return x


# ==============================================================================
# Core Co-Learning Modules
# ==============================================================================

class GeometricParallaxRectification(nn.Module):
    """
    Dynamically aligns structural boundaries between RGB and Event streams.
    
    Inputs:
        - rgb_feat (Tensor): Reduced RGB features [B, H, W, C_e]
        - evt_feat (Tensor): Event features [B, H, W, C_e]
    Outputs:
        - aligned_evt (Tensor): Spatially rectified Event features [B, H, W, C_e]
    """
    def __init__(self, dim, kernel_size=3, max_offset=10.0):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.max_offset = max_offset

        self.displacement_generator = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 2, kernel_size=7, padding=3, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1),
            nn.GELU()
        )

        self.offset_conv = nn.Conv2d(dim * 2, 3 * kernel_size * kernel_size, kernel_size=kernel_size, padding=kernel_size // 2)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        self.proj = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=self.padding, groups=dim)

    def forward(self, rgb_feat: Tensor, evt_feat: Tensor) -> Tensor:
        rgb_spatial = rgb_feat.permute(0, 3, 1, 2)
        evt_spatial = evt_feat.permute(0, 3, 1, 2)

        feat_diff = rgb_spatial - evt_spatial
        combined_context = torch.cat([rgb_spatial, evt_spatial, feat_diff], dim=1)

        offset_features = self.displacement_generator(combined_context)
        routing_params = self.offset_conv(offset_features)

        f_x, f_y, f_m = torch.chunk(routing_params, 3, dim=1)
        continuous_offset = torch.cat((f_x, f_y), dim=1)
        bounded_offset = torch.tanh(continuous_offset) * self.max_offset
        modulation_mask = torch.sigmoid(f_m)

        aligned_evt = deform_conv2d(
            input=evt_spatial,
            offset=bounded_offset,
            weight=self.proj.weight,
            bias=self.proj.bias,
            padding=self.padding,
            mask=modulation_mask
        )
        return aligned_evt.permute(0, 2, 3, 1)


class HarmonicSpectralResonance(nn.Module):
    """
    Executes cross-spectral texture transfer in the complex frequency domain.
    
    Inputs:
        - rgb_feat (Tensor): RGB features [B, H, W, C_e]
        - evt_feat (Tensor): Aligned Event features [B, H, W, C_e]
    Outputs:
        - fused_feat (Tensor): Frequency-injected RGB features [B, H, W, C_e]
    """
    def __init__(self, dim):
        super().__init__()
        self.spectral_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        self.spatial_fuse = nn.Conv2d(dim, dim, 1)

    def forward(self, rgb_feat: Tensor, evt_feat: Tensor) -> Tensor:
        B, H, W, C = rgb_feat.shape
        dtype = rgb_feat.dtype

        rgb_fp32 = rgb_feat.to(torch.float32).permute(0, 3, 1, 2)
        evt_fp32 = evt_feat.to(torch.float32).permute(0, 3, 1, 2)

        fft_rgb = torch.fft.rfft2(rgb_fp32, dim=(-2, -1), norm='ortho')
        fft_evt = torch.fft.rfft2(evt_fp32, dim=(-2, -1), norm='ortho')

        amp_rgb = torch.abs(fft_rgb)
        amp_evt = torch.abs(fft_evt)

        freq_context = torch.mean(amp_rgb, dim=(2, 3))
        gating_weight = self.spectral_gate(freq_context).view(B, C, 1, 1)

        enhanced_amp = amp_rgb + gating_weight * amp_evt
        phase_rgb = torch.angle(fft_rgb)
        
        complex_fusion = torch.polar(enhanced_amp, phase_rgb)
        fused_spatial = torch.fft.irfft2(complex_fusion, s=(H, W), dim=(-2, -1), norm='ortho')
        fused_spatial = fused_spatial.to(dtype).permute(0, 2, 3, 1)

        return rgb_feat + self.spatial_fuse(fused_spatial.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class TransientGlobalRouting(nn.Module):
    """
    Explicitly extracts kinematic prior from the event stream to dynamically route 
    macroscopic photometric context from the RGB stream.
    
    Inputs:
        - rgb_stream (Tensor): Normalised RGB features [B, H, W, C]
        - evt_stream (Tensor): Normalised Event features [B, H, W, C // 2]
    Outputs:
        - rgb_out (Tensor): Routed RGB features [B, H, W, C]
        - evt_out (Tensor): Routed Event features [B, H, W, C // 2]
    """
    def __init__(self, dim, num_head=8, window=7):
        super().__init__()
        self.num_head = num_head
        self.window = window
        self.dim_reduced = dim // 2

        # RGB Projections
        self.rgb_q_proj = nn.Linear(dim, dim)
        self.rgb_dim_reduce = nn.Linear(dim, self.dim_reduced)
        self.rgb_gating_proj = nn.Linear(dim, dim)
        self.rgb_value_proj = nn.Linear(dim, dim)
        self.rgb_spatial_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        
        # Event Projections
        self.evt_spatial_conv = nn.Conv2d(self.dim_reduced, self.dim_reduced, 7, padding=3, groups=self.dim_reduced)
        self.evt_embed_in = nn.Linear(self.dim_reduced, self.dim_reduced)
        self.evt_embed_out = nn.Linear(self.dim_reduced, self.dim_reduced)

        # Core Modules
        self.gpr = GeometricParallaxRectification(dim=self.dim_reduced)
        self.hsr = HarmonicSpectralResonance(self.dim_reduced)

        if window != 0:
            self.transient_prior_extractor = nn.Sequential(
                nn.Conv2d(self.dim_reduced, num_head, kernel_size=3, padding=1),
            )
            nn.init.constant_(self.transient_prior_extractor[0].weight, 0)
            nn.init.constant_(self.transient_prior_extractor[0].bias, 0)

        concat_dim = dim * 2 if window != 0 else dim + self.dim_reduced
        self.fusion_out_rgb = nn.Linear(concat_dim, dim)
        self.fusion_out_evt = nn.Linear(concat_dim, self.dim_reduced)

        if window != 0:
            self.kinematic_query_proj = nn.Linear(self.dim_reduced * 3, self.dim_reduced)
            self.kinematic_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
            self.rgb_kv_proj = nn.Linear(dim, dim)

        self.act = nn.GELU()
        self.norm_rgb = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm_evt = LayerNorm(self.dim_reduced, eps=1e-6, data_format="channels_last")

    def forward(self, rgb_stream: Tensor, evt_stream: Tensor) -> tuple[Tensor, Tensor]:
        B, H, W, C = rgb_stream.size()
        rgb_normed = self.norm_rgb(rgb_stream)
        evt_normed = self.norm_evt(evt_stream)

        # Phase 1: Spatial Dimension Alignment
        rgb_reduced = self.rgb_dim_reduce(rgb_normed)
        evt_context = self.evt_embed_in(evt_normed).permute(0, 3, 1, 2)
        evt_context = self.evt_spatial_conv(evt_context).permute(0, 2, 3, 1)
        evt_context = self.evt_embed_out(evt_context)

        # Phase 2: Geometric Parallax Rectification
        aligned_evt = self.gpr(rgb_reduced, evt_context)

        # Prepare Kinematic Anchor for Routing
        if self.window != 0:
            kinematic_anchor = torch.cat([rgb_normed, aligned_evt], dim=3).permute(0, 3, 1, 2)

        # Phase 3: RGB Key/Value Extraction
        rgb_query_state = self.rgb_q_proj(rgb_normed)
        rgb_value_state = self.act(self.rgb_value_proj(rgb_normed).permute(0, 3, 1, 2))
        rgb_gate_state = self.rgb_gating_proj(self.rgb_spatial_conv(rgb_value_state).permute(0, 2, 3, 1))

        # Phase 4: Cross-Modal Transient Routing (Window Attention)
        if self.window != 0:
            rgb_kv_state = self.rgb_kv_proj(rgb_value_state.permute(0, 2, 3, 1))
            head_dim = C // self.num_head // 2
            
            rgb_kv_state = rgb_kv_state.reshape(B, H * W, 2, self.num_head, head_dim).permute(2, 0, 3, 1, 4)
            rgb_k, rgb_v = rgb_kv_state.unbind(0)

            # Build Kinematic Query
            kinematic_anchor_pooled = self.kinematic_pool(kinematic_anchor).permute(0, 2, 3, 1)
            kinematic_query = self.kinematic_query_proj(kinematic_anchor_pooled)
            kinematic_query = kinematic_query.reshape(B, -1, self.num_head, head_dim).permute(0, 2, 1, 3)

            # Scaled Dot-Product Routing
            scale = head_dim ** -0.5
            routing_scores = (kinematic_query * scale) @ rgb_k.transpose(-2, -1)
            
            # Inject Transient Prior
            transient_prior = self.transient_prior_extractor(aligned_evt.permute(0, 3, 1, 2))
            transient_prior = transient_prior.view(B, self.num_head, H * W).unsqueeze(2)
            
            routing_scores = routing_scores + transient_prior
            routing_probs = routing_scores.softmax(dim=-1)
            
            # Value Aggregation
            dynamic_routing_feat = (routing_probs @ rgb_v)
            dynamic_routing_feat = dynamic_routing_feat.reshape(B, self.num_head, self.window, self.window, head_dim)
            dynamic_routing_feat = dynamic_routing_feat.permute(0, 1, 4, 2, 3).reshape(B, C // 2, self.window, self.window)
            dynamic_routing_feat = F.interpolate(dynamic_routing_feat, (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        # Phase 5: Harmonic Spectral Resonance
        hsr_fused = self.hsr(rgb_reduced, aligned_evt)
        main_rgb_branch = rgb_query_state * rgb_gate_state

        # Phase 6: Dual-Stream Aggregation
        if self.window != 0:
            aggregated_tensor = torch.cat([main_rgb_branch, dynamic_routing_feat, hsr_fused], dim=3)
        else:
            aggregated_tensor = torch.cat([main_rgb_branch, hsr_fused], dim=3)

        rgb_out = self.fusion_out_rgb(aggregated_tensor)
        evt_out = self.fusion_out_evt(aggregated_tensor)

        return rgb_out, evt_out


class EvitaBlock(nn.Module):
    """
    The core intertwined hierarchical building block of the Evita backbone.
    
    Inputs:
        - rgb_feat (Tensor): RGB features [B, H, W, C]
        - evt_feat (Tensor): Event features [B, H, W, C // 2]
    Outputs:
        - out_rgb (Tensor): Updated RGB features [B, H, W, C]
        - out_evt (Tensor): Updated Event features [B, H, W, C // 2]
    """
    def __init__(self, index, dim, num_head, window=7, mlp_ratio=4., drop_path=0., block_index=0, last_block_index=50):
        super().__init__()
        self.index = index
        layer_scale_init_value = 1e-6
        if block_index > last_block_index:
            window = 0
            
        self.tgr = TransientGlobalRouting(dim, num_head, window=window)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp_rgb = MLP(dim, mlp_ratio)
        self.mlp_evt = MLP(dim // 2, mlp_ratio)
        
        self.layer_scale_rgb_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_rgb_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_evt_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim // 2)), requires_grad=True)
        self.layer_scale_evt_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim // 2)), requires_grad=True)

    def forward(self, rgb_feat: Tensor, evt_feat: Tensor) -> tuple[Tensor, Tensor]:
        res_rgb, res_evt = rgb_feat, evt_feat
        rgb_feat, evt_feat = self.tgr(rgb_feat, evt_feat)

        evt_feat = res_evt + self.drop_path(self.layer_scale_evt_1.unsqueeze(0).unsqueeze(0) * evt_feat)
        rgb_feat = res_rgb + self.drop_path(self.layer_scale_rgb_1.unsqueeze(0).unsqueeze(0) * rgb_feat)

        evt_feat = evt_feat + self.drop_path(self.layer_scale_evt_2.unsqueeze(0).unsqueeze(0) * self.mlp_evt(evt_feat))
        rgb_feat = rgb_feat + self.drop_path(self.layer_scale_rgb_2.unsqueeze(0).unsqueeze(0) * self.mlp_rgb(rgb_feat))
        
        return rgb_feat, evt_feat


# ==============================================================================
# Backbone Architecture
# ==============================================================================

class EvitaArchitecture(nn.Module):
    """
    Evita Unified Backbone for Dense RGB-Event Parsing.
    """
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 windows=[7, 7, 7, 7], mlp_ratios=[4, 4, 4, 4], last_block=[50, 50, 50, 50], num_heads=[2, 4, 10, 16],
                 drop_path_rate=0.10, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.downsample_layers_rgb = nn.ModuleList()
        self.downsample_layers_evt = nn.ModuleList()

        # Stem configurations
        stem_rgb = nn.Sequential(
            nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
        )

        stem_evt = nn.Sequential(
            nn.Conv2d(10, dims[0] // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 4),
            nn.GELU(),
            nn.Conv2d(dims[0] // 4, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
        )

        self.downsample_layers_rgb.append(stem_rgb)
        self.downsample_layers_evt.append(stem_evt)

        # Intermediate downsampling
        for i in range(len(dims) - 1):
            stride = 2
            ds_rgb = nn.Sequential(
                nn.BatchNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=stride, padding=1),
            )
            self.downsample_layers_rgb.append(ds_rgb)

            ds_evt = nn.Sequential(
                nn.BatchNorm2d(dims[i] // 2),
                nn.Conv2d(dims[i] // 2, dims[i + 1] // 2, kernel_size=3, stride=stride, padding=1),
            )
            self.downsample_layers_evt.append(ds_evt)

        # Intertwined Co-learning Stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[EvitaBlock(index=cur + j, dim=dims[i], window=windows[i], drop_path=dp_rates[cur + j],
                             num_head=num_heads[i], mlp_ratio=mlp_ratios[i], block_index=depths[i] - j,
                             last_block_index=last_block[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.pred = nn.Linear(dims[-1] // 2 * 3, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, LayerNorm, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, evt_tensor: Tensor, rgb_tensor: Tensor) -> list[Tensor]:
        if rgb_tensor.shape[1] != 3:
            rgb_tensor = rgb_tensor.repeat(1, 3, 1, 1)

        multi_scale_features = []
        for i in range(4):
            rgb_tensor = self.downsample_layers_rgb[i](rgb_tensor)
            evt_tensor = self.downsample_layers_evt[i](evt_tensor)

            rgb_tensor = rgb_tensor.permute(0, 2, 3, 1)
            evt_tensor = evt_tensor.permute(0, 2, 3, 1)

            for blk in self.stages[i]:
                rgb_tensor, evt_tensor = blk(rgb_tensor, evt_tensor)
                
            rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)
            evt_tensor = evt_tensor.permute(0, 3, 1, 2)
            
            multi_scale_features.append(rgb_tensor)

        return multi_scale_features

    def forward(self, x):
        return self.forward_features(x)

# ==============================================================================
# Model Variants & Downstream Wrapper
# ==============================================================================

@register_model
def Evita_Pico(pretrained=False, **kwargs):
    model = EvitaArchitecture(dims=[16, 32, 64, 128], mlp_ratios=[4, 4, 4, 4], depths=[2, 2, 2, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7], drop_path_rate=0.10, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def Evita_Nano(pretrained=False, **kwargs):
    model = EvitaArchitecture(dims=[32, 64, 128, 256], mlp_ratios=[4, 4, 4, 4], depths=[2, 2, 2, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7], drop_path_rate=0.10, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def Evita_Tiny(pretrained=False, **kwargs):
    model = EvitaArchitecture(dims=[32, 64, 128, 256], mlp_ratios=[8, 8, 4, 4], depths=[3, 3, 5, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7], drop_path_rate=0.10, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def Evita_Small(pretrained=False, **kwargs):
    model = EvitaArchitecture(dims=[64, 128, 256, 512], mlp_ratios=[8, 8, 4, 4], depths=[2, 2, 4, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7], drop_path_rate=0.10, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def Evita_Base(pretrained=False, **kwargs):
    model = EvitaArchitecture(dims=[64, 128, 256, 512], mlp_ratios=[8, 8, 4, 4], depths=[3, 3, 12, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7], drop_path_rate=0.15, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def Evita_Large(pretrained=False, **kwargs):
    model = EvitaArchitecture(dims=[96, 192, 288, 576], mlp_ratios=[8, 8, 4, 4], depths=[3, 3, 12, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7], drop_path_rate=0.20, **kwargs)
    model.default_cfg = _cfg()
    return model


class Evita(nn.Module):
    def __init__(self, backbone: str = 'Evita_Large', num_classes: int = 25, pretrained_path: str = None) -> None:
        super().__init__()

        BACKBONE_REGISTRY = {
            'Evita_Pico':  {'builder': Evita_Pico,  'channels': [16, 32, 64, 128],   'embed_dim': 128},
            'Evita_Nano':  {'builder': Evita_Nano,  'channels': [32, 64, 128, 256],  'embed_dim': 256},
            'Evita_Tiny':  {'builder': Evita_Tiny,  'channels': [32, 64, 128, 256],  'embed_dim': 256},
            'Evita_Small': {'builder': Evita_Small, 'channels': [64, 128, 256, 512], 'embed_dim': 512},
            'Evita_Base':  {'builder': Evita_Base,  'channels': [64, 128, 256, 512], 'embed_dim': 512},
            'Evita_Large': {'builder': Evita_Large, 'channels': [96, 192, 288, 576], 'embed_dim': 576},
        }

        if backbone not in BACKBONE_REGISTRY:
            raise ValueError(f"Unsupported backbone: {backbone}. Available options: {list(BACKBONE_REGISTRY.keys())}")

        config = BACKBONE_REGISTRY[backbone]

        self.head_in_channels = config['channels']
        self.num_classes = num_classes
        self.current_embed_dim = config['embed_dim']

        self.backbone = config['builder']()
        
        self.decode_head = SegFormerHead(
            in_channels=self.head_in_channels, 
            embed_dim=self.current_embed_dim, 
            num_classes=self.num_classes
        )

        self.apply(self._init_weights)

        if pretrained_path:
            self._load_pretrained_backbone(pretrained_path)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _load_pretrained_backbone(self, path: str) -> None:
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))

        backbone_state_dict = self.backbone.state_dict()
        new_state_dict = {}

        ckpt_keys_map = {k.replace('module.', ''): k for k in state_dict.keys()}

        for k, v in backbone_state_dict.items():
            k_clean = k.replace('module.', '')
            ckpt_key = ckpt_keys_map.get(k_clean, k)

            if ckpt_key in state_dict:
                ckpt_weight = state_dict[ckpt_key]
                if ckpt_weight.shape == v.shape:
                    new_state_dict[k] = ckpt_weight
                else:
                    print(f"Shape mismatch for {k}: model expects {v.shape}, checkpoint has {ckpt_weight.shape}")

        self.backbone.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained backbone weights from {path}")

    def load_state_dict(self, state_dict, strict=True):
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            error_msg = str(e)
            if 'size mismatch' in error_msg and 'decode_head' in error_msg:
                print("Detected embed_dim mismatch in decode_head! Attempting to fix...")
                fuse_key = 'decode_head.linear_fuse.0.weight'
                if fuse_key in state_dict:
                    true_embed_dim = state_dict[fuse_key].shape[0]
                    print(f"Automatically re-initializing SegFormerHead with correct embed_dim = {true_embed_dim}...")
                    
                    self.current_embed_dim = true_embed_dim
                    self.decode_head = SegFormerHead(
                        in_channels=self.head_in_channels, 
                        embed_dim=self.current_embed_dim, 
                        num_classes=self.num_classes
                    )
                    self.decode_head.to(next(self.parameters()).device)
                    return super().load_state_dict(state_dict, strict=strict)
            raise e

    def forward(self, ev_rep: Tensor, img: Tensor) -> Tensor:
        multi_scale_features = self.backbone.forward_features(ev_rep, img)
        logits = self.decode_head(multi_scale_features)
        logits = F.interpolate(logits, size=img.shape[2:], mode='bilinear', align_corners=False)
        return logits