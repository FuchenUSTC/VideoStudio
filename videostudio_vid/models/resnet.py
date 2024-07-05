# -*- encoding: utf-8 -*-
# auth: Fuchen Long
# mail: longfc.ustc@gmail.com
# date: 2023/06/03
# desc: resnet blocks for video diffusion

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import logging
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.activations import get_activation

logger = logging.get_logger('video-diffusion')  # pylint: disable=invalid-name


class TemporalConvLayer(nn.Module):
    """
    Temporal convolutional layer that can be used for video (sequence of images) input Code mostly copied from:
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016
    """

    def __init__(self, in_dim, out_dim=None, dropout=0.0, zero_init=True, use_more_conv=False):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_more_conv = use_more_conv

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), 
            nn.SiLU(), 
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0))
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        
        if use_more_conv:
            self.conv5 = nn.Sequential(
                nn.GroupNorm(32, out_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
            )            

        # zero out the last layer params,so the conv block is identity
        if zero_init:
            logger.info('zero init temporal conv layers. use more conv: {}'.format(self.use_more_conv))
            if not use_more_conv:
                nn.init.zeros_(self.conv4[-1].weight)
                nn.init.zeros_(self.conv4[-1].bias)
            else:
                nn.init.zeros_(self.conv5[-1].weight)
                nn.init.zeros_(self.conv5[-1].bias)                

    def forward(self, hidden_states, num_frames=1):
        
        hidden_states = (
            hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
        )

        identity = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.conv4(hidden_states)
        if self.use_more_conv:
            hidden_states = self.conv5(hidden_states)

        hidden_states = identity + hidden_states

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
        )
        return hidden_states


class TinyTemporalConvLayer(nn.Module):

    def __init__(self, in_dim, out_dim=None, dropout=0.0, zero_init=True):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), 
            nn.SiLU(), 
            nn.Dropout(dropout),
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0))
        )
        #self.conv2 = nn.Sequential(
        #    nn.GroupNorm(32, in_dim), 
        #    nn.SiLU(), 
        #    nn.Conv3d(in_dim, out_dim, (5, 1, 1), padding=(2, 0, 0))
        #)
        # v4-1: (3, 1, 1) (1, 0, 0)
        # v4-2: (3, 3, 3) (1, 1, 1)
        # v5: (1, 5, 5) (0, 2, 2) + (5, 1, 1) (2, 0, 0)

        # zero out the last layer params,so the conv block is identity
        if zero_init:
            logger.info('zero init tiny temporal conv layers.')
            nn.init.zeros_(self.conv1[-1].weight)
            nn.init.zeros_(self.conv1[-1].bias)

    def forward(self, hidden_states, num_frames=1):
        hidden_states = (
            hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
        )

        identity = hidden_states
        hidden_states = self.conv1(hidden_states)

        hidden_states = identity + hidden_states

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
        )
        return hidden_states


# V2
class TemporalConvLayerV2(nn.Module):
    """
    Temporal convolutional layer that can be used for video (sequence of images) input Code mostly copied from:
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016
    """

    def __init__(self, in_dim, out_dim=None, dropout=0.0, zero_init=True):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), 
            nn.SiLU(), 
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0))
        )

        # zero out the last layer params,so the conv block is identity
        if zero_init:
            logger.info('zero init temporal conv layers.')
            nn.init.zeros_(self.conv1[-1].weight)
            nn.init.zeros_(self.conv1[-1].bias)

    def forward(self, hidden_states, num_frames=1):
        hidden_states = (
            hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
        )

        identity = hidden_states
        hidden_states = self.conv1(hidden_states)

        hidden_states = identity + hidden_states

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
        )
        return hidden_states


class TemporalResnetBlock(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        kernel_size = (3, 1, 1)
        padding = [k // 2 for k in kernel_size]

        self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(0.0)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        self.nonlinearity = get_activation("silu")

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, input_tensor: torch.FloatTensor, temb: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, :, None, None]
            temb = temb.permute(0, 2, 1, 3, 4)
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


# VideoResBlock
class SpatioTemporalResBlock(nn.Module):
    r"""
    A SpatioTemporal Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the spatial resenet.
        temporal_eps (`float`, *optional*, defaults to `eps`): The epsilon to use for the temporal resnet.
        merge_factor (`float`, *optional*, defaults to `0.5`): The merge factor to use for the temporal mixing.
        merge_strategy (`str`, *optional*, defaults to `learned_with_images`):
            The merge strategy to use for the temporal mixing.
        switch_spatial_to_temporal_mix (`bool`, *optional*, defaults to `False`):
            If `True`, switch the spatial and temporal mixing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
        temporal_eps: Optional[float] = None,
        merge_factor: float = 0.5,
        merge_strategy="learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
    ):
        super().__init__()

        self.spatial_res_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=eps,
        )

        self.temporal_res_block = TemporalResnetBlock(
            in_channels=out_channels if out_channels is not None else in_channels,
            out_channels=out_channels if out_channels is not None else in_channels,
            temb_channels=temb_channels,
            eps=temporal_eps if temporal_eps is not None else eps,
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor,
            merge_strategy=merge_strategy,
            switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,
        )
        #logger.info("SpatioTemporalResBlock, Mixer factor: {}, Mixer type: {}".format(merge_factor, merge_strategy))

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ):
        num_frames = image_only_indicator.shape[-1]
        hidden_states = self.spatial_res_block(hidden_states, temb)

        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states_mix = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )
        hidden_states = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )

        if temb is not None:
            temb = temb.reshape(batch_size, num_frames, -1)

        hidden_states = self.temporal_res_block(hidden_states, temb)
        hidden_states = self.time_mixer(
            x_spatial=hidden_states_mix,
            x_temporal=hidden_states,
            image_only_indicator=image_only_indicator,
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
        return hidden_states


class AlphaBlender(nn.Module):
    r"""
    A module to blend spatial and temporal features.

    Parameters:
        alpha (`float`): The initial value of the blending factor.
        merge_strategy (`str`, *optional*, defaults to `learned_with_images`):
            The merge strategy to use for the temporal mixing.
        switch_spatial_to_temporal_mix (`bool`, *optional*, defaults to `False`):
            If `True`, switch the spatial and temporal mixing.
    """

    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
    ):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.switch_spatial_to_temporal_mix = switch_spatial_to_temporal_mix  # For TemporalVAE

        if merge_strategy not in self.strategies:
            raise ValueError(f"merge_strategy needs to be in {self.strategies}")

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif self.merge_strategy == "learned" or self.merge_strategy == "learned_with_images":
            self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))
        else:
            raise ValueError(f"Unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, image_only_indicator: torch.Tensor, ndims: int) -> torch.Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor

        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)

        elif self.merge_strategy == "learned_with_images":
            if image_only_indicator is None:
                raise ValueError("Please provide image_only_indicator to use learned_with_images merge strategy")

            alpha = torch.where(
                image_only_indicator.bool(),
                torch.ones(1, 1, device=image_only_indicator.device),
                torch.sigmoid(self.mix_factor)[..., None],
            )

            # (batch, channel, frames, height, width)
            if ndims == 5:
                alpha = alpha[:, None, :, None, None]
            # (batch*frames, height*width, channels)
            elif ndims == 3:
                alpha = alpha.reshape(-1)[:, None, None]
            else:
                raise ValueError(f"Unexpected ndims {ndims}. Dimensions should be 3 or 5")

        else:
            raise NotImplementedError

        return alpha

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha = self.get_alpha(image_only_indicator, x_spatial.ndim)
        alpha = alpha.to(x_spatial.dtype)

        if self.switch_spatial_to_temporal_mix:
            alpha = 1.0 - alpha

        x = alpha * x_spatial + (1.0 - alpha) * x_temporal
        return x



# Temporal donwsampling
class Downsample3D(nn.Module):
    """A 3D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        conv_cls = nn.Conv2d

        conv = conv_cls(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        conv_time = nn.Sequential(
            nn.GroupNorm(32, self.out_channels), 
            nn.SiLU(), 
            nn.Conv3d(self.out_channels, self.out_channels, (3, 1, 1), stride=(stride, 1, 1), padding=(1, 0, 0))
        )
        self.conv = conv
        self.conv_time_temp = conv_time

    def forward(self, hidden_states: torch.FloatTensor, num_frames=16, temporal_skip=False) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        if not temporal_skip:
            hidden_states = (
                hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
            ) # N C T H W
            hidden_states = self.conv_time_temp(hidden_states) # N C T/2 H W
            num_frames = num_frames // 2
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
                (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
            ) # (Nt) C H W
        return hidden_states, num_frames


class Upsample3D(nn.Module):
    """A 3D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        conv_cls = nn.Conv2d

        conv = None
        conv = conv_cls(self.channels, self.out_channels, 3, padding=1)
        conv_time = nn.Sequential(
            nn.GroupNorm(32, self.out_channels), 
            nn.SiLU(), 
            nn.Conv3d(self.out_channels, self.out_channels, (3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        )

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        self.conv = conv
        self.conv_time_temp = conv_time

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_size: Optional[int] = None,
        num_frames=16,
        temporal_skip: Optional[bool]=False,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        # ---------- spatial interpolation ---------- #
        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)
        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")
        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)
        # Spatial feature fusion
        hidden_states = self.conv(hidden_states)

        # ---------- temporal interpolation ---------- #
        if not temporal_skip:
            hidden_states = (
                hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
            ) # N C T H W
            dtype = hidden_states.dtype
            if dtype == torch.bfloat16:
                hidden_states = hidden_states.to(torch.float32)
            if hidden_states.shape[0] >= 64:
                hidden_states = hidden_states.contiguous()
            if output_size is None:
                hidden_states = F.interpolate(hidden_states, size=(num_frames*2, hidden_states.shape[-2], hidden_states.shape[-1]), mode="nearest")
            else:
                hidden_states = F.interpolate(hidden_states, size=(num_frames*2, output_size[0], output_size[1]), mode="nearest")
            if dtype == torch.bfloat16:
                hidden_states = hidden_states.to(dtype)
            hidden_states = self.conv_time_temp(hidden_states)
            num_frames = num_frames * 2
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
                (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
            ) # (Nt) C H W

        return hidden_states, num_frames
