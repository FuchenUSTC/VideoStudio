# -*- encoding: utf-8 -*-
# auth: Fuchen Long
# mail: longfc.ustc@gmail.com
# date: 2023/07/10
# desc: 3D unet blocks for sd-xl

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import logging
from diffusers.models.resnet import Downsample2D, ResnetBlock2D, Upsample2D
from diffusers.models.transformer_2d import Transformer2DModel

from .transformer_temporal import TransformerTemporalModel
from .resnet import TemporalConvLayer
from .transformer_3d import Transformer3DModel



logger = logging.get_logger('video-diffusion')  # pylint: disable=invalid-name


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    transformer_layers_per_block=1,
    num_attention_heads=None,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    resnet_skip_time_act=False,
    resnet_out_scale_factor=1.0,
    cross_attention_norm=None,
    attention_head_dim=None,
    downsample_type=None,
    temporal_cross_atten=False,
    temporal_transformer_zero_init=True,
    temporal_trans_num_align_spatial=False,
    temporal_res_use_more_conv=False,
    temporal_trans_double_equip=False,
):
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warn(
            f"It is recommended to provide `attention_head_dim` when calling `get_down_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temporal_res_use_more_conv=temporal_res_use_more_conv,
        )
    elif down_block_type == "CrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock3D")
        return CrossAttnDownBlock3D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temporal_cross_atten=temporal_cross_atten,
            temporal_transformer_zero_init=temporal_transformer_zero_init,
            temporal_trans_num_align_spatial=temporal_trans_num_align_spatial,
            temporal_res_use_more_conv=temporal_res_use_more_conv,
            temporal_trans_double_equip=temporal_trans_double_equip,
        )
    elif down_block_type == "FullCrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for FullCrossAttnDownBlock3D")
        return FullCrossAttnDownBlock3D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temporal_cross_atten=temporal_cross_atten,
            temporal_transformer_zero_init=temporal_transformer_zero_init,
        )    
    elif down_block_type == "SmallCrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for SmallCrossAttnDownBlock3D")
        return SmallCrossAttnDownBlock3D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temporal_cross_atten=temporal_cross_atten,
            temporal_transformer_zero_init=temporal_transformer_zero_init,
        )    
 
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    transformer_layers_per_block=1,
    num_attention_heads=None,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    resnet_skip_time_act=False,
    resnet_out_scale_factor=1.0,
    cross_attention_norm=None,
    attention_head_dim=None,
    upsample_type=None,
    temporal_cross_atten=False,
    temporal_transformer_zero_init=True,
    temporal_trans_num_align_spatial=False,
    temporal_res_use_more_conv=False,
    temporal_trans_double_equip=False,
):
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warn(
            f"It is recommended to provide `attention_head_dim` when calling `get_up_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temporal_res_use_more_conv=temporal_res_use_more_conv,
        )
    elif up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")
        return CrossAttnUpBlock3D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temporal_cross_atten=False,
            temporal_transformer_zero_init=temporal_transformer_zero_init,
            temporal_trans_num_align_spatial=temporal_trans_num_align_spatial,
            temporal_res_use_more_conv=temporal_res_use_more_conv,
            temporal_trans_double_equip=temporal_trans_double_equip,
        )
    elif up_block_type == "FullCrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for FullCrossAttnUpBlock3D")
        return FullCrossAttnUpBlock3D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temporal_cross_atten=False,
            temporal_transformer_zero_init=temporal_transformer_zero_init,
        )
    elif up_block_type == "SmallCrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for SmallCrossAttnUpBlock3D")
        return SmallCrossAttnUpBlock3D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temporal_cross_atten=False,
            temporal_transformer_zero_init=temporal_transformer_zero_init,
        )

    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock3DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=True,
        upcast_attention=False,
        temporal_attn_num_head_channels=64,
        temporal_cross_atten=False,
        temporal_transformer_zero_init=True,
        temporal_trans_num_align_spatial=False,
        temporal_res_use_more_conv=False,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        self.temporal_cross_atten = temporal_cross_atten
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        temp_convs = [
            TemporalConvLayer(
                in_channels,
                in_channels,
                dropout=0.1,
                use_more_conv=temporal_res_use_more_conv,
            )
        ]
        attentions = []
        temp_attentions = []

        for _ in range(num_layers):
            attentions.append(
                Transformer2DModel(
                    num_attention_heads,
                    in_channels // num_attention_heads,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                )
            )
            temp_attentions.append(
                TransformerTemporalModel(
                    in_channels // temporal_attn_num_head_channels,
                    temporal_attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block if temporal_trans_num_align_spatial else 1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    double_self_attention= not temporal_cross_atten,
                    zero_init=temporal_transformer_zero_init,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    in_channels,
                    in_channels,
                    dropout=0.1,
                    use_more_conv=temporal_res_use_more_conv,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.temp_convs[0](hidden_states, num_frames=num_frames)
        for attn, temp_attn, resnet, temp_conv in zip(
            self.attentions, self.temp_attentions, self.resnets[1:], self.temp_convs[1:]
        ):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None, num_frames=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            if num_frames is not None:
                                return module(*inputs, return_dict=return_dict, num_frames=num_frames)
                            else:
                                return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    None,
                    **ckpt_kwargs,
                )[0]              
                if self.temporal_cross_atten:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_attn, return_dict=False, num_frames=num_frames),
                        hidden_states,
                        encoder_hidden_states,
                        cross_attention_kwargs,
                    )[0]
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_attn, return_dict=False, num_frames=num_frames), hidden_states)[0]
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(temp_conv), hidden_states, num_frames)
            else:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                if self.temporal_cross_atten:
                    hidden_states = temp_attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        num_frames=num_frames
                    ).sample
                else:
                    hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample
                hidden_states = resnet(hidden_states, temb)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        return hidden_states


class CrossAttnDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        temporal_attn_num_head_channels=64,
        temporal_cross_atten=False,
        temporal_transformer_zero_init=True,
        temporal_trans_num_align_spatial=False,
        temporal_res_use_more_conv=False,
        temporal_trans_double_equip=False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        temp_attentions = []
        temp_convs = []
        if temporal_trans_double_equip: temp_attentions_equip = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        self.temporal_cross_atten = temporal_cross_atten
        self.temporal_trans_double_equip = temporal_trans_double_equip

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                    use_more_conv=temporal_res_use_more_conv,
                )
            )
            attentions.append(
                Transformer2DModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
            )
            temp_attentions.append(
                TransformerTemporalModel(
                    out_channels // temporal_attn_num_head_channels,
                    temporal_attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block if temporal_trans_num_align_spatial else 1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    double_self_attention= not temporal_cross_atten,
                    zero_init=temporal_transformer_zero_init,
                )
            )
            if temporal_trans_double_equip:
                temp_attentions_equip.append(
                    TransformerTemporalModel(
                        out_channels // temporal_attn_num_head_channels,
                        temporal_attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        double_self_attention= not temporal_cross_atten,
                        zero_init=temporal_transformer_zero_init,
                    )
                )
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)
        if temporal_trans_double_equip: self.temp_attentions_equip = nn.ModuleList(temp_attentions_equip)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):
        # TODO(Patrick, William) - attention mask is not used
        output_states = ()

        if not self.temporal_trans_double_equip:
            for resnet, temp_conv, attn, temp_attn in zip(
                self.resnets, self.temp_convs, self.attentions, self.temp_attentions
            ):
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None, num_frames=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                if num_frames is not None:
                                    return module(*inputs, return_dict=return_dict, num_frames=num_frames)
                                else:
                                    return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_conv), hidden_states, num_frames)
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False),
                        hidden_states,
                        encoder_hidden_states,
                        None,  # timestep
                        None,  # class_labels
                        cross_attention_kwargs,
                        attention_mask,
                        None,
                        **ckpt_kwargs,
                    )[0]
                    if self.temporal_cross_atten:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(temp_attn, return_dict=False, num_frames=num_frames),
                            hidden_states,
                            encoder_hidden_states,
                            cross_attention_kwargs,
                        )[0]
                    else:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(temp_attn, return_dict=False, num_frames=num_frames), hidden_states)[0]
                    output_states += (hidden_states,)
                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = temp_conv(hidden_states, num_frames=num_frames)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                    if self.temporal_cross_atten:
                        hidden_states = temp_attn(
                            hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=cross_attention_kwargs,
                            num_frames=num_frames
                        ).sample
                    else:
                        hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample

                    output_states += (hidden_states,)
        else:
            for resnet, temp_conv, attn, temp_attn, temp_attn_equip in zip(
                self.resnets, self.temp_convs, self.attentions, self.temp_attentions, self.temp_attentions_equip
            ):
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None, num_frames=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                if num_frames is not None:
                                    return module(*inputs, return_dict=return_dict, num_frames=num_frames)
                                else:
                                    return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_conv), hidden_states, num_frames)
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False),
                        hidden_states,
                        encoder_hidden_states,
                        None,  # timestep
                        None,  # class_labels
                        cross_attention_kwargs,
                        attention_mask,
                        None,
                        **ckpt_kwargs,
                    )[0]
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_attn, return_dict=False, num_frames=num_frames), hidden_states)[0]
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_attn_equip, return_dict=False, num_frames=num_frames), hidden_states)[0]
                    output_states += (hidden_states,)
                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = temp_conv(hidden_states, num_frames=num_frames)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                    hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample
                    hidden_states = temp_attn_equip(hidden_states, num_frames=num_frames).sample
                    output_states += (hidden_states,)            

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
        temporal_res_use_more_conv=False,
    ):
        super().__init__()
        resnets = []
        temp_convs = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                    use_more_conv=temporal_res_use_more_conv,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None, num_frames=1):
        output_states = ()

        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
              
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(temp_conv), hidden_states, num_frames)
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)
            
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        temporal_attn_num_head_channels=64,
        temporal_cross_atten=False,
        temporal_transformer_zero_init=True,
        temporal_trans_num_align_spatial=False,
        temporal_res_use_more_conv=False,
        temporal_trans_double_equip=False,
    ):
        super().__init__()
        resnets = []
        temp_convs = []
        attentions = []
        temp_attentions = []
        if temporal_trans_double_equip: temp_attentions_equip = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        self.temporal_cross_atten = temporal_cross_atten
        self.temporal_trans_double_equip = temporal_trans_double_equip

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                    use_more_conv=temporal_res_use_more_conv,
                )
            )
            attentions.append(
                Transformer2DModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
            )
            temp_attentions.append(
                TransformerTemporalModel(
                    out_channels // temporal_attn_num_head_channels,
                    temporal_attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block if temporal_trans_num_align_spatial else 1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    double_self_attention= not temporal_cross_atten,
                    zero_init=temporal_transformer_zero_init,
                )
            )
            if temporal_trans_double_equip:
                temp_attentions_equip.append(
                    TransformerTemporalModel(
                        out_channels // temporal_attn_num_head_channels,
                        temporal_attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers= 1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        double_self_attention= not temporal_cross_atten,
                        zero_init=temporal_transformer_zero_init,
                    )
                )
                
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)
        if temporal_trans_double_equip: self.temp_attentions_equip = nn.ModuleList(temp_attentions_equip)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):
        # TODO(Patrick, William) - attention mask is not used
        if not self.temporal_trans_double_equip:
            for resnet, temp_conv, attn, temp_attn in zip(
                self.resnets, self.temp_convs, self.attentions, self.temp_attentions
            ):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None, num_frames=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                if num_frames is not None:
                                    return module(*inputs, return_dict=return_dict, num_frames=num_frames)
                                else:
                                    return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_conv), hidden_states, num_frames)
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False),
                        hidden_states,
                        encoder_hidden_states,
                        None,  # timestep
                        None,  # class_labels
                        cross_attention_kwargs,
                        attention_mask,
                        None,
                        **ckpt_kwargs,
                    )[0]
                    if self.temporal_cross_atten:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(temp_attn, return_dict=False, num_frames=num_frames),
                            hidden_states,
                            encoder_hidden_states,
                            cross_attention_kwargs,
                        )[0]
                    else:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(temp_attn, return_dict=False, num_frames=num_frames), hidden_states)[0]
                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = temp_conv(hidden_states, num_frames=num_frames)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                    if self.temporal_cross_atten:
                        hidden_states = temp_attn(
                            hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=cross_attention_kwargs,
                            num_frames=num_frames
                        ).sample
                    else:
                        hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample
        else:
            for resnet, temp_conv, attn, temp_attn, temp_attn_equip in zip(
                self.resnets, self.temp_convs, self.attentions, self.temp_attentions, self.temp_attentions_equip
            ):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None, num_frames=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                if num_frames is not None:
                                    return module(*inputs, return_dict=return_dict, num_frames=num_frames)
                                else:
                                    return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_conv), hidden_states, num_frames)
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False),
                        hidden_states,
                        encoder_hidden_states,
                        None,  # timestep
                        None,  # class_labels
                        cross_attention_kwargs,
                        attention_mask,
                        None,
                        **ckpt_kwargs,
                    )[0]
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_attn, return_dict=False, num_frames=num_frames), hidden_states)[0]
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_attn_equip, return_dict=False, num_frames=num_frames), hidden_states)[0]
                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = temp_conv(hidden_states, num_frames=num_frames)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                    hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample
                    hidden_states = temp_attn_equip(hidden_states, num_frames=num_frames).sample       
        
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        
        return hidden_states


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        temporal_res_use_more_conv=False,
    ):
        super().__init__()
        resnets = []
        temp_convs = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                    use_more_conv=temporal_res_use_more_conv,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, num_frames=1):
        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(temp_conv), hidden_states, num_frames)
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


# ---- Full Blocks --- #

class UNetFullMidBlock3DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=True,
        upcast_attention=False,
        temporal_attn_num_head_channels=64,
        temporal_cross_atten=False,
        temporal_transformer_zero_init=True,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        self.temporal_cross_atten = temporal_cross_atten
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        temp_convs = [
            TemporalConvLayer(
                in_channels,
                in_channels,
                dropout=0.1,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            attentions.append(
                Transformer3DModel(
                    num_attention_heads,
                    in_channels // num_attention_heads,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    in_channels,
                    in_channels,
                    dropout=0.1,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.temp_convs[0](hidden_states, num_frames=num_frames)
        for attn, resnet, temp_conv in zip(
            self.attentions, self.resnets[1:], self.temp_convs[1:]
        ):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None, num_frames=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            if num_frames is not None:
                                return module(*inputs, return_dict=return_dict, num_frames=num_frames)
                            else:
                                return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False, num_frames=num_frames),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    None,
                    **ckpt_kwargs,
                )[0]              
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(temp_conv), hidden_states, num_frames)
            else:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    num_frames=num_frames,
                ).sample
                hidden_states = resnet(hidden_states, temb)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        return hidden_states


class FullCrossAttnDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        temporal_attn_num_head_channels=64,
        temporal_cross_atten=False,
        temporal_transformer_zero_init=True,
    ):
        super().__init__()
        resnets = []
        attentions = []
        temp_convs = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        self.temporal_cross_atten = temporal_cross_atten

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                )
            )
            attentions.append(
                Transformer3DModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):
        # TODO(Patrick, William) - attention mask is not used
        output_states = ()

        for resnet, temp_conv, attn in zip(
            self.resnets, self.temp_convs, self.attentions
        ):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None, num_frames=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            if num_frames is not None:
                                return module(*inputs, return_dict=return_dict, num_frames=num_frames)
                            else:
                                return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(temp_conv), hidden_states, num_frames)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False, num_frames=num_frames),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    None,
                    **ckpt_kwargs,
                )[0]  
                output_states += (hidden_states,)
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    num_frames=num_frames,
                ).sample

                output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class FullCrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        temporal_attn_num_head_channels=64,
        temporal_cross_atten=False,
        temporal_transformer_zero_init=True,
    ):
        super().__init__()
        resnets = []
        temp_convs = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        self.temporal_cross_atten = temporal_cross_atten

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                )
            )
            attentions.append(
                Transformer3DModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):
        # TODO(Patrick, William) - attention mask is not used
        for resnet, temp_conv, attn in zip(
            self.resnets, self.temp_convs, self.attentions
        ):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None, num_frames=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            if num_frames is not None:
                                return module(*inputs, return_dict=return_dict, num_frames=num_frames)
                            else:
                                return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(temp_conv), hidden_states, num_frames)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False, num_frames=num_frames),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    None,
                    **ckpt_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    num_frames=num_frames,
                ).sample

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


# --- Full Blocks with Down-Up ---- #

class UNetDownUpMidBlock3DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=True,
        upcast_attention=False,
        temporal_attn_num_head_channels=64,
        temporal_cross_atten=False,
        temporal_transformer_zero_init=True,
        temporal_trans_num_align_spatial=False,
        use_temp_conv_in_mid=False,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        self.temporal_cross_atten = temporal_cross_atten
        self.use_temp_conv_in_mid = use_temp_conv_in_mid
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        temp_convs = [
            TemporalConvLayer(
                in_channels,
                in_channels,
                dropout=0.1,
            )
        ]
        attentions = []
        temp_attentions = []

        for _ in range(num_layers):
            attentions.append(
                Transformer2DModel(
                    num_attention_heads,
                    in_channels // num_attention_heads,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                )
            )
            temp_attentions.append(
                TransformerTemporalModel(
                    in_channels // temporal_attn_num_head_channels,
                    temporal_attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block if temporal_trans_num_align_spatial else 1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    double_self_attention= not temporal_cross_atten,
                    zero_init=temporal_transformer_zero_init,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    in_channels,
                    in_channels,
                    dropout=0.1,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)
        
        self.down = Downsample2D(in_channels, use_conv=False, out_channels=in_channels, padding=1, name="op")
        self.up = Upsample2D(in_channels, use_conv=False, out_channels=in_channels)
        if use_temp_conv_in_mid:
            self.temp_mid_conv = TemporalConvLayer(in_channels, in_channels, dropout=0.1)
        else:
            self.temp_mid_attention = TransformerTemporalModel(
                    in_channels // temporal_attn_num_head_channels,
                    temporal_attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    double_self_attention= not temporal_cross_atten,
                    zero_init=temporal_transformer_zero_init,)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.temp_convs[0](hidden_states, num_frames=num_frames)
        for attn, temp_attn, resnet, temp_conv in zip(
            self.attentions, self.temp_attentions, self.resnets[1:], self.temp_convs[1:]
        ):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None, num_frames=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            if num_frames is not None:
                                return module(*inputs, return_dict=return_dict, num_frames=num_frames)
                            else:
                                return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    None,
                    **ckpt_kwargs,
                )[0]              
                if self.temporal_cross_atten:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_attn, return_dict=False, num_frames=num_frames),
                        hidden_states,
                        encoder_hidden_states,
                        cross_attention_kwargs,
                    )[0]
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(temp_attn, return_dict=False, num_frames=num_frames), hidden_states)[0]
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(temp_conv), hidden_states, num_frames)
            else:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                if self.temporal_cross_atten:
                    hidden_states = temp_attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        num_frames=num_frames
                    ).sample
                else:
                    hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample
                hidden_states = resnet(hidden_states, temb)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        # add up-down
        residue = hidden_states
        hidden_states = self.down(hidden_states)
        if self.use_temp_conv_in_mid:
            hidden_states = self.temp_mid_conv(hidden_states, num_frames=num_frames)
        else:
            hidden_states = self.temp_mid_attention(hidden_states, num_frames=num_frames).sample
        hidden_states = self.up(hidden_states)
        hidden_states += residue

        return hidden_states


# --- Only TemporalConv Layers ---- #

class UNetSmallMidBlock3DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=True,
        upcast_attention=False,
        temporal_attn_num_head_channels=64,
        temporal_cross_atten=False,
        temporal_transformer_zero_init=True,
        temporal_trans_num_align_spatial=False,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        self.temporal_cross_atten = temporal_cross_atten
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        temp_convs = [
            TemporalConvLayer(
                in_channels,
                in_channels,
                dropout=0.1,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            attentions.append(
                Transformer2DModel(
                    num_attention_heads,
                    in_channels // num_attention_heads,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    in_channels,
                    in_channels,
                    dropout=0.1,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.temp_convs[0](hidden_states, num_frames=num_frames)
        for attn, resnet, temp_conv in zip(
            self.attentions, self.resnets[1:], self.temp_convs[1:]
        ):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None, num_frames=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            if num_frames is not None:
                                return module(*inputs, return_dict=return_dict, num_frames=num_frames)
                            else:
                                return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    None,
                    **ckpt_kwargs,
                )[0]              
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(temp_conv), hidden_states, num_frames)
            else:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                hidden_states = resnet(hidden_states, temb)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        return hidden_states


class SmallCrossAttnDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        temporal_attn_num_head_channels=64,
        temporal_cross_atten=False,
        temporal_transformer_zero_init=True,
        temporal_trans_num_align_spatial=False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        temp_convs = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        self.temporal_cross_atten = temporal_cross_atten

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                )
            )
            attentions.append(
                Transformer2DModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):
        # TODO(Patrick, William) - attention mask is not used
        output_states = ()

        for resnet, temp_conv, attn in zip(
            self.resnets, self.temp_convs, self.attentions
        ):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None, num_frames=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            if num_frames is not None:
                                return module(*inputs, return_dict=return_dict, num_frames=num_frames)
                            else:
                                return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(temp_conv), hidden_states, num_frames)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    None,
                    **ckpt_kwargs,
                )[0]
                output_states += (hidden_states,)
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class SmallCrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        temporal_attn_num_head_channels=64,
        temporal_cross_atten=False,
        temporal_transformer_zero_init=True,
        temporal_trans_num_align_spatial=False,
    ):
        super().__init__()
        resnets = []
        temp_convs = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        self.temporal_cross_atten = temporal_cross_atten

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                )
            )
            attentions.append(
                Transformer2DModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):
        # TODO(Patrick, William) - attention mask is not used
        for resnet, temp_conv, attn in zip(
            self.resnets, self.temp_convs, self.attentions
        ):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None, num_frames=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            if num_frames is not None:
                                return module(*inputs, return_dict=return_dict, num_frames=num_frames)
                            else:
                                return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(temp_conv), hidden_states, num_frames)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    None,
                    **ckpt_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
