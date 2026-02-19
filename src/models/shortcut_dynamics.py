# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
import torch.nn as nn

import src.models.vision_transformer as vit_encoder
from src.models.utils.modules import Block, CrossAttentionBlock
from src.utils.tensors import trunc_normal_


class ShortcutDynamicsModule(nn.Module):
    """
    ShortcutDynamicsModule: A lightweight Transformer that processes action tokens
    and conditions them on visual features via cross-attention.
    """

    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        context_dim: int,
        depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_swiglu: bool = True,
        use_rope: bool = False,
        max_action_len: int = 100,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.SiLU,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Input projection
        self.action_proj = nn.Linear(action_dim, embed_dim)
        
        # Context projection (if visual features dim != dynamics dim)
        if context_dim != embed_dim:
            self.context_proj = nn.Linear(context_dim, embed_dim)
        else:
            self.context_proj = nn.Identity()

        # Positional embedding for action sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, max_action_len, embed_dim))
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # Self-Attention Block for temporal mixing of actions
            # Note: We use the standard Block from modules which includes SelfAttn + MLP
            self_attn = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.0,
                act_layer=act_layer,
                norm_layer=norm_layer,
                use_sdpa=True,
                use_rope=use_rope,
                wide_silu=use_swiglu
            )
            
            # Cross-Attention Block for conditioning on image features
            # CrossAttentionBlock: (q, x) -> q' via CrossAttn -> Add -> MLP -> Add
            cross_attn = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                act_layer=act_layer,
                norm_layer=norm_layer
            )
            self.layers.append(nn.ModuleList([self_attn, cross_attn]))

        self.norm = norm_layer(embed_dim)
        
        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, actions: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: (B, T_act, D_act)
            context: (B, N_ctx, D_ctx) - Visual tokens
        Returns:
            dynamics_tokens: (B, T_act, embed_dim)
        """
        B, T, _ = actions.shape
        
        # Project inputs
        x = self.action_proj(actions)  # (B, T, D)
        
        if self.pos_embed.shape[1] >= T:
             x = x + self.pos_embed[:, :T, :]
        else:
            raise ValueError(f"max_action_len ({self.pos_embed.shape[1]}) must be >= action sequence length ({T})")

        # Project context if necessary
        context = self.context_proj(context)

        # Apply blocks
        for self_attn, cross_attn in self.layers:
            x = self_attn(x)
            x = cross_attn(x, context)
        
        x = self.norm(x)
        return x


class PointDecoder(nn.Module):
    """
    PointDecoder: Decodes trajectory points from dynamics tokens.
    Functionally similar to AttentiveClassifier but for regression.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        output_dim: int = 3,
        num_point_queries: int = 1,
        depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.SiLU,
    ):
        super().__init__()
        self.num_point_queries = num_point_queries
        
        if input_dim != embed_dim:
            self.input_proj = nn.Linear(input_dim, embed_dim)
        else:
            self.input_proj = nn.Identity()

        self.query_tokens = nn.Parameter(torch.zeros(1, num_point_queries, embed_dim))
        
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                act_layer=act_layer,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])
        
        self.head = nn.Linear(embed_dim, output_dim)
        
        self.apply(self._init_weights)
        trunc_normal_(self.query_tokens, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, dynamics_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dynamics_tokens: (B, T, input_dim)
        Returns:
            points: (B, num_point_queries, output_dim)
        """
        B = dynamics_tokens.shape[0]
        
        # Project input to decoder dimension
        context = self.input_proj(dynamics_tokens)
        
        # Initialize queries
        q = self.query_tokens.repeat(B, 1, 1)
        
        # Cross-attend queries to dynamics context
        for blk in self.blocks:
            q = blk(q, context)
            
        return self.head(q)


class GeometricDynamicsModel(nn.Module):
    """
    GeometricDynamicsModel: Wraps a frozen V-JEPA encoder, a trainable dynamics module,
    and a point decoder.
    """

    def __init__(
        self,
        encoder_config: dict,
        dynamics_config: dict,
        decoder_config: dict,
        img_size: int = 256,
        num_frames: int = 16,
    ):
        super().__init__()

        encoder_config = dict(encoder_config)
        
        # 1. Frozen Encoder
        model_name = encoder_config.pop("model_name")
        # Removing checkpoint_key as it's used by the loader, not the model init
        _ = encoder_config.pop("checkpoint_key", None)
        
        self.encoder = vit_encoder.__dict__[model_name](
            img_size=img_size,
            num_frames=num_frames,
            **encoder_config
        )
        
        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        
        # 2. Dynamics Module
        context_dim = self.encoder.embed_dim
        self.dynamics = ShortcutDynamicsModule(
            context_dim=context_dim,
            **dynamics_config
        )
        
        # 3. Decoder
        dynamics_out_dim = dynamics_config["embed_dim"]
        self.decoder = PointDecoder(
            input_dim=dynamics_out_dim,
            **decoder_config
        )

    def forward(self, video: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, C, T, H, W)
            actions: (B, T_act, D_act)
        Returns:
            points: (B, N_points, 3)
        """
        # Extract features (No grad for frozen backbone)
        with torch.no_grad():
            # VisionTransformer.forward(x) returns unpooled tokens [B, N, D]
            jepa_tokens = self.encoder(video)
            
        # Predict dynamics latent tokens
        dynamics_tokens = self.dynamics(actions, jepa_tokens)
        
        # Decode geometry points
        points = self.decoder(dynamics_tokens)
        
        return points