# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# adapted from:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import os
import sys
import urllib
from functools import partial
from os.path import expanduser

import numpy as np
import six
import timm.models.vision_transformer
import torch
import torch.nn as nn
import torchvision.transforms as T
from einops import rearrange, repeat
from timm.models.vision_transformer import Block, PatchEmbed, resize_pos_embed

from spa.utils.fp16_utils import auto_fp16

from .modules import SimpleUpsample
from .utils import get_2d_sincos_pos_embed


class ToTensorIfNot(T.ToTensor):
    def __call__(self, pic):
        if not torch.is_tensor(pic):
            return super().__call__(pic)
        return pic


class ViTEncoder(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer Encoder with image transforms, used for downstream purpose"""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        del self.head

        self.img_size = kwargs.get("img_size", 224)

        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

        self.image_transform = T.Compose(
            [
                T.Resize(self.img_size, interpolation=T.InterpolationMode.BICUBIC),
                ToTensorIfNot(),
                T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, feature_map=False, cat_cls=True):
        # If not `feature_map` (default), will return [cls] token (b c)
        # otherwise, return reshaped feature map (b c h w)
        # if both `feature_map` and `cat_cls`, concatenate feature map with [cls] token
        x = self.image_transform(x)
        latents = self.forward_features(x)
        if not feature_map:
            return latents[:, 0]
        else:
            h = w = int(latents[:, 1:].shape[1] ** 0.5)
            feature_map = rearrange(
                latents[:, 1:],
                "b (h w) c -> b c h w",
                h=h,
                w=w,
            )

            if cat_cls:
                cls_token = repeat(latents[:, 0:1], "n 1 c -> n c h w", h=h, w=w)
                return torch.cat([feature_map, cls_token], dim=1)
            else:
                return feature_map


class SPAViT(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with SPA's upsampler decoder, used for pre-training"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pretrained_weight=None,
        mask_ratio=0.75,
        out_feature_channels=128,
        **kwargs,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            **kwargs,
        )
        del self.head

        self.in_chans = in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        num_patches = self.patch_embed.num_patches
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.readout_project = nn.Sequential(
            nn.Linear(decoder_embed_dim * 2, decoder_embed_dim, bias=True),
            nn.GELU(approximate="none"),
        )
        self.upsample = nn.Sequential(
            SimpleUpsample(
                decoder_embed_dim, out_feature_channels * 16, upscale_factor=4
            ),
            SimpleUpsample(
                out_feature_channels, out_feature_channels * 16, upscale_factor=4
            ),
        )
        # --------------------------------------------------------------------------

        self.mask_ratio = mask_ratio

        if pretrained_weight is None:
            self.initialize_weights()
        else:
            self.load_weight(pretrained_weight)

    def load_weight(self, pretrained_weight):
        state_dict = torch.load(pretrained_weight, map_location="cpu")
        if state_dict.get("model", None) is not None:
            state_dict = state_dict["model"]
        if state_dict["pos_embed"].shape != self.pos_embed.shape:
            state_dict["pos_embed"] = resize_pos_embed(
                state_dict["pos_embed"],
                self.pos_embed,
                getattr(self, "num_tokens", 1),
                self.patch_embed.grid_size,
            )

        # filter out keys with name decoder or mask_token
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if "decoder" not in k and "mask_token" not in k
        }

        self.load_state_dict(state_dict, strict=False)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x, n_channel=3, p=None):
        """
        x: (N, L, patch_size**2 *n_channel)
        imgs: (N, n_channel, H, W)
        """
        if p is None:
            p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, n_channel))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], n_channel, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    @auto_fp16(apply_to=("x"), out_fp32=True)
    def forward_encoder(self, x, mask_ratio):  # (n, 3, h, w)
        # embed patches
        x = self.patch_embed(x)  # (n, p*p, c)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(
            x, mask_ratio
        )  # (n, p*p*(1-mask_ratio), c)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (n, p*p*(1-mask_ratio) + 1, c)

        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)  # (n, p*p*(1-mask_ratio) + 1, c)

        x = self.norm(x)
        return mask, ids_restore, x

    @auto_fp16(apply_to=("x"), out_fp32=True)
    def forward_decoder(self, latent, ids_restore):
        x = self.decoder_embed(latent)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle

        readout = x[:, :1, :].expand_as(x_)
        x = torch.cat([x_, readout], dim=-1)
        x = self.readout_project(x)
        x = self.unpatchify(x, self.decoder_embed_dim, p=1)  # b c p p
        x = self.upsample(x)  # b c p*n p*n

        return x

    def forward(self, batch_dict, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        imgs = torch.cat(batch_dict["img"], dim=0)
        imgs = imgs.view(-1, *imgs.shape[-3:])

        mask, ids_restore, latent = self.forward_encoder(imgs, mask_ratio)
        feature_map = self.forward_decoder(latent, ids_restore)
        batch_dict["img_features"] = [feature_map]
        return batch_dict


def spa_vit_base_patch16(img_size=224, pretrained=True, **kwargs):
    model = ViTEncoder(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        qkv_bias=True,
        **kwargs,
    )
    if pretrained:
        model = load_pretrained(model, "spa-b")

    return model


def spa_vit_large_patch16(img_size=224, pretrained=True, **kwargs):
    model = ViTEncoder(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        qkv_bias=True,
        **kwargs,
    )
    if pretrained:
        model = load_pretrained(model, "spa-l")

    return model


def load_pretrained(model: nn.Module, ckpt_name: str):
    from collections import OrderedDict

    from huggingface_hub import hf_hub_download

    try:
        import safetensors.torch

        _has_safetensors = True
    except ImportError:
        _has_safetensors = False

    if _has_safetensors:
        from safetensors.torch import load_file

        ckpt_file = hf_hub_download(
            repo_id="HaoyiZhu/SPA", filename=f"{ckpt_name}.safetensors"
        )
        _state_dict = load_file(ckpt_file)
    else:
        ckpt_file = hf_hub_download(
            repo_id="HaoyiZhu/SPA", filename=f"{ckpt_name}.ckpt"
        )
        _state_dict = torch.load(ckpt_file)["state_dict"]

    state_dict = OrderedDict()
    for key, value in _state_dict.items():
        if key.startswith("model.img_backbone.") and (
            "decoder" not in key
            and "head" not in key
            and "upsample" not in key
            and "mask" not in key
            and "readout" not in key
        ):
            state_dict[key.replace("model.img_backbone.", "")] = value

    if state_dict["pos_embed"].shape != model.pos_embed.shape:
        state_dict["pos_embed"] = resize_pos_embed(
            state_dict["pos_embed"],
            model.pos_embed,
            getattr(model, "num_tokens", 1),
            model.patch_embed.grid_size,
        )

    model.load_state_dict(state_dict, strict=True)
    return model
