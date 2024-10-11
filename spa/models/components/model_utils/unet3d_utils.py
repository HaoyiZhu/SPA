from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=False,
        group_norm=False,
    ):
        super().__init__()

        self.downsample = downsample
        self.conv1 = nn.Conv3d(inplanes, planes, 3, stride, padding=1, bias=False)
        self.bn1 = (
            nn.BatchNorm3d(planes, eps=1e-3, momentum=0.01)
            if not group_norm
            else nn.GroupNorm(
                num_groups=planes // 16,
                num_channels=planes,
            )
        )
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = (
            nn.BatchNorm3d(planes, eps=1e-3, momentum=0.01)
            if not group_norm
            else nn.GroupNorm(
                num_groups=planes // 16,
                num_channels=planes,
            )
        )
        self.relu = nn.GELU()  # nn.ReLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv3d(
                    inplanes,
                    planes,
                    1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                (
                    nn.BatchNorm3d(planes, eps=1e-3, momentum=0.01)
                    if not group_norm
                    else nn.GroupNorm(num_groups=planes // 16, num_channels=planes)
                ),
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_layer(x)

        out = out + identity
        out = self.relu(out)

        return out


class SimpleConv3D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, **kwargs
    ):
        super(SimpleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
            nn.GELU(),
        )

    def forward(self, x, **kwargs):
        outs = []
        outs.append(self.conv(x))
        return outs


class BEV23DConv3D(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        grid_size,
        kernel_size=3,
        padding=1,
        stride=1,
        **kwargs
    ):
        super(BEV23DConv3D, self).__init__()
        self.grid_size = grid_size
        self.mid_channels = mid_channels
        self.conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels * grid_size[2],
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            nn.BatchNorm2d(mid_channels * grid_size[2]),
            # nn.ReLU(inplace=True),
            nn.GELU(),
        )
        self.conv3d = nn.Sequential(
            nn.Conv3d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
            nn.GELU(),
        )

    def forward(self, x, **kwargs):
        outs = []
        x = self.conv2d(x)
        # (B, C, Y, X) -> (B, C, Z, Y, X)
        x = x.view(x.shape[0], -1, *torch.flip(self.grid_size, dims=[-1]))
        outs.append(self.conv3d(x))
        return outs


class BEV23DExpConv3D(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        grid_size,
        sdf_channels,
        semantic_channels,
        kernel_size=3,
        padding=1,
        stride=1,
        **kwargs
    ):
        super(BEV23DExpConv3D, self).__init__()
        self.grid_size = grid_size
        self.mid_channels = mid_channels
        self.conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels * grid_size[2],
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            nn.BatchNorm2d(mid_channels * grid_size[2]),
            # nn.ReLU(inplace=True),
            nn.GELU(),
        )
        self.shared_conv3d = BasicBlock3D(mid_channels, mid_channels, downsample=False)
        self.sdf_conv3d = nn.Sequential(
            nn.Conv3d(
                mid_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm3d(mid_channels),
            # nn.Softplus(beta=100),
            nn.GELU(),
            nn.Conv3d(
                mid_channels,
                sdf_channels,
                kernel_size=1,
            ),
        )
        self.semantic_conv3d = nn.Sequential(
            nn.Conv3d(
                mid_channels + sdf_channels - 1,
                mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm3d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Conv3d(
                mid_channels,
                semantic_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x, **kwargs):
        x = self.conv2d(x)
        # (B, C, Y, X) -> (B, C, Z, Y, X)
        assert x.shape[-1] == self.grid_size[0] and x.shape[-2] == self.grid_size[1]
        x = x.view(x.shape[0], -1, *torch.flip(self.grid_size, dims=[-1]))
        x = self.shared_conv3d(x)
        sdf = self.sdf_conv3d(x)
        semantic = self.semantic_conv3d(torch.cat([x, sdf[:, 1:]], dim=1))
        outs = [sdf[:, :1], semantic]
        return outs


class ExpConv3D(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        sdf_channels,
        rgb_channels=None,
        semantic_channels=None,
        kernel_size=3,
        padding=1,
        stride=1,
        group_norm=False,
        **kwargs
    ):
        super(ExpConv3D, self).__init__()

        self.shared_conv3d = BasicBlock3D(
            in_channels, mid_channels, downsample=True, group_norm=group_norm
        )
        self.sdf_conv3d = nn.Sequential(
            nn.Conv3d(
                mid_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            (
                nn.BatchNorm3d(mid_channels)
                if not group_norm
                else nn.GroupNorm(
                    num_groups=mid_channels // 16,
                    num_channels=mid_channels,
                )
            ),
            nn.Softplus(beta=100),
            nn.Conv3d(
                mid_channels,
                sdf_channels,
                kernel_size=1,
            ),
        )
        self.rgb_channels = rgb_channels
        if self.rgb_channels is not None:
            self.rgb_conv3d = nn.Sequential(
                nn.Conv3d(
                    mid_channels + sdf_channels - 1,
                    mid_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                ),
                (
                    nn.BatchNorm3d(mid_channels)
                    if not group_norm
                    else nn.GroupNorm(
                        num_groups=mid_channels // 16,
                        num_channels=mid_channels,
                    )
                ),
                nn.GELU(),
                nn.Conv3d(
                    mid_channels,
                    rgb_channels,
                    kernel_size=1,
                ),
            )
        self.semantic_channels = semantic_channels
        if self.semantic_channels is not None:
            self.semantic_conv3d = nn.Sequential(
                nn.Conv3d(
                    mid_channels + sdf_channels - 1,
                    mid_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                ),
                (
                    nn.BatchNorm3d(mid_channels)
                    if not group_norm
                    else nn.GroupNorm(
                        num_groups=mid_channels // 16,
                        num_channels=mid_channels,
                    )
                ),
                nn.GELU(),
                nn.Conv3d(
                    mid_channels,
                    semantic_channels,
                    kernel_size=1,
                ),
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, **kwargs):
        x = self.shared_conv3d(x)
        sdf = self.sdf_conv3d(x)
        outs = [sdf[:, :1]]
        if self.rgb_channels is not None:
            rgb = self.rgb_conv3d(torch.cat([x, sdf[:, 1:]], dim=1))
            outs.append(rgb)
        if self.semantic_channels is not None:
            semantic = self.semantic_conv3d(torch.cat([x, sdf[:, 1:]], dim=1))
            outs.append(semantic)
        return outs
