import torch
from torch import nn as nn
from torch.autograd import Function

from . import grid_sampler_cuda

padding_mode_enum = {"zeros": 0, "border": 1, "reflection": 2}


class GridSampler3DBackward(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        grid,
        grad_output,
        padding_mode="zeros",
        align_corners=True,
    ):
        ctx.align_corners = align_corners
        ctx.padding_mode = padding_mode
        grad_input, grad_grid = grid_sampler_cuda.grid_sampler_3d_backward(
            grad_output,
            input,
            grid,
            padding_mode_enum[padding_mode],
            ctx.align_corners,
        )
        ctx.save_for_backward(input, grid, grad_output)
        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad2_gard_input, grad2_grad_grid):
        input, grid, grad_output = ctx.saved_tensors

        (
            grad_input,
            grad_grid,
            grad_grad_output,
        ) = grid_sampler_cuda.grid_sampler_3d_backward_backward(
            grad2_gard_input.contiguous(),
            grad2_grad_grid.contiguous(),
            input,
            grid,
            grad_output,
            padding_mode_enum[ctx.padding_mode],
            ctx.align_corners,
        )
        return grad_input, grad_grid, grad_grad_output, None, None


class GridSampler3D(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        grid,
        padding_mode="zeros",
        align_corners=True,
    ):
        output = grid_sampler_cuda.grid_sampler_3d_forward(
            input,
            grid,
            padding_mode_enum[padding_mode],
            align_corners,
        )
        ctx.save_for_backward(input, grid)
        ctx.align_corners = align_corners
        ctx.padding_mode = padding_mode
        return output

    @staticmethod
    def backward(ctx, grad_out):
        input, grid = ctx.saved_tensors
        d_input, d_grid = GridSampler3DBackward.apply(
            input,
            grid,
            grad_out.contiguous(),
            ctx.padding_mode,
            ctx.align_corners,
        )
        return d_input, d_grid, None, None


class GridSampler2DBackward(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        grid,
        grad_output,
        padding_mode="zeros",
        align_corners=True,
    ):
        ctx.align_corners = align_corners
        ctx.padding_mode = padding_mode
        grad_input, grad_grid = grid_sampler_cuda.grid_sampler_2d_backward(
            grad_output,
            input,
            grid,
            padding_mode_enum[padding_mode],
            ctx.align_corners,
        )
        ctx.save_for_backward(input, grid, grad_output)
        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad2_gard_input, grad2_grad_grid):
        input, grid, grad_output = ctx.saved_tensors
        (
            grad_input,
            grad_grid,
            grad_grad_output,
        ) = grid_sampler_cuda.grid_sampler_2d_backward_backward(
            grad2_gard_input.contiguous(),
            grad2_grad_grid.contiguous(),
            input,
            grid,
            grad_output,
            padding_mode_enum[ctx.padding_mode],
            ctx.align_corners,
        )
        return grad_input, grad_grid, grad_grad_output, None, None


class GridSampler2D(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        grid,
        padding_mode="zeros",
        align_corners=True,
    ):
        output = grid_sampler_cuda.grid_sampler_2d_forward(
            input,
            grid,
            padding_mode_enum[padding_mode],
            align_corners,
        )
        ctx.save_for_backward(input, grid)
        ctx.align_corners = align_corners
        ctx.padding_mode = padding_mode
        return output

    @staticmethod
    def backward(ctx, grad_out):
        input, grid = ctx.saved_tensors
        d_input, d_grid = GridSampler2DBackward.apply(
            input,
            grid,
            grad_out.contiguous(),
            ctx.padding_mode,
            ctx.align_corners,
        )
        return d_input, d_grid, None, None
