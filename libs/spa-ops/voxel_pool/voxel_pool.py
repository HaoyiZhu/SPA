import torch

from . import voxel_pool_ext


class VoxelPoolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, coords, ranks, B, Z, Y, X):
        kept = torch.ones(ranks.shape[0], device=feats.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.nonzero(kept, as_tuple=True)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks.shape[0] - interval_starts[-1]
        coords = coords.int()

        out = voxel_pool_ext.voxel_pool_forward(
            feats,
            coords,
            interval_lengths,
            interval_starts,
            B,
            Z,
            Y,
            X,
        )

        ctx.save_for_backward(interval_starts, interval_lengths, coords)
        ctx.saved_shapes = B, Z, Y, X
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, coords = ctx.saved_tensors
        B, Z, Y, X = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        feats_grad = voxel_pool_ext.voxel_pool_backward(
            out_grad,
            coords,
            interval_lengths,
            interval_starts,
            B,
            Z,
            Y,
            X,
        )

        return feats_grad, None, None, None, None, None, None


def voxel_pool(feats, coords, B, Z, Y, X):
    # (bs_idx, z, y, x)
    ranks = (
        coords[:, 0] * (Z * Y * X)
        + coords[:, 1] * (Y * X)
        + coords[:, 2] * X
        + coords[:, 3]
    )
    indices = ranks.argsort()
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]

    x = VoxelPoolFunction.apply(feats, coords, ranks, B, Z, Y, X)

    return x
