import numpy as np
import torch
import torch.nn as nn


class AABBBoxCollider(nn.Module):
    """Module for colliding rays with the scene box to compute near and far values.

    Args:
        scene_box: scene box to apply to dataset
    """

    def __init__(self, near_plane, **kwargs):
        super().__init__()
        self.near_plane = near_plane

    def _intersect_with_aabb(self, rays_o, rays_d, aabb):
        """Returns collection of valid rays within a specified near/far bounding box along with a mask
        specifying which rays are valid

        Args:
            rays_o: (num_rays, 3) ray origins, scaled
            rays_d: (num_rays, 3) ray directions
            aabb: (6, ) This is [min point (x,y,z), max point (x,y,z)], scaled
        """
        # avoid divide by zero
        dir_fraction = 1.0 / (rays_d + 1e-6)

        # x
        t1 = (aabb[0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
        t2 = (aabb[3] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
        # y
        t3 = (aabb[1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
        t4 = (aabb[4] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
        # z
        t5 = (aabb[2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]
        t6 = (aabb[5] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]

        near = torch.max(
            torch.cat(
                [torch.minimum(t1, t2), torch.minimum(t3, t4), torch.minimum(t5, t6)],
                dim=1,
            ),
            dim=1,
        ).values
        far = torch.min(
            torch.cat(
                [torch.maximum(t1, t2), torch.maximum(t3, t4), torch.maximum(t5, t6)],
                dim=1,
            ),
            dim=1,
        ).values

        # clamp to near plane
        near = torch.clamp(near, min=self.near_plane)
        # assert torch.all(nears < fars), "not collide with scene box"
        # fars = torch.maximum(fars, nears + 1e-6)
        mask = near < far
        near[~mask] = 0.0
        far[~mask] = 0.0

        return near, far, mask

    def forward(self, origins, directions, scene_bbox):
        """Intersects the rays with the scene box and updates the near and far values.
        Populates nears and fars fields and returns the ray_bundle.
        Returns:
            nears: (num_rays, 1)
            fars: (num_rays, 1)
        """
        near, far, mask = self._intersect_with_aabb(origins, directions, scene_bbox)
        near = near[..., None]
        far = far[..., None]
        return near, far, mask


class AABBBoxColliderNp:
    """Module for colliding rays with the scene box to compute near and far values.

    Args:
        scene_box: scene box to apply to dataset
    """

    def __init__(self, near_plane, **kwargs):
        self.near_plane = near_plane

    def _intersect_with_aabb(self, rays_o, rays_d, aabb):
        """Returns collection of valid rays within a specified near/far bounding box along with a mask
        specifying which rays are valid

        Args:
            rays_o: (num_rays, 3) ray origins, scaled
            rays_d: (num_rays, 3) ray directions
            aabb: (6, ) This is [min point (x,y,z), max point (x,y,z)], scaled
        """
        # avoid divide by zero
        dir_fraction = 1.0 / (rays_d + 1e-6)

        # x
        t1 = (aabb[0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
        t2 = (aabb[3] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
        # y
        t3 = (aabb[1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
        t4 = (aabb[4] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
        # z
        t5 = (aabb[2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]
        t6 = (aabb[5] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]

        near = np.max(
            np.concatenate(
                [np.minimum(t1, t2), np.minimum(t3, t4), np.minimum(t5, t6)],
                axis=1,
            ),
            axis=1,
        )
        far = np.min(
            np.concatenate(
                [np.maximum(t1, t2), np.maximum(t3, t4), np.maximum(t5, t6)],
                axis=1,
            ),
            axis=1,
        )

        # clamp to near plane
        near = np.clip(near, a_min=self.near_plane, a_max=None)
        mask = near < far
        near[~mask] = 0.0
        far[~mask] = 0.0

        return near, far, mask

    def __call__(self, origins, directions, scene_bbox):
        """Intersects the rays with the scene box and updates the near and far values.
        Populates nears and fars fields and returns the ray_bundle.
        Returns:
            nears: (num_rays, 1)
            fars: (num_rays, 1)
        """
        near, far, mask = self._intersect_with_aabb(origins, directions, scene_bbox)
        near = near[..., None]
        far = far[..., None]
        return near, far, mask
