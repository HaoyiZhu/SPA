from functools import partial

from .base_surface_model import SurfaceModel


class NeuSModel(SurfaceModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sample_and_forward_field(
        self, ray_bundle, volume_feature, scene_bbox, scene_scale
    ):
        sampler_out_dict = self.sampler(
            ray_bundle,
            occupancy_fn=self.field.get_occupancy,
            sdf_fn=partial(
                self.field.get_sdf,
                volume_feature=volume_feature,
                scene_bbox=scene_bbox,
                scene_scale=scene_scale,
            ),
        )
        ray_samples = sampler_out_dict.pop("ray_samples")
        field_outputs = self.field(ray_samples, volume_feature, scene_bbox, scene_scale)
        weights, _ = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs["alphas"]
        )

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,  # (num_rays, num_smaples+num_importance, 1)
            "sampled_points": ray_samples.frustums.get_positions(),  # (num_rays, num_smaples+num_importance, 3)
            **sampler_out_dict,
        }

        return samples_and_field_outputs
