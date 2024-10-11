from collections.abc import Sequence

import torch
import torch.nn as nn


class Metrics(nn.Module):
    def __init__(self, metrics, input_keys, output_keys):
        super().__init__()
        self.metrics = metrics

        self.input_keys = input_keys
        if not isinstance(self.input_keys, Sequence):
            self.input_keys = [self.input_keys] * len(self.metrics)
        else:
            assert len(self.input_keys) == len(self.metrics)

        self.output_keys = output_keys
        if not isinstance(self.output_keys, Sequence):
            self.output_keys = [self.output_keys] * len(self.metrics)
        else:
            assert len(self.output_keys) == len(self.metrics)

        self.metrics = nn.ModuleList(self.metrics)

    def reset(self):
        for c in self.metrics:
            c.reset()

    def compute(self):
        metrics = dict()
        for c, out_k in zip(self.metrics, self.output_keys):
            metrics[out_k] = c.compute()
        return metrics

    def metrics_dict(self):
        metrics = dict()
        for c, out_k in zip(self.metrics, self.output_keys):
            metrics[out_k] = c
        return metrics

    @torch.inference_mode()
    def forward(self, data_dict):
        metrics = dict()
        for c, in_k, out_k in zip(self.metrics, self.input_keys, self.output_keys):
            if isinstance(in_k, Sequence):
                try:
                    metrics[out_k] = c(*[data_dict[k_] for k_ in in_k])
                except KeyError:
                    metrics[out_k] = c(data_dict[in_k])
            else:
                metrics[out_k] = c(data_dict[in_k])
        return metrics
