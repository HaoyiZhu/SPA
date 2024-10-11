from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from lightning import LightningDataModule
from torch.utils.data import BatchSampler
from torch.utils.data import ConcatDataset as _ConcatDataset
from torch.utils.data import DataLoader, Dataset, RandomSampler, default_collate

from .combined_loader import CombinedLoader


class ConcatDataset(_ConcatDataset):
    pass


class ConcatRandomSampler(RandomSampler):
    def __init__(
        self, data_source, replacement=False, num_samples=None, generator=None
    ):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self._generator = generator
        self._samplers = [
            RandomSampler(
                _data_source,
                replacement=replacement,
                num_samples=num_samples,
                generator=generator,
            )
            for _data_source in data_source.datasets
        ]
        self.epoch = 0

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, generator):
        self._generator = generator
        for sampler in self._samplers:
            sampler.generator = generator

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        raise NotImplementedError


class DistributedConcatBatchSampler(BatchSampler):
    def __init__(
        self,
        sampler,
        batch_size,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        super().__init__(sampler, batch_size, drop_last)
        self._batch_samplers = [
            BatchSampler(_sampler, batch_size, drop_last)
            for _sampler in sampler._samplers
        ]
        self._length = [len(batch_sampler) for batch_sampler in self._batch_samplers]
        self._cumulative_sizes = sampler.data_source.cumulative_sizes

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.sampler.epoch)
            self.sampler.generator = g

        consumed = [0] * len(self._length)
        iter_num = 0
        iterators = [iter(batch_sampler) for batch_sampler in self._batch_samplers]

        initial_batch = []
        batch_to_yield = []
        while iter_num < sum(self._length):
            for i, iterator in enumerate(iterators):
                if consumed[i] >= self._length[i]:
                    continue
                batch_indices = next(iterator)
                cumulative_batch_indices = [
                    idx + (self._cumulative_sizes[i - 1] if i > 0 else 0)
                    for idx in batch_indices
                ]
                if iter_num < self.num_replicas:
                    initial_batch.append(cumulative_batch_indices)
                if iter_num % self.num_replicas == self.rank:
                    batch_to_yield = cumulative_batch_indices
                if iter_num % self.num_replicas == self.num_replicas - 1:
                    yield batch_to_yield
                    batch_to_yield = []
                iter_num += 1
                consumed[i] += 1

        if len(batch_to_yield) > 0:
            yield batch_to_yield
        elif iter_num < self.__len__() * self.num_replicas:
            yield initial_batch[self.rank]

    def __len__(self):
        return (sum(self._length) + self.num_replicas - 1) // self.num_replicas


class BaseCatDataModule(LightningDataModule):
    """`LightningDataModule` for basic datasets.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        train,
        val,
        test=None,
        **kwargs,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["train", "val", "test"])

        self.data_train: Optional[Dataset] = train
        self.data_val: Optional[Dataset] = val
        self.data_test: Optional[Dataset] = test

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self.hparams.get("train")
            self.data_val = self.hparams.get("val")
            self.data_test = self.hparams.get("test")

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """

        if hasattr(self.data_train[0], "_collate_fn"):
            _collate_fn = self.data_train[0]._collate_fn
        else:
            _collate_fn = default_collate

        data_train = ConcatDataset(self.data_train)
        sampler = ConcatRandomSampler(data_train)
        batch_sampler = DistributedConcatBatchSampler(
            sampler, batch_size=self.hparams.batch_size_train
        )
        return DataLoader(
            dataset=data_train,
            batch_sampler=batch_sampler,
            num_workers=self.hparams.num_workers,
            persistent_workers=(True if self.hparams.num_workers > 0 else False),
            pin_memory=self.hparams.pin_memory,
            collate_fn=_collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        if hasattr(self.data_val, "_collate_fn"):
            _collate_fn = self.data_val._collate_fn
        else:
            _collate_fn = default_collate

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size_val,
            num_workers=self.hparams.num_workers,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=_collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        if hasattr(self.data_test, "_collate_fn"):
            _collate_fn = self.data_test._collate_fn
        else:
            _collate_fn = default_collate

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size_test,
            num_workers=self.hparams.num_workers,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=_collate_fn,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
