from abc import ABC
from dataclasses import dataclass
import torch
from typing import List, Union

from ._utils import is_valid_device, validate_non_overlapping_shards_metadata

Device = Union[torch.device, int, str]

class PlacementSpec(ABC):
    """
    Base class representing the placement of an entity. Subclasses of this
    class can be used to specify customized placements which might not be
    covered by existing APIs.
    """
    pass


@dataclass
class DevicePlacementSpec(PlacementSpec):
    """
    Associates placement of an entity with a single device. The device can be a
    local device or a remote device specified by one of the following remote
    formats:

        1. "rank:<rank>/<device>" (ex: "rank:0/cuda:0").
        2. "<worker_name>/<device>" (ex: "trainer0/cuda:0").

    Args:
        device(str, :class:`torch.device`): The device to place the entity on.
    """

    device: Device

    def __post_init__(self):
        if not is_valid_device(self.device):
            raise ValueError(f'{self.device} is not a valid device')


class ShardingSpec(PlacementSpec):
    """
    Base class representing sharding specifications. It is special type of
    PlacementSpec.
    """
    pass


@dataclass
class ChunkShardingSpec(ShardingSpec):
    """
    This is a type of PlacementSpec that defines the placement as being sharded
    across multiple devices. In particular, it represents sharding a Tensor
    along a single dimension into equal chunks (similar to :meth:`torch.chunk`).

    The semantics of how a tensor is partitioned is inline with
    :meth:`torch.chunk`, where ``dim`` in torch.chunk corresponds to the
    specified ``dim`` and ``chunks`` in torch.chunk is the number of elements
    in the placement specified.

    Args:
        dim (int or str):
            The dimension to shard on, could be an integer representing the
            dimension or a string in case of named tensors where dimensions are
            named.
        placement(List[Device]):
            Specifies the placement of each shard of the Tensor. The size of
            the list represents the number of shards to be created. This
            parameter can be a list of devices
            (ex: ["rank:0/cuda:0", "rank:1/cuda:1"]) or a list of custom
            placement specs.

            The device can be a local device or a remote device specified by one
            of the following remote formats:

                1. "rank:<rank>/<device>" (ex: "rank:0/cuda:0").
                2. "<worker_name>/<device>" (ex: "trainer0/cuda:0").
    """

    ShardingDim = Union[int, str]

    dim: ShardingDim
    placements: List[Device]

    def __post_init__(self):
        self._verify_dim(self.dim)
        self._verify_devices(self.placements)

    @staticmethod
    def _verify_devices(placements):
        if placements is None or len(placements) == 0:
            raise ValueError(f'None/Empty placement provided: {placements}')
        for dev in placements:
            if not is_valid_device(dev):
                raise ValueError(f'{dev} is not a valid device')

    @staticmethod
    def _verify_dim(dim):
        if not (isinstance(dim, int) or isinstance(dim, str)):
            raise ValueError(f'{dim} needs to either be an int or str')


@dataclass
class ShardMetadata(object):
    """
    Represents a shard of the overall Tensor including its
    offsets, lengths and device placement.

    Args:
        shard_offsets(List[int]): Offsets in the orignal tensor indicating
            the start offsets for this shard. Should have the same rank as
            the original tensor.
        shard_lengths(List[int]): Lengths indicating the length of each
            dimension for this shard. Should have the same rank as the
            original tensor.
        placement(Device):
            Specifies the placement of this shard.

            The placement can be a local device or a remote device specified by one
            of the following remote formats:

                1. "rank:<rank>/<device>" (ex: "rank:0/cuda:0").
                2. "<worker_name>/<device>" (ex: "trainer0/cuda:0").
    """

    __slots__ = ['shard_offsets', 'shard_lengths', 'placement']

    shard_offsets: List[int]
    shard_lengths: List[int]
    placement: Device

    def __post_init__(self):
        if not is_valid_device(self.placement):
            raise ValueError(f'{self.placement} is not a valid device')

        if len(self.shard_offsets) != len(self.shard_lengths):
            raise ValueError(
                f'shard_offsets and shard_lengths should have '
                f'the same number of elements, found {len(self.shard_offsets)} '
                f'and {self.shard_lengths} respectively')

        for i in range(len(self.shard_offsets)):
            if self.shard_offsets[i] < 0:
                raise ValueError('shard_offsets should be >=0')
            if self.shard_lengths[i] <= 0:
                raise ValueError('shard_lengths should be > 0')


@dataclass
class EnumerableShardingSpec(ShardingSpec):
    """
    This is a type of PlacementSpec that allows users to specify a generic
    sharding scheme by enumerating exactly how each shard is laid out.

    Args:
        shards(List[ShardMetadata]): List of :class:`ShardMetadata` objects representing
            each shard. Note that none of the shards should overlap.
    """

    shards: List[ShardMetadata]

    def __post_init__(self):
        if len(self.shards) == 0:
            raise ValueError(f'Empty shard list provided: {self.shards}')

        # Validate each shard has same rank.
        rank = -1
        for shard in self.shards:
            if rank != -1 and rank != len(shard.shard_offsets):
                raise ValueError(f'Found inconsistent ranks for shards: {rank} and {len(shard.shard_offsets)}')
            rank = len(shard.shard_offsets)

        validate_non_overlapping_shards_metadata(self.shards)
