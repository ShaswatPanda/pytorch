import functools

from torch.utils.data import IterDataPipe, functional_datapipe
from typing import Callable, Iterator, Optional, Sized, Tuple, TypeVar, Deque
from collections import deque

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('concat')
class ConcatIterDataPipe(IterDataPipe):
    r""" :class:`ConcatIterDataPipe`.

    Iterable DataPipe to concatenate multiple Iterable DataPipes.
    args:
        datapipes: Iterable DataPipes being concatenated
    """
    datapipes: Tuple[IterDataPipe]
    length: Optional[int]

    def __init__(self, *datapipes: IterDataPipe):
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `IterDataPipe`")
        self.datapipes = datapipes  # type: ignore[assignment]
        self.length = None

    def __iter__(self) -> Iterator:
        for dp in self.datapipes:
            for data in dp:
                yield data

    def __len__(self) -> int:
        if self.length is not None:
            if self.length == -1:
                raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
            return self.length
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            self.length = sum(len(dp) for dp in self.datapipes)
        else:
            self.length = -1
        return len(self)


# This is fake class to show API, going to be replaced by the copy from torchdata
# TODO(VitalyFedyunin): Replace with valid version, documentation and tests
class IterateBuffer(IterDataPipe):

    def __init__(self, buffer):
        self.buffer = buffer

    def __iter__(self):
        for i in self.buffer:
            yield i


@functional_datapipe('fork')
class ForkIterDataPipe(IterDataPipe):
    r""" :class:`ForkIterDataPipe`.

        Iterable DataPipe to create multiple instances of the same Iterable DataPipe.
        args:
            datapipe: Iterable DataPipe being copied
            num_instances: number of instances of the datapipe to create
            buffer_size: maxmium buffer size who restricts how far ahead the fastest child datapipe
             can read relative to the slowest child datapipe
    """
    def __new__(cls, datapipe: IterDataPipe, num_instances: int, buffer_size: int = 1000):
        container = _ForkIterDataPipe(datapipe, num_instances, buffer_size)
        return [ForkIterDataPipeHelper(container, i) for i in range(num_instances)]

    def __init__(self, *arg):
        raise Exception("__init__ called instead of __new__")


class _ForkIterDataPipe(IterDataPipe):

    def __init__(self, datapipe: IterDataPipe, num_instances: int, buffer_size: int = 1000):
        self.main_datapipe = iter(datapipe)
        self.num_instances = num_instances
        self.buffer: Deque[T_co] = deque()
        self.buffer_size = buffer_size
        self.child_pointers = [0] * num_instances  # Indicate the indices of the next element to get
        self.slow_ptr = 0
        self.fast_ptr = 0

    def getNext(self, instance_id) -> T_co:
        stop = False
        while not stop:
            if not self.buffer or self.child_pointers[instance_id] > self.fast_ptr:  # Buffer empty or slower
                self.fast_ptr = self.child_pointers[instance_id]
                if self.fast_ptr - self.slow_ptr > self.buffer_size:
                    raise BufferError(f"ForkIterDataPipe buffer overflow, buffer size {self.buffer_size} is insufficient.")
                try:
                    self.buffer.append(self.main_datapipe.__next__())
                    self.child_pointers[instance_id] += 1
                    yield self.buffer[-1]
                except StopIteration:
                    stop = True
            else:  # the child pointer is slower than or equal to the fast_ptr
                buffer_index = self.child_pointers[instance_id] - self.slow_ptr
                return_val = self.buffer[buffer_index]
                self.child_pointers[instance_id] += 1
                if self.child_pointers[instance_id] - 1 == self.slow_ptr:
                    new_min = min(self.child_pointers)  # Can make this faster if I don't need to call min
                    if self.slow_ptr < new_min:
                        self.slow_ptr = new_min
                        self.buffer.popleft()
                yield return_val


class ForkIterDataPipeHelper(IterDataPipe):

    def __init__(self, main_datapipe: ForkIterDataPipe, instance_id: int):
        self.main_data_pipe = main_datapipe
        self.instance_id = instance_id

    def __iter__(self):
        yield from self.main_data_pipe.getNext(self.instance_id)


@functional_datapipe('demux')
class DemultiplexerIterDataPipe(IterDataPipe):

    # given n = num_instances
    # You want classifier_fn to return (0, ..., n - 1)
    # Check and make sure the output from classifier_fn is within range [0, n-1]
    # We are forcing people to return an int

    def __init__(self, datapipe: IterDataPipe[T_co], classifier_fn: Callable[[T_co], int]):
        self.datapipe = datapipe
        self.classifier_fn = classifier_fn

    # Placeholder implementation
    def __new__(cls, datapipe, instances, classifier_fn):
        result = []
        buffer = list(datapipe)

        def filter_fn(classifier_fn, i, x):
            return classifier_fn(x) == i
        return [IterateBuffer(buffer).filter(functools.partial(filter_fn, classifier_fn, i)) for i in range(instances)]

@functional_datapipe('mux')
class MultiplexerIterDataPipe(IterDataPipe):

    def __init__(self, *datapipes):
        self.datapipes = datapipes

    def __iter__(self):
        iterators = [iter(x) for x in self.datapipes]
        finished = {}
        had_more = True
        while had_more:
            had_more = False
            for i in range(len(iterators)):
                if i not in finished:
                    try:
                        value = iterators[i].__next__()
                        had_more = True
                        yield value
                    except StopIteration:
                        finished[i] = 1


@functional_datapipe('zip')
class ZipIterDataPipe(IterDataPipe[Tuple[T_co]]):
    r""" :class:`ZipIterDataPipe`.

    Iterable DataPipe aggregates elements into a tuple from each of
    the input DataPipe. The output DataPipe is stopped when the
    shortest input DataPipe is exhausted.
    args:
        *datapipes: Iterable DataPipes being aggregated
    """
    datapipes: Tuple[IterDataPipe]
    length: Optional[int]

    def __init__(self, *datapipes: IterDataPipe):
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("All inputs are required to be `IterDataPipe` "
                            "for `ZipIterDataPipe`.")
        super().__init__()
        self.datapipes = datapipes  # type: ignore[assignment]
        self.length = None

    def __iter__(self) -> Iterator[Tuple[T_co]]:
        for data in zip(*self.datapipes):
            yield data

    def __len__(self) -> int:
        if self.length is not None:
            if self.length == -1:
                raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
            return self.length
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            self.length = min(len(dp) for dp in self.datapipes)
        else:
            self.length = -1
        return len(self)
