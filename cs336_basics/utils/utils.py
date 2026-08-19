from pathlib import Path

import h5py
import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor
from typing import BinaryIO, IO

import os


def hdf5_to_npy(
    hdf5_path: str | Path,
    npy_path: str | Path,
    group_name: str = "tokens",
) -> None:
    """Concatenate an HDF5 group's datasets into one memory-mapped NPY file.

    Only one HDF5 dataset is loaded into memory at a time. Dataset names must
    be integer strings, as produced by ``BPEtokenizer.encode_file``.
    """
    hdf5_path = Path(hdf5_path)
    npy_path = Path(npy_path)

    with h5py.File(hdf5_path, "r") as hdf5_file:
        if group_name not in hdf5_file:
            raise KeyError(f"HDF5 group {group_name!r} was not found in {hdf5_path}")

        group = hdf5_file[group_name]
        if not isinstance(group, h5py.Group):
            raise TypeError(f"{group_name!r} in {hdf5_path} is not an HDF5 group")

        try:
            dataset_names = sorted(group.keys(), key=int)
        except ValueError as error:
            raise ValueError(f"All dataset names in {group_name!r} must be integer strings") from error

        if not dataset_names:
            raise ValueError(f"HDF5 group {group_name!r} contains no datasets")
        if not all(isinstance(group[name], h5py.Dataset) for name in dataset_names):
            raise TypeError(f"HDF5 group {group_name!r} must contain only datasets")

        datasets = [group[name] for name in dataset_names]
        total_size = sum(dataset.size for dataset in datasets)
        output_dtype = np.result_type(*(dataset.dtype for dataset in datasets))

        output = np.lib.format.open_memmap(
            npy_path,
            mode="w+",
            dtype=output_dtype,
            shape=(total_size,),
        )
        offset = 0
        try:
            for dataset in datasets:
                array = dataset[...]
                next_offset = offset + array.size
                output[offset:next_offset] = array.reshape(-1)
                offset = next_offset
                del array
        finally:
            output.flush()
            del output


def get_batch(
    x: np.array, 
    batch_size: int, 
    context_length: int,
    device_str: str = "cpu"
) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batchsize context_length"]]:
    starts = np.random.randint(0, len(x) - context_length, size=batch_size)
    input = np.stack([x[i: i + context_length] for i in starts])
    target = np.stack([x[i + 1 : i + context_length + 1] for i in starts])
    return (torch.from_numpy(input).long().to(device_str), torch.from_numpy(target).long().to(device_str))

def get_mmap(
    filename: str
) -> np.memmap:
    return np.load(filename, mmap_mode="r")

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]
):
    dic = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(dic, out)
    
def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device = "cpu",
) -> int:
    dic = torch.load(src, map_location=map_location)
    model.load_state_dict(dic["model"])
    optimizer.load_state_dict(dic["optimizer"])
    return dic["iteration"]
    

if __name__ == "__main__":
    tokenized_path = "/Users/leon34/Desktop/CSdiy/stanfordCS336/Code/Lab/assignment1-basics/data/tokenized"
    ts_train_tokenized_path = os.path.join(tokenized_path, "ts_train_tokenized.hd5")
    owt_train_tokenized_path = os.path.join(tokenized_path, "owt_train_tokenized.hd5")
    
    ts_train_tokenized_npy = os.path.join(tokenized_path, "ts_train_tokenized.npy")
    owt_train_tokenized_npy= os.path.join(tokenized_path, "owt_train_tokenized.npy")
    hdf5_to_npy(ts_train_tokenized_path, ts_train_tokenized_npy)
    hdf5_to_npy(owt_train_tokenized_path, owt_train_tokenized_npy)
