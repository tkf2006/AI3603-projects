"""Memory-mapped, non-overlapping next-token blocks from preprocessed data."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenBlockDataset(Dataset):
    """For block i, return x=tokens[i*B:i*B+B] and y=x shifted by one."""

    def __init__(self, path, block_size=512):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if self.path.stat().st_size % np.dtype("<u2").itemsize:
            raise ValueError("Token file contains a partial uint16 value")
        self.block_size = block_size
        self.tokens = np.memmap(self.path, dtype="<u2", mode="r")
        self.n_blocks = max(0, (len(self.tokens) - 1) // block_size)

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, index):
        if not 0 <= index < self.n_blocks:
            raise IndexError(index)
        start = index * self.block_size
        window = self.tokens[start:start + self.block_size + 1]
        x = torch.from_numpy(np.array(window[:-1], dtype=np.int64, copy=True))
        y = torch.from_numpy(np.array(window[1:], dtype=np.int64, copy=True))
        return x, y
