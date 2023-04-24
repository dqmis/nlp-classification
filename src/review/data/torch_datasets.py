from torch.utils.data import Dataset
import numpy as np
from typing import List, Any, Dict

from transformers.tokenization_utils_base import BatchEncoding

import torch


class AmazonTokensDataset(Dataset):
    def __init__(self, encodings: BatchEncoding, labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index) -> Dict[str, Any]:
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item
