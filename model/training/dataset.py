import numpy as np
import torch
from torch.utils.data import Dataset

from config import config, get_dataset_path


class TokenDataset(Dataset):

    def __init__(self) -> None:
        ## Memory-mapped data file
        self.mmdata: np.memmap | None = None

    def __len__(self) -> int:
        if self.mmdata is None:
            self.mmdata = np.memmap(get_dataset_path(), dtype=np.uint16, mode="r")
        # Eligible examples consist of max_seq_len tokens
        # Example indices range from [0, -max_seq_len]
        return len(self.mmdata) - config["max_seq_len"]

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mmdata is None:
            self.mmdata = np.memmap(get_dataset_path(), dtype=np.uint16, mode="r")

        input_ids = torch.from_numpy(
            self.mmdata[index : index + config["max_seq_len"]].astype(np.int64)
        )
        target_ids = torch.from_numpy(
            self.mmdata[index + 1 : index + config["max_seq_len"] + 1].astype(np.int64)
        )

        # ids: (batch_size, max_seq_len)
        return input_ids, target_ids
