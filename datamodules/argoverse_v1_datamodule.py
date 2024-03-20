from typing import Callable, Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from datasets import ArgoverseV1Dataset

from transforms import HorizontalFlip
from transforms import AgentRandomOcclusion
from transforms import LaneRandomOcclusion


class ArgoverseV1DataModule(pl.LightningDataModule):

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 flip_p: float = 0.5,
                 agent_occlusion_ratio: float = 0.05,
                 lane_occlusion_ratio: float = 0.2,
                 num_historical_steps: int = 20,
                 num_future_steps: int = 30,
                 margin: float = 50,
                 **kwargs) -> None:
        super(ArgoverseV1DataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_transform = Compose([HorizontalFlip(flip_p), AgentRandomOcclusion(agent_occlusion_ratio,num_historical_steps), LaneRandomOcclusion(lane_occlusion_ratio)])
        self.val_transform = LaneRandomOcclusion(0.0)
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.margin = margin

    def prepare_data(self) -> None:
        ArgoverseV1Dataset(self.root, 'train', self.train_transform, self.num_historical_steps, self.num_future_steps, self.margin)
        ArgoverseV1Dataset(self.root, 'val', self.val_transform, self.num_historical_steps, self.num_future_steps, self.margin)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ArgoverseV1Dataset(self.root, 'train', self.train_transform, self.num_historical_steps, self.num_future_steps, self.margin)
        self.val_dataset = ArgoverseV1Dataset(self.root, 'val', self.val_transform, self.num_historical_steps, self.num_future_steps, self.margin)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)
