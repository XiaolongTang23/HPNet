import math
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from utils import wrap_angle

class HorizontalFlip(BaseTransform):
    def __init__(self,
                 flip_p=0.5):
        super(HorizontalFlip, self).__init__()
        self.flip_p=flip_p
    
    def flip_position_and_heading(self, position, heading):
        position[...,0] = -position[...,0]
        angle = wrap_angle(math.pi - heading)
        return position, angle

    def __call__(self, data: HeteroData) -> HeteroData:
        if torch.rand(1).item() < self.flip_p:
            data['agent']['position'], data['agent']['heading'] = self.flip_position_and_heading(data['agent']['position'], data['agent']['heading'])
            data['lane']['position'], data['lane']['heading'] = self.flip_position_and_heading(data['lane']['position'], data['lane']['heading'])
            data['centerline']['position'], data['centerline']['heading'] = self.flip_position_and_heading(data['centerline']['position'], data['centerline']['heading'])
        return data