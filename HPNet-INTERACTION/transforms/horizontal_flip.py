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
    
    def flip_position(self, position):
        position[...,0] = -position[...,0]
        return position

    def flip_heading(self, heading):
        angle = wrap_angle(math.pi - heading)
        return angle

    def __call__(self, data: HeteroData) -> HeteroData:
        if torch.rand(1).item() < self.flip_p:
            data['agent']['position'], data['agent']['heading'], data['agent']['velocity_theta'] = self.flip_position(data['agent']['position']), self.flip_heading(data['agent']['heading']), self.flip_heading(data['agent']['velocity_theta'])
            data['lane']['position'], data['lane']['heading'] = self.flip_position(data['lane']['position']), self.flip_heading(data['lane']['heading'])
            data['polyline']['position'], data['polyline']['heading'] = self.flip_position(data['polyline']['position']), self.flip_heading(data['polyline']['heading'])
            data['polyline']['side'] = 2 - data['polyline']['side']
            data['lane', 'lane']['left_neighbor_edge_index'], data['lane', 'lane']['right_neighbor_edge_index'] = data['lane', 'lane']['right_neighbor_edge_index'], data['lane', 'lane']['left_neighbor_edge_index']
        return data