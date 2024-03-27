import math
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

class LaneRandomOcclusion(BaseTransform):
    def __init__(self, lane_occlusion_ratio=0.1):
        super(LaneRandomOcclusion, self).__init__()
        self.lane_occlusion_ratio = lane_occlusion_ratio

    def _mask_edge_index(self, edge_index, occlusion_index):
        mask = ~torch.isin(edge_index[0], occlusion_index)
        return edge_index[:, mask]

    def __call__(self, data):   
        num_occlusions = int(data['lane']['num_nodes'] * self.lane_occlusion_ratio)
        occlusion_index = torch.randperm(data['lane']['num_nodes'])[:num_occlusions]

        visible_mask = torch.ones(data['lane']['num_nodes'], dtype=torch.bool)
        visible_mask[occlusion_index] = False
        data['lane']['visible_mask'] = visible_mask

        edge_types = ['left_neighbor_edge_index', 'right_neighbor_edge_index', 'predecessor_edge_index', 'successor_edge_index']
        for edge_type in edge_types:
            edge_index = data['lane', 'lane'][edge_type]
            data['lane', 'lane'][edge_type] = self._mask_edge_index(edge_index, occlusion_index)

        return data