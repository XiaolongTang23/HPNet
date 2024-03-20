import math
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

class AgentRandomOcclusion(BaseTransform):
    def __init__(self, agent_occlusion_ratio=0.1, num_historical_steps=20):
        super(AgentRandomOcclusion, self).__init__()
        self.agent_occlusion_ratio = agent_occlusion_ratio
        self.num_historical_steps = num_historical_steps

    def __call__(self, data):
        visible_mask = data['agent']['visible_mask'][:, :self.num_historical_steps]
        visible_position = torch.nonzero(visible_mask == 1)

        num_occlusions = int(visible_position.size(0) * self.agent_occlusion_ratio)
        occlusion_index = torch.randperm(visible_position.size(0))[:num_occlusions]
        occlusion_position = visible_position[occlusion_index]
        for x, y in occlusion_position:
            visible_mask[x, y] = False
        data['agent']['visible_mask'][:, :self.num_historical_steps] = visible_mask

        return data