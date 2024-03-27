from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_dense_adj

from layers import GraphAttention
from layers import TwoLayerMLP
from utils import compute_angles_lengths_2D
from utils import init_weights
from utils import wrap_angle
from utils import transform_point_to_local_coordinate
from utils import generate_reachable_matrix

class MapEncoder(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_hops:int, 
                 num_heads: int,
                 dropout: float) -> None:
        super(MapEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout

        self._l2l_edge_type = ['left_neighbor', 'right_neighbor', 'predecessor', 'successor']

        self.p_emb_layer = TwoLayerMLP(input_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.l_emb_layer = TwoLayerMLP(input_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.p2l_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.l2l_emb_layer = TwoLayerMLP(input_dim=8, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.p2l_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.l2l_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True)

        self.apply(init_weights)

    def forward(self, data: Batch) -> torch.Tensor:
        #embedding
        p_length = data['polyline']['length']
        p_embs = self.p_emb_layer(input=p_length.unsqueeze(-1))        #[(C1,...,Cb),D]

        l_length = data['lane']['length']
        l_input = l_length.unsqueeze(-1)
        l_embs = self.l_emb_layer(input=l_input)   #[(M1,...,Mb),D]

        #edge
        #p2l
        p2l_position_p = data['polyline']['position']               #[(C1,...,Cb),2]
        p2l_position_l = data['lane']['position']                   #[(M1,...,Mb),2]
        p2l_heading_p = data['polyline']['heading']                 #[(C1,...,Cb)]
        p2l_heading_l = data['lane']['heading']                     #[(M1,...,Mb)]
        p2l_type = data['polyline']['side']                         #[(C1,...,Cb)]
        p2l_edge_index = data['polyline', 'lane']['polyline_to_lane_edge_index']    #[2,(C1,...,Cb)]
        p2l_edge_vector = transform_point_to_local_coordinate(p2l_position_p[p2l_edge_index[0]], p2l_position_l[p2l_edge_index[1]], p2l_heading_l[p2l_edge_index[1]])
        p2l_edge_attr_length, p2l_edge_attr_theta = compute_angles_lengths_2D(p2l_edge_vector)
        p2l_edge_attr_heading = wrap_angle(p2l_heading_p[p2l_edge_index[0]] - p2l_heading_l[p2l_edge_index[1]])
        p2l_edge_attr_type = p2l_type[p2l_edge_index[0]]
        p2l_edge_attr_input = torch.stack([p2l_edge_attr_length, p2l_edge_attr_theta, p2l_edge_attr_heading, p2l_edge_attr_type], dim=-1)
        p2l_edge_attr_embs = self.p2l_emb_layer(input = p2l_edge_attr_input)

        #l2l
        l2l_position = data['lane']['position']                     #[(M1,...,Mb),2]
        l2l_heading = data['lane']['heading']                       #[(M1,...,Mb)]
        l2l_edge_index = []
        l2l_edge_attr_type = []
        l2l_edge_attr_hop = []
        
        l2l_left_neighbor_edge_index = data['lane', 'lane']['left_neighbor_edge_index']
        num_left_neighbor_edges = l2l_left_neighbor_edge_index.size(1)
        l2l_edge_index.append(l2l_left_neighbor_edge_index)
        l2l_edge_attr_type.append(F.one_hot(torch.tensor(self._l2l_edge_type.index('left_neighbor')), num_classes=len(self._l2l_edge_type)).to(l2l_left_neighbor_edge_index.device).repeat(num_left_neighbor_edges, 1))
        l2l_edge_attr_hop.append(torch.ones(num_left_neighbor_edges, device=l2l_left_neighbor_edge_index.device))

        l2l_right_neighbor_edge_index = data['lane', 'lane']['right_neighbor_edge_index']
        num_right_neighbor_edges = l2l_right_neighbor_edge_index.size(1)
        l2l_edge_index.append(l2l_right_neighbor_edge_index)
        l2l_edge_attr_type.append(F.one_hot(torch.tensor(self._l2l_edge_type.index('right_neighbor')), num_classes=len(self._l2l_edge_type)).to(l2l_right_neighbor_edge_index.device).repeat(num_right_neighbor_edges, 1))
        l2l_edge_attr_hop.append(torch.ones(num_right_neighbor_edges, device=l2l_right_neighbor_edge_index.device))

        num_lanes = data['lane']['num_nodes']
        l2l_predecessor_edge_index = data['lane', 'lane']['predecessor_edge_index']
        l2l_predecessor_edge_index_all = generate_reachable_matrix(l2l_predecessor_edge_index, self.num_hops, num_lanes)
        for i in range(self.num_hops):
            num_edges_now = l2l_predecessor_edge_index_all[i].size(1)
            l2l_edge_index.append(l2l_predecessor_edge_index_all[i])
            l2l_edge_attr_type.append(F.one_hot(torch.tensor(self._l2l_edge_type.index('predecessor')), num_classes=len(self._l2l_edge_type)).to(l2l_predecessor_edge_index.device).repeat(num_edges_now, 1))
            l2l_edge_attr_hop.append((i + 1) * torch.ones(num_edges_now, device=l2l_predecessor_edge_index.device))
        
        l2l_successor_edge_index = data['lane', 'lane']['successor_edge_index']
        l2l_successor_edge_index_all = generate_reachable_matrix(l2l_successor_edge_index, self.num_hops, num_lanes)
        for i in range(self.num_hops):
            num_edges_now = l2l_successor_edge_index_all[i].size(1)
            l2l_edge_index.append(l2l_successor_edge_index_all[i])
            l2l_edge_attr_type.append(F.one_hot(torch.tensor(self._l2l_edge_type.index('successor')), num_classes=len(self._l2l_edge_type)).to(l2l_successor_edge_index.device).repeat(num_edges_now, 1))
            l2l_edge_attr_hop.append((i + 1) * torch.ones(num_edges_now, device=l2l_successor_edge_index.device))

        l2l_edge_index = torch.cat(l2l_edge_index, dim=1)
        l2l_edge_attr_type = torch.cat(l2l_edge_attr_type, dim=0)
        l2l_edge_attr_hop = torch.cat(l2l_edge_attr_hop, dim=0)
        l2l_edge_vector = transform_point_to_local_coordinate(l2l_position[l2l_edge_index[0]], l2l_position[l2l_edge_index[1]], l2l_heading[l2l_edge_index[1]])
        l2l_edge_attr_length, l2l_edge_attr_theta = compute_angles_lengths_2D(l2l_edge_vector)
        l2l_edge_attr_heading = wrap_angle(l2l_heading[l2l_edge_index[0]] - l2l_heading[l2l_edge_index[1]])
        l2l_edge_attr_input = torch.cat([l2l_edge_attr_length.unsqueeze(-1), l2l_edge_attr_theta.unsqueeze(-1), l2l_edge_attr_heading.unsqueeze(-1), l2l_edge_attr_hop.unsqueeze(-1), l2l_edge_attr_type.long()], dim=-1)
        l2l_edge_attr_embs = self.l2l_emb_layer(input=l2l_edge_attr_input)

        #attention
        #c2l
        l_embs = self.p2l_attn_layer(x = [p_embs, l_embs], edge_index = p2l_edge_index, edge_attr = p2l_edge_attr_embs)         #[(M1,...,Mb),D]

        #l2l
        l_embs = self.l2l_attn_layer(x = l_embs, edge_index = l2l_edge_index, edge_attr = l2l_edge_attr_embs)                   #[(M1,...,Mb),D]

        return l_embs