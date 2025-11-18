from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse

from layers import GraphAttention
from layers import TwoLayerMLP
from utils import compute_angles_lengths_2D
from utils import init_weights
from utils import wrap_angle
from utils import drop_edge_between_samples
from utils import transform_point_to_local_coordinate
from utils import transform_point_to_global_coordinate
from utils import transform_traj_to_global_coordinate
from utils import transform_traj_to_local_coordinate


class Backbone(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 duration: int,
                 a2a_radius: float,
                 l2a_radius: float,
                 num_attn_layers: int,
                 num_modes: int,
                 num_heads: int,
                 dropout: float) -> None:
        super(Backbone, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.duration = duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.num_attn_layers = num_attn_layers
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.dropout = dropout

        self.mode_tokens = nn.Embedding(num_modes, hidden_dim)

        # Agent embedding with 4 features: velocity_length, velocity_theta, length, width
        self.a_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.l2m_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.t2m_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.m2m_h_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.m2m_a_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        # 3D edge features for m2m_s
        self.m2m_s_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.l2m_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.t2m_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)

        self.m2m_h_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.m2m_a_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        # Use edge attributes for m2m_s
        self.m2m_s_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])

        self.traj_propose = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.num_future_steps * 2)

        self.proposal_to_anchor = TwoLayerMLP(input_dim=self.num_future_steps * 2, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.l2n_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.t2n_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)

        self.n2n_h_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.n2n_a_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        # Use edge attributes for n2n_s
        self.n2n_s_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])

        self.traj_refine = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.num_future_steps * 2)

        self.apply(init_weights)

    def forward(self, data: Batch, l_embs: torch.Tensor) -> torch.Tensor:
        # Initialization
        a_velocity_length = data['agent']['velocity_length']
        a_velocity_theta = data['agent']['velocity_theta']
        a_length = data['agent']['length'].unsqueeze(-1).repeat_interleave(self.num_historical_steps, -1)
        a_width = data['agent']['width'].unsqueeze(-1).repeat_interleave(self.num_historical_steps, -1)
        a_input = torch.stack([a_velocity_length, a_velocity_theta, a_length, a_width], dim=-1)
        a_embs = self.a_emb_layer(input=a_input)

        num_all_agent = a_length.size(0)
        m_embs = self.mode_tokens.weight.unsqueeze(0).repeat_interleave(self.num_historical_steps, 0)
        m_embs = m_embs.unsqueeze(1).repeat_interleave(num_all_agent, 1).reshape(-1, self.hidden_dim)

        m_batch = data['agent']['batch'].unsqueeze(1).repeat_interleave(self.num_modes, 1)
        m_position = data['agent']['position'][:, :self.num_historical_steps].unsqueeze(2).repeat_interleave(self.num_modes, 2)
        m_heading = data['agent']['heading'][:, :self.num_historical_steps].unsqueeze(2).repeat_interleave(self.num_modes, 2)
        m_valid_mask = data['agent']['visible_mask'][:, :self.num_historical_steps].unsqueeze(2).repeat_interleave(self.num_modes, 2)

        # ALL EDGE
        # t2m edge
        t2m_position_t = data['agent']['position'][:, :self.num_historical_steps].reshape(-1, 2)
        t2m_position_m = m_position.reshape(-1, 2)
        t2m_heading_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1)
        t2m_heading_m = m_heading.reshape(-1)
        t2m_valid_mask_t = data['agent']['visible_mask'][:, :self.num_historical_steps]
        t2m_valid_mask_m = m_valid_mask.reshape(num_all_agent, -1)
        t2m_valid_mask = t2m_valid_mask_t.unsqueeze(2) & t2m_valid_mask_m.unsqueeze(1)
        t2m_edge_index = dense_to_sparse(t2m_valid_mask)[0]
        t2m_edge_index = t2m_edge_index[:, torch.floor(t2m_edge_index[1] / self.num_modes) >= t2m_edge_index[0]]
        t2m_edge_index = t2m_edge_index[:, torch.floor(t2m_edge_index[1] / self.num_modes) - t2m_edge_index[0] <= self.duration]
        t2m_edge_vector = transform_point_to_local_coordinate(t2m_position_t[t2m_edge_index[0]], t2m_position_m[t2m_edge_index[1]], t2m_heading_m[t2m_edge_index[1]])
        t2m_edge_attr_length, t2m_edge_attr_theta = compute_angles_lengths_2D(t2m_edge_vector)
        t2m_edge_attr_heading = wrap_angle(t2m_heading_t[t2m_edge_index[0]] - t2m_heading_m[t2m_edge_index[1]])
        t2m_edge_attr_interval = t2m_edge_index[0] - torch.floor(t2m_edge_index[1] / self.num_modes)
        t2m_edge_attr_input = torch.stack([t2m_edge_attr_length, t2m_edge_attr_theta, t2m_edge_attr_heading, t2m_edge_attr_interval], dim=-1)
        t2m_edge_attr_embs = self.t2m_emb_layer(input=t2m_edge_attr_input)

        # l2m edge
        l2m_position_l = data['lane']['position']
        l2m_position_m = m_position.reshape(-1, 2)
        l2m_heading_l = data['lane']['heading']
        l2m_heading_m = m_heading.reshape(-1)
        l2m_batch_l = data['lane']['batch']
        l2m_batch_m = m_batch.unsqueeze(1).repeat_interleave(self.num_historical_steps, 1).reshape(-1)
        l2m_valid_mask_l = data['lane']['visible_mask']
        l2m_valid_mask_m = m_valid_mask.reshape(-1)
        l2m_valid_mask = l2m_valid_mask_l.unsqueeze(1) & l2m_valid_mask_m.unsqueeze(0)
        l2m_valid_mask = drop_edge_between_samples(l2m_valid_mask, batch=(l2m_batch_l, l2m_batch_m))
        l2m_edge_index = dense_to_sparse(l2m_valid_mask)[0]
        l2m_edge_index = l2m_edge_index[:, torch.norm(l2m_position_l[l2m_edge_index[0]] - l2m_position_m[l2m_edge_index[1]], p=2, dim=-1) < self.l2a_radius]
        l2m_edge_vector = transform_point_to_local_coordinate(l2m_position_l[l2m_edge_index[0]], l2m_position_m[l2m_edge_index[1]], l2m_heading_m[l2m_edge_index[1]])
        l2m_edge_attr_length, l2m_edge_attr_theta = compute_angles_lengths_2D(l2m_edge_vector)
        l2m_edge_attr_heading = wrap_angle(l2m_heading_l[l2m_edge_index[0]] - l2m_heading_m[l2m_edge_index[1]])
        l2m_edge_attr_input = torch.stack([l2m_edge_attr_length, l2m_edge_attr_theta, l2m_edge_attr_heading], dim=-1)
        l2m_edge_attr_embs = self.l2m_emb_layer(input=l2m_edge_attr_input)

        # mode edge
        # m2m_a_edge
        m2m_a_position = m_position.permute(1, 2, 0, 3).reshape(-1, 2)
        m2m_a_heading = m_heading.permute(1, 2, 0).reshape(-1)
        m2m_a_batch = data['agent']['batch']
        m2m_a_valid_mask = m_valid_mask.permute(1, 2, 0).reshape(self.num_historical_steps * self.num_modes, -1)
        m2m_a_valid_mask = m2m_a_valid_mask.unsqueeze(2) & m2m_a_valid_mask.unsqueeze(1)
        m2m_a_valid_mask = drop_edge_between_samples(m2m_a_valid_mask, m2m_a_batch)
        m2m_a_edge_index = dense_to_sparse(m2m_a_valid_mask)[0]
        m2m_a_edge_index = m2m_a_edge_index[:, m2m_a_edge_index[1] != m2m_a_edge_index[0]]
        m2m_a_edge_index = m2m_a_edge_index[:, torch.norm(m2m_a_position[m2m_a_edge_index[1]] - m2m_a_position[m2m_a_edge_index[0]], p=2, dim=-1) < self.a2a_radius]
        m2m_a_edge_vector = transform_point_to_local_coordinate(m2m_a_position[m2m_a_edge_index[0]], m2m_a_position[m2m_a_edge_index[1]], m2m_a_heading[m2m_a_edge_index[1]])
        m2m_a_edge_attr_length, m2m_a_edge_attr_theta = compute_angles_lengths_2D(m2m_a_edge_vector)
        m2m_a_edge_attr_heading = wrap_angle(m2m_a_heading[m2m_a_edge_index[0]] - m2m_a_heading[m2m_a_edge_index[1]])
        m2m_a_edge_attr_input = torch.stack([m2m_a_edge_attr_length, m2m_a_edge_attr_theta, m2m_a_edge_attr_heading], dim=-1)
        m2m_a_edge_attr_embs = self.m2m_a_emb_layer(input=m2m_a_edge_attr_input)

        # m2m_h
        m2m_h_position = m_position.permute(2, 0, 1, 3).reshape(-1, 2)
        m2m_h_heading = m_heading.permute(2, 0, 1).reshape(-1)
        m2m_h_valid_mask = m_valid_mask.permute(2, 0, 1).reshape(-1, self.num_historical_steps)
        m2m_h_valid_mask = m2m_h_valid_mask.unsqueeze(2) & m2m_h_valid_mask.unsqueeze(1)
        m2m_h_edge_index = dense_to_sparse(m2m_h_valid_mask)[0]
        m2m_h_edge_index = m2m_h_edge_index[:, m2m_h_edge_index[1] > m2m_h_edge_index[0]]
        m2m_h_edge_index = m2m_h_edge_index[:, m2m_h_edge_index[1] - m2m_h_edge_index[0] <= self.duration]
        m2m_h_edge_vector = transform_point_to_local_coordinate(m2m_h_position[m2m_h_edge_index[0]], m2m_h_position[m2m_h_edge_index[1]], m2m_h_heading[m2m_h_edge_index[1]])
        m2m_h_edge_attr_length, m2m_h_edge_attr_theta = compute_angles_lengths_2D(m2m_h_edge_vector)
        m2m_h_edge_attr_heading = wrap_angle(m2m_h_heading[m2m_h_edge_index[0]] - m2m_h_heading[m2m_h_edge_index[1]])
        m2m_h_edge_attr_interval = m2m_h_edge_index[0] - m2m_h_edge_index[1]
        m2m_h_edge_attr_input = torch.stack([m2m_h_edge_attr_length, m2m_h_edge_attr_theta, m2m_h_edge_attr_heading, m2m_h_edge_attr_interval], dim=-1)
        m2m_h_edge_attr_embs = self.m2m_h_emb_layer(input=m2m_h_edge_attr_input)

        # m2m_s edge with 3D features
        m2m_s_position = m_position.transpose(0, 1).reshape(-1, 2)
        m2m_s_heading = m_heading.transpose(0, 1).reshape(-1)
        m2m_s_valid_mask = m_valid_mask.transpose(0, 1).reshape(-1, self.num_modes)
        m2m_s_valid_mask = m2m_s_valid_mask.unsqueeze(2) & m2m_s_valid_mask.unsqueeze(1)
        m2m_s_edge_index = dense_to_sparse(m2m_s_valid_mask)[0]
        m2m_s_edge_index = m2m_s_edge_index[:, m2m_s_edge_index[0] != m2m_s_edge_index[1]]
        m2m_s_edge_vector = transform_point_to_local_coordinate(m2m_s_position[m2m_s_edge_index[0]], m2m_s_position[m2m_s_edge_index[1]], m2m_s_heading[m2m_s_edge_index[1]])
        m2m_s_edge_attr_length, m2m_s_edge_attr_theta = compute_angles_lengths_2D(m2m_s_edge_vector)
        m2m_s_edge_attr_heading = wrap_angle(m2m_s_heading[m2m_s_edge_index[0]] - m2m_s_heading[m2m_s_edge_index[1]])
        m2m_s_edge_attr_input = torch.stack([m2m_s_edge_attr_length, m2m_s_edge_attr_theta, m2m_s_edge_attr_heading], dim=-1)
        m2m_s_edge_attr_embs = self.m2m_s_emb_layer(input=m2m_s_edge_attr_input)

        # ALL ATTENTION
        # t2m attention
        t_embs = a_embs.reshape(-1, self.hidden_dim)
        m_embs_t = self.t2m_attn_layer(x=[t_embs, m_embs], edge_index=t2m_edge_index, edge_attr=t2m_edge_attr_embs)

        # l2m attention
        m_embs_l = self.l2m_attn_layer(x=[l_embs, m_embs], edge_index=l2m_edge_index, edge_attr=l2m_edge_attr_embs)

        m_embs = m_embs_t + m_embs_l
        m_embs = m_embs.reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        # moda attention
        for i in range(self.num_attn_layers):
            # m2m_a
            m_embs = m_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(1, 2).reshape(-1, self.hidden_dim)
            m_embs = self.m2m_a_attn_layers[i](x=m_embs, edge_index=m2m_a_edge_index, edge_attr=m2m_a_edge_attr_embs)
            # m2m_h
            m_embs = m_embs.reshape(self.num_historical_steps, self.num_modes, num_all_agent, self.hidden_dim).permute(1, 2, 0, 3).reshape(-1, self.hidden_dim)
            m_embs = self.m2m_h_attn_layers[i](x=m_embs, edge_index=m2m_h_edge_index, edge_attr=m2m_h_edge_attr_embs)
            # m2m_s with edge attributes
            m_embs = m_embs.reshape(self.num_modes, num_all_agent, self.num_historical_steps, self.hidden_dim).transpose(0, 2).reshape(-1, self.hidden_dim)
            m_embs = self.m2m_s_attn_layers[i](x=m_embs, edge_index=m2m_s_edge_index, edge_attr=m2m_s_edge_attr_embs)
        m_embs = m_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)

        # generate traj
        traj_propose = self.traj_propose(m_embs).reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.num_future_steps, 2)
        traj_propose = transform_traj_to_global_coordinate(traj_propose, m_position, m_heading)

        # generate anchor
        proposal = traj_propose.detach()

        n_batch = m_batch
        n_position = proposal[:, :, :, self.num_future_steps // 2, :]
        _, n_heading = compute_angles_lengths_2D(proposal[:, :, :, self.num_future_steps // 2, :] - proposal[:, :, :, self.num_future_steps // 2 - 1, :])
        n_valid_mask = m_valid_mask

        proposal = transform_traj_to_local_coordinate(proposal, n_position, n_heading)
        anchor = self.proposal_to_anchor(proposal.reshape(-1, self.num_future_steps * 2))
        n_embs = anchor

        # t2n edge
        t2n_position_t = data['agent']['position'][:, :self.num_historical_steps].reshape(-1, 2)
        t2n_position_n = n_position.reshape(-1, 2)
        t2n_heading_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1)
        t2n_heading_n = n_heading.reshape(-1)
        t2n_valid_mask_t = data['agent']['visible_mask'][:, :self.num_historical_steps]
        t2n_valid_mask_n = n_valid_mask.reshape(num_all_agent, -1)
        t2n_valid_mask = t2n_valid_mask_t.unsqueeze(2) & t2n_valid_mask_n.unsqueeze(1)
        t2n_edge_index = dense_to_sparse(t2n_valid_mask)[0]
        t2n_edge_index = t2n_edge_index[:, torch.floor(t2n_edge_index[1] / self.num_modes) >= t2n_edge_index[0]]
        t2n_edge_index = t2n_edge_index[:, torch.floor(t2n_edge_index[1] / self.num_modes) - t2n_edge_index[0] <= self.duration]
        t2n_edge_vector = transform_point_to_local_coordinate(t2n_position_t[t2n_edge_index[0]], t2n_position_n[t2n_edge_index[1]], t2n_heading_n[t2n_edge_index[1]])
        t2n_edge_attr_length, t2n_edge_attr_theta = compute_angles_lengths_2D(t2n_edge_vector)
        t2n_edge_attr_heading = wrap_angle(t2n_heading_t[t2n_edge_index[0]] - t2n_heading_n[t2n_edge_index[1]])
        t2n_edge_attr_interval = t2n_edge_index[0] - torch.floor(t2n_edge_index[1] / self.num_modes) - self.num_future_steps // 2
        t2n_edge_attr_input = torch.stack([t2n_edge_attr_length, t2n_edge_attr_theta, t2n_edge_attr_heading, t2n_edge_attr_interval], dim=-1)
        t2n_edge_attr_embs = self.t2m_emb_layer(input=t2n_edge_attr_input)

        # l2n edge
        l2n_position_l = data['lane']['position']
        l2n_position_n = n_position.reshape(-1, 2)
        l2n_heading_l = data['lane']['heading']
        l2n_heading_n = n_heading.reshape(-1)
        l2n_batch_l = data['lane']['batch']
        l2n_batch_n = n_batch.unsqueeze(1).repeat_interleave(self.num_historical_steps, 1).reshape(-1)
        l2n_valid_mask_l = data['lane']['visible_mask']
        l2n_valid_mask_n = n_valid_mask.reshape(-1)
        l2n_valid_mask = l2n_valid_mask_l.unsqueeze(1) & l2n_valid_mask_n.unsqueeze(0)
        l2n_valid_mask = drop_edge_between_samples(l2n_valid_mask, batch=(l2n_batch_l, l2n_batch_n))
        l2n_edge_index = dense_to_sparse(l2n_valid_mask)[0]
        l2n_edge_index = l2n_edge_index[:, torch.norm(l2n_position_l[l2n_edge_index[0]] - l2n_position_n[l2n_edge_index[1]], p=2, dim=-1) < self.l2a_radius]
        l2n_edge_vector = transform_point_to_local_coordinate(l2n_position_l[l2n_edge_index[0]], l2n_position_n[l2n_edge_index[1]], l2n_heading_n[l2n_edge_index[1]])
        l2n_edge_attr_length, l2n_edge_attr_theta = compute_angles_lengths_2D(l2n_edge_vector)
        l2n_edge_attr_heading = wrap_angle(l2n_heading_l[l2n_edge_index[0]] - l2n_heading_n[l2n_edge_index[1]])
        l2n_edge_attr_input = torch.stack([l2n_edge_attr_length, l2n_edge_attr_theta, l2n_edge_attr_heading], dim=-1)
        l2n_edge_attr_embs = self.l2m_emb_layer(input=l2n_edge_attr_input)

        # mode edge
        # n2n_a_edge
        n2n_a_position = n_position.permute(1, 2, 0, 3).reshape(-1, 2)
        n2n_a_heading = n_heading.permute(1, 2, 0).reshape(-1)
        n2n_a_batch = data['agent']['batch']
        n2n_a_valid_mask = n_valid_mask.permute(1, 2, 0).reshape(self.num_historical_steps * self.num_modes, -1)
        n2n_a_valid_mask = n2n_a_valid_mask.unsqueeze(2) & n2n_a_valid_mask.unsqueeze(1)
        n2n_a_valid_mask = drop_edge_between_samples(n2n_a_valid_mask, n2n_a_batch)
        n2n_a_edge_index = dense_to_sparse(n2n_a_valid_mask)[0]
        n2n_a_edge_index = n2n_a_edge_index[:, n2n_a_edge_index[1] != n2n_a_edge_index[0]]
        n2n_a_edge_index = n2n_a_edge_index[:, torch.norm(n2n_a_position[n2n_a_edge_index[1]] - n2n_a_position[n2n_a_edge_index[0]], p=2, dim=-1) < self.a2a_radius]
        n2n_a_edge_vector = transform_point_to_local_coordinate(n2n_a_position[n2n_a_edge_index[0]], n2n_a_position[n2n_a_edge_index[1]], n2n_a_heading[n2n_a_edge_index[1]])
        n2n_a_edge_attr_length, n2n_a_edge_attr_theta = compute_angles_lengths_2D(n2n_a_edge_vector)
        n2n_a_edge_attr_heading = wrap_angle(n2n_a_heading[n2n_a_edge_index[0]] - n2n_a_heading[n2n_a_edge_index[1]])
        n2n_a_edge_attr_input = torch.stack([n2n_a_edge_attr_length, n2n_a_edge_attr_theta, n2n_a_edge_attr_heading], dim=-1)
        n2n_a_edge_attr_embs = self.m2m_a_emb_layer(input=n2n_a_edge_attr_input)

        # n2n_h edge
        n2n_h_position = n_position.permute(2, 0, 1, 3).reshape(-1, 2)
        n2n_h_heading = n_heading.permute(2, 0, 1).reshape(-1)
        n2n_h_valid_mask = n_valid_mask.permute(2, 0, 1).reshape(-1, self.num_historical_steps)
        n2n_h_valid_mask = n2n_h_valid_mask.unsqueeze(2) & n2n_h_valid_mask.unsqueeze(1)
        n2n_h_edge_index = dense_to_sparse(n2n_h_valid_mask)[0]
        n2n_h_edge_index = n2n_h_edge_index[:, n2n_h_edge_index[1] > n2n_h_edge_index[0]]
        n2n_h_edge_index = n2n_h_edge_index[:, n2n_h_edge_index[1] - n2n_h_edge_index[0] <= self.duration]
        n2n_h_edge_vector = transform_point_to_local_coordinate(n2n_h_position[n2n_h_edge_index[0]], n2n_h_position[n2n_h_edge_index[1]], n2n_h_heading[n2n_h_edge_index[1]])
        n2n_h_edge_attr_length, n2n_h_edge_attr_theta = compute_angles_lengths_2D(n2n_h_edge_vector)
        n2n_h_edge_attr_heading = wrap_angle(n2n_h_heading[n2n_h_edge_index[0]] - n2n_h_heading[n2n_h_edge_index[1]])
        n2n_h_edge_attr_interval = n2n_h_edge_index[0] - n2n_h_edge_index[1]
        n2n_h_edge_attr_input = torch.stack([n2n_h_edge_attr_length, n2n_h_edge_attr_theta, n2n_h_edge_attr_heading, n2n_h_edge_attr_interval], dim=-1)
        n2n_h_edge_attr_embs = self.m2m_h_emb_layer(input=n2n_h_edge_attr_input)

        # n2n_s edge with 3D features
        n2n_s_position = n_position.transpose(0, 1).reshape(-1, 2)
        n2n_s_heading = n_heading.transpose(0, 1).reshape(-1)
        n2n_s_valid_mask = n_valid_mask.transpose(0, 1).reshape(-1, self.num_modes)
        n2n_s_valid_mask = n2n_s_valid_mask.unsqueeze(2) & n2n_s_valid_mask.unsqueeze(1)
        n2n_s_edge_index = dense_to_sparse(n2n_s_valid_mask)[0]
        n2n_s_edge_index = n2n_s_edge_index[:, n2n_s_edge_index[0] != n2n_s_edge_index[1]]
        n2n_s_edge_vector = transform_point_to_local_coordinate(n2n_s_position[n2n_s_edge_index[0]], n2n_s_position[n2n_s_edge_index[1]], n2n_s_heading[n2n_s_edge_index[1]])
        n2n_s_edge_attr_length, n2n_s_edge_attr_theta = compute_angles_lengths_2D(n2n_s_edge_vector)
        n2n_s_edge_attr_heading = wrap_angle(n2n_s_heading[n2n_s_edge_index[0]] - n2n_s_heading[n2n_s_edge_index[1]])
        n2n_s_edge_attr_input = torch.stack([n2n_s_edge_attr_length, n2n_s_edge_attr_theta, n2n_s_edge_attr_heading], dim=-1)
        n2n_s_edge_attr_embs = self.m2m_s_emb_layer(input=n2n_s_edge_attr_input)

        # t2n attention
        t_embs = a_embs.reshape(-1, self.hidden_dim)
        n_embs_t = self.t2n_attn_layer(x=[t_embs, n_embs], edge_index=t2n_edge_index, edge_attr=t2n_edge_attr_embs)

        # l2m attention
        n_embs_l = self.l2n_attn_layer(x=[l_embs, n_embs], edge_index=l2n_edge_index, edge_attr=l2n_edge_attr_embs)

        n_embs = n_embs_t + n_embs_l
        n_embs = n_embs.reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        # moda attention
        for i in range(self.num_attn_layers):
            # m2m_a
            n_embs = n_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(1, 2).reshape(-1, self.hidden_dim)
            n_embs = self.n2n_a_attn_layers[i](x=n_embs, edge_index=n2n_a_edge_index, edge_attr=n2n_a_edge_attr_embs)
            # m2m_h
            n_embs = n_embs.reshape(self.num_historical_steps, self.num_modes, num_all_agent, self.hidden_dim).permute(1, 2, 0, 3).reshape(-1, self.hidden_dim)
            n_embs = self.n2n_h_attn_layers[i](x=n_embs, edge_index=n2n_h_edge_index, edge_attr=n2n_h_edge_attr_embs)
            # m2m_s with edge attributes
            n_embs = n_embs.reshape(self.num_modes, num_all_agent, self.num_historical_steps, self.hidden_dim).transpose(0, 2).reshape(-1, self.hidden_dim)
            n_embs = self.n2n_s_attn_layers[i](x=n_embs, edge_index=n2n_s_edge_index, edge_attr=n2n_s_edge_attr_embs)
        n_embs = n_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)

        # generate refinement
        traj_refine = self.traj_refine(n_embs).reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.num_future_steps, 2)
        traj_output = transform_traj_to_global_coordinate(proposal + traj_refine, n_position, n_heading)

        return traj_propose, traj_output
