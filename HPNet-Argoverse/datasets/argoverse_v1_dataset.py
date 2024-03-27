import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from itertools import permutations
from itertools import product

import numpy as np
import pandas as pd
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm

from utils import compute_angles_lengths_2D
from utils import transform_point_to_local_coordinate
from utils import get_index_of_A_in_B


class ArgoverseV1Dataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 num_historical_steps: int = 20,
                 num_future_steps:int = 30,
                 margin: float = 50) -> None:
        self.root = root

        if split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test'
        else:
            raise ValueError(split + ' is not valid')
        
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [os.path.splitext(name)[0] + '.pt' for name in self.raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, name) for name in self.processed_file_names]
        
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps
        self.margin = margin

        self._turn_direction_type = ['NONE', 'LEFT', 'RIGHT']
        super(ArgoverseV1Dataset, self).__init__(root=root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'processed_data')
    
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        map_api = ArgoverseMap()
        for raw_path in tqdm(self.raw_paths):
            df = pd.read_csv(raw_path)
            data = dict()      
            scenario_id = self.get_scenario_id(raw_path)
            city = self.get_city(df)
            data['city'] = city
            data['scenario_id'] = scenario_id
            data.update(self.get_features(df, map_api, self.margin, city))
            torch.save(data, os.path.join(self.processed_dir, scenario_id +'.pt'))
    
    @staticmethod
    def get_scenario_id(raw_path: str) -> str:
        return os.path.splitext(os.path.basename(raw_path))[0]

    @staticmethod
    def get_city(df: pd.DataFrame) -> str:
        return df['CITY_NAME'].values[0]

    def get_features(self, 
                     df: pd.DataFrame,
                     map_api: ArgoverseMap,
                     margin: float,
                     city: str) -> Dict:
        data = {
            'agent': {},
            'lane': {},
            'centerline': {},
            ('centerline', 'lane'): {},
            ('lane', 'lane'): {}
        }
        #AGENT
        # filter out actors that are unseen during the historical time steps
        timestep_ids = list(np.sort(df['TIMESTAMP'].unique()))
        historical_timestamps = timestep_ids[:self.num_historical_steps]
        historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
        agent_ids = list(historical_df['TRACK_ID'].unique())
        num_agents = len(agent_ids)
        df = df[df['TRACK_ID'].isin(agent_ids)]
        
        agent_index = agent_ids.index(df[df['OBJECT_TYPE'] == 'AGENT']['TRACK_ID'].values[0])

        # initialization
        visible_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        length_mask = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.bool)
        agent_position = torch.zeros(num_agents, self.num_steps, 2, dtype=torch.float)
        agent_heading = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.float)
        agent_length = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.float)
        
        for track_id, track_df in df.groupby('TRACK_ID'):
            agent_idx = agent_ids.index(track_id)
            agent_steps = [timestep_ids.index(timestamp) for timestamp in track_df['TIMESTAMP']]

            visible_mask[agent_idx, agent_steps] = True

            length_mask[agent_idx, 0] = False
            length_mask[agent_idx, 1:] = ~(visible_mask[agent_idx, 1:self.num_historical_steps] & visible_mask[agent_idx, :self.num_historical_steps-1])

            agent_position[agent_idx, agent_steps] = torch.from_numpy(np.stack([track_df['X'].values, track_df['Y'].values], axis=-1)).float()
            motion = torch.cat([agent_position.new_zeros(1,2), agent_position[agent_idx,1:] - agent_position[agent_idx,:-1]], dim=0)
            length, heading = compute_angles_lengths_2D(motion)
            agent_length[agent_idx] = length[:self.num_historical_steps]
            agent_heading[agent_idx] = heading[:self.num_historical_steps]
            agent_length[agent_idx, length_mask[agent_idx]] = 0
            agent_heading[agent_idx, length_mask[agent_idx]] = 0

        data['agent']['num_nodes'] = num_agents
        data['agent']['agent_index'] = agent_index
        data['agent']['visible_mask'] = visible_mask
        data['agent']['position'] = agent_position
        data['agent']['heading'] = agent_heading
        data['agent']['length'] = agent_length
        
        #MAP
        positions = agent_position[:,:self.num_historical_steps][visible_mask[:,:self.num_historical_steps]].reshape(-1,2)
        left_boundary = min(positions[:,0])
        right_boundary = max(positions[:,0])
        down_boundary = min(positions[:,1])
        up_boundary = max(positions[:,1])
        lane_ids = map_api.get_lane_ids_in_xy_bbox((left_boundary + right_boundary) / 2, (down_boundary + up_boundary) / 2, city, max((right_boundary - left_boundary) / 2, (up_boundary - down_boundary) / 2) + margin)
        
        num_lanes = len(lane_ids)
        lane_position = torch.zeros(num_lanes, 2, dtype=torch.float)
        lane_heading = torch.zeros(num_lanes, dtype=torch.float)
        lane_length = torch.zeros(num_lanes, dtype=torch.float)
        lane_is_intersection = torch.zeros(num_lanes, dtype=torch.uint8)
        lane_turn_direction = torch.zeros(num_lanes, dtype=torch.uint8)
        lane_traffic_control = torch.zeros(num_lanes, dtype=torch.uint8)

        num_centerlines = torch.zeros(num_lanes, dtype=torch.long)
        centerline_position: List[Optional[torch.Tensor]] = [None] * num_lanes
        centerline_heading: List[Optional[torch.Tensor]] = [None] * num_lanes
        centerline_length: List[Optional[torch.Tensor]] = [None] * num_lanes

        lane_adjacent_edge_index = []
        lane_predecessor_edge_index = []
        lane_successor_edge_index = []
        for lane_id in lane_ids: 
            lane_idx = lane_ids.index(lane_id)

            centerlines = torch.from_numpy(map_api.get_lane_segment_centerline(lane_id, city)[:, :2]).float()
            num_centerlines[lane_idx] = centerlines.size(0) - 1
            centerline_position[lane_idx] = (centerlines[1:] + centerlines[:-1]) / 2
            centerline_vectors = centerlines[1:] - centerlines[:-1]
            centerline_length[lane_idx], centerline_heading[lane_idx] = compute_angles_lengths_2D(centerline_vectors)

            lane_length[lane_idx] = centerline_length[lane_idx].sum()
            center_index = int(num_centerlines[lane_idx]/2)
            lane_position[lane_idx] = centerlines[center_index]
            lane_heading[lane_idx] = torch.atan2(centerlines[center_index + 1, 1] - centerlines[center_index, 1], 
                                                 centerlines[center_index + 1, 0] - centerlines[center_index, 0])
            
            lane_is_intersection[lane_idx] = torch.tensor(map_api.lane_is_in_intersection(lane_id, city), dtype=torch.uint8) 
            lane_turn_direction[lane_idx] = torch.tensor(self._turn_direction_type.index(map_api.get_lane_turn_direction(lane_id, city)), dtype=torch.uint8)
            lane_traffic_control[lane_idx] = torch.tensor(map_api.lane_has_traffic_control_measure(lane_id, city), dtype=torch.uint8)

            lane_adjacent_ids = map_api.get_lane_segment_adjacent_ids(lane_id, city)
            lane_adjacent_idx = get_index_of_A_in_B(lane_adjacent_ids, lane_ids)
            if len(lane_adjacent_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_adjacent_idx, dtype=torch.long), torch.full((len(lane_adjacent_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_adjacent_edge_index.append(edge_index)
            lane_predecessor_ids = map_api.get_lane_segment_predecessor_ids(lane_id, city)
            lane_predecessor_idx = get_index_of_A_in_B(lane_predecessor_ids, lane_ids)
            if len(lane_predecessor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_predecessor_idx, dtype=torch.long), torch.full((len(lane_predecessor_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_predecessor_edge_index.append(edge_index)
            lane_successor_ids = map_api.get_lane_segment_successor_ids(lane_id, city)
            lane_successor_idx = get_index_of_A_in_B(lane_successor_ids, lane_ids)
            if len(lane_successor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_successor_idx, dtype=torch.long), torch.full((len(lane_successor_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_successor_edge_index.append(edge_index)

        data['lane']['num_nodes'] = num_lanes
        data['lane']['position'] = lane_position
        data['lane']['length'] = lane_length
        data['lane']['heading'] = lane_heading
        data['lane']['is_intersection'] = lane_is_intersection
        data['lane']['turn_direction'] = lane_turn_direction
        data['lane']['traffic_control'] = lane_traffic_control

        data['centerline']['num_nodes'] = num_centerlines.sum().item()
        data['centerline']['position'] = torch.cat(centerline_position, dim=0)
        data['centerline']['heading'] = torch.cat(centerline_heading, dim=0)
        data['centerline']['length'] = torch.cat(centerline_length, dim=0)

        centerline_to_lane_edge_index = torch.stack([torch.arange(num_centerlines.sum(), dtype=torch.long), torch.arange(num_lanes, dtype=torch.long).repeat_interleave(num_centerlines)], dim=0)
        data['centerline', 'lane']['centerline_to_lane_edge_index'] = centerline_to_lane_edge_index

        if len(lane_adjacent_edge_index) != 0:
            lane_adjacent_edge_index = torch.cat(lane_adjacent_edge_index, dim=1)
        else:
            lane_adjacent_edge_index = torch.tensor([[], []], dtype=torch.long)
        lane_predecessor_edge_index = torch.cat(lane_predecessor_edge_index, dim=1)
        lane_successor_edge_index = torch.cat(lane_successor_edge_index, dim=1)

        data['lane', 'lane']['adjacent_edge_index'] = lane_adjacent_edge_index
        data['lane', 'lane']['predecessor_edge_index'] = lane_predecessor_edge_index
        data['lane', 'lane']['successor_edge_index'] = lane_successor_edge_index
        
        return data

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx: int) -> HeteroData:     
        return HeteroData(torch.load(self.processed_paths[idx]))