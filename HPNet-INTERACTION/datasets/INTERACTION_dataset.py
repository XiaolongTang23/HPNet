import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from itertools import permutations
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm

import lanelet2
from lanelet2.projection import UtmProjector

from utils import compute_angles_lengths_2D
from utils import transform_point_to_local_coordinate
from utils import get_index_of_A_in_B
from utils import wrap_angle


class INTERACTIONDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 num_historical_steps: int = 10,
                 num_future_steps:int = 30) -> None:
        self.root = root

        if split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test_multi-agent'
        else:
            raise ValueError(split + ' is not valid')
        
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = []
        for raw_path in tqdm(self.raw_paths):
            raw_dir, raw_file_name = os.path.split(raw_path)
            scenario_name = os.path.splitext(raw_file_name)[0]
            scenario_name = '_'.join(scenario_name.split('_')[:-1])
            df = pd.read_csv(raw_path)
            for case_id in tqdm(df['case_id'].unique()):
                self._processed_file_names.append(scenario_name + '_' + str(int(case_id)) + '.pt')
        self._processed_paths = [os.path.join(self.processed_dir, name) for name in self.processed_file_names]
        
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps

        self.projector = UtmProjector(lanelet2.io.Origin(0, 0))
        self.traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                           lanelet2.traffic_rules.Participants.Vehicle)

        self._agent_type = ['car', 'pedestrian/bicycle']
        self._polyline_side = ['left','center','right']
        super(INTERACTIONDataset, self).__init__(root=root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory+'_processed')
    
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
        for raw_path in tqdm(self.raw_paths):
            #map
            raw_dir, raw_file_name = os.path.split(raw_path)
            scenario_name = os.path.splitext(raw_file_name)[0]
            scenario_name = '_'.join(scenario_name.split('_')[:-1])
            base_dir = os.path.dirname(raw_dir)
            map_path = os.path.join(base_dir,'maps',scenario_name+'.osm')
            map_api = lanelet2.io.load(map_path, self.projector)
            routing_graph = lanelet2.routing.RoutingGraph(map_api, self.traffic_rules)
            
            #agent
            df = pd.read_csv(raw_path)
            for case_id, case_df in tqdm(df.groupby('case_id')):
                data = dict()      
                data['scenario_name'] = scenario_name
                data['case_id'] = int(case_id)
                data.update(self.get_features(case_df, map_api, routing_graph))
                torch.save(data, os.path.join(self.processed_dir, scenario_name + '_' + str(int(case_id)) + '.pt'))

    def get_features(self, 
                     df: pd.DataFrame,
                     map_api,
                     routing_graph) -> Dict:
        data = {
            'agent': {},
            'lane': {},
            'polyline': {},
            ('polyline', 'lane'): {},
            ('lane', 'lane'): {}
        }
        #AGENT
        # filter out actors that are unseen during the historical time steps
        timestep_ids = list(np.sort(df['timestamp_ms'].unique()))
        historical_timestamps = timestep_ids[:self.num_historical_steps]
        historical_df = df[df['timestamp_ms'].isin(historical_timestamps)]
        agent_ids = list(historical_df['track_id'].unique())
        num_agents = len(agent_ids)
        df = df[df['track_id'].isin(agent_ids)]

        # initialization
        agent_id = torch.zeros(num_agents, dtype=torch.uint8)
        visible_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        agent_position = torch.zeros(num_agents, self.num_steps, 2, dtype=torch.float)
        agent_heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        agent_velocity = torch.zeros(num_agents, self.num_steps, 2, dtype=torch.float)
        agent_velocity_length = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.float)
        agent_velocity_theta = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.float)
        agent_length = torch.zeros(num_agents, dtype=torch.float)
        agent_width = torch.zeros(num_agents, dtype=torch.float)
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        agent_category = torch.zeros(num_agents, dtype=torch.uint8)
        agent_interset = torch.zeros(num_agents, dtype=torch.uint8)
        
        for track_id, track_df in df.groupby('track_id'):
            agent_idx = agent_ids.index(track_id)
            agent_id[agent_idx] = track_id
            agent_steps = [timestep_ids.index(timestamp) for timestamp in track_df['timestamp_ms']]

            visible_mask[agent_idx, agent_steps] = True

            agent_type_name = track_df['agent_type'].values[0]
            agent_type[agent_idx] = torch.tensor(self._agent_type.index(agent_type_name), dtype=torch.uint8)

            if agent_type_name == 'car':
                agent_length[agent_idx] = track_df['length'].values[0]
                agent_width[agent_idx] = track_df['width'].values[0]

            if 'track_to_predict' in track_df.columns:
                agent_category[agent_idx] = torch.tensor(track_df['track_to_predict'].values[0], dtype=torch.uint8)
                agent_interset[agent_idx] = torch.tensor(track_df['interesting_agent'].values[0], dtype=torch.uint8)
            elif agent_type_name == 'car':
                agent_category[agent_idx] = torch.tensor(1, dtype=torch.uint8)


            agent_position[agent_idx, agent_steps] = torch.from_numpy(np.stack([track_df['x'].values, track_df['y'].values], axis=-1)).float()

            agent_velocity[agent_idx, agent_steps] = torch.from_numpy(np.stack([track_df['vx'].values, track_df['vy'].values], axis=-1)).float()
            velocity_length, velocity_theta = compute_angles_lengths_2D(agent_velocity[agent_idx])
            agent_velocity_length[agent_idx] = velocity_length[:self.num_historical_steps]

            if agent_type_name == 'car':
                agent_heading[agent_idx,agent_steps] = torch.from_numpy(track_df['psi_rad'].values).float()
                agent_velocity_theta[agent_idx] = wrap_angle(velocity_theta[:self.num_historical_steps] - agent_heading[agent_idx,:self.num_historical_steps])
            else:
                agent_heading[agent_idx] = velocity_theta
                agent_velocity_theta[agent_idx] = 0


        data['agent']['id'] = agent_id
        data['agent']['num_nodes'] = num_agents
        data['agent']['visible_mask'] = visible_mask
        data['agent']['position'] = agent_position
        data['agent']['heading'] = agent_heading
        data['agent']['velocity_length'] = agent_velocity_length
        data['agent']['velocity_theta'] = agent_velocity_theta
        data['agent']['length'] = agent_length
        data['agent']['width'] = agent_width
        data['agent']['type'] = agent_type
        data['agent']['category'] = agent_category
        data['agent']['interest'] = agent_interset
        
        #MAP
        lane_ids = []
        for lane in map_api.laneletLayer: 
            lane_ids.append(lane.id)
        
        num_lanes = len(lane_ids)
        lane_id = torch.zeros(num_lanes, dtype=torch.float)
        lane_position = torch.zeros(num_lanes, 2, dtype=torch.float)
        lane_heading = torch.zeros(num_lanes, dtype=torch.float)
        lane_length = torch.zeros(num_lanes, dtype=torch.float)

        num_polylines = torch.zeros(num_lanes, dtype=torch.long)
        polyline_position: List[Optional[torch.Tensor]] = [None] * num_lanes
        polyline_heading: List[Optional[torch.Tensor]] = [None] * num_lanes
        polyline_length: List[Optional[torch.Tensor]] = [None] * num_lanes
        polyline_side: List[Optional[torch.Tensor]] = [None] * num_lanes

        lane_left_neighbor_edge_index = []
        lane_right_neighbor_edge_index = []
        lane_predecessor_edge_index = []
        lane_successor_edge_index = []

        for lane in map_api.laneletLayer: 
            lane_idx = lane_ids.index(lane.id)
            lane_id[lane_idx] = lane.id

            points = [np.array([pt.x, pt.y]) for pt in lane.centerline]
            centerline = torch.from_numpy(np.array(points)).float()

            center_index = int((centerline.size(0)-1)/2)
            lane_position[lane_idx] = centerline[center_index, :2]
            lane_heading[lane_idx] = torch.atan2(centerline[center_index+1, 1] - centerline[center_index, 1], 
                                                 centerline[center_index+1, 0] - centerline[center_index, 0])
            lane_length[lane_idx] = torch.norm(centerline[1:] - centerline[:-1], p=2, dim=-1).sum()
            
            points = [np.array([pt.x, pt.y]) for pt in lane.leftBound]
            left_boundary = torch.from_numpy(np.array(points)).float()
            points = [np.array([pt.x, pt.y]) for pt in lane.rightBound]
            right_boundary = torch.from_numpy(np.array(points)).float()
            left_vector = left_boundary[1:] - left_boundary[:-1]      
            right_vector = right_boundary[1:] - right_boundary[:-1]
            centerline_vector = centerline[1:] - centerline[:-1]
            polyline_position[lane_idx] = torch.cat([(left_boundary[1:] + left_boundary[:-1])/2, (right_boundary[1:] + right_boundary[:-1])/2, (centerline[1:] + centerline[:-1])/2], dim=0)
            polyline_length[lane_idx], polyline_heading[lane_idx] = compute_angles_lengths_2D(torch.cat([left_vector, right_vector, centerline_vector],dim=0))
            num_left_polyline = len(left_vector)
            num_right_polyline = len(right_vector)
            num_centerline_polyline = len(centerline_vector)
            polyline_side[lane_idx] = torch.cat(
                [torch.full((num_left_polyline,), self._polyline_side.index('left'), dtype=torch.uint8),
                 torch.full((num_right_polyline,), self._polyline_side.index('right'), dtype=torch.uint8),
                 torch.full((num_centerline_polyline,), self._polyline_side.index('center'), dtype=torch.uint8)], dim=0)

            num_polylines[lane_idx] = num_left_polyline + num_right_polyline + num_centerline_polyline

            lane_left_neighbor_lane = routing_graph.left(lane)
            lane_left_neighbor_id = [lane_left_neighbor_lane.id] if lane_left_neighbor_lane else []
            lane_left_neighbor_idx = get_index_of_A_in_B(lane_left_neighbor_id, lane_ids)
            if len(lane_left_neighbor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_left_neighbor_idx, dtype=torch.long), torch.full((len(lane_left_neighbor_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_left_neighbor_edge_index.append(edge_index)
            lane_right_neighbor_lane = routing_graph.right(lane)
            lane_right_neighbor_id = [lane_right_neighbor_lane.id] if lane_right_neighbor_lane else []
            lane_right_neighbor_idx = get_index_of_A_in_B(lane_right_neighbor_id, lane_ids)
            if len(lane_right_neighbor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_right_neighbor_idx, dtype=torch.long), torch.full((len(lane_right_neighbor_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_right_neighbor_edge_index.append(edge_index)
            lane_predecessor_lanes = routing_graph.previous(lane)
            lane_predecessor_ids = [ll.id for ll in lane_predecessor_lanes] if lane_predecessor_lanes else []
            lane_predecessor_idx = get_index_of_A_in_B(lane_predecessor_ids, lane_ids)
            if len(lane_predecessor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_predecessor_idx, dtype=torch.long), torch.full((len(lane_predecessor_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_predecessor_edge_index.append(edge_index)
            lane_successor_lanes = routing_graph.following(lane)
            lane_successor_ids = [ll.id for ll in lane_successor_lanes] if lane_successor_lanes else []
            lane_successor_idx = get_index_of_A_in_B(lane_successor_ids, lane_ids)
            if len(lane_successor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_successor_idx, dtype=torch.long), torch.full((len(lane_successor_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_successor_edge_index.append(edge_index)

        data['lane']['id'] = lane_id
        data['lane']['num_nodes'] = num_lanes
        data['lane']['position'] = lane_position
        data['lane']['length'] = lane_length
        data['lane']['heading'] = lane_heading

        data['polyline']['num_nodes'] = num_polylines.sum().item()
        data['polyline']['position'] = torch.cat(polyline_position, dim=0)
        data['polyline']['heading'] = torch.cat(polyline_heading, dim=0)
        data['polyline']['length'] = torch.cat(polyline_length, dim=0)
        data['polyline']['side'] = torch.cat(polyline_side, dim=0)

        polyline_to_lane_edge_index = torch.stack([torch.arange(num_polylines.sum(), dtype=torch.long), torch.arange(num_lanes, dtype=torch.long).repeat_interleave(num_polylines)], dim=0)
        data['polyline', 'lane']['polyline_to_lane_edge_index'] = polyline_to_lane_edge_index

        if len(lane_left_neighbor_edge_index) != 0:
            lane_left_neighbor_edge_index = torch.cat(lane_left_neighbor_edge_index, dim=1)
        else:
            lane_left_neighbor_edge_index = torch.tensor([[], []], dtype=torch.long)
        if len(lane_right_neighbor_edge_index) != 0:
            lane_right_neighbor_edge_index = torch.cat(lane_right_neighbor_edge_index, dim=1)
        else:
            lane_right_neighbor_edge_index = torch.tensor([[], []], dtype=torch.long)
        if len(lane_predecessor_edge_index) != 0:
            lane_predecessor_edge_index = torch.cat(lane_predecessor_edge_index, dim=1)
        else:
            lane_right_neighbor_edge_index = torch.tensor([[], []], dtype=torch.long)
        if len(lane_successor_edge_index) != 0:
            lane_successor_edge_index = torch.cat(lane_successor_edge_index, dim=1)
        else:
            lane_successor_edge_index = torch.tensor([[], []], dtype=torch.long)
        
        data['lane', 'lane']['left_neighbor_edge_index'] = lane_left_neighbor_edge_index
        data['lane', 'lane']['right_neighbor_edge_index'] = lane_right_neighbor_edge_index
        data['lane', 'lane']['predecessor_edge_index'] = lane_predecessor_edge_index
        data['lane', 'lane']['successor_edge_index'] = lane_successor_edge_index

        return data

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> HeteroData:     
        return HeteroData(torch.load(self.processed_paths[idx]))