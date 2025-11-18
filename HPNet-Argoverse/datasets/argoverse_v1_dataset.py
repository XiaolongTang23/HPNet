import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from itertools import permutations
from itertools import product
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm

from utils import compute_angles_lengths_2D
from utils import transform_point_to_local_coordinate
from utils import get_index_of_A_in_B
from utils import wrap_angle


class ArgoverseV1Dataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 num_historical_steps: int = 20,
                 num_future_steps: int = 30) -> None:
        self.root = root

        if split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'val'
        else:
            raise ValueError(split + ' is not valid')

        self._raw_file_names = [f for f in os.listdir(self.raw_dir) if f.endswith('.csv')]
        self._processed_file_names = []
        for raw_file_name in tqdm(self._raw_file_names):
            raw_path = os.path.join(self.raw_dir, raw_file_name)
            scenario_name = os.path.splitext(raw_file_name)[0]
            df = pd.read_csv(raw_path)
            for case_id in df['case_id'].unique():
                self._processed_file_names.append(scenario_name + '_' + str(int(case_id)) + '.pt')
        self._processed_paths = [os.path.join(self.processed_dir, name) for name in self.processed_file_names]

        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps

        self._polyline_side = ['left', 'center', 'right']
        super(ArgoverseV1Dataset, self).__init__(root=root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory + '_processed')

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
        for raw_file_name in tqdm(self._raw_file_names):
            raw_path = os.path.join(self.raw_dir, raw_file_name)
            scenario_name = os.path.splitext(raw_file_name)[0]

            # Load map
            map_path = os.path.join(self.root, 'maps', scenario_name + '.osm')
            map_data = self.load_osm_map(map_path)

            # Load trajectory data
            df = pd.read_csv(raw_path)
            for case_id, case_df in tqdm(df.groupby('case_id')):
                data = dict()
                data['scenario_name'] = scenario_name
                data['case_id'] = int(case_id)
                data.update(self.get_features(case_df, map_data))
                torch.save(data, os.path.join(self.processed_dir, scenario_name + '_' + str(int(case_id)) + '.pt'))

    def load_osm_map(self, map_path: str) -> Dict:
        """Load OSM map file with xy coordinates"""
        tree = ET.parse(map_path)
        root = tree.getroot()

        # Parse nodes
        nodes = {}
        for node in root.findall('.//node'):
            node_id = int(node.get('id'))
            x = float(node.get('x'))
            y = float(node.get('y'))
            nodes[node_id] = np.array([x, y])

        # Parse ways (lanes)
        ways = {}
        for way in root.findall('.//way'):
            way_id = int(way.get('id'))
            nd_refs = [int(nd.get('ref')) for nd in way.findall('nd')]
            ways[way_id] = nd_refs

        return {'nodes': nodes, 'ways': ways}

    def get_features(self,
                     df: pd.DataFrame,
                     map_data: Dict) -> Dict:
        data = {
            'agent': {},
            'lane': {},
            'polyline': {},
            ('polyline', 'lane'): {},
            ('lane', 'lane'): {}
        }

        # AGENT
        # Filter out actors that are unseen during the historical time steps
        frame_ids = list(np.sort(df['frame_id'].unique()))
        historical_frames = frame_ids[:self.num_historical_steps]
        historical_df = df[df['frame_id'].isin(historical_frames)]
        agent_ids = list(historical_df['track_id'].unique())
        num_agents = len(agent_ids)
        df = df[df['track_id'].isin(agent_ids)]

        # Initialization
        agent_id = torch.zeros(num_agents, dtype=torch.long)
        visible_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        agent_position = torch.zeros(num_agents, self.num_steps, 2, dtype=torch.float)
        agent_heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        agent_velocity = torch.zeros(num_agents, self.num_steps, 2, dtype=torch.float)
        agent_velocity_length = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.float)
        agent_velocity_theta = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.float)
        agent_length = torch.zeros(num_agents, dtype=torch.float)
        agent_width = torch.zeros(num_agents, dtype=torch.float)
        agent_category = torch.zeros(num_agents, dtype=torch.uint8)

        for track_id, track_df in df.groupby('track_id'):
            agent_idx = agent_ids.index(track_id)
            agent_id[agent_idx] = track_id
            agent_steps = [frame_ids.index(frame_id) for frame_id in track_df['frame_id']]

            visible_mask[agent_idx, agent_steps] = True

            # Set category based on agent_type
            agent_type_name = track_df['agent_type'].values[0]
            if agent_type_name == 'agent':
                agent_category[agent_idx] = torch.tensor(1, dtype=torch.uint8)
            else:
                agent_category[agent_idx] = torch.tensor(0, dtype=torch.uint8)

            agent_length[agent_idx] = track_df['length'].values[0]
            agent_width[agent_idx] = track_df['width'].values[0]

            agent_position[agent_idx, agent_steps] = torch.from_numpy(
                np.stack([track_df['x'].values, track_df['y'].values], axis=-1)).float()

            agent_velocity[agent_idx, agent_steps] = torch.from_numpy(
                np.stack([track_df['vx'].values, track_df['vy'].values], axis=-1)).float()

            velocity_length, velocity_theta = compute_angles_lengths_2D(agent_velocity[agent_idx])
            agent_velocity_length[agent_idx] = velocity_length[:self.num_historical_steps]

            # Use psi_rad directly for heading
            agent_heading[agent_idx, agent_steps] = torch.from_numpy(track_df['psi_rad'].values).float()

            # Compute velocity_theta as angle between velocity direction and heading
            agent_velocity_theta[agent_idx] = wrap_angle(
                velocity_theta[:self.num_historical_steps] - agent_heading[agent_idx, :self.num_historical_steps])

        data['agent']['id'] = agent_id
        data['agent']['num_nodes'] = num_agents
        data['agent']['visible_mask'] = visible_mask
        data['agent']['position'] = agent_position
        data['agent']['heading'] = agent_heading
        data['agent']['velocity_length'] = agent_velocity_length
        data['agent']['velocity_theta'] = agent_velocity_theta
        data['agent']['length'] = agent_length
        data['agent']['width'] = agent_width
        data['agent']['category'] = agent_category

        # MAP
        nodes = map_data['nodes']
        ways = map_data['ways']

        lane_ids = list(ways.keys())
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

        for lane_idx, way_id in enumerate(lane_ids):
            lane_id[lane_idx] = way_id

            # Get centerline points from way nodes
            nd_refs = ways[way_id]
            points = np.array([nodes[nd_ref] for nd_ref in nd_refs if nd_ref in nodes])

            if len(points) < 2:
                # Skip invalid lanes
                lane_position[lane_idx] = torch.zeros(2)
                lane_heading[lane_idx] = 0
                lane_length[lane_idx] = 0
                num_polylines[lane_idx] = 0
                polyline_position[lane_idx] = torch.zeros(0, 2)
                polyline_heading[lane_idx] = torch.zeros(0)
                polyline_length[lane_idx] = torch.zeros(0)
                polyline_side[lane_idx] = torch.zeros(0, dtype=torch.uint8)
                continue

            centerline = torch.from_numpy(points).float()

            # Lane center position and heading
            center_index = int((centerline.size(0) - 1) / 2)
            lane_position[lane_idx] = centerline[center_index, :2]

            if center_index + 1 < centerline.size(0):
                lane_heading[lane_idx] = torch.atan2(
                    centerline[center_index + 1, 1] - centerline[center_index, 1],
                    centerline[center_index + 1, 0] - centerline[center_index, 0])

            lane_length[lane_idx] = torch.norm(centerline[1:] - centerline[:-1], p=2, dim=-1).sum()

            # Generate polylines from centerline (as left, right, center)
            # For simplicity, we use centerline only and generate approximate boundaries
            centerline_vector = centerline[1:] - centerline[:-1]

            # Use centerline as the main polyline
            polyline_position[lane_idx] = (centerline[1:] + centerline[:-1]) / 2
            polyline_length[lane_idx], polyline_heading[lane_idx] = compute_angles_lengths_2D(centerline_vector)

            num_centerline_polyline = len(centerline_vector)
            polyline_side[lane_idx] = torch.full((num_centerline_polyline,),
                                                  self._polyline_side.index('center'),
                                                  dtype=torch.uint8)
            num_polylines[lane_idx] = num_centerline_polyline

        # Filter out empty polylines
        valid_polylines = [p for p in polyline_position if p is not None and p.numel() > 0]
        valid_headings = [h for h in polyline_heading if h is not None and h.numel() > 0]
        valid_lengths = [l for l in polyline_length if l is not None and l.numel() > 0]
        valid_sides = [s for s in polyline_side if s is not None and s.numel() > 0]

        data['lane']['id'] = lane_id
        data['lane']['num_nodes'] = num_lanes
        data['lane']['position'] = lane_position
        data['lane']['length'] = lane_length
        data['lane']['heading'] = lane_heading

        if valid_polylines:
            data['polyline']['num_nodes'] = num_polylines.sum().item()
            data['polyline']['position'] = torch.cat(valid_polylines, dim=0)
            data['polyline']['heading'] = torch.cat(valid_headings, dim=0)
            data['polyline']['length'] = torch.cat(valid_lengths, dim=0)
            data['polyline']['side'] = torch.cat(valid_sides, dim=0)
        else:
            data['polyline']['num_nodes'] = 0
            data['polyline']['position'] = torch.zeros(0, 2)
            data['polyline']['heading'] = torch.zeros(0)
            data['polyline']['length'] = torch.zeros(0)
            data['polyline']['side'] = torch.zeros(0, dtype=torch.uint8)

        # Polyline to lane edge index
        if num_polylines.sum() > 0:
            polyline_to_lane_edge_index = torch.stack([
                torch.arange(num_polylines.sum(), dtype=torch.long),
                torch.arange(num_lanes, dtype=torch.long).repeat_interleave(num_polylines)
            ], dim=0)
        else:
            polyline_to_lane_edge_index = torch.tensor([[], []], dtype=torch.long)
        data['polyline', 'lane']['polyline_to_lane_edge_index'] = polyline_to_lane_edge_index

        # Lane-to-lane edges (empty for now as we don't have routing info from simple OSM)
        data['lane', 'lane']['left_neighbor_edge_index'] = torch.tensor([[], []], dtype=torch.long)
        data['lane', 'lane']['right_neighbor_edge_index'] = torch.tensor([[], []], dtype=torch.long)
        data['lane', 'lane']['predecessor_edge_index'] = torch.tensor([[], []], dtype=torch.long)
        data['lane', 'lane']['successor_edge_index'] = torch.tensor([[], []], dtype=torch.long)

        return data

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> HeteroData:
        return HeteroData(torch.load(self.processed_paths[idx]))
