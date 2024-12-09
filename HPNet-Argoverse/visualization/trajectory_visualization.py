import matplotlib.pyplot as plt
from argoverse.map_representation.map_api import ArgoverseMap
import numpy as np
import torch
import os
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch

num_historical_steps = 20

def trajectory_visualization(data:Batch, traj_output: torch.tensor, is_test: torch.bool=False) -> None:
    batch_size = len(data['scenario_id'])
    city_name = data['city']

    agent_batch = data['agent']['batch']
    agent_position = data['agent']['position'].detach()
    agent_position = unbatch(agent_position, agent_batch)
    num_modes = traj_output.size(2)
    traj_output = traj_output.detach()
    traj_output = unbatch(traj_output[:,-1], agent_batch)
    agent_index = data['agent']['agent_index']

    map_api = ArgoverseMap()
    
    for i in range(batch_size):
        plt.figure()
        agent_position_i = agent_position[i][agent_index[i]].squeeze(0)
        agent_historical_position = agent_position_i[:num_historical_steps].cpu().numpy()
        agent_future_position = agent_position_i[num_historical_steps:].cpu().numpy()
        agent_prediction_position = traj_output[i][agent_index[i]].squeeze(0).cpu().numpy()

        if ~is_test:
            x_min = min(np.min(agent_historical_position[:,0]), np.min(agent_future_position[:,0]), np.min(agent_prediction_position[:,:,0]))
            x_max = max(np.max(agent_historical_position[:,0]), np.max(agent_future_position[:,0]), np.max(agent_prediction_position[:,:,0]))
            y_min = min(np.min(agent_historical_position[:,1]), np.min(agent_future_position[:,1]), np.min(agent_prediction_position[:,:,1]))
            y_max = max(np.max(agent_historical_position[:,1]), np.max(agent_future_position[:,1]), np.max(agent_prediction_position[:,:,1]))
        else:
            x_min = min(np.min(agent_historical_position[:,0]), np.min(agent_prediction_position[:,:,0]))
            x_max = max(np.max(agent_historical_position[:,0]), np.max(agent_prediction_position[:,:,0]))
            y_min = min(np.min(agent_historical_position[:,1]), np.min(agent_prediction_position[:,:,1]))
            y_max = max(np.max(agent_historical_position[:,1]), np.max(agent_prediction_position[:,:,1]))
        plt.xlim(x_min-5, x_max+5)
        plt.ylim(y_min-5, y_max+5)

        #Map
        #lane
        lane_ids = map_api.get_lane_ids_in_xy_bbox((x_min + x_max) / 2, (y_min + y_max) / 2, city_name[i], max((x_max - x_min) / 2, (y_max - y_min) / 2) + 5)
        for lane_id in lane_ids:
            lane_boundary = map_api.get_lane_segment_polygon(lane_id, city_name[i])[:,:2]
            plt.plot(
                lane_boundary[:, 0],
                lane_boundary[:, 1],
                "-",
                color="#E0E0E0",
                alpha=1,
                linewidth=1,
                zorder=0,
            )

        #history
        plt.plot(
            agent_historical_position[:, 0],
            agent_historical_position[:, 1],
            "-",
            color="green",
            alpha=1,
            linewidth=1,
            label="Historical Trajectory",
            zorder=2
        )
        plt.scatter(
            agent_historical_position[-1, 0],
            agent_historical_position[-1, 1],
            color="green",
            alpha=1,
            s=10,
            zorder=2
        )

        # GT
        if ~is_test:
            plt.plot(
                agent_future_position[:, 0],
                agent_future_position[:, 1],
                "-",
                color="red",
                alpha=1,
                linewidth=1,
                label="Future Trajectory",
                zorder=2
            )
            plt.scatter(
                agent_future_position[-1, 0],
                agent_future_position[-1, 1],
                color="red",
                alpha=1,
                s=10,
                zorder=2
            )

        #predict
        for j in range(num_modes):
            plt.plot(
                agent_prediction_position[j,:,0],
                agent_prediction_position[j,:,1],
                "-",
                color="blue",
                alpha=0.5,
                linewidth=1,
                label="Predicted Trajectory",
                zorder=1
            )
            plt.scatter(
                agent_prediction_position[j, -1, 0],
                agent_prediction_position[j, -1, 1],
                color="blue",
                alpha=0.5,
                s=10,
                zorder=1
            )
        
        plt.axis("off")
        if is_test:
            os.makedirs('test_output/visualization', exist_ok=True)
            plt.savefig(f'test_output/visualization/{data["scenario_id"][i]}.png')
        else:
            os.makedirs('visualization/trajectory', exist_ok=True)
            plt.savefig(f'visualization/trajectory/{data["scenario_id"][i]}.png')
        plt.close()
