import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
import math
import pandas as pd

from losses import HuberTrajLoss
from losses import HuberYawLoss
from metrics import minJointADE
from metrics import minJointFDE
from modules import Backbone
from modules import MapEncoder

from utils import generate_target
from utils import generate_predict_mask
from utils import compute_angles_lengths_2D

#torch.set_float32_matmul_precision('high')

class HPNet(pl.LightningModule):

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 duration: int,
                 a2a_radius: float,
                 l2a_radius: float,
                 num_visible_steps: int,
                 num_modes: int,
                 num_attn_layers: int,
                 num_hops: int,
                 num_heads: int,
                 dropout: float,
                 lr: float,
                 weight_decay: float,
                 warmup_epochs: int,
                 T_max: int,
                 **kwargs) -> None:
        super(HPNet, self).__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.duration = duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.num_visible_steps = num_visible_steps
        self.num_modes = num_modes
        self.num_attn_layers = num_attn_layers
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max

        self.Backbone = Backbone(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            duration=duration,
            a2a_radius=a2a_radius,
            l2a_radius=l2a_radius,
            num_attn_layers=num_attn_layers,
            num_modes=num_modes,
            num_heads=num_heads,
            dropout=dropout
        )
        self.MapEncoder = MapEncoder(
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            num_heads=num_heads,
            dropout=dropout
        )

        self.reg_loss_traj = HuberTrajLoss()

        self.min_joint_ade = minJointADE()
        self.min_joint_fde = minJointFDE()

        self._columns = ['case_id', 'track_id', 'frame_id', 'timestamp_ms', 'interesting_agent',
                         'x1', 'y1', 'psi_rad1',
                         'x2', 'y2', 'psi_rad2',
                         'x3', 'y3', 'psi_rad3',
                         'x4', 'y4', 'psi_rad4',
                         'x5', 'y5', 'psi_rad5',
                         'x6', 'y6', 'psi_rad6']
        self.test_output = dict()

    def forward(self, 
                data: Batch):
        lane_embs = self.MapEncoder(data=data)
        pred = self.Backbone(data=data, l_embs=lane_embs)
        return pred

    def training_step(self,data,batch_idx):
        traj_propose, traj_output = self(data)               #[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F],[(N1,...,Nb),H,K]
        
        agent_mask = data['agent']['category'] == 1
        traj_propose = traj_propose[agent_mask]
        traj_output = traj_output[agent_mask]

        target_traj, target_mask = generate_target(position=data['agent']['position'], 
                                                   mask=data['agent']['visible_mask'],
                                                   num_historical_steps=self.num_historical_steps,
                                                   num_future_steps=self.num_future_steps)  #[(N1,...Nb),H,F,2],[(N1,...Nb),H,F]

        target_traj = target_traj[agent_mask]
        target_mask = target_mask[agent_mask]
        
        agent_batch = data['agent']['batch'][agent_mask]
        batch_size = len(data['case_id'])
        errors = (torch.norm(traj_propose - target_traj.unsqueeze(2), p=2, dim=-1) * target_mask.unsqueeze(2)).sum(dim=-1)  #[(n1,...nb),H,K]
        joint_errors = [error.sum(dim=0, keepdim=True) for error in unbatch(errors, agent_batch)]
        joint_errors = torch.cat(joint_errors, dim=0)    #[b,H,K]

        num_agent_pre_batch = torch.bincount(agent_batch)
        best_mode_index = joint_errors.argmin(dim=-1)     #[b,H]
        best_mode_index = best_mode_index.repeat_interleave(num_agent_pre_batch, 0)     #[(N1,...Nb),H]
        traj_best_propose = traj_propose[torch.arange(traj_propose.size(0))[:, None], torch.arange(traj_propose.size(1))[None, :], best_mode_index]   #[(n1,...nb),H,F,2]
        traj_best_output = traj_output[torch.arange(traj_output.size(0))[:, None], torch.arange(traj_output.size(1))[None, :], best_mode_index]   #[(n1,...nb),H,F,2]
        
        predict_mask = generate_predict_mask(data['agent']['visible_mask'][agent_mask,:self.num_historical_steps], self.num_visible_steps)   #[(n1,...nb),H]
        targ_mask = target_mask[predict_mask]                               #[Na,F]
        traj_pro = traj_best_propose[predict_mask]                          #[Na,F,2]
        traj_ref = traj_best_output[predict_mask]                           #[Na,F,2]
        targ_traj = target_traj[predict_mask]                               #[Na,F,2]

        reg_loss_traj_propose = self.reg_loss_traj(traj_pro[targ_mask], targ_traj[targ_mask]) 
        reg_loss_traj_refine = self.reg_loss_traj(traj_ref[targ_mask], targ_traj[targ_mask])  
        loss = reg_loss_traj_propose + reg_loss_traj_refine
        self.log('train_reg_loss_traj_propose', reg_loss_traj_propose, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_reg_loss_traj_refine', reg_loss_traj_refine, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        return loss

    def validation_step(self,data,batch_idx):
        traj_propose, traj_output = self(data)               #[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2]
        
        agent_mask = data['agent']['category'] == 1
        traj_propose = traj_propose[agent_mask]
        traj_output = traj_output[agent_mask]

        target_traj, target_mask = generate_target(position=data['agent']['position'], 
                                                   mask=data['agent']['visible_mask'],
                                                   num_historical_steps=self.num_historical_steps,
                                                   num_future_steps=self.num_future_steps)  #[(N1,...Nb),H,F,2],[(N1,...Nb),H,F]
        target_traj = target_traj[agent_mask]
        target_mask = target_mask[agent_mask]
        
        agent_batch = data['agent']['batch'][agent_mask]
        batch_size = len(data['case_id'])
        errors = (torch.norm(traj_propose - target_traj.unsqueeze(2), p=2, dim=-1) * target_mask.unsqueeze(2)).sum(dim=-1)  #[(n1,...nb),H,K]
        joint_errors = [error.sum(dim=0, keepdim=True) for error in unbatch(errors, agent_batch)]
        joint_errors = torch.cat(joint_errors, dim=0)    #[b,H,K]

        num_agent_pre_batch = torch.bincount(agent_batch)
        best_mode_index = joint_errors.argmin(dim=-1)     #[b,H]
        best_mode_index = best_mode_index.repeat_interleave(num_agent_pre_batch, 0)     #[(N1,...Nb),H]
        traj_best_propose = traj_propose[torch.arange(traj_propose.size(0))[:, None], torch.arange(traj_propose.size(1))[None, :], best_mode_index]   #[(n1,...nb),H,F,2]
        traj_best_output = traj_output[torch.arange(traj_output.size(0))[:, None], torch.arange(traj_output.size(1))[None, :], best_mode_index]   #[(n1,...nb),H,F,2]
        
        predict_mask = generate_predict_mask(data['agent']['visible_mask'][agent_mask,:self.num_historical_steps], self.num_visible_steps)   #[(n1,...nb),H]
        targ_mask = target_mask[predict_mask]                               #[Na,F]
        traj_pro = traj_best_propose[predict_mask]                          #[Na,F,2]
        traj_ref = traj_best_output[predict_mask]                           #[Na,F,2]
        targ_traj = target_traj[predict_mask]                               #[Na,F,2]
        
        reg_loss_traj_propose = self.reg_loss_traj(traj_pro[targ_mask], targ_traj[targ_mask]) 
        reg_loss_traj_refine = self.reg_loss_traj(traj_ref[targ_mask], targ_traj[targ_mask])   
        loss = reg_loss_traj_propose + reg_loss_traj_refine
        self.log('val_reg_loss_traj_propose', reg_loss_traj_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_reg_loss_traj_refine', reg_loss_traj_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        visible_mask = data['agent']['visible_mask'][agent_mask]                      #[(n1,...nb),H+F]
        visible_num = visible_mask.sum(dim=-1)                                        #[(n1,...nb)]
        scored_mask = visible_num == self.num_historical_steps + self.num_future_steps
        scored_predict_traj = unbatch(traj_output[scored_mask,-1], agent_batch[scored_mask])                   #[(n1,K,F,2),...,(nb,K,F,2)]
        scored_target_traj = unbatch(target_traj[scored_mask,-1], agent_batch[scored_mask])                    #[(n1,F,2),...,(nb,F,2)]
        scored_target_mask = unbatch(target_mask[scored_mask,-1], agent_batch[scored_mask])                    #[(n1,F),...,(nb,F)]

        self.min_joint_ade.update(scored_predict_traj, scored_target_traj, scored_target_mask)
        self.min_joint_fde.update(scored_predict_traj, scored_target_traj, scored_target_mask)
        self.log('val_minJointADE', self.min_joint_ade, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('val_minJointFDE', self.min_joint_fde, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)


    def test_step(self,data,batch_idx):
        traj_propose, traj_output = self(data)               #[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F]

        agent_mask = data['agent']['category'] == 1
        agent_batch = data['agent']['batch'][agent_mask]
        batch_size = len(data['case_id'])
        num_agent_pre_batch = torch.bincount(agent_batch)

        scenario_name = data['scenario_name']   #[b]
        case_id = data['case_id']               #[b]
        agent_id = data['agent']['id'][agent_mask]
        agent_interset = data['agent']['interest'][agent_mask]
        traj_output = traj_output[agent_mask, -1]
        tep = torch.cat([data['agent']['position'][agent_mask, -1:].unsqueeze(1).repeat_interleave(self.num_modes,1), traj_output], dim=-2) #[(n1+...+nb),K,F+1,2]
        _, yaw_output = compute_angles_lengths_2D(tep[:,:,1:] - tep[:,:,:-1])   #[(n1+...+nb),K,F]

        scored_agent_id = unbatch(agent_id, agent_batch)                        #[n1,...nb]
        scored_agent_interset = unbatch(agent_interset, agent_batch)            #[n1,...nb]
        scored_predict_traj = unbatch(traj_output, agent_batch)           #[(n1,K,F,2),...,(nb,K,F,2)]
        scored_predict_yaw = unbatch(yaw_output, agent_batch)             #[(n1,K,F),...,(nb,K,F)]
        
        case_id = case_id.cpu().numpy()
        scored_agent_id = [agent_id.cpu().numpy() for agent_id in scored_agent_id]
        scored_agent_interset = [agent_interset.cpu().numpy() for agent_interset in scored_agent_interset]
        scored_predict_traj = [predict_traj.cpu().numpy() for predict_traj in scored_predict_traj]
        scored_predict_yaw = [predict_yaw.cpu().numpy() for predict_yaw in scored_predict_yaw]
        
        scored_frame_id = list(range(30))
        scored_frame_id = [id + 11 for id in scored_frame_id]
        scored_timestamp_ms = [frame_id * 100 for frame_id in scored_frame_id]

        for i in range(batch_size):
            rows = []
            for j in range(num_agent_pre_batch[i]):
                for k in range(self.num_future_steps):
                    row = [case_id[i], scored_agent_id[i][j], scored_frame_id[k], scored_timestamp_ms[k], scored_agent_interset[i][j],
                        scored_predict_traj[i][j,0,k,0], scored_predict_traj[i][j,0,k,1], scored_predict_yaw[i][j,0,k],
                        scored_predict_traj[i][j,1,k,0], scored_predict_traj[i][j,1,k,1], scored_predict_yaw[i][j,1,k],
                        scored_predict_traj[i][j,2,k,0], scored_predict_traj[i][j,2,k,1], scored_predict_yaw[i][j,2,k],
                        scored_predict_traj[i][j,3,k,0], scored_predict_traj[i][j,3,k,1], scored_predict_yaw[i][j,3,k],
                        scored_predict_traj[i][j,4,k,0], scored_predict_traj[i][j,4,k,1], scored_predict_yaw[i][j,4,k],
                        scored_predict_traj[i][j,5,k,0], scored_predict_traj[i][j,5,k,1], scored_predict_yaw[i][j,5,k]]
                    rows.append(row)

            if scenario_name[i] in self.test_output:
                self.test_output[scenario_name[i]] = self.test_output[scenario_name[i]] + rows
            else:
                self.test_output[scenario_name[i]] = rows

    def on_test_end(self):
        for key, value in self.test_output.items():
            df = pd.DataFrame(value, columns=self._columns)
            df['track_to_predict'] = 1
            df.to_csv('./test_output/' + key + '_sub.csv', index=False)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        
        warmup_epochs = self.warmup_epochs
        T_max = self.T_max

        def warmup_cosine_annealing_schedule(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs + 1) / (T_max - warmup_epochs + 1)))

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_annealing_schedule),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HPNet')
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--num_historical_steps', type=int, default=10)
        parser.add_argument('--num_future_steps', type=int, default=30)
        parser.add_argument('--duration', type=int, default=10)
        parser.add_argument('--a2a_radius', type=float, default=80)
        parser.add_argument('--l2a_radius', type=float, default=80)
        parser.add_argument('--num_visible_steps', type=int, default=3)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_attn_layers', type=int, default=3)
        parser.add_argument('--num_hops', type=int, default=4)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--warmup_epochs', type=int, default=4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
