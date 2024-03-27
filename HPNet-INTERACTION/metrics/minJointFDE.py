from typing import List
import torch
from torchmetrics import Metric

class minJointFDE(Metric):
    def __init__(self, 
                 dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self,
               scored_predict_traj: List[torch.Tensor],                         #[(n1,K,F,2),...,(nb,K,F,2)]
               scored_target_traj: List[torch.Tensor],                          #[(n1,F,2),...,(nb,F,2)]
               scored_target_mask) -> None:                                     #[(n1,F),...,(nb,F)]
        batch_size = len(scored_predict_traj)
        for i in range(batch_size):
            errors = torch.norm(scored_predict_traj[i] - scored_target_traj[i].unsqueeze(1), p=2, dim=-1)     #[ni,K,F]
            errors = errors.transpose(0,1)                                          #[K,ni,F]
            errors = errors[:,:,-1][:,scored_target_mask[i][:,-1]]                  #[K,ni-valid]
            joint_fde = errors.mean(dim=1)                                          #[K]
            min_joint_fde = joint_fde.min(dim=0).values                         #1
            self.sum = self.sum + min_joint_fde
        self.count = self.count + batch_size
    
    def compute(self) -> torch.Tensor:
        return self.sum / self.count