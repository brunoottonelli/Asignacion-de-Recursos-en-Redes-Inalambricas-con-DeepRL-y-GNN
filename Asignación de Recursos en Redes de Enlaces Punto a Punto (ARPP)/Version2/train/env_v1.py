import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch_geometric.data import Data, Batch
import torch_geometric as pyg
from utils import graphs_to_tensor, get_gnn_inputs, graphs_to_tensor_synthetic, get_rates, objective_function, power_constraint

class WirelessEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                building_id=990, 
                b5g=False, 
                num_links=5, 
                num_features=1, 
                sigma=1e-4,
                synthetic=False, 
                train=True,
                sanity_check=False
                ):
        super().__init__()
        
        self.building_id = building_id
        self.b5g = b5g
        self.num_links = num_links
        self.synthetic = synthetic
        self.train = train
        self.sanity_check = sanity_check
        
        if synthetic or sanity_check:
            filename = 'sc_graphs.pkl' if sanity_check else 'synthetic_graphs.pkl'
            x_tensor, channel_matrix_tensor = graphs_to_tensor_synthetic(num_links, num_features=num_features, b5g=b5g, building_id=building_id, filename=filename)
            self.dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
            if train:
                self.dataset = self.dataset
        else:
            x_tensor, channel_matrix_tensor = graphs_to_tensor(train=train, num_links=num_links, num_features=num_features, b5g=b5g, building_id=building_id)
            self.dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
            self.dataset = self.dataset
            
        self.num_samples = len(self.dataset)
        self.current_idx = 0
        self.indices = np.arange(self.num_samples)
        
        if self.train:
            np.random.shuffle(self.indices)
            
        self.action_space = None 
        self.observation_space = None

        max_antenna_power_dbm = 6 
        self.max_antenna_power_mw = 10 ** (max_antenna_power_dbm / 10) 
        self.pmax_per_ap = self.max_antenna_power_mw*0.6* torch.ones((num_links,))
        self.sigma = sigma       

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.current_idx >= self.num_samples:
             self.current_idx = 0
             if self.train:
                 np.random.shuffle(self.indices)
        
        data_idx = self.indices[self.current_idx]
        self.data = self.dataset[data_idx]
        self.current_idx += 1
        
        return self.data, {}

    def step(self, action):
        phi = action
        if not isinstance(phi, torch.Tensor):
            phi = torch.tensor(phi)
            
        channel_matrix = self.data.matrix
        
        if phi.dim() == 2:
             phi = phi.unsqueeze(0)
        
        mat = channel_matrix
        if mat.dim() == 2:
            mat = mat.unsqueeze(0)
            
        if hasattr(self.data, 'batch') and self.data.batch is not None:
             batch_size = phi.shape[0]
             N = self.num_links
             mat = mat.view(batch_size, N, N)

        rates = get_rates(phi, mat, self.sigma, self.max_antenna_power_mw)
        power_constr = power_constraint(phi, self.pmax_per_ap)
        sum_rate = objective_function(rates) 
        reward = sum_rate 
        
        info = {
            'power_constr': power_constr,
            'rates': rates,
            'sum_rate': sum_rate
        }
        
        terminated = True 
        truncated = False
        
        return self.data, reward, terminated, truncated, info

    def set_dataloader(self, batch_size=1):
        pass
