import pickle
import numpy as np
import networkx as nx

import torch



from torch_geometric.loader import DataLoader

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train')))

from utils import get_rates
from utils import graphs_to_tensor
from utils import get_gnn_inputs, objective_function


from utils import graphs_to_tensor_synthetic

def run(building_id=990, b5g=False, num_links=3, num_channels=11, num_layers=5, K=3,
        batch_size=64, epochs=200, eps=5e-5, mu_lr=5e-4, synthetic=1, rn=100, rn1=100,
         sigma=1e-4, max_antenna_power_dbm=6, baseline_type=2):   
    """
    baseline_type:
    -1: Run ALL baselines (Random and Fixed Power)
    0: GNN (requires model)
    2: Random
    3: Fixed Power
    """
    torch.manual_seed(rn)
    np.random.seed(rn1) 

    if synthetic:
        x_tensor, channel_matrix_tensor = graphs_to_tensor_synthetic(
            num_links=num_links, num_features=1, b5g=b5g, building_id=building_id
        )
        num_links = x_tensor.shape[1] 
        dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
        dataloader = DataLoader(dataset[:7000], batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        x_tensor, channel_matrix_tensor = graphs_to_tensor(
            train=True, num_links=num_links, num_features=1, b5g=b5g, building_id=building_id
        )
        num_links = x_tensor.shape[1] 
        dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    max_antenna_power_mw = 10 ** (max_antenna_power_dbm / 10)
    pmax_per_ap = max_antenna_power_mw * 0.8
    
    baselines_to_run = []
    if baseline_type == -1:
        baselines_to_run = [2, 3]
    else:
        baselines_to_run = [baseline_type]

    for b_type in baselines_to_run:
        b_name = ['gnn', 'unused', 'random', 'fixed_power'][b_type]
        print(f"\n--- Starting Baseline: {b_name.upper()} ---")
        objective_function_values = []

        if b_type == 3:
            fixed_channels = torch.randint(0, num_channels, (1, num_links))
        
        for epoc in range(epochs):
            for batch_idx, data in enumerate(dataloader):
                channel_matrix_batch = data.matrix.view(batch_size, num_links, num_links)
                
                if b_type == 1:
                    continue
                elif b_type == 2:
                    p_tx_max = pmax_per_ap / max_antenna_power_mw
                    probs = torch.zeros(batch_size, num_links, num_channels + 1, device=channel_matrix_batch.device)
                    probs[:, :, 0] = 1.0 - p_tx_max
                    probs[:, :, 1:] = p_tx_max / num_channels
                    actions = torch.distributions.Categorical(probs=probs).sample()
                    
                    phi = torch.zeros(batch_size, num_links, num_channels, device=channel_matrix_batch.device)
                    active_mask = (actions > 0)
                    active_channels = actions[active_mask] - 1
                    phi[active_mask, active_channels] = max_antenna_power_mw
                elif b_type == 3:
                    phi = torch.zeros(batch_size, num_links, num_channels, device=channel_matrix_batch.device)
                    
                    channels_batch = fixed_channels.expand(batch_size, -1).to(channel_matrix_batch.device)
                    
                    phi.scatter_(-1, channels_batch.unsqueeze(-1), pmax_per_ap)
                else:
                    continue 

                rates = get_rates(phi, channel_matrix_batch, sigma, p0=max_antenna_power_mw)
                sum_rate = -objective_function(rates)
                
                if batch_idx % 10 == 0:
                    objective_function_values.extend(sum_rate.detach().cpu().numpy().tolist())
                
        banda = ['2_4', '5']
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, '..', 'results', f"{banda[b5g]}_{building_id}", 'torch_results') + '/'
        if not os.path.exists(path):
            os.makedirs(path)
            
        file_name = f"{path}baseline_{b_name}_trajectory.pkl"
        with open(file_name, 'wb') as archivo:
            pickle.dump(objective_function_values, archivo)
        print(f"Results saved to {file_name}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--building_id', type=int, default=856)
    parser.add_argument('--b5g', type=int, default=0)
    parser.add_argument('--num_links', type=int, default=20)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eps', type=float, default=5e-4)
    parser.add_argument('--mu_lr', type=float, default=5e-4)
    parser.add_argument('--synthetic', type=int, default=0)
    parser.add_argument('--baseline', type=int, default=-1, help="-1: All, 0: GNN, 2: Random, 3: Fixed_Power")

    args = parser.parse_args()
    
    run(building_id=args.building_id, b5g=args.b5g, num_links=args.num_links, 
        num_channels=args.num_channels, num_layers=args.num_layers, K=args.k, 
        batch_size=args.batch_size, epochs=args.epochs, eps=args.eps, mu_lr=args.mu_lr, 
        baseline_type=args.baseline, synthetic=args.synthetic)

