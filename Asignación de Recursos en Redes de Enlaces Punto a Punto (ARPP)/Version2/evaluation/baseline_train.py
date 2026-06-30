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

def run(building_id=990, b5g=False, num_links=3, num_channels=11, 
        batch_size=64, epochs=200, synthetic=1, rn=100, rn1=100,
        max_antenna_power_dbm=6, policy='random', **kwargs):   
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
    pmax_per_ap = max_antenna_power_mw * 0.61
    sigma = 1e-4
    
    b_name = policy
    print(f"\n--- Starting Baseline: {b_name.upper()} ---")
    objective_function_values = []
    
    if policy == 'fixed':
        # Sorteamos los canales fijos por AP antes de empezar
        fixed_channels = torch.randint(0, num_channels, (num_links,))
    
    for epoc in range(epochs):
        for batch_idx, data in enumerate(dataloader):
            channel_matrix_batch = data.matrix.view(batch_size, num_links, num_links)
            
            p_tx_max = pmax_per_ap / max_antenna_power_mw
            
            if policy == 'random':
                power_levels_factors = [0.5, 1.0]
                num_power_levels = len(power_levels_factors)
                output_dim = (num_channels * num_power_levels) + 1
                
                probs = torch.zeros(batch_size, num_links, output_dim, device=channel_matrix_batch.device)
                probs[:, :, 0] = 1.0 - p_tx_max
                probs[:, :, 1:] = p_tx_max / (num_channels * num_power_levels)
                
                actions = torch.distributions.Categorical(probs=probs).sample()
                
                phi = torch.zeros(batch_size, num_links, num_channels, device=channel_matrix_batch.device)
                active_mask = (actions > 0)
                
                val_actions = actions[active_mask] - 1
                selected_channels = val_actions // num_power_levels
                selected_power_indices = val_actions % num_power_levels
                
                factors_tensor = torch.tensor(power_levels_factors, device=channel_matrix_batch.device)
                selected_powers = max_antenna_power_mw * factors_tensor[selected_power_indices]
                
                phi[active_mask, selected_channels] = selected_powers
                
            elif policy == 'fixed':
                phi = torch.zeros(batch_size, num_links, num_channels, device=channel_matrix_batch.device)
                
                # Sorteamos si el AP transmite o no en este instante
                active_mask = (torch.rand(batch_size, num_links, device=channel_matrix_batch.device) < p_tx_max)
                
                # Sorteamos el nivel de potencia
                power_levels_factors = torch.tensor([0.5, 1.0], device=channel_matrix_batch.device)
                num_power_levels = len(power_levels_factors)
                power_indices = torch.randint(0, num_power_levels, (batch_size, num_links), device=channel_matrix_batch.device)
                
                selected_powers = max_antenna_power_mw * power_levels_factors[power_indices]
                final_powers = selected_powers * active_mask.float()
                
                # Asignamos la potencia al canal que tiene fijo
                channels_batch = fixed_channels.expand(batch_size, -1).to(channel_matrix_batch.device)
                phi.scatter_(-1, channels_batch.unsqueeze(-1), final_powers.unsqueeze(-1))

            rates = get_rates(phi, channel_matrix_batch, sigma, p0=max_antenna_power_mw)
            sum_rate = -objective_function(rates)
            
            if batch_idx % 10 == 0:
                objective_function_values.append(sum_rate)
            
    banda = ['2_4', '5']
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, '..', 'results', f"{banda[b5g]}_{building_id}", 'torch_resultss') + '/'
    if not os.path.exists(path):
        os.makedirs(path)
        
    file_name = f"{path}baseline_{b_name}_{epochs}.pkl"
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
    parser.add_argument('--epochs', type=int, default=700)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--synthetic', type=int, default=0)
    parser.add_argument('--policy', type=str, default='random', choices=['random', 'fixed'])

    args = parser.parse_args()
    
    run(building_id=args.building_id, b5g=args.b5g, num_links=args.num_links, 
        num_channels=args.num_channels, 
        batch_size=args.batch_size, epochs=args.epochs,
        synthetic=args.synthetic, policy=args.policy)

