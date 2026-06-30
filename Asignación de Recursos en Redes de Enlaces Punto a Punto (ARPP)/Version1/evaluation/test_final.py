import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train')))
from torch_geometric.loader import DataLoader
from torch.distributions import Categorical
import torch.nn.functional as F

from gnn import GNN
from utils import graphs_to_tensor, graphs_to_tensor_synthetic, get_gnn_inputs
from utils import power_constraint, get_rates, objective_function

def evaluate(weights_path, output_path=None, building_id=84, b5g=0, num_links=5, num_layers=4, K=5, batch_size=64, synthetic=False):
    """
    Versión simplificada del test, pensada para correr en un bucle FOR masivo sin gráficos ni ventanas.
    Guarda los resultados de testeo (la distribución del sum rate) en un '.pkl' y devuelve su ruta.
    """

    exp_dir = os.path.dirname(os.path.abspath(weights_path))
    
    if output_path is None:
        output_path = os.path.join(exp_dir, 'test_results.pkl')
    elif not os.path.isabs(output_path):
        output_path = os.path.join(exp_dir, output_path)

    if synthetic:
        x_tensor, channel_matrix_tensor = graphs_to_tensor_synthetic(num_links, num_features=1, b5g=b5g, building_id=building_id)
        dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
        test_dataset = dataset[7000:]
    else:
        x_tensor, channel_matrix_tensor = graphs_to_tensor(train=False, num_links=num_links, num_features=1, b5g=b5g, building_id=building_id)
        test_dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
 
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    num_channels = 3 
    power_levels_factors = [1.0] 
    num_power_levels = len(power_levels_factors)

    input_dim = 1
    hidden_dim = 16
    output_dim = (num_channels * num_power_levels) + 1  
    dropout = False

    max_antenna_power_dbm = 6
    max_antenna_power_mw = 10 ** (max_antenna_power_dbm / 10)
    pmax_per_ap = max_antenna_power_mw * 0.7

    model = GNN(input_dim, hidden_dim, output_dim, num_layers, dropout, K)
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        return None

    print(f"  [Test] Cargando pesos desde {weights_path}...")
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    sigma = 1e-4
    all_sum_rates = []

    print(f"  [Test] Iniciando Evaluación para Building ID {building_id}...")
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            channel_matrix_batch = data.matrix.view(batch_size, num_links, num_links)

            logits = model(data.x, data.edge_index, data.edge_attr)    
            
            psi = logits.view(batch_size, num_links, output_dim)
            probs = F.softmax(psi, dim=-1) 
            
            m = Categorical(probs=probs)  
            actions = m.sample()  

            phi = torch.zeros(batch_size, num_links, num_channels, device=probs.device)
            active_mask = (actions > 0)

            val_actions = actions[active_mask] - 1
            selected_channels = val_actions
            selected_power_indices = 0 
            
            factors_tensor = torch.tensor(power_levels_factors, device=probs.device)
            selected_powers = max_antenna_power_mw * factors_tensor[selected_power_indices]
            
            phi[active_mask, selected_channels] = selected_powers 

            rates = get_rates(phi, channel_matrix_batch, sigma,p0=max_antenna_power_mw)

            sum_rates = -objective_function(rates) 
            all_sum_rates.append(sum_rates)

    print(f"  [Test] Guardando resultados per-sample en {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(all_sum_rates, f)

    return output_path
