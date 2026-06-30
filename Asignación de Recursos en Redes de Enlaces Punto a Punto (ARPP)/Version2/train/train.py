import pickle
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluation')))

from plot_results import plot_results
from gnn import GNN
from utils import mu_update
from env_v1 import WirelessEnv

def run(building_id=990, b5g=False, num_links=5, num_layers=5, K=5, batch_size=64, epochs=100, eps=5e-4, mu_lr=5e-4, synthetic=1, sanity_check=0, rn=100, rn1=100):   

    torch.manual_seed(rn)
    np.random.seed(rn1) 

    env = WirelessEnv(building_id=building_id, b5g=b5g, num_links=num_links, num_features=1, sigma=1e-4, synthetic=synthetic, train=True, sanity_check=sanity_check)
    
    dataloader = DataLoader(env.dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    mu_k = torch.ones((1, num_links), requires_grad = False)
    epochs = epochs

    max_antenna_power_dbm = 6
    max_antenna_power_mw = 10 ** (max_antenna_power_dbm / 10)

    num_channels = 3
    
    power_levels_factors = [0.5, 1.0] 
    num_power_levels = len(power_levels_factors)

    input_dim = 1
    hidden_dim = 16
    output_dim = (num_channels * num_power_levels) + 1
    
    num_layers = num_layers 
    dropout = True


    gnn_model = GNN(input_dim, hidden_dim, output_dim, num_layers, dropout, K)

    optimizer = optim.Adam(gnn_model.parameters(), lr= mu_lr)

    for name, param in gnn_model.named_parameters():
        param.requires_grad = True
        if param.requires_grad:
            print(name, param.data)

    objective_function_values = []
    power_constraint_values = []
    loss_values = []
    mu_k_values = []
    probs_values = []
    plt.ion() 
    fig_mu, ax_mu = plt.subplots(figsize=(10, 6))
    ax_mu.set_xlabel('Steps (x10 batches)')
    ax_mu.set_ylabel('mu_k values')
    ax_mu.set_title('Real-time mu_k per link')
    lines_mu = [ax_mu.plot([], [], label=f'Link {i}')[0] for i in range(num_links)]
    
    for epoc in range(epochs):
        print("Epoc number: {}".format(epoc))
        
        for batch_idx, data in enumerate(dataloader):

            psi = gnn_model.forward(data.x, data.edge_index, data.edge_attr)

            psi = psi.view(batch_size, num_links, output_dim) 
            
            probs = torch.softmax(psi, dim=-1)  
            m = Categorical(probs=probs)  
            actions = m.sample()            
            log_p = m.log_prob(actions)     
            log_p_sum = log_p.sum(dim=1).unsqueeze(-1)      
            
            phi = torch.zeros(batch_size, num_links, num_channels, device=probs.device)
            active_mask = (actions > 0)
            
            val_actions = actions[active_mask] - 1
            
            selected_channels = val_actions // num_power_levels
            selected_power_indices = val_actions % num_power_levels
            
            factors_tensor = torch.tensor(power_levels_factors, device=probs.device)
            selected_powers = max_antenna_power_mw * factors_tensor[selected_power_indices]
            
            phi[active_mask, selected_channels] = selected_powers

            env.data = data 
            
            obs, reward, terminated, truncated, info = env.step(phi)
            
            power_constr_per_ap = info['power_constr'] 
            
            sum_rate = reward

            power_constr_mean = torch.mean(power_constr_per_ap, dim = 0)

            sum_rate_mean = torch.mean(sum_rate, dim = 0)
            
            mu_k = mu_update(mu_k, power_constr_per_ap, eps)
            
            penalty_per_ap = power_constr_per_ap * mu_k

            total_penalty = penalty_per_ap.sum(dim=1).unsqueeze(-1) 

            cost = sum_rate.unsqueeze(-1) + total_penalty 

            cost_baseline = cost - cost.mean(dim=0, keepdim=True).detach()

            entropy = m.entropy().sum(dim=1).unsqueeze(-1)

            loss = cost_baseline * log_p_sum - 0.3 * entropy

            loss_mean = torch.mean(loss, dim = 0)
        
            loss_mean.backward() 
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=1.0)
            optimizer.step()                                                   
            optimizer.zero_grad()                                                   
                                      
            
            if batch_idx%10 == 0:
                probs_mean_batch = probs.mean(dim=0)  
                probs_values.append(probs_mean_batch.detach().numpy())
                
                power_constraint_values.append(power_constr_mean.detach().numpy())
                objective_function_values.append((-sum_rate_mean).detach().numpy())
                loss_values.append(loss_mean.squeeze(-1).detach().numpy())
                mu_k_values.append(mu_k.squeeze(-1).detach().numpy())    
                
                history_np = np.array(mu_k_values).reshape(len(mu_k_values), -1) 
                x_data = np.arange(history_np.shape[0])
                for i in range(num_links):
                    lines_mu[i].set_data(x_data, history_np[:, i])
                ax_mu.relim()
                ax_mu.autoscale_view()
                fig_mu.canvas.draw()
                fig_mu.canvas.flush_events()

    path = plot_results(
            building_id=building_id,
            b5g=b5g,
            normalized_psi=probs,
            normalized_psi_values=probs_values,
            num_layers=num_layers,
            K=K,
            batch_size=batch_size,
            epochs=epochs,
            rn=rn,
            rn1=rn1,
            eps=eps,
            mu_lr = mu_lr,
            objective_function_values=objective_function_values,
            power_constraint_values=power_constraint_values,
            loss_values=loss_values,
            mu_k_values=mu_k_values,
            num_links = num_links,
            train=True,
            power_levels_factors=power_levels_factors
        )


    file_name = path + '/objective_function_values_train_' + str(epochs) + '.pkl'
    with open(file_name, 'wb') as archivo:
        pickle.dump(objective_function_values, archivo)

    torch.save(gnn_model.state_dict(), path + '/gnn_model_weights.pth')
    
    return path

if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser(description= 'System configuration')

    parser.add_argument('--building_id', type=int, default=856)
    parser.add_argument('--b5g', type=int, default=0)
    parser.add_argument('--num_links', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=700)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eps', type=float, default=3e-3)
    parser.add_argument('--mu_lr', type=float, default=9e-4)
    parser.add_argument('--synthetic', type=int, default=0) 
    parser.add_argument('--sanity_check', type=int, default=0)
    args = parser.parse_args()
    
    print(f'building_id: {args.building_id}')
    print(f'b5g: {args.b5g}')
    print(f'num_links: {args.num_links}')
    print(f'num_layers: {args.num_layers}')
    print(f'k: {args.k}')
    print(f'epochs: {args.epochs}')
    print(f'batch_size: {args.batch_size}')
    print(f'eps: {args.eps}')
    print(f'mu_lr: {args.mu_lr}')
    print(f'synthetic: {args.synthetic}')
    
    run(building_id=args.building_id, b5g=args.b5g, num_links=args.num_links, num_layers=args.num_layers, K=args.k, batch_size=args.batch_size, epochs=args.epochs, eps=args.eps, mu_lr=args.mu_lr, synthetic=args.synthetic, sanity_check=args.sanity_check)
