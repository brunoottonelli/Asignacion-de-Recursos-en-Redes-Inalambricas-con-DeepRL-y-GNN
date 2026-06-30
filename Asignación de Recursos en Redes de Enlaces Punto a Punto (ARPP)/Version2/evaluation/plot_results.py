import numpy as np
import matplotlib.pyplot as plt
import os
import torch


def plot_results(building_id, b5g, normalized_psi, normalized_psi_values=[], power_values=[], 
                num_layers=5, K=3, batch_size=64, epochs=100, rn=100, rn1=100, eps=5e-5, 
                mu_lr=1e-4, objective_function_values=[], power_constraint_values=[], 
                loss_values=[], mu_k_values=[], baseline=0, mark=0, train=True, num_links=None,
                num_power_levels=None, power_levels=None, power_levels_factors=None):
    """
    Function that plots training results with correct interpretation of probabilities per link.
    
    Args:
        normalized_psi_values: List of arrays with shape [num_links, num_actions]
                              where num_actions = num_channels + 1 (including "no TX")
    """

    band = ['2_4', '5']
    eps_str = str(f"{eps:.0e}")
    mu_lr_str = str(f"{mu_lr:.0e}")
    batch_size_str = str(batch_size)

    # Agrega esto:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_ROOT = os.path.join(BASE_DIR, '..', 'results')  

    subfolder = f"{band[b5g]}_{building_id}/torch_results/n_layers{num_layers}_order{K}"
    if train:
        if mark:
            exp_name = f"mark_{eps_str}_{mu_lr_str}_{batch_size}_{epochs}_{rn}_{rn1}"
        else:
            exp_name = f"ceibal_train_temporal_{eps_str}_{mu_lr_str}_{batch_size}_{epochs}_{rn}_{rn1}"
        if baseline == 0:
            result_path = os.path.join(RESULTS_ROOT, subfolder, exp_name)
        else:
            result_path = os.path.join(RESULTS_ROOT, subfolder, exp_name + f"_baseline{baseline}")
    else:
        exp_name = f"ceibal_val_{eps_str}_{mu_lr_str}_{batch_size}_{epochs}_{rn}_{rn1}"
        result_path = os.path.join(RESULTS_ROOT, subfolder, exp_name)
    path = result_path 

    if not os.path.exists(path):
        os.makedirs(path)

    # =========================================================================
    # Métricas
    # =========================================================================
    
    if len(objective_function_values) > 0:
        plt.figure(figsize=(16, 9))
        plt.title('Función Objetivo (Sum Rate)', fontsize=16, fontweight='bold')
        plt.xlabel('Iteraciones (x10)', fontsize=14)
        plt.ylabel('Sum Rate', fontsize=14)
        plt.plot(objective_function_values, linewidth=2, color='blue')
        plt.grid(alpha=0.3)
        image_name = f'objective_function_{eps_str}_{mu_lr_str}_{batch_size_str}.png'
        plt.savefig(os.path.join(path, image_name), dpi=150, bbox_inches='tight')
        plt.close()

    if len(power_constraint_values) > 0:
        plt.figure(figsize=(16, 9))
        plt.title('Restricción de Potencia', fontsize=16, fontweight='bold')
        plt.xlabel('Iteraciones (x10)', fontsize=14)
        plt.ylabel('Violación de Restricción', fontsize=14)
        plt.plot(power_constraint_values, linewidth=2, color='red')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Objetivo (≤0)')
        # plt.ylim(-2, 1)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        image_name = f'power_constraint_{eps_str}_{mu_lr_str}_{batch_size_str}.png'
        plt.savefig(os.path.join(path, image_name), dpi=150, bbox_inches='tight')
        plt.close()

    if len(loss_values) > 0:
        plt.figure(figsize=(16, 9))
        plt.title('Loss', fontsize=16, fontweight='bold')
        plt.xlabel('Iteraciones (x10)', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.plot(loss_values, linewidth=2, color='green')
        plt.grid(alpha=0.3)
        image_name = f'loss_{eps_str}_{mu_lr_str}_{batch_size_str}.png'
        plt.savefig(os.path.join(path, image_name), dpi=150, bbox_inches='tight')
        plt.close()

    if len(mu_k_values) > 0:
        plt.figure(figsize=(16, 9))
        plt.title('Multiplicadores de Lagrange (μ_k)', fontsize=16, fontweight='bold')
        plt.xlabel('Iteraciones (x10)', fontsize=14)
        plt.ylabel('μ_k', fontsize=14)
        
        mu_k_array = np.array(mu_k_values)
        if mu_k_array.ndim == 1:
            plt.plot(mu_k_array, linewidth=2, label='Todos los links')
        else:
            for link_idx in range(mu_k_array.shape[1]):
                plt.plot(mu_k_array[:, link_idx], linewidth=2, label=f'Link {link_idx}')
        
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        image_name = f'mu_k_{eps_str}_{mu_lr_str}_{batch_size_str}.png'
        plt.savefig(os.path.join(path, image_name), dpi=150, bbox_inches='tight')
        plt.close()


    
    if len(normalized_psi_values) > 0:
        normalized_psi_array = np.array(normalized_psi_values)  
        
        num_links_data = normalized_psi_array.shape[1]
        num_actions = normalized_psi_array.shape[2]
        
        if num_links is None:
            num_links = num_links_data

        print(f"\nShape de probabilidades: {normalized_psi_array.shape}")
        print(f"  - Iteraciones guardadas: {normalized_psi_array.shape[0]}")
        print(f"  - Número de links: {num_links}")
        print(f"  - Número de acciones: {num_actions}")

        fig, axes = plt.subplots(num_links, 1, figsize=(16, 5*num_links))
        if num_links == 1:
            axes = [axes]
        
        for link_idx in range(num_links):
            ax = axes[link_idx]
            
            for action_idx in range(num_actions):
                if action_idx == 0:
                    label = 'No TX'
                    color = 'black'
                    linestyle = '--'
                    linewidth = 2
                else:
                    if power_levels_factors is not None:
                        num_power_levels_inner = len(power_levels_factors)
                        val_action = action_idx - 1
                        channel = val_action // num_power_levels_inner
                        power_idx = val_action % num_power_levels_inner
                        factor = power_levels_factors[power_idx]
                        
                        label = f'Ch {channel} ({factor*100:.0f}%)'
                        color = f'C{channel}'
                        linestyle = '-' if factor == 1.0 else ':'
                    else:
                        channel = action_idx - 1
                        label = f'Canal {channel}'
                        color = f'C{channel}'
                        linestyle = '-'
                    
                    linewidth = 2
                
                ax.plot(normalized_psi_array[:, link_idx, action_idx], 
                       label=label, color=color, linestyle=linestyle, linewidth=linewidth)
            
            ax.set_title(f'Link {link_idx}', fontsize=13, fontweight='bold')
            ax.set_xlabel('Iteraciones (x10)', fontsize=11)
            ax.set_ylabel('Probabilidad', fontsize=11)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'all_links_policies.png'), dpi=150, bbox_inches='tight')
        plt.close()


        
        print("\n" + "="*70)
        print("ANÁLISIS DE POLÍTICA FINAL")
        print("="*70)
        
        final_probs = normalized_psi_array[-1]
        
        for link_idx in range(num_links):
            probs = final_probs[link_idx, :]
            best_action = np.argmax(probs)
            
            if best_action == 0:
                action_name = "No TX"
            else:
                if power_levels_factors is not None:
                    num_power_levels_inner = len(power_levels_factors)
                    val_action = best_action - 1
                    channel = val_action // num_power_levels_inner
                    factor = power_levels_factors[val_action % num_power_levels_inner]
                    action_name = f"Ch {channel} ({factor*100:.0f}%)"
                else:
                    action_name = f"Canal {best_action - 1}"
            
            print(f"\nLink {link_idx}:")
            print(f"  Acción preferida: {action_name} (p={probs[best_action]:.4f})")
            print(f"  Distribución completa:")
            
            for action_idx in range(num_actions):
                if action_idx == 0:
                    name = "No TX   "
                else:
                    if power_levels_factors is not None:
                        num_power_levels_inner = len(power_levels_factors)
                        val_action = action_idx - 1
                        channel = val_action // num_power_levels_inner
                        factor = power_levels_factors[val_action % num_power_levels_inner]
                        name = f"Ch {channel} ({factor*100:.0f}%)"
                    else:
                        name = f"Canal {action_idx-1}"
                
                bar = '█' * int(probs[action_idx] * 50)
                line = f"    {name:12s}: {probs[action_idx]:.4f} {bar}"
                print(line)
        
        print("\n" + "-"*70)
        print("ANÁLISIS DE CONFLICTOS DE CANAL")
        print("-"*70)
        
        if power_levels_factors is not None:
            num_physical_channels = (num_actions - 1) // len(power_levels_factors)
        else:
            num_physical_channels = num_actions - 1

        for channel in range(num_physical_channels):
            links_using_ch = []
            
            for link_idx in range(num_links):
                prob_usage = 0.0
                
                if power_levels_factors is not None:
                    num_power_levels_inner = len(power_levels_factors)
                    start_action = 1 + channel * num_power_levels_inner
                    end_action = start_action + num_power_levels_inner
                    
                    prob_usage = np.sum(final_probs[link_idx, start_action:end_action])
                else:
                    action_idx = channel + 1
                    prob_usage = final_probs[link_idx, action_idx]

                if prob_usage > 0.3:  
                    links_using_ch.append((link_idx, prob_usage))
            
            if len(links_using_ch) > 1:
                msg = f"  ⚠ Canal {channel}: Usado por {len(links_using_ch)} links (INTERFERENCIA)"
                for link_id, prob in links_using_ch:
                    msg += f"\n      Link {link_id} (p={prob:.3f})"
            elif len(links_using_ch) == 1:
                link_id, prob = links_using_ch[0]
                msg = f"  ✓ Canal {channel}: Usado solo por Link {link_id} (p={prob:.3f})"
            else:
                msg = f"  - Canal {channel}: No usado"
            
            print(msg)
        
        print("\n" + "="*70)

    return path