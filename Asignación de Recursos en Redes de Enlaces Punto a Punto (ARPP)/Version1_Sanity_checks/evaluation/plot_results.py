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

    # Se eliminan los plots de objective, power_constraint, loss y mu_k para que en results quede solo all_links.

    if len(normalized_psi_values) > 0:
        normalized_psi_array = np.array(normalized_psi_values)  # [iterations, num_links, num_actions]
        
        num_links_data = normalized_psi_array.shape[1]
        num_actions = normalized_psi_array.shape[2]
        
        # Use provided num_links if available, else derive from data
        if num_links is None:
            num_links = num_links_data

        print(f"\nShape de probabilidades: {normalized_psi_array.shape}")
        print(f"  - Iteraciones guardadas: {normalized_psi_array.shape[0]}")
        print(f"  - Número de enlaces: {num_links}")
        print(f"  - Número de acciones: {num_actions}")

        # =====================================================================
        # PLOT 2: Todas las políticas juntas 
        # =====================================================================
        
        fig, axes = plt.subplots(1, num_links, figsize=(6*num_links, 5))
        if num_links == 1:
            axes = [axes]
        
        for link_idx in range(num_links):
            ax = axes[link_idx]
            
            for action_idx in range(num_actions):
                if action_idx == 0:
                    label = 'No transmitir'
                    color = 'black'
                    linestyle = '--'
                    linewidth = 2
                else:
                    if power_levels_factors is not None:
                        num_power_levels_inner = len(power_levels_factors)
                        # Restamos 1 porque 0 es NoTX
                        val_action = action_idx - 1
                        channel = val_action // num_power_levels_inner
                        power_idx = val_action % num_power_levels_inner
                        factor = power_levels_factors[power_idx]
                        
                        label = f'Canal {channel}'
                        # Color base por canal, estilo por potencia
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
            
            # ax.set_title(f'Enlace {link_idx}', fontsize=13, fontweight='bold')
            ax.set_xlabel('Iteraciones (x10)', fontsize=11)
            ax.set_ylabel('Probabilidad', fontsize=11)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'interferencia_total.pdf'), dpi=150, bbox_inches='tight')
        plt.close()        
        
        # # =====================================================================
        # # PLOT 3: Mapa espacial de asignación de canales de la red ad-hoc (Estilo Académico de Tesis)
        # # =====================================================================
        # # Fondo blanco puro y estilo minimalista científico
        # fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
        # ax.set_facecolor('white')
        
        # # Posiciones de los Transmisores (y=0) y Receptores (y=1)
        # d_horizontal = 4.0
        # h_vertical = 1.5 # Mayor separación vertical para claridad académica
        # transmitters = np.array([[0, 0], [d_horizontal, 0], [2*d_horizontal, 0]])
        # receivers = np.array([[0, h_vertical], [d_horizontal, h_vertical], [2*d_horizontal, h_vertical]])
        
        # # Nodos representados como círculos perfectos minimalistas (Gris elegante y contorno sobrio)
        # ax.scatter(transmitters[:, 0], transmitters[:, 1], 
        #            color='#F2F4F7', edgecolor='#4A5568', linewidth=1.5, 
        #            s=600, zorder=5, label='Transmisores (Tx)')
        
        # ax.scatter(receivers[:, 0], receivers[:, 1], 
        #            color='#FFFFFF', edgecolor='#4A5568', linewidth=1.5, 
        #            s=600, zorder=5, label='Receptores (Rx)')
        
        # # Etiquetas internas del nodo en una tipografía limpia y formal
        # for idx in range(num_links):
        #     ax.text(transmitters[idx, 0], transmitters[idx, 1], f'$Tx_{idx}$', 
        #             color='#2D3748', fontsize=11, ha='center', va='center', fontweight='normal')
        #     ax.text(receivers[idx, 0], receivers[idx, 1], f'$Rx_{idx}$', 
        #             color='#2D3748', fontsize=11, ha='center', va='center', fontweight='normal')
        
        # # Obtener la asignación final y sus probabilidades
        # final_probs = normalized_psi_array[-1] # Shape [num_links, num_actions]
        
        # # Paleta académica específica del usuario
        # # Amatista ('#9B59B6'), Turquesa/Teal ('#1ABC9C'), Ámbar ('#F39C12')
        # colors_map = {0: '#A0AEC0', 1: '#9B59B6', 2: '#1ABC9C', 3: '#F39C12'} 
        # labels_map = {0: 'No TX', 1: 'Canal 0', 2: 'Canal 1', 3: 'Canal 2'}
        
        # assigned_channels = []
        # # En una red ad-hoc, el Link i conecta Tx_i con Rx_i
        # for idx in range(num_links):
        #     probs = final_probs[idx, :]
        #     best_action = np.argmax(probs)
        #     prob_best = probs[best_action]
        #     assigned_channels.append(best_action)
            
        #     # Enlace directo
        #     color = colors_map.get(best_action, '#000000')
        #     linestyle = '--' if best_action == 0 else '-'
        #     # Líneas más finas y estilizadas tipo publicación/tesis
        #     linewidth = 1.5 + 3.0 * prob_best 
            
        #     ax.plot([transmitters[idx, 0], receivers[idx, 0]], 
        #             [transmitters[idx, 1], receivers[idx, 1]], 
        #             color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.9, zorder=2,
        #             label=f'Enlace {idx}: {labels_map.get(best_action, f"Canal {best_action-1}")} ($p={prob_best:.2f}$)')
            
        # # Dibujar líneas punteadas finas de interferencia (física) Tx-Rx cruzada
        # # Usamos un color gris-rojizo suave y fino (#E53E3E)
        # interf_color = '#E53E3E'
        # interf_lines = [
        #     (transmitters[1], receivers[0]),
        #     (transmitters[1], receivers[2]),
        #     (transmitters[0], receivers[1]),
        #     (transmitters[2], receivers[1])
        # ]
        
        # for idx, (p1, p2) in enumerate(interf_lines):
        #     lbl = 'Interferencia física' if idx == 0 else None
        #     ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
        #             color=interf_color, linestyle=':', linewidth=1.2, alpha=0.5, zorder=1, label=lbl)
        
        # # Título y etiquetas de ejes estilizados para LaTeX/Tesis
        # ax.set_title('Topología de la Red y Asignación de Canales Optimizada', 
        #              fontsize=13, fontweight='bold', color='#1A202C', pad=15)
        
        # ax.set_xlim(-1.5, 2*d_horizontal + 1.5)
        # ax.set_ylim(-0.5, h_vertical + 0.5)
        # ax.axis('off')
        
        # # Leyenda de tesis con formato discreto y elegante en la base
        # handles, labels = ax.get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # ax.legend(by_label.values(), by_label.keys(), loc='upper center', 
        #           bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=True, facecolor='white', 
        #           edgecolor='#E2E8F0', fontsize=9, shadow=False)
        
        # plt.savefig(os.path.join(path, 'spatial_network_allocation.png'), dpi=300, bbox_inches='tight')
        # plt.close()


        # =====================================================================
        # ANÁLISIS FINAL (solo en consola)
        # =====================================================================
        
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
            
            # Imprimir en consola
            print(f"\nEnlace {link_idx}:")
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