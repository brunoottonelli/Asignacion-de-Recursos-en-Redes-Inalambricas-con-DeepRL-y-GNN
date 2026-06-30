import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from baseline_train import run as run_baseline

def main():
    buildings = {
        84: 26,
        814: 24, 
        67: 22, 
        800: 15, 
        838: 15, 
        1141: 15, 
        201: 10, 
        211: 10,
        990: 5
    }

    b5g = 0
    batch_size = 64
    epochs = 1   
    baseline_type = -1 
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    baseline_results_dir = os.path.abspath(os.path.join(base_dir, '..', 'results', 'generalization_test', 'baselines'))
    os.makedirs(baseline_results_dir, exist_ok=True)

    print("="*70)
    print(" INICIANDO EVALUACIÓN MULTI-BUILDING DE BASELINES (RANDOM & FIXED_POWER)")
    print("="*70)

    random_pkl_paths = {}
    fixed_power_pkl_paths = {}

    for b_id, links in buildings.items():
        print(f"\n[{'-'*10} EVALUANDO BASELINES EN BUILDING ID: {b_id} (APs: {links}) {'-'*10}]")
        
        run_baseline(
            building_id=b_id,
            b5g=b5g,
            num_links=links,
            num_channels=3,
            batch_size=batch_size,
            epochs=epochs,
            synthetic=0,
            baseline_type=baseline_type
        )
        
        banda = '2_4' if b5g == 0 else '5'
        results_root = os.path.abspath(os.path.join(base_dir, '..', 'results'))
        random_path = os.path.join(results_root, f"{banda}_{b_id}", "torch_results", "baseline_random_trajectory.pkl")
        fixed_power_path = os.path.join(results_root, f"{banda}_{b_id}", "torch_results", "baseline_fixed_power_trajectory.pkl")
        
        if os.path.exists(random_path):
            random_pkl_paths[b_id] = random_path
        if os.path.exists(fixed_power_path):
            fixed_power_pkl_paths[b_id] = fixed_power_path
            
    print("\n" + "="*70)
    print(" GENERANDO BOXPLOTS DE LOS BASELINES")
    print("="*70)
        
    for baseline_name, paths_dict in [('Random', random_pkl_paths), ('Fixed_Power', fixed_power_pkl_paths)]:
        plot_data = []
        labels = []
        
        for b_id in buildings.keys():
            if b_id in paths_dict:
                pkl_path = paths_dict[b_id]
                try:
                    with open(pkl_path, 'rb') as file:
                        data = pickle.load(file)
                        flat_list = []
                        for d in data:
                            if hasattr(d, 'detach'):
                                flat_list.extend(d.detach().cpu().numpy().flatten())
                            elif isinstance(d, np.ndarray):
                                flat_list.extend(d.flatten())
                            elif isinstance(d, (list, tuple)):
                                flat_list.extend(d)
                            else:
                                flat_list.append(d)
                        plot_data.append(np.array(flat_list))
                        labels.append(f"Build {b_id}\n({buildings[b_id]} APs)")
                except Exception as e:
                    print(f"Error procesando el archivo de datos para {baseline_name} - Building {b_id}: {e}")

        if len(plot_data) > 0:
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.set_ylabel('Sum Rate Capacity', fontsize=12, fontweight='bold')
            ax.set_title(f'Prueba de Baselines: {baseline_name} Allocation', fontsize=15, fontweight='bold')
            
            bplot = ax.boxplot(plot_data, patch_artist=True, labels=labels)
            
            cmap = matplotlib.colormaps.get_cmap('viridis')
            colors = [cmap(i / len(plot_data)) for i in range(len(plot_data))]
            
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
                patch.set_edgecolor('black')

            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
            final_plot_path = os.path.join(baseline_results_dir, f'{baseline_name.lower()}_baseline_boxplot.png')
            
            plt.tight_layout()
            plt.savefig(final_plot_path, dpi=200, bbox_inches='tight')
            print(f"✓ Gráfico guardado para {baseline_name}: {final_plot_path}")
            plt.close()
        else:
            print(f"No se encontraron métricas suficientes para graficar el comportamiento de {baseline_name}.")

if __name__ == "__main__":
    main()
