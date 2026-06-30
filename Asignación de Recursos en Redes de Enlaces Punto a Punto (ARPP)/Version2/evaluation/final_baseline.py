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
        856: 20,
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
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    baseline_results_dir = os.path.abspath(os.path.join(base_dir, '..', 'results', 'generalization_test', 'baselines'))
    os.makedirs(baseline_results_dir, exist_ok=True)

    print("="*70)
    print(" INICIANDO EVALUACIÓN MULTI-BUILDING DE BASELINES (RANDOM Y FIXED)")
    print("="*70)

    random_pkl_paths = {}
    fixed_pkl_paths = {}

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
            policy='random'
        )
        
        run_baseline(
            building_id=b_id,
            b5g=b5g,
            num_links=links,
            num_channels=3,
            batch_size=batch_size,
            epochs=epochs,
            synthetic=0,
            policy='fixed'
        )
        
        banda = '2_4' if b5g == 0 else '5'
        random_path = os.path.join(base_dir, '..', 'results', f"{banda}_{b_id}", 'torch_resultss', f"baseline_random_{epochs}.pkl")
        fixed_path = os.path.join(base_dir, '..', 'results', f"{banda}_{b_id}", 'torch_resultss', f"baseline_fixed_{epochs}.pkl")
        
        if os.path.exists(random_path):
            random_pkl_paths[b_id] = random_path
        if os.path.exists(fixed_path):
            fixed_pkl_paths[b_id] = fixed_path
            
    print("\n" + "="*70)
    print(" GENERANDO BOXPLOTS DE LOS BASELINES")
    print("="*70)
        
    print("\n" + "="*70)
    print(" GENERANDO BOXPLOT COMBINADO DE LOS BASELINES")
    print("="*70)
        
    plot_data = []
    labels = []
    colors = []
    
    for b_id in buildings.keys():
        if b_id in random_pkl_paths and b_id in fixed_pkl_paths:
            try:
                with open(random_pkl_paths[b_id], 'rb') as file:
                    data = pickle.load(file)
                    flattened_data = np.concatenate([d.detach().cpu().numpy() if hasattr(d, 'detach') else np.mean(d) for d in data]).flatten()
                    plot_data.append(flattened_data)
                    labels.append(f"{b_id}\nRnd")
                    colors.append('lightgreen')
                    
                with open(fixed_pkl_paths[b_id], 'rb') as file:
                    data = pickle.load(file)
                    flattened_data = np.concatenate([d.detach().cpu().numpy() if hasattr(d, 'detach') else np.mean(d) for d in data]).flatten()
                    plot_data.append(flattened_data)
                    labels.append(f"{b_id}\nFix")
                    colors.append('lightblue')
            except Exception as e:
                print(f"Error procesando el archivo de datos para Building {b_id}: {e}")

    if len(plot_data) > 0:
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.set_ylabel('Sum Rate Capacity', fontsize=12, fontweight='bold')
        ax.set_title('Comparativa de Baselines: Random vs Fixed Allocation', fontsize=15, fontweight='bold')
        
        bplot = ax.boxplot(plot_data, patch_artist=True, tick_labels=labels)
        
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor('black')

        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        import matplotlib.patches as mpatches
        rand_patch = mpatches.Patch(color='lightgreen', label='Random Channel & Power')
        fix_patch = mpatches.Patch(color='lightblue', label='Fixed Channel, Random Power')
        ax.legend(handles=[rand_patch, fix_patch])
        
        final_plot_path = os.path.join(baseline_results_dir, 'combined_baselines_boxplot.png')
        
        plt.tight_layout()
        plt.savefig(final_plot_path, dpi=200, bbox_inches='tight')
        print(f"✓ Gráfico combinado guardado: {final_plot_path}")
        plt.close()
    else:
        print("No se encontraron métricas suficientes para graficar el comportamiento.")

if __name__ == "__main__":
    main()
