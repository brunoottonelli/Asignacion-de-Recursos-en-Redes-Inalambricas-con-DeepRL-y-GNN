import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from test_final import evaluate as run_test

def main():
    # Edificios de validación objetivo
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

    # Arquitectura GNN 
    num_layers = 4
    K = 4
    b5g = 0
    batch_size = 32
    
    # 1. Definir los pesos del Building que queremos cargar
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    weights_path = os.path.abspath(os.path.join(
        base_dir, '..', 'results', '2_4_856', 'torch_results', 'n_layers4_order4', 
        'ceibal_train_temporal_3e-05_9e-04_32_200_100_100', 'gnn_model_weights.pth'
    ))

    if not os.path.exists(weights_path):
        print(f"ERROR: No se encontró el archivo de pesos origen en:\n {weights_path}")
        print("Por favor, asegúrate de que la ruta sea correcta.")
        return

    generalization_results_dir = os.path.abspath(os.path.join(base_dir, '..', 'results', 'temporal_test'))
    os.makedirs(generalization_results_dir, exist_ok=True)

    test_pkl_paths = {}


    for b_id, links in buildings.items():
        print(f"\n[{'-'*10} EVALUANDO GENERALIZACIÓN EN BUILDING ID: {b_id} (APs: {links}) {'-'*10}]")
    
        specific_output_path = os.path.join(generalization_results_dir, f'test_results_b{b_id}.pkl')
        
        pkl_output_path = run_test(
            weights_path=weights_path,
            output_path=specific_output_path, 
            building_id=b_id,
            b5g=b5g,
            num_links=links,
            num_layers=num_layers,
            K=K,
            batch_size=batch_size,
            synthetic=False,
        )
        
        if pkl_output_path is not None:
            test_pkl_paths[b_id] = pkl_output_path
            print(f"       ✓ Evaluación exitosa. Datos en: {pkl_output_path}")

    print("\n" + "="*70)
    print(" GENERANDO BOXPLOT FINAL DE GENERALIZACIÓN")
    print("="*70)
    
    plot_data = []
    labels = []

    for b_id, pkl_path in test_pkl_paths.items():
        try:
            with open(pkl_path, 'rb') as file:
                data = pickle.load(file)
                flattened_data = np.concatenate(data).flatten()
                plot_data.append(flattened_data)
                
                if b_id == 856:
                    labels.append(f"Build {b_id}\n({buildings[b_id]} APs)\n(ORIGEN)")
                else:
                    labels.append(f"Build {b_id}\n({buildings[b_id]} APs)")
        except Exception as e:
            print(f"Error procesando el archivo de datos para Building {b_id}: {e}")

    if len(plot_data) > 0:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_ylabel('Sum Rate Capacity', fontsize=12, fontweight='bold')
        ax.set_title('Prueba de Generalización GNN (Pesos originados en B856)', fontsize=15, fontweight='bold')
        
        bplot = ax.boxplot(plot_data, patch_artist=True, labels=labels)
        
        cmap = matplotlib.colormaps.get_cmap('viridis')
        colors = [cmap(i / len(plot_data)) for i in range(len(plot_data))]
        
        colors[0] = 'crimson'
        
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor('black')

        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        final_plot_path = os.path.join(generalization_results_dir, 'generalization_boxplot.png')
        
        plt.tight_layout()
        print(f"\n¡Prueba de Generalización Completada!")
        print(f"El gráfico de resultados se ha guardado en:\n{final_plot_path}")

        plt.close()
    else:
        print("No se encontraron métricas suficientes para graficar.")

    print("\n" + "="*70)
    print(" GENERANDO PLOTS INDIVIDUALES POR EDIFICIO")
    print("="*70)
    
    plots_dir = os.path.join(generalization_results_dir, 'individual_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    for b_id, pkl_path in test_pkl_paths.items():
        try:
            with open(pkl_path, 'rb') as file:
                data = pickle.load(file)
                data_means = [d.detach().cpu().numpy().mean() if hasattr(d, 'detach') else np.mean(d) for d in data]
                
                plt.figure(figsize=(10, 5))
                plt.plot(data_means, color='blue', alpha=0.7)
                
                plt.xlabel('Batch', fontsize=10)
                plt.ylabel('Mean Sum Rate Capacity', fontsize=10)
                plt.title(f'Rendimiento Temporal - Building {b_id} ({buildings[b_id]} APs)', fontsize=12)
                plt.grid(True, alpha=0.3)
                
                plot_filename = os.path.join(plots_dir, f'plot_b{b_id}.png')
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"       ✓ Guardado plot individual: {plot_filename}")
                
        except Exception as e:
            print(f"Error generando plot individual para Building {b_id}: {e}")

    print(f"\n¡Proceso finalizado! Los plots individuales están en: {plots_dir}")

if __name__ == "__main__":
    main()
