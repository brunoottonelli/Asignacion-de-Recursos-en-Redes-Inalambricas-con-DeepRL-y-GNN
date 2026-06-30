import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_comparison(building_id, b5g, num_layers, K, batch_size, epochs, eps, mu_lr, rn=100, rn1=100):
    band = ['2_4', '5']
    eps_str = f"{eps:.0e}"
    mu_lr_str = f"{mu_lr:.0e}"
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, '..', 'results')
    
    subfolder = f"{band[b5g]}_{building_id}/torch_results/n_layers{num_layers}_order{K}"
    exp_name = f"ceibal_train_temporal_{eps_str}_{mu_lr_str}_{batch_size}_{epochs}_{rn}_{rn1}"
    gnn_path = os.path.join(results_root, subfolder, exp_name, 'gnn_objective_trajectory.pkl')
    
    baseline_path = os.path.join(results_root, f"{band[b5g]}_{building_id}", 'torch_results', f"baseline_random_trajectory.pkl")
    baseline_fp_path = os.path.join(results_root, f"{band[b5g]}_{building_id}", 'torch_results', f"baseline_fixed_power_trajectory.pkl")

    data_gnn = []
    if os.path.exists(gnn_path):
        with open(gnn_path, 'rb') as f:
            data_gnn = pickle.load(f)
        print(f"✅ Cargado GNN desde: {gnn_path}")
    else:
        print(f"❌ No se encontró archivo GNN en: {gnn_path}")

    data_random = []
    if os.path.exists(baseline_path):
        with open(baseline_path, 'rb') as f:
            data_random = pickle.load(f)
        print(f"✅ Cargado Random desde: {baseline_path}")
    else:
        alt_path = os.path.join(results_root, f"{band[b5g]}_{building_id}", 'torch_resultss', f"baseline_random_trajectory.pkl")
        if os.path.exists(alt_path):
             with open(alt_path, 'rb') as f:
                data_random = pickle.load(f)
             print(f"✅ Cargado Random desde ruta alternativa: {alt_path}")
        else:
            print(f"❌ No se encontró archivo Random en: {baseline_path}")

    data_fp = []
    if os.path.exists(baseline_fp_path):
        with open(baseline_fp_path, 'rb') as f:
            data_fp = pickle.load(f)
        print(f"✅ Cargado Fixed Power desde: {baseline_fp_path}")
    else:
        alt_fp_path = os.path.join(results_root, f"{band[b5g]}_{building_id}", 'torch_resultss', f"baseline_fixed_power_trajectory.pkl")
        if os.path.exists(alt_fp_path):
             with open(alt_fp_path, 'rb') as f:
                data_fp = pickle.load(f)
             print(f"✅ Cargado Fixed Power desde ruta alternativa: {alt_fp_path}")
        else:
            print(f"❌ No se encontró archivo Fixed Power en: {baseline_fp_path}")

    plt.style.use('default')
    color_gnn = '#1ABC9C'  
    color_rand = '#9B59B6' 
    color_fp = '#E74C3C'
    
    if len(data_gnn) > 0 or len(data_random) > 0 or len(data_fp) > 0:
        plt.figure(figsize=(15, 6))
        
        if len(data_gnn) > 0:
            y_gnn = [x.mean().item() if hasattr(x, 'mean') else (np.mean(x) if isinstance(x, np.ndarray) else x) for x in data_gnn]
            plt.plot(y_gnn, label='REINFORCE (GNN)', color=color_gnn, linewidth=2.5)
            
        if len(data_random) > 0:
            y_rand = [x.mean().item() if hasattr(x, 'mean') else (np.mean(x) if isinstance(x, np.ndarray) else x) for x in data_random]
            plt.plot(y_rand, label='Política Aleatoria ', color=color_rand, linestyle='--', linewidth=2)

        if len(data_fp) > 0:
            y_fp = [x.mean().item() if hasattr(x, 'mean') else (np.mean(x) if isinstance(x, np.ndarray) else x) for x in data_fp]
            plt.plot(y_fp, label='Potencia Fija', color=color_fp, linestyle='-.', linewidth=2)

        # plt.title(f'Análisis de Convergencia del Entrenamiento - Edificio {building_id}', fontsize=14, fontweight='bold')
        plt.xlabel('Iteraciones (x10)', fontsize=20)
        plt.ylabel('Capacidad', fontsize=20)
        
        plt.legend(loc='lower right', frameon=True, edgecolor='black')
        plt.grid(True, which='both', linestyle=':', alpha=0.5)
        
        plt.gca().tick_params(direction='in', top=True, right=True)
        
        output_name = os.path.join(results_root, 'primer_problema_train_reinforce_canales.pdf')
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"\n📑 Gráfico estilo Paper guardado en: {output_name}")
        plt.show()
    else:
        print("No hay datos para graficar.")

if __name__ == '__main__':
    plot_comparison(
        building_id=856,
        b5g=0,
        num_layers=4,
        K=4,
        batch_size=32,
        epochs=200,
        eps=3e-5,
        mu_lr=9e-4
    )
