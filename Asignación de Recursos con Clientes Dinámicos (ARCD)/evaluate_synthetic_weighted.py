import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.data_loader import load_distributions
from model.network_graph_env import NetworkGraphEnv
from model.gnn_model import GNN

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--building_id", nargs='+', default=["990", "44", "211", "79", "405", "856", "838"], help="Lista de building IDs para evaluar")
    p.add_argument("--timesteps", type=int, default=600)
    p.add_argument("--episodes", type=int, default=50, help="Number of episodios para evaluación")
    p.add_argument("--save_dir", type=str, default="outputs/synthetic_models")
    p.add_argument("--exp_name", type=str, default="ase_completo", help="Nombre del experimento a evaluar")
    p.add_argument("--mean_arrival", type=float, default=3.0)
    p.add_argument("--base_n_aps", type=int, default=5, help="Número de APs con los que se entrenó para escalar mean_arrival")
    p.add_argument("--mean_dur", type=float, default=15.0)
    p.add_argument("--decision_period", type=int, default=1)
    p.add_argument("--delta_mean", type=int, default=2)
    p.add_argument("--warmup_steps", type=int, default=1, help="Timesteps de calentamiento antes de iniciar rl.")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num_gnn_layers", type=int, default=2)
    p.add_argument("--num_fc_layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--pe_dim", type=int, default=4)
    p.add_argument("--pe_method", type=str, default="laplacian", choices=["laplacian", "adjacency"])
    p.add_argument("--ap_feature_mode", type=str, default="ase", choices=["ase", "static_id", "none"])
    p.add_argument("--path_mac_to_building", type=str, default="data/from_mac_to_building_id.csv")
    p.add_argument("--path_datos_vecinos", type=str, default="data/datos_resto_del_anio/datos_vecinos_Abril.csv.gz")
    p.add_argument("--pe_save_load_path", type=str, default="data/matrices_atenuacion/matriz_b{building_id}.pkl")
    p.add_argument("--available_channels", nargs='+', type=int, default=[1, 2, 3], help="Lista de canales disponibles separados por espacio.")
    p.add_argument("--tx_powers_dbm",      nargs='+', type=float, default=[5,15,20], help="Lista de potencias disponibles en dBm separadas por espacio.")
    p.add_argument("--signal_noise_std", type=float, default=2.0, help="Desviación estándar del ruido Gaussiano a sumar a la señal (en dB).")
    p.add_argument("--pomdp", action="store_true", default=False, help="Habilita modo POMDP cegando a la red del epsilon_t futuro y usando variables pasadas")
    return p.parse_args()



def get_average_interference_matrix(env, num_aps, n_channels, n_powers):
    """Ejecuta 1 episodio aleatorio para acumular y promediar la matriz de interferencia W_norm"""
    W_avg = np.zeros((num_aps, num_aps))
    obs, _ = env.reset(seed=env.random_seed)
    
    count = 0
    done = False
    while not done:
        ch_idx = torch.randint(0, n_channels, (num_aps,))
        pwr_idx = torch.randint(0, n_powers, (num_aps,))
        action = torch.stack([ch_idx, pwr_idx], dim=1)
        
        edge_attr = obs.edge_attr_dict[('ap', 'interferes', 'ap')].squeeze().cpu().numpy()
        W = edge_attr.reshape((num_aps, num_aps))
        W_avg += W
        count += 1
        
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        
    return W_avg / max(1, count)

def get_optimal_coloring(W, num_channels, num_powers, cache_path=None):
    N = W.shape[0]
    print(f"Búsqueda heurística (Greedy + Local Search) para {N} APs en espacio de {(num_channels * num_powers)**N:.2e} combinaciones...")
    
    # Greedy para canales
    node_degrees = np.sum(W, axis=1)
    nodes_sorted = np.argsort(-node_degrees)
    channels = np.zeros(N, dtype=int)
    
    for u in nodes_sorted:
        channel_costs = np.zeros(num_channels)
        for v in range(N):
            if u != v:
                channel_costs[channels[v]] += W[u, v]
        channels[u] = np.argmin(channel_costs)
        
    powers = np.full(N, num_powers - 1, dtype=int)
    
    best_channels = channels.copy()
    best_powers = powers.copy()
    
    def eval_cost(ch, pw):
        cost = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if ch[i] == ch[j]:
                    cost += W[i, j] * (pw[i] + pw[j] + 1)
        penalty_low_power = np.sum((num_powers - 1) - pw) * 0.5 
        return cost + penalty_low_power
        
    best_cost = eval_cost(best_channels, best_powers)
    
    # Refinamiento
    for _ in range(5000):
        new_ch = best_channels.copy()
        new_pw = best_powers.copy()
        
        node_to_change = np.random.randint(N)
        if np.random.rand() < 0.5:
            new_ch[node_to_change] = np.random.randint(num_channels)
        else:
            new_pw[node_to_change] = np.random.randint(num_powers)
            
        new_cost = eval_cost(new_ch, new_pw)
        if new_cost < best_cost:
            best_cost = new_cost
            best_channels = new_ch
            best_powers = new_pw

    result = (torch.tensor(best_channels, dtype=torch.long), 
              torch.tensor(best_powers, dtype=torch.long), 
              best_cost)
              
    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
            
    return result

def evaluate_policy(env, model, num_aps, n_channels, n_powers, use_random=False, static_action=None):
    obs, _ = env.reset()
    total_reward = 0.0
    total_rate = 0.0
    done = False
    while not done:
        if static_action is not None:
            # Acción Estática
            action = static_action
        elif use_random:
            # Aleatorio Dinámico
            ch_idx = torch.randint(0, n_channels, (num_aps,))
            pwr_idx = torch.randint(0, n_powers, (num_aps,))
            action = torch.stack([ch_idx, pwr_idx], dim=1)
        else:
            # Política GNN
            data = obs
            with torch.no_grad():
                joint_logits, _ = model(
                    data.x_dict,
                    data.edge_index_dict,
                    data.edge_attr_dict if hasattr(data, 'edge_attr_dict') else None,
                    getattr(data, 'delta_t', None),
                    alpha=1.0
                )
                joint_acts = torch.argmax(joint_logits, dim=1)
                
            action = torch.stack([joint_acts // n_powers, joint_acts % n_powers], dim=1)
            
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        total_rate += info.get("total_rate", 0.0)
        obs = next_state
        
    return total_reward, total_rate
 
def main():
    args = parse_args()
    
    building_ids = args.building_id if isinstance(args.building_id, list) else [args.building_id]
    
    all_episodes_data = []
    all_summary_data = []
    
    for b_id in building_ids:
        print(f"\n=========================================")
        print(f"Evaluando Building ID: {b_id}")
        print(f"=========================================")
    
        print(f"Cargando distribuciones (building_id='{b_id}')...")
        distributions = load_distributions(building_id=b_id, verbose=False)
    
        available_channels = args.available_channels
        tx_powers_dbm      = args.tx_powers_dbm
        n_aps              = len(distributions[0].blocks[0].datos)
    
        print(f"Detectados {n_aps} APs.")
        
        adjusted_mean_arrival = args.mean_arrival * (n_aps / args.base_n_aps)
        print(f"Ajustando mean_arrival para densidad constante: {args.mean_arrival} -> {adjusted_mean_arrival:.2f}")
    
        env = NetworkGraphEnv(
            distributions=distributions,
            n_aps=n_aps,
            building_id=b_id,
            mean_arrival=adjusted_mean_arrival,
            mean_duration=args.mean_dur,
            total_timesteps=args.timesteps,
            decision_period=args.decision_period,
            delta_mean=args.delta_mean,
            available_channels=available_channels,
            tx_powers_dbm=tx_powers_dbm,
            random_seed=args.seed,
            warmup_steps=args.warmup_steps,
            signal_noise_std=args.signal_noise_std,
            pe_dim=args.pe_dim,
            pe_method=args.pe_method,
            ap_feature_mode=args.ap_feature_mode,
            path_mac_to_building=args.path_mac_to_building,
            path_datos_vecinos=args.path_datos_vecinos,
            pe_save_load_path=args.pe_save_load_path,
            pomdp=args.pomdp,
        )
    
 
    
        model = GNN(
            hidden_channels=args.hidden,
            num_aps=n_aps,
            out_channels_ch=len(available_channels),
            out_channels_pwr=len(tx_powers_dbm),
            num_gnn_layers=args.num_gnn_layers,
            num_fc_layers=args.num_fc_layers,
            pe_dim=args.pe_dim,
            ap_feature_mode=args.ap_feature_mode,
        )
    
        exp_dir = os.path.join(args.save_dir, args.exp_name)
        model_path = os.path.join(exp_dir, "best_model_ppo.pt")
    
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model.eval()
            print("Modelo cargado exitosamente.")
        except Exception as e:
            print(f"Error cargando el modelo: {e}")
            print("Continuando con pesos no inicializados (aleatorios) para el GNN solo como prueba.")
 
        # Baseline Estático
        cache_dir = os.path.join(exp_dir, "cache_coloring")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"coloring_b{b_id}_{n_aps}aps.pkl")
        
        print("\nConstruyendo matriz de interferencia AP-AP...")
        env._adm.random_seed = args.seed
        env.reset(seed=args.seed)
        W = get_average_interference_matrix(env, n_aps, len(available_channels), len(tx_powers_dbm))
        
        optimal_channels, optimal_powers, optimal_cost = get_optimal_coloring(
            W, num_channels=len(available_channels), num_powers=len(tx_powers_dbm), cache_path=cache_file
        )
        print(f"Coloración estática encontrada - Interferencia: {optimal_cost:.4f}")
        
        static_action = torch.stack([optimal_channels, optimal_powers], dim=1).to(env.device)
 
        scores_random_rate = []
        scores_gnn_rate = []
        scores_color_rate = []
    
        print(f"\nEvaluando baselines durante {args.episodes} episodios...")
        for ep in range(args.episodes):
            # Offset de semillas
            ep_seed = args.seed + 10000 + ep
            
            # Graph Coloring
            random.seed(ep_seed)
            np.random.seed(ep_seed)
            torch.manual_seed(ep_seed)
            env._adm.random_seed = ep_seed
            env.reset(seed=ep_seed)
            _, color_rate = evaluate_policy(env, model, n_aps, len(available_channels), len(tx_powers_dbm), static_action=static_action)
            scores_color_rate.append(color_rate)
        
            # Random
            random.seed(ep_seed)
            np.random.seed(ep_seed)
            torch.manual_seed(ep_seed)
            env._adm.random_seed = ep_seed
            env.reset(seed=ep_seed)
            _, rand_rate = evaluate_policy(env, model, n_aps, len(available_channels), len(tx_powers_dbm), use_random=True)
            scores_random_rate.append(rand_rate)
        
            # GNN
            random.seed(ep_seed)
            np.random.seed(ep_seed)
            torch.manual_seed(ep_seed)
            env._adm.random_seed = ep_seed
            env.reset(seed=ep_seed)
            _, gnn_rate = evaluate_policy(env, model, n_aps, len(available_channels), len(tx_powers_dbm), use_random=False)
            scores_gnn_rate.append(gnn_rate)
            
            all_episodes_data.append({
                "building_id": b_id,
                "n_aps": n_aps,
                "episode": ep + 1,
                "color_rate": float(color_rate),
                "random_rate": float(rand_rate),
                "gnn_rate": float(gnn_rate)
            })
        
            print(f"Episodio {ep+1:2d}/{args.episodes} -> Color: {color_rate:9.2f} | Random: {rand_rate:9.2f} | GNN: {gnn_rate:9.2f}")

        print("\n" + "=" * 60)
        print("                      RESULTADOS                      ")
        print("=" * 60)
        print(f"Promedio Graph Coloring : {np.mean(scores_color_rate):8.2f} ± {np.std(scores_color_rate):8.2f}")
        print(f"Promedio Random         : {np.mean(scores_random_rate):8.2f} ± {np.std(scores_random_rate):8.2f}")
        print(f"Promedio GNN (PPO)      : {np.mean(scores_gnn_rate):8.2f} ± {np.std(scores_gnn_rate):8.2f}")
        print("=" * 60)
        
        all_summary_data.append({
            "building_id": b_id,
            "n_aps": n_aps,
            "color_mean": float(np.mean(scores_color_rate)),
            "color_std": float(np.std(scores_color_rate)),
            "random_mean": float(np.mean(scores_random_rate)),
            "random_std": float(np.std(scores_random_rate)),
            "gnn_mean": float(np.mean(scores_gnn_rate)),
            "gnn_std": float(np.std(scores_gnn_rate))
        })

        # Generación de Boxplot
        eval_dir = os.path.join(exp_dir, "evals/ajustados")
        os.makedirs(eval_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
    
        box = plt.boxplot([scores_color_rate, scores_random_rate, scores_gnn_rate], patch_artist=True, 
                          tick_labels=['Graph Coloring\n(Static)', 'Random\n(Dynamic)', 'GNN\n(PPO)'],
                          medianprops=dict(color='black', linewidth=2),
                          whiskerprops=dict(linewidth=1.5),
                          capprops=dict(linewidth=1.5),
                          flierprops=dict(marker='o', alpha=0.3, markersize=4))
    
        colors = ['#b3ffb3', '#ff9999', '#99ccff']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.ylabel('Total Sum Rate')
        plt.title(f'Baseline Comparison ({args.episodes} episodios)\nExp: {args.exp_name} | Edificio: {b_id}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
        plot_path = os.path.join(eval_dir, f"comparison_boxplot_{args.exp_name}_b{b_id}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nBoxplot absoluto guardado exitosamente en '{plot_path}'.")

        # Generación de Boxplot Relativo (vs Random)
        diff_gnn = [(g - r) / r * 100 for g, r in zip(scores_gnn_rate, scores_random_rate)]
        diff_color = [(c - r) / r * 100 for c, r in zip(scores_color_rate, scores_random_rate)]

        plt.figure(figsize=(6, 6))
        box2 = plt.boxplot([diff_color, diff_gnn], patch_artist=True, 
                          tick_labels=['Graph Coloring\n(vs Random)', 'GNN (PPO)\n(vs Random)'],
                          medianprops=dict(color='black', linewidth=2),
                          whiskerprops=dict(linewidth=1.5),
                          capprops=dict(linewidth=1.5),
                          flierprops=dict(marker='o', alpha=0.3, markersize=4))
    
        colors2 = ['#b3ffb3', '#99ccff']
        for patch, color in zip(box2['boxes'], colors2):
            patch.set_facecolor(color)
        
        plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Baseline (Random)')
        plt.ylabel('Mejora Relativa (%)')
        plt.title(f'Desempeño Relativo vs Aleatorio ({args.episodes} episodios)\nExp: {args.exp_name} | Edificio: {b_id}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
    
        plot_path2 = os.path.join(eval_dir, f"comparison_relative_boxplot_{args.exp_name}_b{b_id}.png")
        plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
        print(f"Boxplot relativo guardado exitosamente en '{plot_path2}'.")

        plt.close('all')

    if len(all_episodes_data) > 0:
        exp_dir = os.path.join(args.save_dir, args.exp_name)
        eval_dir = os.path.join(exp_dir, "evals/ajustados")
        os.makedirs(eval_dir, exist_ok=True)
        
        df_ep = pd.DataFrame(all_episodes_data)
        df_ep.to_csv(os.path.join(eval_dir, "episodes_results.csv"), index=False)
        
        df_sum = pd.DataFrame(all_summary_data)
        df_sum.to_csv(os.path.join(eval_dir, "summary_results.csv"), index=False)
        print(f"\nResultados guardados en CSV dentro de {eval_dir}/")

if __name__ == "__main__":
    main()
