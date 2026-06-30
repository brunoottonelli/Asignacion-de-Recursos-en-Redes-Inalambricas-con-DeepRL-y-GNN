from __future__ import annotations
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from data.data_loader import load_distributions
from model.network_graph_env import NetworkGraphEnv
from model.gnn_model import GNN

def set_seed(seed: int):
    if seed is not None:
        import os
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

def compute_returns(rewards: list[float], gamma: float = 0.95, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Calcula el retorno descontado futuro (G_t) para cada step de la trayectoria."""
    T = len(rewards)
    returns = torch.zeros(T, dtype=torch.float32, device=device)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

def run_episode(env: NetworkGraphEnv, model: GNN, training: bool, seed: int | None = None, device: torch.device = torch.device("cpu"), alpha: float = 1.0):
    obs, _ = env.reset(seed=seed)
    
    obs_history = []
    act_history = []
    rewards:   list[float]        = []
    ep_rates:  list[float]        = []
    ep_ch_history: list[np.ndarray] = []
    ep_pwr_history: list[np.ndarray] = []

    done = False
    while not done:
        data = obs.to(device)
        
        if training:
            # En REINFORCE guardamos la observación para hacer forward en batch al final
            obs_history.append(data)

        with torch.no_grad():
            # No necesitamos backprop en este step individual, lo haremos en batch luego.
            # El GNN también devuelve un "value" (Critic) pero REINFORCE no lo necesita, así que lo ignoramos.
            joint_logits, _ = model(data, alpha=alpha)

        n_powers = len(env.tx_powers_dbm)
        joint_dist  = Categorical(logits=joint_logits)
        joint_acts  = joint_dist.sample()
        
        if training:
            act_history.append(joint_acts)

        action = torch.stack([joint_acts // n_powers, joint_acts % n_powers], dim=1)
        ep_ch_history.append(action[:, 0].cpu().numpy())
        ep_pwr_history.append(action[:, 1].cpu().numpy())

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rewards.append(reward)
        ep_rates.append(info.get("avg_user_rate", 0.0))

    return {
        "obs_history": obs_history,
        "act_history": act_history,
        "rewards":     rewards,
        "ep_rates":    ep_rates,
        "ep_ch_history": ep_ch_history,
        "ep_pwr_history": ep_pwr_history,
    }

def train(args: argparse.Namespace):
    """Loop principal de inicialización y entrenamiento (REINFORCE / Vanilla Policy Gradient)."""
    if args.seed is not None:
        set_seed(args.seed)
        
    device = torch.device(args.device)
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*70}")
    print(f"  [{start_dt}] Iniciando ejecución...")
    print(f"  Marco de Optimización Analítica: REINFORCE (Vanilla Policy Gradient)")
    print(f"  Dispositivo de cómputo: {device}")
    print(f"  Topología Física: Edificio {args.building_id} | Horizonte Temporal E: {args.episodes} episodios")
    print(f"  Hiperparámetros de Regularización: c_ent={args.entropy_coef} | update_episodes={args.update_episodes}")
    print(f"{'='*70}\n")

    distributions = load_distributions(building_id=args.building_id, verbose=True)

    available_channels = args.available_channels
    tx_powers_dbm      = args.tx_powers_dbm
    n_aps              = len(distributions[0].blocks[0].datos)

    env = NetworkGraphEnv(
        distributions=distributions,
        n_aps=n_aps,
        building_id=args.building_id,
        mean_arrival=args.mean_arrival,
        mean_duration=args.mean_dur,
        total_timesteps=args.timesteps,
        decision_period=args.decision_period,
        delta_mean=args.delta_mean,
        available_channels=available_channels,
        tx_powers_dbm=tx_powers_dbm,
        random_seed=args.seed,
        warmup_steps=args.warmup_steps,
        signal_noise_std=args.signal_noise_std,
        device=device,
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
    ).to(device)

    dummy_obs, _ = env.reset()
    model(dummy_obs.to(device), alpha=1.0)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\n[Baseline] Evaluando política aleatoria durante {args.update_episodes} episodios...")
    random_rewards = []
    for ep in range(args.update_episodes):
        ep_seed = args.seed + ep if args.seed is not None else None
        env.reset(seed=ep_seed)
        done = False
        ep_reward = 0.0
        while not done:
            action = env.action_space.sample()
            _, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            done = term or trunc
        random_rewards.append(ep_reward)
    
    random_baseline_value = float(np.mean(random_rewards))
    print(f"[Baseline] Retorno promedio aleatorio calculado: {random_baseline_value:.2f}\n")

    save_dir = Path(args.save_dir) / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_G_train = -float("inf")  
    metrics      = []
    episodes_list = []
    rewards_list = []
    baselines_list = []
    start_update = 0

    if args.resume:
        model_path = save_dir / "final_model_reinforce.pt"
        if not model_path.exists():
            model_path = save_dir / "best_model_reinforce.pt"
        
        if model_path.exists():
            print(f"[!] Resumiendo entrenamiento desde {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        metrics_path = save_dir / "training_metrics_reinforce.csv"
        if metrics_path.exists():
            print(f"[!] Cargando métricas desde {metrics_path}")
            df_metrics = pd.read_csv(metrics_path)
            metrics = df_metrics.to_dict('records')
            if len(metrics) > 0:
                start_update = metrics[-1]["update"]
                best_G_train = max([m["return"] for m in metrics])
                episodes_list = [m["update"] for m in metrics]
                rewards_list = [m["return"] for m in metrics]
                for i in range(len(rewards_list)):
                    ma_len = min(i + 1, 50)
                    baselines_list.append(float(np.mean(rewards_list[i + 1 - ma_len : i + 1])))
    
    plt.ion()
    fig_reward, ax_reward = plt.subplots(figsize=(10, 5))
    if hasattr(fig_reward.canvas.manager, 'set_window_title'):
        fig_reward.canvas.manager.set_window_title('Live Reward per Episode')
    
    t_global_start = time.time()
    update_episodes = args.update_episodes
    
    try:
        for update in range(start_update, args.episodes // update_episodes):
            model.train()
            alpha = 1.0 if update >= args.cl_start_update else 0.0
        
            batch_obs_history = []
            batch_act_history = []
            batch_returns = []
        
            batch_rates = 0.0
            batch_G_train = 0.0
        
            # Fase 1: Recolección de Experiencia (Rollout)
            for b in range(update_episodes):
                episode = update * update_episodes + b
                ep_seed = args.seed + episode if args.seed is not None else None
            
                traj = run_episode(env, model, training=True, seed=ep_seed, device=device, alpha=alpha)
            
                batch_obs_history.extend(traj["obs_history"])
                batch_act_history.extend(traj["act_history"])
            
                # Calcular G_t de REINFORCE
                returns_t = compute_returns(traj["rewards"], gamma=args.gamma, device=device)
                batch_returns.append(returns_t)
            
                G_train = sum(traj["rewards"])
                batch_G_train += G_train
                batch_rates += sum(traj["ep_rates"])
            
                if b == update_episodes - 1:
                    ch_arr = np.stack(traj["ep_ch_history"])
                    diag_parts = []
                    for i in range(n_aps):
                        most_ch_idx = int(np.bincount(ch_arr[:, i]).argmax())
                        most_ch = available_channels[most_ch_idx]
                        diag_parts.append(f"AP{i}: Ch{most_ch}")
                    diag_str = " | ".join(diag_parts)
                
            avg_G_train = batch_G_train / update_episodes
        
            if avg_G_train > best_G_train:
                best_G_train = avg_G_train
                torch.save(model.state_dict(), save_dir / "best_model_reinforce.pt")
        
            # Fase 2: Actualización de la Política (REINFORCE)
            all_returns = torch.cat(batch_returns).detach().to(device)
            
            # Normalización de Retornos (Baseline simple y estabilización)
            # Aunque REINFORCE original no normaliza, en la práctica es crucial para reducir varianza.
            all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-8)

            from torch_geometric.data import Batch
            batch_obs = Batch.from_data_list(batch_obs_history).to(device)
            batch_act_tensor = torch.cat(batch_act_history).to(device)
            
            # Forward pass en batch sobre todo lo recolectado para calcular los gradientes
            joint_logits, _ = model(batch_obs, alpha=alpha)
            joint_dist = Categorical(logits=joint_logits)
            
            new_log_probs_raw = joint_dist.log_prob(batch_act_tensor)
            new_entropies_raw = joint_dist.entropy()
            
            batch_size = len(batch_obs_history)
            # Sumar log_probs de todos los APs para tener la log_prob de la acción conjunta
            log_probs = new_log_probs_raw.view(batch_size, n_aps).sum(dim=1)
            entropies = new_entropies_raw.view(batch_size, n_aps).sum(dim=1)
            
            # Loss REINFORCE: - E[ G_t * log_pi(a|s) ] + penalidad_entropia
            loss_actor = -(log_probs * all_returns).mean()
            loss_entropy = -entropies.mean()
            
            total_loss = loss_actor + args.entropy_coef * loss_entropy
            
            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            # Métricas
            metrics.append({
                "update":         update + 1,
                "return":         avg_G_train,
                "G_val":          np.nan,
                "total_rate":     batch_rates / update_episodes,
                "loss_actor":     loss_actor.item(),
                "loss":           total_loss.item(),
                "entropy":        -loss_entropy.item(),
                "returns_mean":   all_returns.mean().item(),
                "grad_norm":      grad_norm.item(),
                "sec":            time.time() - t_global_start,
            })

            # Gráfico en tiempo real
            episodes_list.append(update + 1)
            rewards_list.append(avg_G_train)
        
            ma_len = min(len(rewards_list), 50)
            baselines_list.append(float(np.mean(rewards_list[-ma_len:])))

            if ax_reward is not None:
                ax_reward.clear()
                ax_reward.plot(episodes_list, rewards_list, color='#8000ff', marker='o', label='Retorno Medio (Batch)')
                ax_reward.plot(episodes_list, baselines_list, color='#ff8000', linestyle='--', label='Media Móvil (50)')
                ax_reward.axhline(random_baseline_value, color='red', linestyle=':', linewidth=2, label=f'Random Baseline ({random_baseline_value:.2f})')
                ax_reward.set_title('Convergencia del Retorno por Update (REINFORCE)')
                ax_reward.set_xlabel('Update (Batch de Episodios)')
                ax_reward.set_ylabel('Retorno Acumulado Promedio')
                ax_reward.legend()
                ax_reward.grid(True, alpha=0.3)
                plt.pause(0.001)

            print(f"Update {update+1:3d} (Ep {(update+1)*update_episodes}) | G_t(Avg): {avg_G_train:8.2f} | Rate: {batch_rates/update_episodes:8.1f} | L(a): {loss_actor.item():7.4f} | Ent: {-loss_entropy.item():.4f}")
            print(f"          | Dec. dom.: {diag_str}")

    except KeyboardInterrupt:
        print("\n[!] Entrenamiento interrumpido por el usuario (Ctrl+C).")
        print("    Guardando progreso de las métricas y modelo actual...")
    print(f"\n{'='*70}")
    duracion = time.time() - t_global_start
    print(f"  🏁 Entrenamiento Exitoso  |  Llevó: {duracion/60:.1f} m  | Mejor G_train: {best_G_train:.2f}")

    torch.save(model.state_dict(), save_dir / "final_model_reinforce.pt")
    df_metrics = pd.DataFrame(metrics)
    metrics_path = save_dir / "training_metrics_reinforce.csv"
    df_metrics.to_csv(metrics_path, index=False)
    print(f"  💾 Modelos alojados en: {save_dir}/")
    print(f"  💾 Métricas crudas en:  {metrics_path}")
    print(f"{'='*70}\n")

    try:
        final_plot_path = save_dir / "training_plot_final_reinforce.png"
        fig_reward.savefig(final_plot_path, bbox_inches='tight', dpi=300)
        print(f"  📊 Gráfica final guardada en: {final_plot_path}")
    except Exception as e:
        pass

    plt.ioff()
    plt.show(block=False)

    return model, df_metrics


def parse_args():
    p = argparse.ArgumentParser(description="NetROML – REINFORCE + GNN")
    p.add_argument("--seed",          type=int,  default=42)

    p.add_argument("--exp_name", type=str, default="ase_powers_reinforce", help="Nombre del experimento.")
    p.add_argument("--building_id",   default="990",                           type=str)
    p.add_argument("--dist_joblib",   default="data/distributions_990.joblib", type=str)
    p.add_argument("--step2_csv",     default="data/dataset_990_step2.csv",    type=str)
    p.add_argument("--save_dir",      default="outputs/synthetic_models", type=str)
    p.add_argument("--path_mac_to_building", type=str, default="data/from_mac_to_building_id.csv")
    p.add_argument("--path_datos_vecinos", type=str, default="data/datos_resto_del_anio/datos_vecinos_Abril.csv.gz")
    p.add_argument("--pe_save_load_path", type=str, default="data/matrices_atenuacion/matriz_b{building_id}.pkl")

    p.add_argument("--episodes",      type=int,   default=6000)
    p.add_argument("--timesteps",     type=int,   default=600)
    p.add_argument("--warmup_steps",  type=int,   default= 1)
    p.add_argument("--decision_period", type=int, default=1)

    p.add_argument("--available_channels", nargs='+', type=int, default=[1, 2, 3])
    p.add_argument("--tx_powers_dbm",      nargs='+', type=float, default=[5, 15, 20])
    
    p.add_argument("--mean_arrival",  type=float, default=3.0)
    p.add_argument("--mean_dur",      type=float, default=15.0)
    p.add_argument("--delta_mean",    type=int,   default=2)
    p.add_argument("--signal_noise_std", type=float, default=2.0)

    p.add_argument("--lr",            type=float, default=4e-4)
    p.add_argument("--gamma",         type=float, default=0.3)
    p.add_argument("--entropy_coef",  type=float, default=0.005)
    p.add_argument("--update_episodes", type=int, default=10, help="Número de episodios por update de REINFORCE.")
    
    p.add_argument("--hidden",        type=int,   default=128)
    p.add_argument("--num_gnn_layers", type=int,   default=2)
    p.add_argument("--num_fc_layers",  type=int,   default=3)
    p.add_argument("--device",        default="cpu", type=str)
    
    p.add_argument("--ap_feature_mode", type=str, default="ase", choices=["ase", "static_id", "none"])
    p.add_argument("--pe_dim",        type=int,   default=4)
    p.add_argument("--pe_method",     type=str,   default="laplacian", choices=["laplacian", "adjacency"])
    p.add_argument("--cl_start_update", type=int, default=0)
    
    p.add_argument("--pomdp",         action="store_true", help="Habilita modo POMDP cegando a la red del epsilon_t")
    p.add_argument("--resume",        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
