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
        # Para hacer determinista a mps (y cuda si se usa algun backend en particular)
        torch.use_deterministic_algorithms(True, warn_only=True)

def compute_gae(rewards: list[float], values: list[torch.Tensor], gamma: float = 0.95, lam: float = 0.99, device: torch.device = torch.device("cpu")) -> tuple[torch.Tensor, torch.Tensor]:
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32, device=device)
    gae = 0.0
    
    val_tensors = [v.detach().to(device) for v in values]
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    
    for t in reversed(range(T)):
        next_val = val_tensors[t+1] if t+1 < T else 0.0
        delta = rewards_t[t] + gamma * next_val - val_tensors[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        
    returns = advantages + torch.stack(val_tensors)
    return advantages, returns


def run_episode(env: NetworkGraphEnv, model: GNN, training: bool, seed: int | None = None, device: torch.device = torch.device("cpu"), alpha: float = 1.0):

    obs, _ = env.reset(seed=seed)
    
    obs_history = []
    act_history = []
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    values:    list[torch.Tensor] = []
    rewards:   list[float]        = []
    ep_rates:  list[float]        = []
    ep_ch_history: list[np.ndarray] = []
    ep_pwr_history: list[np.ndarray] = []

    done = False
    while not done:
        data = obs.to(device)
        
        if training:
            # Guardamos la observación para las épocas de PPO
            obs_history.append(data)

        if training:
            joint_logits, value = model(data, alpha=alpha)
        else:
            with torch.no_grad():
                joint_logits, value = model(data, alpha=alpha)

        n_powers = len(env.tx_powers_dbm)
        joint_dist  = Categorical(logits=joint_logits)
        joint_acts  = joint_dist.sample()

        log_prob  = joint_dist.log_prob(joint_acts).sum()
        entropy_t = joint_dist.entropy().sum()
        
        if training:
            # Guardamos la acción tomada
            act_history.append(joint_acts)

        log_probs.append(log_prob)
        entropies.append(entropy_t)
        values.append(value.squeeze())

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
        "rewards":   rewards,
        "log_probs": log_probs,
        "entropies": entropies,
        "values":    values,
        "ep_rates":  ep_rates,
        "ep_ch_history": ep_ch_history,
        "ep_pwr_history": ep_pwr_history,
    }


def train(args: argparse.Namespace):
    """Loop principal de inicialización y entrenamiento (PPO)."""
    if args.seed is not None:
        set_seed(args.seed)
        
    device = torch.device(args.device)
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*70}")
    print(f"  [{start_dt}] Iniciando ejecución...")
    print(f"  Marco de Optimización Analítica: Proximal Policy Optimization (PPO)")
    print(f"  Dispositivo de cómputo: {device}")
    print(f"  Topología Física: Edificio {args.building_id} | Horizonte Temporal E: {args.episodes} episodios")
    print(f"  Hiperparámetros de Regularización: c_vf={args.vf_coef} | c_ent={args.entropy_coef} | PPO_epochs={args.ppo_epochs} | clip={args.clip_param}")
    print(f"{'='*70}\n")

    distributions = load_distributions(
        building_id=args.building_id,
        verbose=True,
    )

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
    dummy_data = dummy_obs.to(device)
    model(dummy_data, alpha=1.0)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Evaluación inicial de la política aleatoria
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
        model_path = save_dir / "final_model_ppo.pt"
        if not model_path.exists():
            model_path = save_dir / "best_model_ppo.pt"
        
        if model_path.exists():
            print(f"[!] Resumiendo entrenamiento desde {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        metrics_path = save_dir / "training_metrics_ppo.csv"
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

    loss_actor_epoch  = 0.0
    loss_critic_epoch = 0.0
    total_loss_epoch  = 0.0
    entropy_epoch     = 0.0
    grad_norm_epoch   = 0.0

    update_episodes = args.update_episodes # Número de episodios a acumular antes de hacer backprop
    
    try:
        for update in range(start_update, args.episodes // update_episodes):
            model.train()
            
            alpha = 1.0 if update >= args.cl_start_update else 0.0
        
            batch_obs_history = []
            batch_act_history = []
            batch_advantages = []
            batch_returns = []
            batch_log_probs = []
        
            batch_rates = 0.0
            batch_G_train = 0.0
        
            for b in range(update_episodes):
                episode = update * update_episodes + b
                ep_seed = args.seed + episode if args.seed is not None else None
            
                traj = run_episode(env, model, training=True, seed=ep_seed, device=device, alpha=alpha)
            
                batch_obs_history.extend(traj["obs_history"])
                batch_act_history.extend(traj["act_history"])
                batch_log_probs.extend(traj["log_probs"])
            
                advantages, returns_t = compute_gae(traj["rewards"], traj["values"], gamma=args.gamma, lam=0.95, device=device)
                batch_advantages.append(advantages)
                batch_returns.append(returns_t)
            
                G_train = sum(traj["rewards"])
                batch_G_train += G_train
                batch_rates += sum(traj["ep_rates"])
            
                # Generar diagnóstico de decisiones más frecuentes del último episodio
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
                torch.save(model.state_dict(), save_dir / "best_model_ppo.pt")
        
            # Actualizar Modelo
            all_returns = torch.cat(batch_returns).detach().to(device)
            all_advantages = torch.cat(batch_advantages).detach().to(device)
            all_old_log_probs = torch.stack(batch_log_probs).detach().to(device)
        
            # PPO: Normalización de Advantages
            all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
        
            loss_actor_epoch  = 0.0
            loss_critic_epoch = 0.0
            total_loss_epoch  = 0.0
            entropy_epoch     = 0.0
            grad_norm_epoch   = 0.0
 
            from torch_geometric.data import Batch
            batch_obs = Batch.from_data_list(batch_obs_history).to(device)
            batch_act_tensor = torch.cat(batch_act_history).to(device)
 
            # PPO: Múltiples épocas de actualización
            for _ in range(args.ppo_epochs):
            
                joint_logits, new_values_t = model(batch_obs, alpha=alpha)
                joint_dist = Categorical(logits=joint_logits)
            
                new_log_probs_raw = joint_dist.log_prob(batch_act_tensor)
                new_entropies_raw = joint_dist.entropy()
            
                batch_size = len(batch_obs_history)
                new_log_probs_t = new_log_probs_raw.view(batch_size, n_aps).sum(dim=1)
                new_entropies_t = new_entropies_raw.view(batch_size, n_aps).sum(dim=1)
            
                # Ratio de PPO (r_t = pi_new / pi_old)
                ratio = torch.exp(new_log_probs_t - all_old_log_probs)
            
                # Clipped Surrogate Objective
                surr1 = ratio * all_advantages
                surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * all_advantages
            
                loss_actor = -torch.min(surr1, surr2).mean()
                loss_critic = F.mse_loss(new_values_t, all_returns)
                loss_entropy = -new_entropies_t.mean()
            
                total_loss = loss_actor + args.vf_coef * loss_critic + args.entropy_coef * loss_entropy
            
                optimizer.zero_grad()
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            
                loss_actor_epoch  += loss_actor.item()
                loss_critic_epoch += loss_critic.item()
                total_loss_epoch  += total_loss.item()
                entropy_epoch     += loss_entropy.item()
                grad_norm_epoch   += grad_norm.item()
            
            # Promediar métricas de las épocas
            loss_actor_epoch  /= args.ppo_epochs
            loss_critic_epoch /= args.ppo_epochs
            total_loss_epoch  /= args.ppo_epochs
            entropy_epoch     /= args.ppo_epochs
            grad_norm_epoch   /= args.ppo_epochs
 
            # Métricas
            metrics.append({
                "update":         update + 1,
                "return":         avg_G_train,
                "G_val":          np.nan,
                "total_rate":     batch_rates / update_episodes,
                "loss_actor":     loss_actor_epoch,
                "loss_critic":    loss_critic_epoch,
                "loss":           total_loss_epoch,
                "entropy":        -entropy_epoch,
                "advantage_mean": all_advantages.mean().item(),
                "advantage_std":  all_advantages.std().item() if len(all_advantages) > 1 else 0.0,
                "grad_norm":      grad_norm_epoch,
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
                ax_reward.set_title('Convergencia del Retorno por Update (PPO)')
                ax_reward.set_xlabel('Update (x5 Episodios)')
                ax_reward.set_ylabel('Retorno Acumulado Promedio')
                ax_reward.legend()
                ax_reward.grid(True, alpha=0.3)
                plt.pause(0.001)

            # Imprimir logging académico en terminal
            last_ep = (update + 1) * update_episodes
            print(f"Update {update+1:3d} (Ep {last_ep}) | G_t(Avg): {avg_G_train:8.2f} | Rate(Avg): {batch_rates/update_episodes:8.1f} | L(a): {loss_actor_epoch:7.4f} | L(c): {loss_critic_epoch:7.4f}")
            print(f"          | Dec. dom. (último ep): {diag_str}")

    except KeyboardInterrupt:
        print("\n[!] Entrenamiento interrumpido por el usuario (Ctrl+C).")
        print("    Guardando progreso de las métricas y modelo actual...")
    print(f"\n{'='*70}")
    duracion = time.time() - t_global_start
    print(f"  🏁 Entrenamiento Exitoso  |  Llevó: {duracion/60:.1f} m  | Mejor G_train: {best_G_train:.2f}")

    torch.save(model.state_dict(), save_dir / "final_model_ppo.pt")
    df_metrics = pd.DataFrame(metrics)
    metrics_path = save_dir / "training_metrics_ppo.csv"
    df_metrics.to_csv(metrics_path, index=False)
    print(f"  💾 Modelos alojados en: {save_dir}/")
    print(f"  💾 Métricas crudas en:  {metrics_path}")
    print(f"{'='*70}\n")

    try:
        plot_script = Path(__file__).parent / "plots_code" / "plot_training.py"
        if plot_script.exists():
            import subprocess
            print(f"[Análisis] Generando gráficas de convergencia científica...")
            subprocess.run([
                sys.executable, str(plot_script),
                "--csv", str(metrics_path),
                "--out", str(save_dir)
            ], check=False)
    except Exception as e:
        print(f"[Aviso] No se pudieron generar las gráficas automáticamente: {e}")

    # Guardar la gráfica en tiempo real que se generó
    try:
        final_plot_path = save_dir / "training_plot_final.png"
        fig_reward.savefig(final_plot_path, bbox_inches='tight', dpi=300)
        print(f"  📊 Gráfica final guardada en: {final_plot_path}")
    except Exception as e:
        print(f"[Aviso] No se pudo guardar la gráfica en tiempo real: {e}")

    plt.ioff()
    plt.show(block=False)

    return model, df_metrics


def parse_args():

    p = argparse.ArgumentParser(description="NetROML – PPO + GNN")
    p.add_argument("--seed",          type=int,  default=42)

    p.add_argument("--exp_name", type=str, default="ase_completo", help="Nombre del experimento (crea una carpeta dentro de save_dir).")
    p.add_argument("--building_id",   default="990",                           type=str)
    p.add_argument("--dist_joblib",   default="data/distributions_990.joblib", type=str)
    p.add_argument("--step2_csv",     default="data/dataset_990_step2.csv",    type=str)
    p.add_argument("--save_dir",      default="outputs/synthetic_models", type=str)
    p.add_argument("--path_mac_to_building", type=str, default="data/from_mac_to_building_id.csv")
    p.add_argument("--path_datos_vecinos", type=str, default="data/datos_resto_del_anio/datos_vecinos_Abril.csv.gz")
    p.add_argument("--pe_save_load_path", type=str, default="data/matrices_atenuacion/matriz_b{building_id}.pkl")

    p.add_argument("--episodes",      type=int,   default=7000)
    p.add_argument("--timesteps",     type=int,   default=600)
    p.add_argument("--warmup_steps",  type=int,   default= 1, help="Timesteps de calentamiento antes de iniciar rl.")
    p.add_argument("--decision_period", type=int, default=1)
    p.add_argument("--pomdp", action="store_true", default=True, help="Habilita modo POMDP cegando a la red del epsilon_t futuro y usando variables pasadas")

    # DECISIONES
    p.add_argument("--available_channels", nargs='+', type=int, default=[1, 2, 3], help="Lista de canales disponibles separados por espacio.")
    p.add_argument("--tx_powers_dbm",      nargs='+', type=float, default=[5, 15, 20], help="Lista de potencias disponibles en dBm separadas por espacio.")
    
    # SIMULACIÓN DE USUARIOS
    p.add_argument("--mean_arrival",  type=float, default=3.0)
    p.add_argument("--mean_dur",      type=float, default=15.0)
    p.add_argument("--delta_mean",    type=int,   default=2, help="Intervalo promedio (delta) entre arribos de ráfagas.")
    p.add_argument("--signal_noise_std", type=float, default=2.0, help="Desviación estándar del ruido Gaussiano a sumar a la señal (en dB).")

    # PARAMETROS DE APRENDIZAJE
    p.add_argument("--lr",            type=float, default=4e-4)
    p.add_argument("--gamma",         type=float, default=0.3)
    p.add_argument("--entropy_coef",  type=float, default=0.005, help="Coeficiente para incentivar exploración (bono de entropía).")
    p.add_argument("--vf_coef",       type=float, default=0.3, help="Peso del término Critic (L_critic) en la loss total A2C.")
    p.add_argument("--update_episodes", type=int, default=10,   help="Número de episodios a acumular antes de hacer backprop (Rollout Buffer).")
    p.add_argument("--ppo_epochs",    type=int,   default=4,    help="Número de pasadas de PPO por episodio.")
    p.add_argument("--clip_param",    type=float, default=0.15,  help="Epsilon para el surrogate objective (clipping).")
    
    # PARAMETROS DE LA RED
    p.add_argument("--hidden",        type=int,   default=128)
    p.add_argument("--num_gnn_layers", type=int,   default=2)
    p.add_argument("--num_fc_layers",  type=int,   default=3)
    p.add_argument("--device",        default="cpu", type=str, help="cpu or mps")
    
    # PARAMETROS DEL ASE Y CURRICULUM LEARNING
    p.add_argument("--ap_feature_mode", type=str, default="ase", choices=["ase", "static_id", "none"])
    p.add_argument("--pe_dim",        type=int,   default=4,    help="Dimensiones del Positional Encoding (0 para desactivar).")
    p.add_argument("--pe_method",     type=str,   default="laplacian", choices=["laplacian", "adjacency"])
    p.add_argument("--cl_start_update", type=int, default=0, help="Update a partir del cual se habilita alpha para AP features (Curriculum Learning).")
    
    p.add_argument("--resume",        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True, help="Continuar entrenamiento desde el último checkpoint")
    args = p.parse_args()
    return args


if __name__ == "__main__":
    train(parse_args())
