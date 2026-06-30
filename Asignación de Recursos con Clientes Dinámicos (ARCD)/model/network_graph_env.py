from __future__ import annotations
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from simulation.arrival_departure_model import ArrivalDepartureModel
from simulation.wifi_physics import (
    asignaciones_AP, calcular_sinr, calcular_rate,
    actualizar_avg_rate, UMBRAL_5G, UMBRAL_CONEXION, UMBRAL_HANDOVER, SIGMA_DBM,STICKY_FULL,
    STICKY_STD, STICKY_LITE, StickyMode,
)
from simulation.graph_builder import construir_grafo_timestep
from utils.positional_encoding import obtener_matriz_atenuacion_historica, calcular_positional_encoding



class NetworkGraphEnv(gym.Env):

    metadata = {"render.modes": []}

    def __init__(
        self,
        distributions,
        n_aps: int,
        building_id,
        mean_arrival: float = 2.0,
        mean_duration: float = 10.0,
        total_timesteps: int = 100,
        decision_period: int = 1,
        delta_mean: int = 5,
        available_channels: list[int] | None = None,
        tx_powers_dbm: list[float] | None = None,
        sigma_dbm: float = SIGMA_DBM,
        umbral_5g: float = UMBRAL_5G,
        umbral_conexion: float = UMBRAL_CONEXION,
        umbral_handover: float = UMBRAL_HANDOVER,
        sticky_mode: StickyMode = STICKY_STD,
        sticky: bool | None = None,
        random_seed: int | None = None,
        device: torch.device | str = "cpu",
        warmup_steps: int = 50,
        signal_noise_std: float = 0.0,
        ap_feature_mode: str = 'ase',
        pe_dim: int = 4,
        pe_method: str = 'laplacian',
        path_mac_to_building: str = "data/from_mac_to_building_id.csv",
        path_datos_vecinos: str = "data/datos_resto_del_anio/datos_vecinos_Abril.csv.gz",
        pe_save_load_path: str = "data/matrices_atenuacion/matriz_b{building_id}.pkl",
        pomdp: bool = False,
    ):
        super().__init__()
        self.pomdp = pomdp

        if sticky is not None:
            sticky_mode = STICKY_STD if sticky else STICKY_LITE

        self.distributions   = distributions
        self.n_aps           = n_aps
        self.mean_arrival    = mean_arrival
        self.mean_duration   = mean_duration
        self.total_timesteps = total_timesteps
        self.decision_period = max(int(decision_period), 1)
        self.delta_mean      = delta_mean
        self.sigma_dbm       = sigma_dbm
        self.umbral_5g       = umbral_5g
        self.umbral_conexion = umbral_conexion
        self.umbral_handover = umbral_handover
        self.sticky_mode     = sticky_mode
        self.random_seed     = random_seed
        self.device          = torch.device(device)
        self.warmup_steps    = warmup_steps
        self.signal_noise_std = signal_noise_std

        # Positional Encoding
        self.building_id = building_id
        self.pe_dim = pe_dim
        self.ap_feature_mode = ap_feature_mode
        
        if self.ap_feature_mode == 'ase' and self.pe_dim > 0:
            # Cargar matriz histórica
            matriz_base, _ = obtener_matriz_atenuacion_historica(
                building_id=self.building_id,
                path_mac_to_building=path_mac_to_building,
                path_datos_vecinos=path_datos_vecinos,
                save_load_path=pe_save_load_path,
                load_if_exists=True
            )
            # Calcular PE
            pe_numpy = calcular_positional_encoding(matriz_base, d_dimensiones=self.pe_dim, method=pe_method)
            self.ap_pos_encoding = torch.tensor(pe_numpy, dtype=torch.float32, device=self.device)
        else:
            self.ap_pos_encoding = None


        self._dist_tensors = []
        for dist in self.distributions:
            blocks = []
            for b in dist.blocks:
                vals = torch.tensor(
                    b.datos.iloc[:, 1:].values.astype(np.float32),
                    dtype=torch.float32,
                    device=self.device
                )
                vals = torch.nan_to_num(vals, nan=float('nan'), posinf=float('nan'), neginf=float('nan'))
                blocks.append(vals)
            if blocks:
                self._dist_tensors.append(torch.stack(blocks))
            else:
                self._dist_tensors.append(torch.empty((0, self.n_aps, 2), dtype=torch.float32, device=self.device))

        self.available_channels = torch.tensor(
            available_channels or [1, 6, 11],
            dtype=torch.long, device=self.device,
        )
        self.tx_powers_dbm = torch.tensor(
            tx_powers_dbm or [1.0],
            dtype=torch.float32, device=self.device,
        )
        self.n_channels = len(self.available_channels)
        self.n_powers   = len(self.tx_powers_dbm)

        self.action_space = spaces.MultiDiscrete(
            [self.n_channels, self.n_powers] * self.n_aps
        )
        self.observation_space = spaces.Discrete(1) 

        self._adm = ArrivalDepartureModel(
            distribuciones=self.distributions,
            mean_arrival=self.mean_arrival,
            mean_duration=self.mean_duration,
            total_timesteps=self.total_timesteps,
            random_seed=self.random_seed,
            delta_mean=self.delta_mean,
        )
        self.current_step     = 0
        self._delta_t: int    = 0 
        self._eventos         = []
        self._grilla_gan:     torch.Tensor | None = None
        self._grilla_RSSI:    torch.Tensor | None = None
        self._asignaciones:   torch.Tensor | None = None
        self._avg_rates:      torch.Tensor | None = None
        self._estado_sticky                        = None
        self._decisiones_APs: torch.Tensor | None = None
        self._n_rate_obs      = 0

    def reset(self, seed: int | None = None, options=None):

        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        rng = np.random.default_rng(seed)

        self._adm.random_seed = seed
        self._eventos, self._deltas = self._adm.simulate_all_events()

        if len(self._eventos) == 0:
            self._grilla_gan = torch.full(
                (0, self.total_timesteps, self.n_aps, 2),
                float('nan'), dtype=torch.float32, device=self.device,
            )
        else:
            self._grilla_gan = torch.full(
                (len(self._eventos), self.total_timesteps, self.n_aps, 2),
                float('nan'), dtype=torch.float32, device=self.device,
            )
            for fila, ev in enumerate(self._eventos):
                dist_tensor = self._dist_tensors[ev.distribution_idx]
                n_blocks = dist_tensor.shape[0]
                if n_blocks > 0:
                    # Comportamiento estático por defecto
                    base_idx = rng.integers(0, n_blocks)
                    base_block = dist_tensor[base_idx].clone()
                    trayectoria = base_block.unsqueeze(0).expand(ev.duration, -1, -1)
                    self._grilla_gan[fila, ev.arrival_time:ev.departure_time] = trayectoria

        if self.signal_noise_std > 0.0:
            noise = torch.randn_like(self._grilla_gan) * self.signal_noise_std
            self._grilla_gan = self._grilla_gan + noise

        ch_idx  = torch.randint(0, self.n_channels, (self.n_aps,), device=self.device)
        pwr_idx = torch.randint(0, self.n_powers,   (self.n_aps,), device=self.device)
        self._decisiones_APs = torch.stack([ch_idx, pwr_idx], dim=1)

        # RSSI y Asociación inicial
        self._actualizar_RSSI(0, 1)
        n_cli = self._grilla_RSSI.shape[0]
        self._asignaciones = torch.full(
            (n_cli, self.total_timesteps, 3), float('nan'), device=self.device
        )
        
        # Asociación inicial
        asig_t0, self._estado_sticky = asignaciones_AP(
            self._grilla_RSSI[:, 0:1, :, :],
            umbral_5g=self.umbral_5g,
            umbral_conexion=self.umbral_conexion,
            umbral_handover=self.umbral_handover,
            sticky_mode=self.sticky_mode,
        )
        self._asignaciones[:, 0:1, :] = asig_t0

        # ID aleatorio por episodio
        self._ap_random_id = torch.rand((self.n_aps, 1), dtype=torch.float32, device=self.device)

        # Sincronizar contadores
        self._avg_rates   = torch.zeros(len(self._eventos), dtype=torch.float32, device=self.device)
        self._n_rate_obs  = 0
        self.current_step = 0
        self._delta_t     = self._adm.get_delta_t_past(0) if self.pomdp else self._adm.get_delta_t(0)

        obs = self._build_obs()

        # Periodo de calentamiento
        if self.warmup_steps > 0:
            for _ in range(self.warmup_steps):
                ch_idx  = torch.randint(0, self.n_channels, (self.n_aps,), device=self.device)
                pwr_idx = torch.randint(0, self.n_powers,   (self.n_aps,), device=self.device)
                action = torch.stack([ch_idx, pwr_idx], dim=1)
                obs, _, term, _, _ = self.step(action)
                if term:
                    break

        return obs, {}

    def step(self, action):

        t_start = self.current_step
        if t_start >= self.total_timesteps:
            return self._build_obs(), 0.0, True, False, {}

        self._decisiones_APs = self._parse_action(action)
        t_end = min(t_start + self.decision_period, self.total_timesteps)
        self._actualizar_RSSI(t_start, t_end)
        
        # Canales físicos
        canales_por_AP = self.available_channels[self._decisiones_APs[:, 0]]

        # Asociación en bloque
        asig_bloque, self._estado_sticky = asignaciones_AP(
            self._grilla_RSSI[:, t_start:t_end, :, :],
            umbral_5g=self.umbral_5g,
            umbral_conexion=self.umbral_conexion,
            umbral_handover=self.umbral_handover,
            sticky_mode=self.sticky_mode,
            estado_previo=self._estado_sticky,
        )
        self._asignaciones[:, t_start:t_end, :] = asig_bloque

        # Calcular SINR y Rate en bloque
        sinr_bloque = calcular_sinr(
            self._grilla_RSSI[:, t_start:t_end, :, :],
            asig_bloque,
            canales_por_AP,
            self.sigma_dbm,
        )
        rate_bloque = calcular_rate(sinr_bloque)
        last_rate_ahora = rate_bloque[:, -1]

        # Cómputo de recompensas
        rate_sum_bloque = rate_bloque.nansum(dim=0)
        n_activos_bloque = (~torch.isnan(rate_bloque)).sum(dim=0).float()
        
        rewards_t = rate_sum_bloque / torch.clamp(n_activos_bloque, min=1.0)
        rewards_t = torch.where(n_activos_bloque > 0, rewards_t, torch.zeros_like(rewards_t))
        
        reward_sum = rewards_t.sum().item()
        rate_sum = rate_sum_bloque.sum().item()

        # Promedio acumulado (avg_rates)
        for t_idx in range(t_end - t_start):
            self._n_rate_obs += 1
            self._avg_rates = actualizar_avg_rate(self._avg_rates, rate_bloque[:, t_idx], self._n_rate_obs)

        self.current_step = t_end
        terminated = (self.current_step >= self.total_timesteps)
        
        # Actualizar delta_t
        if terminated:
            self._delta_t = 0
        else:
            self._delta_t = self._adm.get_delta_t_past(self.current_step) if self.pomdp else self._adm.get_delta_t(self.current_step)

        info = self._generate_info(t_start, t_end, last_rate_ahora, reward_sum, rate_sum)

        return self._build_obs(), float(reward_sum), terminated, False, info

    def _generate_info(self, t_start, t_end, last_rate, reward, rate_sum):
        t_last = t_end - 1
        n_activos = (~torch.isnan(self._asignaciones[:, t_last, 0])).sum().item() if t_last >= 0 else 0
        eps_list = self._adm.get_epsilon_t(t_last) if t_last >= 0 else []
        eps_mean = float(np.mean(eps_list)) if eps_list else 0.0
        
        avg_rate = (rate_sum / n_activos) if n_activos > 0 else 0.0

        return {
            "total_rate": rate_sum,
            "avg_user_rate": avg_rate,
            "n_active_clients": n_activos,
            "timestep": t_last,
            "delta_t": self._delta_t,
            "epsilon_t_mean": eps_mean,
            "sticky_mode": str(self.sticky_mode),
        }

    def _actualizar_RSSI(self, t_start: int, t_end: int):
        if self._grilla_RSSI is None or self._grilla_RSSI.shape != self._grilla_gan.shape:
            self._grilla_RSSI = torch.empty_like(self._grilla_gan)
            
        potencias = self.tx_powers_dbm[self._decisiones_APs[:, 1]]
        p = potencias.view(1, 1, -1, 1).to(self._grilla_gan.dtype)
        self._grilla_RSSI[:, t_start:t_end, :, :] = self._grilla_gan[:, t_start:t_end, :, :] + p

    def _build_obs(self):
        t_last = max(0, self.current_step - 1) if self.current_step > 0 else 0
        t = min(t_last, self.total_timesteps - 1)
        return construir_grafo_timestep(
            t=t,
            asignaciones_t=self._asignaciones[:, t, :],
            grilla_RSSI_t=self._grilla_RSSI[:, t, :, :],
            decisiones_APs=self._decisiones_APs,
            avg_rates=self._avg_rates,
            eventos=self._eventos,
            available_channels=self.available_channels,
            tx_powers_dbm=self.tx_powers_dbm,
            max_timesteps=self.total_timesteps,
            delta_t=self._delta_t,
            ap_pos_encoding=self.ap_pos_encoding,
            pomdp=self.pomdp,
        )

    def _parse_action(self, action) -> torch.Tensor:
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action.astype(np.int64))
        elif not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long)
            
        action = action.long().to(self.device)
        if action.dim() == 1:
            action = action.reshape(self.n_aps, 2)
        return action
