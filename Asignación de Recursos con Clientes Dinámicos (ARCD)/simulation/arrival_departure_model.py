
from __future__ import annotations
import numpy as np
from data.clases import Distribution, ClientEvent


class ArrivalDepartureModel:

    def __init__(
        self,
        distribuciones: list[Distribution],
        mean_arrival:    float = 2.0,
        mean_duration:   float = 10.0,
        total_timesteps: int   = 100,
        delta_mean:      int   = 5,
        random_seed:     int | None = None,
    ):
        self.distribuciones   = distribuciones
        self.mean_arrival     = mean_arrival
        self.mean_duration    = mean_duration
        self.total_timesteps  = total_timesteps
        self.random_seed      = random_seed
        self.delta_mean       = delta_mean

        self.available_clients = list(range(len(self.distribuciones)))
        self.events: list[ClientEvent] = []
        self.deltas: list[int] = []

        self._delta_t_array: np.ndarray | None = None
        self._delta_t_past_array: np.ndarray | None = None

    def simulate_all_events(self) -> list[ClientEvent]:
        rng = np.random.default_rng(self.random_seed)
        self.events = []
        
        t = 0
        while t < self.total_timesteps:
            # delta de tiempo hasta la próxima llegada (Distribución Normal)
            delta = max(1, int(np.round(rng.normal(loc=self.delta_mean, scale=self.delta_mean * 0.3))))  
            for i in range(delta):
                self.deltas.append(delta-i)
            t += delta
            
            if t >= self.total_timesteps:
                break
            
            cant_cli = max(1, int(np.round(rng.exponential(self.mean_arrival))))

            for _ in range(cant_cli):
                # duración del cliente (Exponencial)
                duration   = max(1, int(np.round(rng.exponential(self.mean_duration))))    
                departure  = min(t + duration, self.total_timesteps)   
                dist_idx   = rng.choice(self.available_clients)     

                self.events.append(ClientEvent(
                    distribution_idx = int(dist_idx),
                    arrival_time     = t,
                    departure_time   = departure,
                    duration         = departure - t,
                ))

        # Construir cache de delta_t
        self._build_delta_t_array()
        self._build_delta_t_past_array()

        return self.events, self.deltas

    def get_active_clients(self, timestep: int) -> list[ClientEvent]:
        return [e for e in self.events if e.arrival_time <= timestep < e.departure_time]

    def get_epsilon_t(self, timestep: int) -> list[int]:
        return [e.departure_time - timestep for e in self.get_active_clients(timestep)]

    def get_delta_t(self, timestep: int) -> int:
        idx = min(timestep, len(self._delta_t_array) - 1)
        return int(self._delta_t_array[idx])

    def get_delta_t_past(self, timestep: int) -> int:
        idx = min(timestep, len(self._delta_t_past_array) - 1)
        return int(self._delta_t_past_array[idx])

    def _build_delta_t_array(self):
        T = self.total_timesteps
        delta = np.zeros(T, dtype=np.int32)

        # Arribos únicos ordenados
        arrivals = sorted({e.arrival_time for e in self.events})
        if not arrivals:
            self._delta_t_array = delta
            return

        j = 0
        for t in range(T):
            while j < len(arrivals) and arrivals[j] <= t:
                j += 1
            delta[t] = (arrivals[j] - t) if j < len(arrivals) else 0

        self._delta_t_array = delta

    def _build_delta_t_past_array(self):
        T = self.total_timesteps
        delta_past = np.zeros(T, dtype=np.int32)

        # Arribos únicos ordenados
        arrivals = sorted({e.arrival_time for e in self.events})
        if not arrivals:
            self._delta_t_past_array = delta_past
            return

        j = 0
        last_arrival = 0
        for t in range(T):
            while j < len(arrivals) and arrivals[j] <= t:
                last_arrival = arrivals[j]
                j += 1
            delta_past[t] = t - last_arrival

        self._delta_t_past_array = delta_past
