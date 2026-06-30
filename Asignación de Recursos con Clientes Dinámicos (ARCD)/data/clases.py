from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class ClientEvent:
    """Representa el ciclo de vida de una sesión cliente en el entorno de simulación."""
    distribution_idx: int
    arrival_time:     int
    departure_time:   int
    duration:         int
    sticky_ap:        int | None = None


@dataclass
class Block:
    """Instantánea de estado de las propiedades físicas del canal para un cliente."""
    block_idx:        int
    distribution_idx: int
    datos:            pd.DataFrame = field(default_factory=pd.DataFrame)

    def __repr__(self) -> str:
        return f"Block(id={self.block_idx}, client_id={self.distribution_idx}, ap_count={len(self.datos)})"


@dataclass
class Distribution:
    """Trayectoria histórica y caracterización física de un cliente WiFi."""
    distribution_idx: int
    mac_client:       str
    blocks:           np.ndarray = field(default_factory=lambda: np.array([], dtype=object))

    def __len__(self) -> int:
        return len(self.blocks)

    def __repr__(self) -> str:
        return f"Distribution(idx={self.distribution_idx}, mac={self.mac_client}, t_len={len(self.blocks)})"
