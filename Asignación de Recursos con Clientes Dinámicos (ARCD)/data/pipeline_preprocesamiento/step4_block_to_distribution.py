from __future__ import annotations
from collections import defaultdict
import numpy as np
from ..clases import Block, Distribution


def blocks_to_distributions(
    blocks: list[Block],
    idx_to_mac: dict[int, str],
) -> list[Distribution]:
    """Agrupa una colección de bloques espaciales en trayectorias temporales (Distribution)."""
    # Agrupar por id de trayectoria
    groups: dict[int, list[Block]] = defaultdict(list)
    for block in blocks:
        groups[block.distribution_idx].append(block)

    distributions: list[Distribution] = []
    
    # Procesar en orden
    for dist_idx in sorted(groups.keys()):
        # Ordenar por bloque
        client_blocks = sorted(groups[dist_idx], key=lambda b: b.block_idx)
        
        distributions.append(Distribution(
            distribution_idx = dist_idx,
            mac_client       = idx_to_mac[dist_idx],
            blocks           = np.array(client_blocks, dtype=object),
        ))
        
    return distributions
