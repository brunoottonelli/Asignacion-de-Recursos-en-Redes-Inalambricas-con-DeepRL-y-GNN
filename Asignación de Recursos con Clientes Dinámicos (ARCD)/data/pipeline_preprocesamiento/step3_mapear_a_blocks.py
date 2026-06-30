from __future__ import annotations
import pandas as pd
from ..clases import Block


def build_datos(block_df: pd.DataFrame, ap_list: list[str]) -> pd.DataFrame:
    """Construye la matriz de estado (datos) para un bloque temporal específico."""
    # Agrupar por AP y banda
    pivot = (
        block_df
        .groupby(['mac_ap', 'banda'])['gain']
        .max(min_count=1)
        .unstack(level='banda')
    )
    
    # Asegurar bandas
    for band_idx in [0, 1]:
        if band_idx not in pivot.columns:
            pivot[band_idx] = pd.NA
            
    # Reindexar con la lista de APs
    pivot = pivot.reindex(ap_list)
    pivot.index.name = 'mac_ap'
    pivot = pivot.reset_index()
    
    pivot = pivot.rename(columns={0: 'G_2_4', 1: 'G_5'})
    pivot.columns.name = None
            
    return pivot[['mac_ap', 'G_2_4', 'G_5']].reset_index(drop=True)


def mapear_a_blocks(df: pd.DataFrame, ap_list: list[str]) -> list[Block]:
    """Transforma el DataFrame imputado en una colección de objetos Block."""
    blocks: list[Block] = []
    
    # Agrupar por cliente y tiempo
    for (dist_idx, blk_idx), group in df.groupby(['distribution_idx', 'block_idx'], sort=True):
        blocks.append(Block(
            block_idx        = int(blk_idx),
            distribution_idx = int(dist_idx),
            datos            = build_datos(group, ap_list),
        ))
        
    return blocks
