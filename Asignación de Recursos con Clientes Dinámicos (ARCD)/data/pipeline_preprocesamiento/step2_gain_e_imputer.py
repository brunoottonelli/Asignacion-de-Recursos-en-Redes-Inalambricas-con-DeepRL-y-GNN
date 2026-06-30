from __future__ import annotations
import itertools
import pandas as pd
import numpy as np

# Constantes de espectro WiFi
ALL_BANDAS  = [0, 1]   # 0: 2.4 GHz, 1: 5 GHz
ALL_ANTENAS = [0, 1]   # Diversidad de antena


def agregar_gain(df: pd.DataFrame, P_TX: float) -> pd.DataFrame:
    """Calcula la ganancia física a partir del RSSI reportado."""
    df = df.copy()
    df['gain'] = df['rssi'] - P_TX
    return df


def imputer(
    df: pd.DataFrame,
    ap_list: list[str],
    bandas: list[int] = ALL_BANDAS,
    antenas: list[int] = ALL_ANTENAS,
    rssi_fill: float = np.nan,
    P_TX: float = 14.0,
) -> pd.DataFrame:
    """Realiza la imputación matricial para garantizar homogeneidad dimensional."""
    # Combinaciones esperadas
    expected_combos = pd.DataFrame(
        list(itertools.product(ap_list, bandas, antenas)),
        columns=['mac_ap', 'banda', 'antena'],
    )
    
    # Claves únicas de bloques
    block_keys = ['mac_cliente', 'datetime', 'distribution_idx', 'block_idx']
    blocks     = df[block_keys].drop_duplicates()
    
    # Producto cartesiano
    full_tmpl = blocks.merge(expected_combos, how='cross')

    # Fusionar con datos reales
    df_merged = full_tmpl.merge(
        df[block_keys + ['mac_ap', 'banda', 'antena', 'rssi', 'gain']],
        on=block_keys + ['mac_ap', 'banda', 'antena'],
        how='left',
    )
    
    df_merged['rssi'] = df_merged['rssi'].fillna(rssi_fill)
    # Calcular ganancia
    df_merged['gain'] = df_merged['rssi'] - P_TX

    # Selección de columnas y orden
    cols = ['datetime', 'distribution_idx', 'mac_cliente', 'mac_ap',
            'rssi', 'gain', 'antena', 'banda', 'block_idx']
            
    return (
        df_merged[cols]
        .sort_values(['mac_cliente', 'datetime', 'mac_ap', 'banda', 'antena'])
        .reset_index(drop=True)
    )
