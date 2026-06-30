from __future__ import annotations
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Imports locales
from .clases import Distribution
from .pipeline_preprocesamiento.step1_crear_dataset import crear_dataset, load_mapping
from .pipeline_preprocesamiento.step2_gain_e_imputer import agregar_gain, imputer, ALL_BANDAS, ALL_ANTENAS
from .pipeline_preprocesamiento.step3_mapear_a_blocks import mapear_a_blocks
from .pipeline_preprocesamiento.step4_block_to_distribution import blocks_to_distributions

# Config global
ALL_MONTHS   = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
DEFAULT_P_TX = 14.0  # Potencia de transmisión estándar en dBm


def run_pipeline(
    building_id: str,
    data_dir: str | None = None,
    mapping_file: str | None = None,
    months: list[str] | None = None,
    P_TX: float = DEFAULT_P_TX,
    rssi_fill: float = np.nan,
    step2_csv_path: str | None = None,
    distributions_path: str | None = None,
    verbose: bool = True,
) -> list[Distribution]:
    """Ejecuta el pipeline completo de preprocesamiento para un edificio específico."""
    months = months or ALL_MONTHS

    def log(msg: str):
        if verbose:
            print(msg)

    log(f"\n{'='*70}")
    log(f"  🚀 NetROML Preprocessing Pipeline – Building: {building_id}")
    log(f"{'='*70}\n")

    # Caché Joblib
    if distributions_path and os.path.exists(distributions_path):
        log(f"▶ Detección de caché (Joblib): {distributions_path}")
        log(f"  ✓ Saltando pipeline completo. Cargando objetos serializados...")
        distributions = joblib.load(distributions_path)
        log(f"  ✓ Distribuciones cargadas: {len(distributions):,}\n")
        return distributions

    # Carga de topología
    ap_set  = load_mapping(mapping_file, building_id)
    ap_list = sorted(list(ap_set))

    # Caché CSV
    if step2_csv_path and os.path.exists(step2_csv_path):
        log(f"▶ Detección de caché (CSV): {step2_csv_path}")
        log(f"  ✓ Saltando Steps 1 y 2. Cargando base de datos consolidada...")
        df_s2 = pd.read_csv(step2_csv_path)
        log(f"  ✓ Registros: {len(df_s2):,}  |  Clientes únicos: {df_s2['mac_cliente'].nunique()}\n")
    else:
        # Step 1: Extracción y consolidación
        log("▶ [STEP 1/4] Extracción, Filtrado y Deduplicación")
        if data_dir is None or mapping_file is None:
            raise ValueError("Se requieren 'data_dir' y 'mapping_file' para procesar datos crudos.")
        
        df_s1 = crear_dataset(
            building_id=building_id,
            data_dir=data_dir,
            mapping_file=mapping_file,
            output_file=None,
            months=months,
        )
        log(f"  ✓ Registros extraídos: {len(df_s1):,}\n")

        # Step 2: Imputación y ganancia
        log("▶ [STEP 2/4] Imputación Matricial y Cálculo de Gain (RSSI -> Gain)")
        
        df_s2 = agregar_gain(df_s1, P_TX)
        df_s2 = imputer(df_s2, ap_list, ALL_BANDAS, ALL_ANTENAS, rssi_fill, P_TX)
        log(f"  ✓ Malla completa: {len(ap_list)} APs  |  Filas totales: {len(df_s2):,}\n")

        if step2_csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(step2_csv_path)), exist_ok=True)
            df_s2.to_csv(step2_csv_path, index=False)
            log(f"  💾 Caché intermedia guardada en: {step2_csv_path}\n")

    # Step 3: Crear bloques
    log("▶ [STEP 3/4] Mapeo de Entidades Lógicas (Block Objects)")
    blocks = mapear_a_blocks(df_s2, ap_list)
    log(f"  ✓ Bloques espaciales generados: {len(blocks):,}\n")

    # Step 4: Crear distribuciones
    log("▶ [STEP 4/4] Agrupación de Trayectorias (Distribution Objects)")
    idx_to_mac    = df_s2.groupby('distribution_idx')['mac_cliente'].first().to_dict()
    distributions = blocks_to_distributions(blocks, idx_to_mac)
    log(f"  ✓ Trayectorias de clientes completadas: {len(distributions):,}\n")

    log(f"{'='*70}")
    log(f"  ✅ Pipeline finalizado con éxito.")
    log(f"  📊 Resumen: {len(distributions):,} clientes | {len(blocks):,} bloques procesados.")
    log(f"{'='*70}\n")

    if distributions_path:
        os.makedirs(os.path.dirname(os.path.abspath(distributions_path)), exist_ok=True)
        joblib.dump(distributions, distributions_path)
        log(f"  💾 Dataset final persistido en: {distributions_path}\n")

    return distributions
