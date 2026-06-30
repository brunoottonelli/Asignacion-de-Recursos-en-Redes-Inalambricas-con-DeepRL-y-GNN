import os
import joblib
import numpy as np
import pandas as pd
import scipy.linalg as la

def obtener_matriz_atenuacion_historica(
    building_id,
    path_mac_to_building: str = "data/from_mac_to_building_id.csv",
    path_datos_vecinos: str = "data/datos_resto_del_anio/datos_vecinos_Abril.csv.gz",
    save_load_path: str = "data/matrices_atenuacion/matriz_b{building_id}.pkl",
    load_if_exists: bool = True,
):
    """Carga o calcula la matriz histórica de atenuaciones (ganancias) entre APs."""
    if save_load_path:
        save_load_path = save_load_path.format(building_id=building_id)
        
    # Cargar cache si existe
    if load_if_exists and save_load_path and os.path.exists(save_load_path):
        print(f"Cargando matriz histórica desde {save_load_path}...")
        datos = joblib.load(save_load_path)
        return datos['matriz_np'], datos['aps_del_edificio']
    
    print("Calculando matriz histórica desde los CSVs originales...")
    
    df_mac_edificios = pd.read_csv(path_mac_to_building)
    columnas_necesarias = ['MAC_AP', 'MAC_vecino', 'Tx_power_vecino', 'Atenuacion'] 
    
    compression = 'gzip' if str(path_datos_vecinos).endswith('.gz') else None
    df_vecinos = pd.read_csv(
        path_datos_vecinos, 
        compression=compression,
        usecols=columnas_necesarias
    )
    
    # Identificar APs
    df_mac_edificios['building_id'] = df_mac_edificios['building_id'].astype(str)
    aps_del_edificio = np.sort(df_mac_edificios[df_mac_edificios['building_id'] == str(building_id)]['MAC_AP'].unique())
    
    # Filtrar vecinos
    df_filtrado = df_vecinos[
        (df_vecinos['MAC_AP'].isin(aps_del_edificio)) &
        (df_vecinos['MAC_vecino'].isin(aps_del_edificio))
    ].copy()
    
    # Agrupar y promediar
    df_agrupado = df_filtrado.groupby(['MAC_AP', 'MAC_vecino'], as_index=False)[['Atenuacion', 'Tx_power_vecino']].mean()
    df_agrupado['Atenuacion'] = df_agrupado['Atenuacion'].round()
    
    # Pivotar a matriz NxN
    matriz_pd = df_agrupado.pivot_table(
        index='MAC_AP', 
        columns='MAC_vecino', 
        values='Atenuacion'
    )
    
    matriz_pd = matriz_pd.reindex(index=aps_del_edificio, columns=aps_del_edificio).fillna(-np.inf)
    matriz_np = matriz_pd.to_numpy()
    
    # Forzar simetría
    matriz_segura = np.nan_to_num(matriz_np, neginf=-100.0)
    matriz_simetrica = (matriz_segura + matriz_segura.T) / 2.0
    matriz_simetrica[matriz_simetrica <= -99.9] = -np.inf
    
    # Guardar
    if save_load_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_load_path)), exist_ok=True)
        joblib.dump({'matriz_np': matriz_simetrica, 'aps_del_edificio': aps_del_edificio}, save_load_path)
        print(f"Matriz calculada y guardada en {save_load_path}")
        
    return matriz_simetrica, aps_del_edificio


def calcular_positional_encoding(matriz_atenuacion, d_dimensiones=4, method='laplacian'):
    """Calcula el embedding posicional para los APs basado en la matriz de atenuación."""
    assert method in ['laplacian', 'adjacency'], "method debe ser 'laplacian' o 'adjacency'"
    
    N = matriz_atenuacion.shape[0]
    
    # Pesos de adyacencia [0, 1]
    mat_segura = np.nan_to_num(matriz_atenuacion, neginf=-100.0)
    A = np.clip((mat_segura + 100.0) / 70.0, 0.0, 1.0)
    
    # Simetría sin self-loops
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    
    if method == 'laplacian':
        # D^{-1/2}
        grados = np.sum(A, axis=1)
        d_inv_sqrt = np.power(np.maximum(grados, 1e-12), -0.5)
        
        # Laplaciano Normalizado
        I = np.eye(N)
        L = I - (A * d_inv_sqrt[:, np.newaxis] * d_inv_sqrt[np.newaxis, :])
        
        # Eigen-descomposición
        eigenvalues, eigenvectors = la.eigh(L)
        PE = eigenvectors[:, 1 : d_dimensiones + 1]
        
    elif method == 'adjacency':
        # Eigen-descomposición de Adyacencia
        eigenvalues, eigenvectors = la.eigh(A)
        PE = eigenvectors[:, -d_dimensiones:][:, ::-1]
        
    else:
        raise ValueError("Método no reconocido.")
        
    # Rellenar con ceros si falta dimensión
    if PE.shape[1] < d_dimensiones:
        padding = np.zeros((N, d_dimensiones - PE.shape[1]), dtype=PE.dtype)
        PE = np.hstack([PE, padding])
            
    return PE
