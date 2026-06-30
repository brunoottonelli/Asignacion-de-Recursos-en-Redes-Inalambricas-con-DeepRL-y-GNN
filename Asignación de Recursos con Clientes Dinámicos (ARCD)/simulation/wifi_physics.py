from __future__ import annotations
import numpy as np
import torch
from typing import Literal

# Constantes físicas
UMBRAL_5G:       float = -200.0   # Preferir 5 GHz
UMBRAL_CONEXION: float = -200.0   # Sensibilidad mínima
UMBRAL_HANDOVER: float = -85.0    # Gatillo de handover
SIGMA_DBM:       float = -90.0   # Ruido térmico

# Modos Sticky
STICKY_FULL = 'full'    # Nunca cambia de AP
STICKY_STD  = 'sticky'  # Cambia bajo umbral
STICKY_LITE = 'lite'    # Cambia si hay uno mejor

StickyMode = Literal['full', 'sticky', 'lite']


def crear_grilla(eventos, distribuciones, total_timesteps: int, rng=None) -> np.ndarray:

    if rng is None:
        rng = np.random.default_rng()
    n = len(eventos)
    grilla = np.full((n, total_timesteps), None, dtype=object)
    for fila, ev in enumerate(eventos):
        bloques = distribuciones[ev.distribution_idx].blocks
        sorteados = rng.choice(bloques, size=ev.duration, replace=True)
        grilla[fila, ev.arrival_time:ev.departure_time] = sorteados
    return grilla


def convertir_grilla_a_tensor(grilla: np.ndarray) -> torch.Tensor:

    primer = next((b for b in grilla.flat if b is not None), None)
    if primer is None:
        return torch.full((grilla.shape[0], grilla.shape[1], 0, 2), float('nan'))
    
    cant_APs = len(primer.datos)
    n_ev, T = grilla.shape
    out = torch.full((n_ev, T, cant_APs, 2), float('nan'), dtype=torch.float32)
    
    for (i, j), bloque in np.ndenumerate(grilla):
        if bloque is not None:
            # Validación de seguridad: el bloque debe tener la dimensión esperada
            if len(bloque.datos) == cant_APs:
                vals = torch.tensor(
                    bloque.datos.iloc[:, 1:].values.astype(np.float32),
                    dtype=torch.float32,
                )
                out[i, j] = vals
                
    # Normalizar: -inf → NaN (compatibilidad con .joblib legacy)
    out = torch.nan_to_num(out, nan=float('nan'), posinf=float('nan'), neginf=float('nan'))
    return out


def obtener_grilla_RSSIs(
    grilla_ganancias: torch.Tensor,
    potencias_APs_dBm: torch.Tensor,
) -> torch.Tensor:

    p = potencias_APs_dBm.view(1, 1, -1, 1).to(grilla_ganancias.dtype)
    return grilla_ganancias + p


def asignaciones_AP(grilla_RSSI: torch.Tensor,
                    umbral_5g: float = UMBRAL_5G,
                    umbral_conexion: float = UMBRAL_CONEXION,
                    umbral_handover: float = UMBRAL_HANDOVER,
                    sticky_mode: StickyMode = STICKY_FULL,
                    estado_previo=None,
                    # Retrocompatibilidad: sticky=True → STICKY_STD, sticky=False → STICKY_LITE
                    sticky: bool | None = None,
                    only_2_4: bool = True,
                   ):

    # Retrocompatibilidad booleana
    if sticky is not None:
        sticky_mode = STICKY_STD if sticky else STICKY_LITE

    cant_cli, cant_timesteps, cant_ap, cant_bandas = grilla_RSSI.shape
    dev = grilla_RSSI.device

    if only_2_4:
        grilla_RSSI = grilla_RSSI[:, :, :, 0]
        grilla_RSSI = grilla_RSSI.unsqueeze(-1)
        cant_bandas = 1

    else:
        # HALLO EL AP IDEAL POR TIMESTEP EN 5GHz
        rssi_5g = grilla_RSSI[:, :, :, 1]
        # torch.max propaga NaNs. Usamos un valor centinela -999 para el cálculo.
        rssi_5g_safe = rssi_5g.nan_to_num(-999.0)
        max_5g, argmax_5g = rssi_5g_safe.max(dim=2) # El argmax me dice con que AP es
                                                    # la banda ya se que es 1 por ser 5G

    # Analizamos el mejor RSSI para todas las bandas en conjunto
    grilla_plana = grilla_RSSI.reshape(cant_cli, cant_timesteps, cant_ap * cant_bandas)
    grilla_plana_safe = grilla_plana.nan_to_num(-999.0)
    max_all_rssi, argmax_flat = grilla_plana_safe.max(dim=2)
    
    argmax_all_ap   = argmax_flat // cant_bandas
    argmax_all_band = argmax_flat % cant_bandas


    # Una posición es válida si existe al menos un AP por encima del umbral de conexión
    mask_active = (max_all_rssi >= umbral_conexion)
    # Pongo en nan las no válidas
    argmax_all_ap_f   = argmax_all_ap.float().masked_fill(~mask_active, float('nan'))
    argmax_all_band_f = argmax_all_band.float().masked_fill(~mask_active, float('nan'))

    if only_2_4:
        ideal_ap   = argmax_all_ap_f
        ideal_band = argmax_all_band_f
        ideal_rssi = max_all_rssi.masked_fill(~mask_active, float('nan'))

    else:
        argmax_5g_f = argmax_5g.float().masked_fill(~mask_active, float('nan'))
        # preferir 5G
        cumple_5g  = (max_5g >= umbral_5g) & mask_active
        ideal_ap   = torch.where(cumple_5g, argmax_5g_f, argmax_all_ap_f)
        ideal_band = torch.where(cumple_5g, torch.ones_like(argmax_5g_f), argmax_all_band_f)
        ideal_rssi = torch.where(cumple_5g, max_5g, max_all_rssi).masked_fill(~mask_active, float('nan'))

        

    # Modo LITE: mejor AP disponible
    if sticky_mode == STICKY_LITE:
        asig = torch.stack([ideal_ap, ideal_band, ideal_rssi], dim=-1)
        ultimo_estado = (ideal_ap[:, -1].clone(), ideal_band[:, -1].clone())
        return asig, ultimo_estado

    

    # Modos FULL y STICKY

    # Inicializamos
    assigned_ap   = torch.full((cant_cli, cant_timesteps), float('nan'), device=dev)
    assigned_band = torch.full((cant_cli, cant_timesteps), float('nan'), device=dev)
    assigned_rssi = torch.full((cant_cli, cant_timesteps), float('nan'), device=dev)

    if estado_previo is not None:
        current_ap   = estado_previo[0].clone()
        current_band = estado_previo[1].clone()
    else:
        current_ap   = torch.full((cant_cli,), float('nan'), device=dev)
        current_band = torch.full((cant_cli,), float('nan'), device=dev)


    # Índice base para vectorizar sobre clientes
    idx_cli = torch.arange(cant_cli, device=dev)


    for t in range(cant_timesteps):
        active_now = ~torch.isnan(ideal_rssi[:, t])
        ap_safe    = current_ap.nan_to_num(0).long()
        band_safe  = current_band.nan_to_num(0).long()
        rssi_current  = grilla_RSSI[idx_cli, t, ap_safe, band_safe]

        # Si el cliente estaba desconectado antes del "safe", vuelvo a ponerlo en NaN.
        rssi_current[torch.isnan(current_ap)] = float('nan')


        # Veo que clientes precisan cambiar su asignación (pasaron un umbral o son nuevos)
        was_connected = ~torch.isnan(current_ap)

        if sticky_mode == STICKY_FULL:
            # Full Sticky: solo cambia AP en el momento del nacimiento (primer slot activo)
            # Una vez conectado, NUNCA cambia → change_mask solo para recién llegados
            change_mask = ~was_connected & active_now

        else:
            # Sticky estándar: cambia si RSSI <= umbral_handover, si es NaN (canal perdido), o si es nuevo
            is_too_low = (rssi_current <= umbral_handover) | torch.isnan(rssi_current)
            change_mask = (was_connected & is_too_low) | (~was_connected & active_now)


        current_ap   = torch.where(change_mask, ideal_ap[:, t],   current_ap)
        current_band = torch.where(change_mask, ideal_band[:, t], current_band)
        current_rssi = torch.where(change_mask, ideal_rssi[:, t], rssi_current.nan_to_num(nan=float('nan')))

        current_ap[~active_now]   = float('nan')
        current_band[~active_now] = float('nan')
        current_rssi[~active_now] = float('nan')

        assigned_ap[:, t]   = current_ap
        assigned_band[:, t] = current_band
        assigned_rssi[:, t] = current_rssi

    asig = torch.stack([assigned_ap, assigned_band, assigned_rssi], dim=-1)
    ultimo_estado = (current_ap.clone(), current_band.clone())
    
    return asig, ultimo_estado


def db_to_linear(x: torch.Tensor) -> torch.Tensor:

    # Filtramos nans para evitar warnings en pow
    mask_nan = torch.isnan(x)
    x_safe = x.nan_to_num(-999.0) # RSSI extremadamente bajo -> ~0 mW
    
    out = torch.pow(10.0, x_safe / 10.0)
    out[mask_nan] = float('nan')

    return out


def calcular_sinr(grilla_RSSI: torch.Tensor,
                  asignaciones: torch.Tensor,
                  canales_por_AP: torch.Tensor,
                  sigma_dbm: float = SIGMA_DBM,
                 ) -> torch.Tensor:
    cant_cli, cant_timesteps, cant_ap, cant_bandas = grilla_RSSI.shape

    dev = grilla_RSSI.device

    ap_idx   = asignaciones[:, :, 0]
    band_idx = asignaciones[:, :, 1]
    rssi_dBm = asignaciones[:, :, 2]
    mask_nan = torch.isnan(ap_idx)


    # NUMERADOR:
    # -------------
    numerador = db_to_linear(rssi_dBm)


    # DENOMINADOR:
    # -------------

    ap_safe   = ap_idx.nan_to_num(0).long()
    band_safe = band_idx.nan_to_num(0).long()

    # Para cada cliente y timestep, agarrar el RSSI de todos los APs en la banda elegida
    band_exp = band_safe.unsqueeze(2).unsqueeze(3).expand(-1, -1, cant_ap, 1)
    rssi_en_banda = grilla_RSSI.gather(3, band_exp).squeeze(3)
    rssi_en_banda_lin = db_to_linear(rssi_en_banda).nan_to_num(0.0)

    # Identificar APs activos
    ap_is_active = torch.zeros((cant_timesteps, cant_ap), dtype=torch.bool, device=dev)
    valid_mask = ~mask_nan
    if valid_mask.any():
        t_idx = torch.arange(cant_timesteps, device=dev).unsqueeze(0).expand(cant_cli, -1)
        ap_is_active[t_idx[valid_mask], ap_safe[valid_mask]] = True

    # Interferencia = suma de APs en el mismo canal que están activos, menos la señal del AP propio
    canal_cli = canales_por_AP[ap_safe]  # (cant_cli, cant_timesteps)
    interf_mask = ap_is_active.unsqueeze(0) & (canales_por_AP.unsqueeze(0).unsqueeze(0) == canal_cli.unsqueeze(2))
    
    total_interf_canal = (rssi_en_banda_lin * interf_mask.float()).sum(dim=2)
    interf_lin = (total_interf_canal - numerador.nan_to_num(0.0)).clamp(min=0.0)

    sigma_lin = db_to_linear(torch.tensor(sigma_dbm, dtype=torch.float32, device=dev))

    denominador = (interf_lin + sigma_lin + 1e-30)


    # SINR:
    # -------------

    sinr = numerador / denominador    
    sinr = sinr.masked_fill(mask_nan, float('nan'))  #(cant_cli, cant_timesteps)

    return sinr     




def calcular_rate(sinr: torch.Tensor) -> torch.Tensor:
    rate = torch.log(1.0 + sinr.nan_to_num(nan=0.0))
    return rate.masked_fill(torch.isnan(sinr), float('nan'))




def actualizar_avg_rate(
    avg_prev: torch.Tensor,
    nuevo_rate: torch.Tensor,
    n: int,
) -> torch.Tensor:
    new_val = nuevo_rate.nan_to_num(nan=0.0)
    avg_new = ((n - 1) * avg_prev + new_val) / n
    return torch.where(torch.isnan(nuevo_rate), avg_prev, avg_new)


def calcular_reward(rate_t: torch.Tensor) -> float:
    return rate_t.nansum().item()

# -------------------------------------------------------


# def nuevo_calcular_reward(rates: torch.Tensor) -> float:
#     r"""
    
#     """

#     # sumo los rates de todos los clientes en un mismo timestep
#     rate_por_timestep = rates.nansum(dim=0)       # (cant_timesteps,)
#     reward = rate_por_timestep.nanmean().item()   # promedio temporal

#     return reward