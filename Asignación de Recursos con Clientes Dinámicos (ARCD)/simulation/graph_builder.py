from __future__ import annotations
import torch
from torch_geometric.data import HeteroData

# Umbral de visibilidad RF: se pasa por parámetro (por defecto -200 dBm para conectar todo)
# UMBRAL_VIS = -200.0   # [dBm]

def build_interference_matrix(rssi_banda: torch.Tensor, ap_activos: torch.Tensor, n_APs: int) -> torch.Tensor:
    """
    Construye la matriz de interferencia entre APs basada en las interacciones actuales de los clientes.
    """
    dev = rssi_banda.device
    I = torch.zeros((n_APs, n_APs), device=dev, dtype=torch.float32)
    
    U_t = rssi_banda.shape[0]
    if U_t > 0:
        # Convertir señales en dBm a una escala lineal (gain log -> mw)
        noise_floor_linear = 10 ** (-100.0 / 10.0)
        rssi_linear = 10 ** (rssi_banda / 10.0)
        rssi_linear = torch.nan_to_num(rssi_linear, nan=noise_floor_linear)
        rssi_linear = torch.clamp(rssi_linear, min=noise_floor_linear)
        
        # Para cada usuario, ignorar la contribución a su propio AP
        mask_diferente = torch.ones((U_t, n_APs), dtype=torch.bool, device=dev)
        mask_diferente[torch.arange(U_t, device=dev), ap_activos] = False
        
        rssi_interference = rssi_linear * mask_diferente
        
        # Acumular la interferencia recibida por los clientes asignados a un AP, proveniente de otros APs
        I.index_add_(0, ap_activos, rssi_interference)
        
        # Calcular el promedio por cliente para no depender de la cantidad de usuarios
        clientes_por_ap = torch.bincount(ap_activos, minlength=n_APs).unsqueeze(1).float()
        clientes_por_ap = torch.clamp(clientes_por_ap, min=1.0)
        I = I / clientes_por_ap
        
    # Simetrizar (promedio mutuo) para capturar la superposición en el canal
    W = (I + I.T) / 2.0
    
    # Normalizamos transformando de vuelta a dBm
    clip_min_linear = 10 ** (-100.0 / 10.0)
    W_safe = torch.clamp(W, min=clip_min_linear)
    W_dbm = 10 * torch.log10(W_safe)
    
    # Llevamos de [-100, -30] a [0, 1]
    W_norm = ((W_dbm + 100.0) / 70.0).clamp(0.0, 1.0)
    
    return W_norm

def construir_grafo_timestep(
    t: int,
    asignaciones_t: torch.Tensor,
    grilla_RSSI_t: torch.Tensor,
    decisiones_APs: torch.Tensor,
    avg_rates: torch.Tensor,
    eventos,
    available_channels: torch.Tensor,
    tx_powers_dbm: torch.Tensor,
    max_timesteps: int,
    delta_t: int = 0,
    umbral_vis: float = -200.0,
    ap_pos_encoding: torch.Tensor | None = None,
    pomdp: bool = False,
) -> HeteroData:

    _, n_APs, _ = grilla_RSSI_t.shape
    n_ch = len(available_channels)
    n_pwr = len(tx_powers_dbm)
    dev  = grilla_RSSI_t.device

    data = HeteroData()

    ap_asignado_t = asignaciones_t[:, 0]   # (n_cli,) — fila de A_t
    mask_activos  = ~torch.isnan(ap_asignado_t)

    # One-hot encoding de la decisión anterior
    ch_onehot = torch.nn.functional.one_hot(decisiones_APs[:, 0], num_classes=n_ch).float()
    pwr_onehot = torch.nn.functional.one_hot(decisiones_APs[:, 1], num_classes=n_pwr).float()
    action_features = torch.cat([ch_onehot, pwr_onehot], dim=-1)

    ap_x_base = torch.cat([
        torch.zeros((n_APs, 2), dtype=torch.float32, device=dev),
        action_features
    ], dim=-1)

    if ap_pos_encoding is not None:
        data['ap'].x = torch.cat([ap_x_base, ap_pos_encoding], dim=-1)
    else:
        data['ap'].x = ap_x_base
        
    data['ap'].id = torch.arange(n_APs, dtype=torch.long, device=dev)

    # Nodos Cliente
    activos_idx = torch.where(mask_activos)[0]
    n_activos   = len(activos_idx)

    if n_activos == 0:
        data['client'].x = torch.empty((0, 1), dtype=torch.float32, device=dev)
        data['ap', 'connects', 'client'].edge_index = torch.empty((2, 0), dtype=torch.long, device=dev)
        data['ap', 'connects', 'client'].edge_attr  = torch.empty((0, 1), dtype=torch.float32, device=dev)
        data['client', 'connected_to', 'ap'].edge_index = torch.empty((2, 0), dtype=torch.long, device=dev)
        
        # Inicializar grafo AP-AP por defecto (pesos en cero para todos con todos)
        ap_idx_1 = torch.arange(n_APs, device=dev).repeat_interleave(n_APs)
        ap_idx_2 = torch.arange(n_APs, device=dev).repeat(n_APs)
        data['ap', 'interferes', 'ap'].edge_index = torch.stack([ap_idx_1, ap_idx_2], dim=0)
        data['ap', 'interferes', 'ap'].edge_attr  = torch.zeros((n_APs * n_APs, 1), dtype=torch.float32, device=dev)
        
        data.delta_t = torch.tensor([delta_t], dtype=torch.float32, device=dev)
        return data


    banda_t    = asignaciones_t[activos_idx, 1]
    

    if pomdp:
        epsilon_t = torch.tensor(
            [t - eventos[i.item()].arrival_time for i in activos_idx],
            dtype=torch.float32, device=dev,
        )
    else:
        epsilon_t = torch.tensor(
            [eventos[i.item()].departure_time - t for i in activos_idx],
            dtype=torch.float32, device=dev,
        )

    ap_activos = ap_asignado_t[activos_idx].long()    
    band_safe  = banda_t.nan_to_num(0).long()
    


    data['client'].x = epsilon_t.unsqueeze(1)

    rssi_banda = grilla_RSSI_t[activos_idx, :, :].gather(
        2, band_safe.view(-1, 1, 1).expand(-1, n_APs, 1)
    ).squeeze(2)  

    vis_cli_local, vis_ap = torch.where(rssi_banda > umbral_vis)

    if len(vis_ap) > 0:
        edge_index_down = torch.stack([vis_ap, vis_cli_local], dim=0)
        edge_attr_down  = ((rssi_banda[vis_cli_local, vis_ap] + 100.0) / 70.0).unsqueeze(1)
    else:
        edge_index_down = torch.empty((2, 0), dtype=torch.long, device=dev)
        edge_attr_down  = torch.empty((0, 1), dtype=torch.float32, device=dev)

    data['ap', 'connects', 'client'].edge_index = edge_index_down
    data['ap', 'connects', 'client'].edge_attr  = edge_attr_down

    local_idx = torch.arange(n_activos, device=dev)
    data['client', 'connected_to', 'ap'].edge_index = torch.stack(
        [local_idx, ap_activos], dim=0
    )

    W_norm = build_interference_matrix(rssi_banda, ap_activos, n_APs)
    
    interf_max = W_norm.max(dim=1, keepdim=True).values
    interf_mean = W_norm.mean(dim=1, keepdim=True)
    interf_features = torch.cat([interf_max, interf_mean, action_features], dim=-1)
    
    if ap_pos_encoding is not None:
        data['ap'].x = torch.cat([interf_features, ap_pos_encoding], dim=-1)
    else:
        data['ap'].x = interf_features

    ap_idx_1 = torch.arange(n_APs, device=dev).repeat_interleave(n_APs)
    ap_idx_2 = torch.arange(n_APs, device=dev).repeat(n_APs)
    
    edge_attr_ap_ap = W_norm[ap_idx_1, ap_idx_2].unsqueeze(1)
    
    data['ap', 'interferes', 'ap'].edge_index = torch.stack([ap_idx_1, ap_idx_2], dim=0)
    data['ap', 'interferes', 'ap'].edge_attr  = edge_attr_ap_ap

    data.delta_t = torch.tensor([delta_t], dtype=torch.float32, device=dev)

    return data
