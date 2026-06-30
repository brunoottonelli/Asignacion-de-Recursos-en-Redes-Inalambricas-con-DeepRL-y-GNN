import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, Linear, global_mean_pool

class GNN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_aps: int,
        out_channels_ch: int = 3,
        out_channels_pwr: int = 1,
        num_gnn_layers: int = 1,
        num_fc_layers: int = 2,
        pe_dim: int = 4,
        ap_feature_mode: str = 'ase',
    ):
        super().__init__()
        self.num_aps = num_aps
        self.out_channels_ch = out_channels_ch
        self.out_channels_pwr = out_channels_pwr
        self.num_gnn_layers = num_gnn_layers
        self.num_fc_layers = num_fc_layers
        self.pe_dim = pe_dim
        self.ap_feature_mode = ap_feature_mode

        # Entradas: interferencia + onehot canales y potencias
        dim_base = 2 + self.out_channels_ch + self.out_channels_pwr
        self.ap_feat_encoder = Linear(dim_base, hidden_channels)
        
        if self.ap_feature_mode == 'ase' and self.pe_dim > 0:
            D1 = hidden_channels // 2
            D2 = hidden_channels
            
            # Red phi
            self.signnet_phi = nn.Sequential(
                nn.Linear(1, D1),
                nn.ReLU(),
                nn.Linear(D1, D1)
            )
            
            # Red rho
            self.signnet_rho = nn.Sequential(
                nn.Linear(pe_dim * D1, D2),
                nn.ReLU(),
                nn.Linear(D2, D2)
            )
            
        elif self.ap_feature_mode == 'static_id':
            self.ap_embedding = nn.Embedding(self.num_aps, hidden_channels)

        self.client_encoder = Linear(-1, hidden_channels)

        # Capas GNN dinámicas
        self.convs = nn.ModuleList()
        for _ in range(num_gnn_layers):
            conv = HeteroConv({
                ('ap',     'connects',    'client'): GATConv((-1, -1), hidden_channels, edge_dim=1, add_self_loops=False),
                ('client', 'connected_to', 'ap')   : SAGEConv((-1, -1), hidden_channels),
                ('ap',     'interferes',  'ap')    : GATConv((-1, -1), hidden_channels, edge_dim=1, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)

        joint_actions = out_channels_ch * out_channels_pwr
        
        # Cabezas lineales
        self.joint_head = Linear(hidden_channels + 1, joint_actions)
        self.value_head = Linear(hidden_channels + 1, 1)

    def forward(self, x_dict, edge_index_dict=None, edge_attr_dict=None, delta_t=None, alpha=1.0):
        from torch_geometric.data import HeteroData, Batch
        ap_batch = None
        ap_id = None
        if isinstance(x_dict, (HeteroData, Batch)):
            data = x_dict
            x_dict = data.x_dict
            edge_index_dict = data.edge_index_dict
            edge_attr_dict = data.edge_attr_dict
            if hasattr(data, 'delta_t'):
                delta_t = data.delta_t
            if hasattr(data['ap'], 'batch'):
                ap_batch = data['ap'].batch
            if hasattr(data['ap'], 'id'):
                ap_id = data['ap'].id

        dim_base = 2 + self.out_channels_ch + self.out_channels_pwr
        interf_features = x_dict['ap'][:, 0:dim_base]
        ap_x = self.ap_feat_encoder(interf_features)
        
        if self.ap_feature_mode == 'ase' and self.pe_dim > 0:
            pe = x_dict['ap'][:, dim_base:dim_base + self.pe_dim] # (batch * n_APs, pe_dim)
            n, k = pe.shape
            
            # Preparar para phi
            pe_expanded = pe.unsqueeze(-1)
            
            # Simetría invariante al signo
            pe_inv = self.signnet_phi(pe_expanded) + self.signnet_phi(-pe_expanded)
            pe_inv = pe_inv.view(n, -1)
            
            # Embedding de SignNet
            pe_emb = self.signnet_rho(pe_inv)
            
            ap_x = ap_x + alpha * pe_emb
            
        elif self.ap_feature_mode == 'static_id':
            if ap_id is None:
                ap_id = torch.arange(self.num_aps, dtype=torch.long, device=ap_x.device)
                if ap_batch is not None:
                    batch_size = ap_batch.max().item() + 1
                    ap_id = ap_id.repeat(batch_size)
            ap_emb = self.ap_embedding(ap_id)
            ap_x = ap_x + alpha * ap_emb
        
        x = {
            'ap': ap_x
        }
        if 'client' in x_dict:
            x['client'] = self.client_encoder(x_dict['client'])

        # Capas GNN
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index_dict, edge_attr_dict=edge_attr_dict)
            if i != self.num_gnn_layers - 1:
                x = {key: F.relu(val) for key, val in x.items()}
                
        ap_embeddings = F.relu(x['ap']) 

        if ap_batch is None:
            n_total_aps = x_dict['ap'].size(0)
            ap_batch = torch.zeros(n_total_aps, dtype=torch.long, device=x_dict['ap'].device)

        # delta_t
        if delta_t is None:
            # Fallback si no hay delta_t
            batch_size = ap_batch.max().item() + 1
            delta_t = torch.zeros(batch_size, 1, dtype=torch.float32, device=ap_embeddings.device)
        else:
            if delta_t.dim() == 1:
                delta_t = delta_t.unsqueeze(1)
 
        # Normalizar delta_t
        delta_t_norm = torch.log1p(delta_t.float().clamp(min=0.0))
 
        # Broadcast de delta_t
        delta_t_broadcasted = delta_t_norm[ap_batch]
        
        # Concatenar
        ap_embeddings_concat = torch.cat([ap_embeddings, delta_t_broadcasted], dim=-1)
 
        joint_logits = self.joint_head(ap_embeddings_concat)
 
        # Pooling
        state_emb = global_mean_pool(ap_embeddings_concat, ap_batch)
        value = self.value_head(state_emb).squeeze(-1)

        return joint_logits, value
