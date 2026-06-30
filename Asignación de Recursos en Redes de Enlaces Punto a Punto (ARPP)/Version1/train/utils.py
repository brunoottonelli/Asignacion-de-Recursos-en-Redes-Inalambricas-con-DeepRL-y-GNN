from torch_geometric.nn import LayerNorm, Sequential
from torch_geometric.nn.conv import MessagePassing

import random
import pickle
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TAGConv
from torch_geometric.nn import GCNConv
import os
import matplotlib.pyplot as plt

import scipy.io

def transform_matrix(adj_matrix, all = True):
    """Transforma la matriz de adyacencia para modelar la interferencia."""
    if all:
        # Logic 1: Define pairs based on strongest connection (Simple matching)
        
        # Number of nodes in the graph
        nodos = adj_matrix.shape[0]
        lista_parejas = []
        
        # Iterate over each node to find its "pair" (likely strongest AP-Client link)
        for i in range(nodos):
            # Find the index of the neighbor with highest edge weight (max h)
            receptor = np.argmax(adj_matrix[i,:])
            # Get the max value
            valor_maximo = adj_matrix[i,receptor]
            transmisor = i
            # Store tuple: [max_h, transmitter_idx, receiver_idx]
            elemento = [valor_maximo, transmisor, receptor]
            lista_parejas.append(elemento)

        # Initialize Channel Matrix H with zeros
        H = np.zeros_like(adj_matrix)

        # Fill the matrix H
        for i in range(nodos):
            for j in range(nodos):
                if (i == j):
                    # Diagonal elements: Direct link channel gain (signal strength)
                    # Uses the max h found for this node as a transmitter
                    H[i,j] = lista_parejas[i][0]
                else:
                    # Off-diagonal: Interference channels
                    # Find who is the receiver for transmitter 'j'
                    k = 0
                    nodo_receptor = 0
                    for k in range(nodos):
                        # Search for the pair where 'j' is the transmitter
                        if (lista_parejas[k][1] == j):
                            # Identify the receiver node for 'j'
                            nodo_receptor = lista_parejas[k][2]
                            if (nodo_receptor == i):
                                # Special case handling?
                                H[i,j] = 0.005
                            else:
                                # H[i,j] is interference from transmitter 'i' to receiver of 'j'
                                # Here it takes adj_matrix[i, nodo_receptor]
                                H[i,j] = adj_matrix[i,nodo_receptor]
        return H
    
    else:
        num_nodes = adj_matrix.shape[0]
        nodos = list(np.arange(num_nodes))
        random.seed(42)
        random.shuffle(nodos)
        
        half_nodes= num_nodes // 2
        nodos_tx = nodos[:half_nodes]
        nodos_rx = nodos[half_nodes:]

        lista_parejas = []
        for nodo_tx in nodos_tx:
            h_canal = []
            for nodo_rx in nodos_rx:
                h_canal.append(adj_matrix[nodo_tx,nodo_rx])
            index_pareja = np.argmax(h_canal)
            valor_maximo = adj_matrix[nodo_tx, nodos_rx[index_pareja]]
            elemento = [valor_maximo, nodo_tx, nodos_rx[index_pareja]]
            lista_parejas.append(elemento)

        H = np.zeros((num_nodes,num_nodes))

        for i in np.arange(half_nodes):
            nodo_tx = lista_parejas[i][1]
            for j in np.arange(half_nodes):
                if (i == j):
                    H[i,j] = lista_parejas[i][0]
                else:
                    nodo_tx_a_j = nodos_tx[j]
                    receptor_j = -1
                    for k in np.arange(half_nodes):
                        if lista_parejas[k][1] == nodo_tx_a_j:
                            receptor_j = k
                    H[i,j] = adj_matrix[nodos_tx[i], nodos_rx[receptor_j]]
        return H
    
def graphs_to_tensor(train=True, num_links=5, num_features=1, b5g=False, building_id=990):
    """Carga los datos de grafos desde archivos pickle y los convierte en tensores."""
    band = ['2_4', '5']
    path = os.path.expanduser("~/wireless-learning/graphs/") + str(band[b5g]) + '_' + str(building_id) + '/'

    if (train):
        file_name = 'train_' + str(band[b5g]) + '_graphs_' + str(building_id) + '.pkl'
        with open(path + file_name, 'rb') as archivo:
            graphs = pickle.load(archivo)
    else:
        file_name = 'val_' + str(band[b5g]) + '_graphs_' + str(building_id) + '.pkl'
        with open(path + file_name, 'rb') as archivo:
            graphs = pickle.load(archivo)
            
    x_list = []
    channel_matrix_list = []

    x = torch.zeros((num_links, 1)) 
    
    for graph in graphs:
        adj_matrix = nx.adjacency_matrix(graph, weight = 'Atenuacion')
        adj_matrix = adj_matrix.toarray()
        
        channel_matrix = transform_matrix(adj_matrix, all = True)
        channel_matrix = channel_matrix/1e-3
        
        channel_matrix_list.append(torch.tensor(channel_matrix.T))
        x_list.append(x)

    channel_matrix_tensor = torch.stack(channel_matrix_list)
    x_tensor = torch.stack(x_list)
    return x_tensor, channel_matrix_tensor


def graphs_to_tensor_synthetic(num_links, num_features = 1, b5g = False, building_id = 990, filename='synthetic_graphs.pkl'):
    """
    Loads SYNTHETIC graph data and converts them to Tensors.
    Similar to graphs_to_tensor but assumes simple matrix structure in pickle.
    """
    band = ['2_4', '5']
    path = os.path.expanduser("~/wireless-learning/graphs/") + str(band[b5g]) + '_' + str(building_id) + '/'
    
    with open(path + filename, 'rb') as archivo:
        graphs = pickle.load(archivo)
        
    x_list = []
    channel_matrix_list = []
    x = torch.zeros((num_links,num_features)) 
    
    for graph in graphs:
        # For synthetic, graph is already a matrix/tensor
        channel_matrix_list.append(torch.tensor(graph))
        x_list.append(x)
        
    channel_matrix_tensor = torch.stack(channel_matrix_list)
    x_tensor = torch.stack(x_list)
    return x_tensor, channel_matrix_tensor


def get_gnn_inputs(x_tensor, channel_matrix_tensor, K=4):
    """
    Convierte tensores crudos en objetos Data de PyG.
    
    Esta versión calcula analíticamente las diagonales de las potencias 
    de la matriz de desplazamiento del grafo (S^k). Esto extrae directamente 
    la información de los ciclos cerrados (longitud 0 a K) con cero varianza
    para preservar la equivarianza y romper la simetría determinísticamente.
    
    Features por nodo [num_links, K+1]:
      x_struct[:, 0] -> S^0: identidad (diag(I) = 1 para todo nodo)
      x_struct[:, 1] -> S^1: ganancia directa (diag S)
      x_struct[:, 2] -> S^2: ciclos de longitud 2 (interferencia mutua)
      x_struct[:, k] -> S^k: ciclos de longitud k
    """
    input_list = []
    size = channel_matrix_tensor.shape[0]
    
    for i in range(size):
        channel_matrix = channel_matrix_tensor[i, :, :].float()
        
        norm = torch.linalg.norm(channel_matrix, ord=2)
        channel_matrix_norm = channel_matrix / (norm + 1e-12)
        
        N = channel_matrix_norm.shape[0]
        diags = [torch.ones(N, 1, dtype=channel_matrix_norm.dtype,
                            device=channel_matrix_norm.device)]  # S^0
        S_power = channel_matrix_norm
        for k in range(1, K + 1):
            if k > 1:
                S_power = torch.matmul(S_power, channel_matrix_norm)
            diags.append(torch.diag(S_power).unsqueeze(1))
        
        x_struct = torch.cat(diags, dim=1)
        
        edge_index = channel_matrix_norm.nonzero(as_tuple=False).t()
        edge_attr = channel_matrix_norm[edge_index[0], edge_index[1]].unsqueeze(1).float()
        
        input_list.append(Data(
            x=x_struct, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            matrix=channel_matrix
        ))
        
    return input_list



def objective_function(rates):
    """
    Computes the objective function: Negative Sum Rate (Cost).
    We want to Maximize Sum Rate, so we Minimize Negative Sum Rate.
    """
    sum_rate = -torch.sum(rates, dim=1)
    return sum_rate

def power_constraint(phi, pmax_per_ap):
    """
    Computes the power constraint violation.
    Constraint: Sum(phi) <= pmax.
    Violation = Sum(phi) - pmax.
    """
   # print(phi) #(64,3,3)
    sum_phi_per_link = torch.sum(phi, dim=2)  
   # print(sum_phi_per_link) #(64,3)
    return (sum_phi_per_link - pmax_per_ap)

def mu_update(mu_k, power_constr, eps):
    """Actualiza el multiplicador de Lagrange (mu_k) para las restricciones de potencia."""
    mu_k = mu_k.detach()
    mu_k_update = eps * torch.mean(power_constr, dim = 0)
    mu_k = mu_k + mu_k_update
    mu_k = torch.max(mu_k, torch.tensor(0.0))
    return mu_k


def get_rates(phi, H, sigma, p0=0.01):
    batch_size, num_links, num_channels = phi.shape
    diagH = torch.diagonal(H, dim1=1, dim2=2) 
    
    total_snr_effective = torch.zeros((batch_size, num_links), device=phi.device)

    for ch in range(num_channels):
        p_ch = phi[:, :, ch] 
        
        signal = diagH * p_ch
        
        total_received_power = torch.matmul(H, p_ch.unsqueeze(-1)).squeeze(-1)
        
        interference_same_channel = total_received_power - signal
        
        penalty = (interference_same_channel * p_ch) / p0
        
        denom = sigma + penalty
        
        snr_ch = signal / (denom + 1e-12)
        
        total_snr_effective += snr_ch

    rates = torch.log1p(total_snr_effective)

    return rates

