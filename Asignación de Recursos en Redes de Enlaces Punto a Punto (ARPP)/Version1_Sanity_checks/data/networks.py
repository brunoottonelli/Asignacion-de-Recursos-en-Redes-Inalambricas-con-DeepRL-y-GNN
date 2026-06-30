import numpy as np
import scipy.io
import pdb
from scipy import sparse
import pickle


def build_adhoc_network(pl, d_horizontal=8.0, h_vertical=1.0):
    """
    Estructura fija
    Transmisores (1, 2, 3) abajo, Receptores (4, 5, 6) arriba.
    """
    num_channels = 3 
    # Posiciones fijas (x, y)
    # Transmisores en la fila de abajo (y=0)
    transmitters = np.array([
        [0, 0],              
        [d_horizontal , 0],   
        [2*d_horizontal, 0]  
    ])

    # Receptores en la fila de arriba (y=h_vertical)
    receivers = np.array([
        [0, h_vertical],             
        [d_horizontal  , h_vertical],  
        [2*d_horizontal, h_vertical] 
    ])

    L = np.zeros((num_channels, num_channels))

    for i in range(num_channels):
        for j in range(num_channels):
            dist = np.linalg.norm(transmitters[i,:] - receivers[j,:])
            L[i,j] = np.power(dist, -pl)


    L[0, 2] = 0.005 # Tx 1 interfiere fuerte a Rx 3
    L[2, 0] = 0.05 # Tx 3 interfiere un poco menos a Rx 1

    return L


    