import pickle
import numpy as np
import os
from networks import build_adhoc_network
import torch

rn=100
rn1=100
torch.manual_seed(rn)
np.random.seed(rn1) 


num_graphs = 64000
num_links = 3
pl = 1e-5
i=0
graphs = []
for _ in range(num_graphs):
    pl = np.random.rand()
    L = build_adhoc_network(pl,9.0, 2.0)
    if i==2:
        print(L)
    graphs.append(L)
    i+=1


# Guarda el archivo en la carpeta local de datos de Sanity Checks
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'sc_graphs.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(graphs, f)

print(f"Guardados {num_graphs} grafos sintéticos de {num_links} linkscle en {output_path}")