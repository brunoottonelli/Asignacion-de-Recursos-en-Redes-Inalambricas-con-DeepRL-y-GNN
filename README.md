<div align="center">
  <img src="logo.png" alt="NetROML Logo" width="400"/>
  <h1>🌐 NetROML Framework</h1>
  <p><strong>Optimización de Recursos Inalámbricos mediante Graph Neural Networks (GNNs) y Deep Reinforcement Learning (DRL)</strong></p>

  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](#)
  [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](#)
  [![PyG](https://img.shields.io/badge/PyG-Graph_Neural_Networks-3D106B.svg?style=for-the-badge)](#)
  [![Gymnasium](https://img.shields.io/badge/Gymnasium-RL_Environments-000000.svg?style=for-the-badge)](#)
</div>

---

## 📖 Resumen del Proyecto

Este repositorio contiene el framework de investigación **NetROML**, diseñado para optimizar de manera descentralizada y autónoma los recursos en redes inalámbricas complejas. La topología de la red se modela como un **grafo espacial** que alimenta a una **Graph Neural Network (GNN)**, y mediante **Deep Reinforcement Learning (DRL)**, los agentes aprenden políticas de decisión eficientes para mitigar la interferencia y maximizar el rendimiento global.

El repositorio está estructurado para resolver **dos problemas**:

1. **ARPP (Asignación de Recursos en redes de enlaces Punto a Punto)**: Optimización estática del canal y potencia de transmisión en redes representadas como grafos de enlaces.
2. **ARCD (Asignación de Recursos con Clientes Dinámicos)**: Optimización en tiempo real sobre redes dinámicas que simulan la llegada y salida de clientes.

---

## 📂 Estructura General del Repositorio

El espacio de trabajo se organiza de la siguiente manera:

```text
.
├── Asignación de Recursos en Redes de Enlaces Punto a Punto (ARPP)/  # Problema 1: Enlaces estáticos
│   ├── Version1/                       # Canal únicamente (REINFORCE)
│   ├── Version2/                       # Canal y Potencia Conjuntos (REINFORCE)
│   └── Version1_Sanity_checks/          # Pruebas de funcionamiento
│
├── Asignación de Recursos con Clientes Dinámicos (ARCD)/             # Problema 2: Clientes dinámicos
│   ├── A2C/                            # Implementación de Advantage Actor-Critic
│   ├── PPO/                            # Implementación de Proximal Policy Optimization
│   ├── REINFORCE/                      # Implementación de REINFORCE
│   ├── data/                           # Utilidades de carga y procesamiento de datos
│   ├── model/                          # Entorno de red y modelos de GNN
│   ├── simulation/                     # Simulación física de Wi-Fi y arribo de clientes
│   ├── utils/                          # Encodings posicionales
│   └── evaluate_synthetic_weighted.py  # Script de evaluación cruzada de baselines
│
├── logo.png                            # Logotipo del framework
└── README.md                           # Documentación principal (este archivo)
```

---

## Problema 1: Asignación de Recursos en Redes de Enlaces Punto a Punto (ARPP)

### ❓ ¿Qué hace?
Resuelve la interferencia en redes estáticas modelando la topología como un **grafo de interferencia**, donde cada nodo del grafo representa un **enlace inalámbrico** (par AP-AP) y las aristas representan el canal físico de interferencia mutua. El objetivo es asignar frecuencias (canales) y potencias de transmisión mediante el algoritmo **REINFORCE** para maximizar el *Sum Rate* total.

* **Versión 1**: Los enlaces tienen potencia fija y la GNN solo optimiza los canales en bandas de 2.4GHz / 5GHz.
* **Versión 2**: Optimización conjunta de la tupla **(Canal, Potencia de Transmisión)** (ej. 100%, 50%, o Apagado).

### 🛠️ ¿Qué hacer? (Instrucciones de Uso)

#### 1. Entrenar el Modelo
Accede a la carpeta de la versión deseada (`Version1` o `Version2`) e inicia el script de entrenamiento indicando el edificio e hiperparámetros:
```bash
cd "Asignación de Recursos en Redes de Enlaces Punto a Punto (ARPP)/Version1/train"
python train.py --building_id 856 --b5g 0 --num_links 20 --epochs 200 --batch_size 32
```
*(Para la Versión 2, usa el mismo comando dentro de `Version2/train/` pero ajustando las epochs recomendadas a `700`)*.

#### 2. Correr Baselines Comparativos
Para generar las métricas de control (Asignación Aleatoria y Potencia Fija):
```bash
cd "../evaluation"
python final_baseline.py
```

#### 3. Evaluar Generalización en Nuevos Edificios
Para comprobar la capacidad de adaptación de la GNN en mapas/topologías de edificios que no vio en el entrenamiento:
```bash
python final_generalization.py
```

#### 4. Graficar y Visualizar Resultados
Para cruzar las curvas de convergencia de REINFORCE vs Baselines:
```bash
python plot_comparison.py
```

---

##  Problema 2: Asignación de Recursos con Clientes Dinámicos (ARCD)

### ❓ ¿Qué hace?
Aborda un escenario realista donde los clientes de la red Wi-Fi **llegan y se van dinámicamente** a lo largo del tiempo. Simula la física de propagación de señales y el comportamiento temporal del tráfico. Para ello, implementa algoritmos de DRL avanzados para coordinar canales y potencias de APs en tiempo real.

Admite tres arquitecturas de aprendizaje:
* **REINFORCE**
* **A2C (Advantage Actor-Critic)**
* **PPO (Proximal Policy Optimization)**

### 🛠️ ¿Qué hacer? (Instrucciones de Uso)

#### 1. Entrenamiento de Algoritmos
Puedes entrenar la red usando cualquiera de los tres enfoques disponibles. Por ejemplo, para entrenar con **PPO**:
```bash
cd "Asignación de Recursos con Clientes Dinámicos (ARCD)/PPO"
python ppo_train_synthetic.py --building_id 990 --episodes 6000 --device cpu
```
Para entrenar con **A2C**:
```bash
cd "../A2C"
python a2c_train_synthetic.py --building_id 990 --episodes 6000
```
Para entrenar con **REINFORCE**:
```bash
cd "../REINFORCE"
python reinforce_train_synthetic.py --building_id 990 --episodes 6000
```

#### 2. Evaluación Multivariable de Baselines (Coloración vs GNN)
Una vez entrenado un modelo (ej. PPO), puedes evaluar su desempeño comparativo frente a un baseline estático heurístico (**Graph Coloring**) y dinámico (**Random**) en una lista de edificios:
```bash
cd ..
python evaluate_synthetic_weighted.py --building_id 990 44 211 --episodes 50 --save_dir "outputs/synthetic_models" --exp_name "ase_completo"
```
Este script cargará el modelo guardado en `best_model_ppo.pt`, ejecutará las simulaciones y generará gráficos comparativos absolutos y relativos (boxplots) en el directorio `outputs/synthetic_models/ase_completo/evals/ajustados/`.

---

## ⚙️ Requisitos y Configuración de Rutas

### Instalar Dependencias
Se recomienda el uso de un entorno virtual para instalar los paquetes necesarios:
```bash
pip install torch numpy networkx matplotlib pandas scipy gymnasium
pip install torch_geometric
```

### Rutas de Datos Clave
Los conjuntos de datos (archivos `.pkl` y matrices de atenuación) deben estar organizados de acuerdo a lo que asume el código:
* **ARPP**: Espera las topologías en `~/wireless-learning/graphs/`
* **ARCD**: Utiliza los archivos ubicados en la carpeta `data/` del subdirectorio (ej. `data/matrices_atenuacion/`, `data/from_mac_to_building_id.csv`).

