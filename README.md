# Redes Neuronales II — Backpropagation, funciones de activación y clasificación binaria / multiclase

Actividad de la especialización en Inteligencia Artificial. Implementación de redes neuronales **perceptrón, de una capa y multicapa** con backpropagation (NumPy), clasificación **binaria** sobre datos reales y clasificación **multiclase en MNIST** con **TensorFlow + Keras** y **PyTorch** (sólo capas densas, **sin redes convolucionales**).

## Notebooks

| # | Notebook | Contenido | Colab |
|---|---|---|---|
| 01 | [`01_perceptron_capas_backprop.ipynb`](notebooks/01_perceptron_capas_backprop.ipynb) | Funciones de activación (Sigmoide, ReLU, Tanh, Softmax) y derivadas · Perceptrón · Red de una capa · MLP con backpropagation · *Gradient checking* · XOR y `make_moons` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/redes-neuronales-ii/blob/main/notebooks/01_perceptron_capas_backprop.ipynb) |
| 02 | [`02_clasificacion_binaria_breast_cancer.ipynb`](notebooks/02_clasificacion_binaria_breast_cancer.ipynb) | Clasificación binaria (Breast Cancer Wisconsin) · Sigmoide vs ReLU · Validación cruzada de hiperparámetros · *Early stopping* · Métricas y ROC · Réplica en Keras | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/redes-neuronales-ii/blob/main/notebooks/02_clasificacion_binaria_breast_cancer.ipynb) |
| 03 | [`03_mnist_tensorflow_keras.ipynb`](notebooks/03_mnist_tensorflow_keras.ipynb) | MNIST multiclase con MLP 784-256-128-10 en TensorFlow + Keras | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/redes-neuronales-ii/blob/main/notebooks/03_mnist_tensorflow_keras.ipynb) |
| 04 | [`04_mnist_pytorch.ipynb`](notebooks/04_mnist_pytorch.ipynb) | Réplica del MLP en PyTorch con bucle de entrenamiento explícito · Comparación Keras vs PyTorch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/redes-neuronales-ii/blob/main/notebooks/04_mnist_pytorch.ipynb) |

> Ejecute los notebooks en orden. El 04 compara con los resultados que guarda el 03 (`results/keras_mnist.json`), por lo que en Colab conviene ejecutar el 03 antes en la misma sesión; si no, el 04 funciona igual y omite la comparación.

## Resultados principales

**Clasificación binaria: Breast Cancer, conjunto de test (114 muestras)**

| Modelo | Exactitud | F1 | Recall malignos | ROC-AUC |
|---|---|---|---|---|
| Perceptrón (NumPy) | 0.9474 | 0.9583 | 0.9286 | 0.9861 |
| Red de una capa (NumPy) | **0.9825** | **0.9861** | **0.9762** | **0.9964** |
| MLP 30-16-1 ReLU optimizado (NumPy) | 0.9649 | 0.9722 | 0.9524 | 0.9884 |
| MLP 30-16-1 ReLU (Keras) | 0.9386 | 0.9496 | 0.9762 | 0.9924 |

**Clasificación multiclase: MNIST, conjunto de test (10 000 imágenes)**

| Framework | Parámetros | Épocas (early stopping) | Exactitud test | Errores |
|---|---|---|---|---|
| TensorFlow + Keras | 235 146 | 12 | **98.22 %** | 178 |
| PyTorch | 235 146 | 10 | 97.96 % | 204 |

Backpropagation manual verificado numéricamente: error relativo entre el gradiente analítico y el numérico del orden de 1e-10.

## Estructura

```
redes-neuronales-ii/
├── notebooks/        # 4 notebooks ejecutados (con salidas)
│   ├── figures/      # gráficas generadas por los notebooks
│   └── results/      # métricas (json/csv) y modelos entrenados (.keras, .pt)
├── src/nn_numpy.py   # Perceptrón, red de una capa y MLP NumPy reutilizables
├── docs/             # documento técnico (PDF)
└── requirements.txt
```

## Ejecución local

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

MNIST se descarga automáticamente (`keras.datasets.mnist` y `torchvision.datasets.MNIST`). Todas las semillas están fijadas (`SEED = 42`).
