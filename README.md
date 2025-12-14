# ResNet-18 para Clasificación de Imágenes Médicas con MedMNIST - TFM Laura Monne

Repositorio oficial: [https://github.com/LauraMonne/TFM_Laura_Monne](https://github.com/LauraMonne/TFM_Laura_Monne)

Este proyecto implementa una arquitectura **ResNet-18** estándar para clasificación de imágenes médicas usando los datasets **BloodMNIST**, **RetinaMNIST** y **BreastMNIST** del repositorio **MedMNIST**.

## Descripción del Proyecto

El proyecto incluye:
- Entrenamiento de **modelos ResNet-18 independientes por dominio biomédico**
- Clasificación supervisada sobre datasets MedMNIST
- Generación de explicaciones post-hoc mediante métodos XAI:
  - Grad-CAM
  - Grad-CAM++
  - Integrated Gradients
  - Saliency Maps
- Evaluación cuantitativa de la explicabilidad con la librería **Quantus**
- Análisis cualitativo y cuantitativo reproducible mediante notebooks

Este repositorio está diseñado para reproducibilidad científica, en coherencia con la memoria del TFM.

## Arquitectura ResNet-18

La arquitectura base es **ResNet-18** [He et al., 2016], adaptada dinámicamente al número de clases de cada dataset.
Componentes principales:
- Capa inicial: Conv2D (7×7) + BatchNorm + ReLU + MaxPooling
- Cuatro bloques residuales:
  - Layer1: 2 bloques, 64 canales
  - Layer2: 2 bloques, 128 canales
  - Layer3: 2 bloques, 256 canales
  - Layer4: 2 bloques, 512 canales
- Global Average Pooling
- Capa fully connected final adaptada al número de clases del dataset

La capa layer4 se utiliza como capa objetivo para métodos CAM, siguiendo la práctica habitual en la literatura.

## Datasets Utilizados (MedMNIST v2)

| Dataset | Clases | Canales | Dominio |
|---------|----------------------|-------------------|---------------|
| BloodMNIST | 8 | RGB(3) | Hematología |
| RetinaMNIST | 5 | RGB(3) | Retinopatía diabética |
| BreastMNIST | 2 | Escala de grises (1) | Ecografía mamaria |

Los splits oficiales **train / validation / test** proporcionados por MedMNIST v2 se utilizan sin modificaciones.

## Instalación

### Requisitos

Las dependencias del proyecto están especificadas en `requirements.txt`.
Principales librerías:
- PyTorch
- torchvision
- medmnist
- numpy
- matplotlib
- scikit-learn
- captum
- quantus

### Instalación
```bash
git clone https://github.com/LauraMonne/TFM_Laura_Monne.git
cd TFM_Laura_Monne
pip install -r requirements.txt
```

### Uso Rápido

1. **Preparar los datos**:
```bash
python prepare_data.py
```
Descarga y prepara los datasets MedMNIST, aplicando normalización y reescalado a 224×224 píxeles.

2. **Entrenamiento de modelos (uno por dataset)**:
```bash
python train.py --dataset blood
python train.py --dataset retina
python train.py --dataset breast
```
Se genera un checkpoint final por dataset:
- `results/best_model_blood.pth`
- `results/best_model_retina.pth`
- `results/best_model_breast.pth`

Además de métricas, curvas de entrenamiento y matrices de confusión.

3. **Generación de explicaciones XAI**:
```bash
python xai_explanations.py
```
Aplica métodos de explicabilidad post-hoc sobre un subconjunto controlado del conjunto de test y guarda:
- Mapas XAI (PNG)
- Metadatos estructurados (`explanations_results_<dataset>.json`)

4. **Evaluación cuantitativa de la explicabilidad**:
```bash
python quantus_evaluation.py --dataset blood
python quantus_evaluation.py --dataset retina
python quantus_evaluation.py --dataset breast
```

Calcula métricas de:
- Fidelidad
- Robustez
- Complejidad
- Localización

Los resultados se procesan posteriormente en `notebooks/quantus_eval.ipynb`

## Estructura del Proyecto

```
TFM_Laura_Monne/
│
├── data/                     # Datasets MedMNIST
├── models/
│   └── resnet18_adaptive.py
├── results/
│   ├── best_model_blood.pth
│   ├── best_model_retina.pth
│   └── best_model_breast.pth
├── outputs/
│   ├── gradcam/
│   ├── gradcampp/
│   ├── integrated_gradients/
│   ├── saliency/
│   └── explanations_results_<dataset>.json
├── notebooks/
│   ├── traint18.ipynb
│   ├── xai_analisis.ipynb
│   └── quantus_eval.ipynb
├── prepare_data.py
├── data_utils.py
├── train.py
├── xai_explanations.py
├── quantus_evaluation.py
├── XAI_README.md
├── requirements.txt
└── README.md
```

## Reproducibilidad

Todos los experimentos pueden reproducirse fijando la semilla aleatoria:

```` bash
from train import set_seed
set_seed(42)
````
Los artefactos generados (modelos, explicaciones, métricas) están completamente trazados mediante ficheros JSON y scripts versionados.


##  Referencias Principales

- He et al., Deep Residual Learning for Image Recognition, CVPR 2016
- Selvaraju et al., Grad-CAM, ICCV 2017
- Chattopadhay et al., Grad-CAM++, WACV 2018
- Ma et al., MedMNIST v2, Scientific Data 2022
- Hedström et al., Quantus, JMLR 2023

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

de un trabajo académico sobre clasificación de imágenes médicas usando redes neuronales convolucionales.
