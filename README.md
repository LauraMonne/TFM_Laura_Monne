# Clasificación de Imágenes Médicas con MedMNIST - TFM Laura Monne

Repositorio oficial: [https://github.com/LauraMonne/TFM_Laura_Monne](https://github.com/LauraMonne/TFM_Laura_Monne)

Este proyecto implementa arquitecturas **ResNet-18** y **VGG16 Small** para clasificación de imágenes médicas usando los datasets **BloodMNIST**, **RetinaMNIST** y **BreastMNIST** del repositorio **MedMNIST**.

## Descripción del Proyecto

El proyecto incluye:
- Entrenamiento de **modelos ResNet-18 y VGG16 Small independientes por dominio biomédico**
- Clasificación supervisada sobre datasets MedMNIST
- **Comparación de arquitecturas** en términos de parámetros y métricas de clasificación
- Generación de explicaciones post-hoc mediante métodos XAI:
  - Grad-CAM
  - Grad-CAM++
  - Integrated Gradients
  - Saliency Maps
- Evaluación cuantitativa de la explicabilidad con la librería **Quantus**
- **Comparación de métricas de explicabilidad** entre ambos modelos
- Análisis cualitativo y cuantitativo reproducible mediante notebooks

Este repositorio está diseñado para reproducibilidad científica, en coherencia con la memoria del TFM.

## Arquitecturas Implementadas

### ResNet-18

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
- ~11M parámetros

La capa layer4 se utiliza como capa objetivo para métodos CAM, siguiendo la práctica habitual en la literatura.

### VGG16 Small

Implementación de **VGG16 reducido** con Batch Normalization para comparar con ResNet-18:
- Cinco bloques convolucionales: 32 → 64 → 128 → 256 → 256 filtros
- Batch Normalization después de cada capa convolucional
- MaxPooling después de cada bloque
- Capas densas: 512 → 256 → num_classes con Dropout
- ~3.7M parámetros (~66% menos que ResNet18)

Ver documentación completa en [VGG16_README.md](VGG16_README.md)

## Datasets Utilizados (MedMNIST v2)

| Dataset       | Clases | Canales | Dominio                |
|--------------|--------|---------|------------------------|
| BloodMNIST   | 8      | RGB (3) | Hematología            |
| RetinaMNIST  | 5      | RGB (3) | Retinopatía diabética  |
| BreastMNIST  | 2      | 1 (gris)| Ecografía mamaria      |

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

### Ejecución del pipeline

1. **Preparación de los datos**:
```bash
python prepare_data.py
```
Descarga y prepara los datasets MedMNIST, aplicando normalización y reescalado a 224×224 píxeles.

2. **Entrenamiento de modelos (uno por dataset)**:

**ResNet-18:**
```bash
python train.py --dataset blood
python train.py --dataset retina
python train.py --dataset breast
```

**VGG16 Small:**
```bash
python train_vgg16.py --dataset blood
python train_vgg16.py --dataset retina
python train_vgg16.py --dataset breast
```

Se genera un checkpoint final por dataset y modelo:
- ResNet-18: `results/best_model_{dataset}.pth`
- VGG16 Small: `results/best_model_vgg16_{dataset}.pth`

Además de métricas, curvas de entrenamiento y matrices de confusión.

3. **Comparación de modelos**:
```bash
python compare_models.py --dataset blood
python compare_models.py --dataset retina
python compare_models.py --dataset breast
```
Genera reportes comparativos de arquitectura, métricas de clasificación y gráficas.

4. **Generación de explicaciones XAI**:
```bash
python xai_explanations.py
```
Aplica métodos de explicabilidad post-hoc sobre un subconjunto controlado del conjunto de test y guarda:
- Mapas XAI (PNG)
- Metadatos estructurados (`explanations_results_<dataset>.json`)

5. **Evaluación cuantitativa de la explicabilidad**:
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
├── data/                          # Datasets MedMNIST
├── results/
│   ├── best_model_blood.pth       # ResNet-18 checkpoints
│   ├── best_model_retina.pth
│   ├── best_model_breast.pth
│   ├── best_model_vgg16_blood.pth # VGG16 checkpoints
│   ├── best_model_vgg16_retina.pth
│   ├── best_model_vgg16_breast.pth
│   ├── comparison_report_*.json   # Comparaciones
│   └── comparison_vgg16_resnet18_*.png
├── outputs/
│   ├── gradcam/
│   ├── gradcampp/
│   ├── integrated_gradients/
│   ├── saliency/
│   └── explanations_results_<dataset>.json
├── notebooks/
│   ├── 1. Notebook entrenamiento ResNet-18.ipynb
│   ├── 2. XAI_analisis.ipynb
│   └── 3. Quantus_eval.ipynb
├── resnet18.py                    # Implementación ResNet-18
├── vgg16.py                       # Implementación VGG16 Small
├── train.py                       # Entrenamiento ResNet-18
├── train_vgg16.py                 # Entrenamiento VGG16
├── compare_models.py              # Comparación de modelos
├── prepare_data.py                # Preparación de datos
├── data_utils.py                  # Utilidades de datos
├── dataset_wrapper.py             # Wrapper de datasets
├── xai_explanations.py            # Generación de explicaciones
├── quantus_evaluation.py          # Evaluación Quantus
├── README.md                      # Esta documentación
├── VGG16_README.md                # Documentación VGG16
├── XAI_README.md                  # Documentación XAI
└── requirements.txt               # Dependencias
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

Este repositorio forma parte de un **trabajo académico sobre clasificación de imágenes biomédicas y explicabilidad en Deep Learning**, desarrollado como Trabajo Final de Máster.
