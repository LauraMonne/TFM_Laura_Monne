# VGG16 Small - Implementación y Comparación con ResNet18

## Descripción

Este módulo implementa un modelo **VGG16 reducido** (VGG16 Small) para clasificación de imágenes médicas, diseñado para ser comparado con ResNet18 en términos de:
- Métricas de clasificación (accuracy, loss)
- Número de parámetros
- Métricas de explicabilidad (XAI)

## Arquitectura VGG16 Small

La arquitectura VGG16 Small es una versión reducida del VGG16 original, con las siguientes características:

### Bloques Convolucionales

| Bloque | Capas Conv | Filtros | Salida tras MaxPool |
|--------|-----------|---------|---------------------|
| 1      | 2         | 32      | 112×112             |
| 2      | 2         | 64      | 56×56               |
| 3      | 3         | 128     | 28×28               |
| 4      | 3         | 256     | 14×14               |
| 5      | 3         | 256     | 7×7                 |

**Comparación con VGG16 original:**
- VGG16 original: 64 → 128 → 256 → 512 → 512 filtros
- VGG16 Small: 32 → 64 → 128 → 256 → 256 filtros (reducción ~50%)

### Características Principales

1. **Batch Normalization**: Añadida después de cada capa convolucional para mejorar la estabilidad del entrenamiento
2. **Adaptive Pooling**: Permite flexibilidad en el tamaño de entrada
3. **Capas Densas**: 
   - Linear (256×7×7 → 512) + ReLU + Dropout(0.5)
   - Linear (512 → 256) + ReLU + Dropout(0.5)
   - Linear (256 → num_classes)
4. **Soporte Multi-canal**: Arquitectura adaptativa para RGB (3 canales) y escala de grises (1 canal)

### Parámetros del Modelo

```
VGG16 Small:     ~3.7M parámetros
ResNet18:        ~11M parámetros
Reducción:       ~66%
```

## Uso

### 1. Entrenar VGG16 Small

```bash
# BloodMNIST (8 clases)
python train_vgg16.py --dataset blood

# RetinaMNIST (5 clases)
python train_vgg16.py --dataset retina

# BreastMNIST (2 clases)
python train_vgg16.py --dataset breast
```

**Opciones adicionales:**
```bash
# Especificar número de épocas
python train_vgg16.py --dataset blood --epochs 100
```

### 2. Comparar con ResNet18

```bash
# Generar comparación completa
python compare_models.py --dataset blood
python compare_models.py --dataset retina
python compare_models.py --dataset breast
```

Este script genera:
- Reporte JSON con métricas comparativas
- Gráficas de curvas de entrenamiento
- Preparación para análisis de explicabilidad

### 3. Análisis de Explicabilidad

Para comparar métricas de explicabilidad:

```bash
# 1. Generar explicaciones XAI (si aún no existen)
python xai_explanations.py --dataset blood --model vgg16

# 2. Evaluar con Quantus
python quantus_evaluation.py --dataset blood --model vgg16

# 3. Comparar resultados en notebook
jupyter notebook "3. Quantus_eval.ipynb"
```

## Archivos Generados

### Entrenamiento
```
results/
├── best_model_vgg16_blood.pth          # Mejor checkpoint (BloodMNIST)
├── best_model_vgg16_retina.pth         # Mejor checkpoint (RetinaMNIST)
├── best_model_vgg16_breast.pth         # Mejor checkpoint (BreastMNIST)
├── training_results_vgg16_blood.json   # Métricas y config
├── training_history_vgg16_blood.png    # Curvas de entrenamiento
├── confusion_matrix_vgg16_blood.png    # Matriz de confusión
└── preds_test_vgg16_blood.npz          # Predicciones en test
```

### Comparación
```
results/
├── comparison_report_blood.json              # Reporte JSON
├── comparison_vgg16_resnet18_blood.png      # Gráficas comparativas
└── comparison_report_retina.json            # (uno por dataset)
```

## Configuración de Entrenamiento

### BloodMNIST (8 clases)
```python
{
  "batch_size": 64,
  "epochs": 50,
  "learning_rate": 1e-3,
  "weight_decay": 1e-4,
  "early_stopping_patience": 10,
  "use_class_weights": True
}
```

### RetinaMNIST (5 clases)
```python
{
  "batch_size": 32,
  "epochs": 50,
  "learning_rate": 1e-4,
  "weight_decay": 2e-4,
  "early_stopping_patience": 10,
  "use_focal_loss": True,
  "focal_gamma": 2.0
}
```

### BreastMNIST (2 clases)
```python
{
  "batch_size": 32,
  "epochs": 50,
  "learning_rate": 5e-4,
  "weight_decay": 2e-4,
  "early_stopping_patience": 12,
  "use_focal_loss": True,
  "focal_gamma": 2.0
}
```

## Comparación VGG16 vs ResNet18

### Ventajas de VGG16 Small

1. **Menor complejidad**: ~66% menos parámetros que ResNet18
2. **Arquitectura más simple**: Más fácil de interpretar
3. **Explicabilidad**: Estructura secuencial puede facilitar análisis XAI
4. **Menor costo computacional**: Menos memoria y tiempo de entrenamiento

### Ventajas de ResNet18

1. **Conexiones residuales**: Mejor flujo de gradientes
2. **Mejor para datasets muy profundos**: Evita degradación
3. **Transfer learning**: Disponible pre-entrenado en ImageNet

## Métricas de Explicabilidad

Las siguientes métricas de Quantus pueden usarse para comparar ambos modelos:

### Fidelidad
- **Faithfulness Correlation**: Correlación entre importancia y cambio en predicción
- **Pixel Flipping**: Degradación al eliminar píxeles importantes

### Robustez
- **Local Lipschitz Estimate**: Sensibilidad a perturbaciones
- **Max-Sensitivity**: Variación máxima de explicaciones

### Localización
- **Pointing Game**: Precisión en localizar región relevante
- **Attribution Localization**: Concentración de atribuciones

### Complejidad
- **Sparseness**: Grado de dispersión de la explicación
- **Complexity**: Número de características relevantes

## Estructura de Archivos

```
TFM_Laura_Monne/
├── vgg16.py                    # Implementación del modelo VGG16 Small
├── train_vgg16.py             # Script de entrenamiento
├── compare_models.py          # Script de comparación
├── VGG16_README.md            # Esta documentación
├── results/                   # Resultados de entrenamiento y comparación
├── outputs/                   # Explicaciones XAI (si se generan)
└── data/                      # Datasets MedMNIST
```

## Requisitos

Las dependencias están especificadas en `requirements.txt`:
```
torch>=2.0.0
torchvision>=0.15.0
medmnist>=3.0.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
tqdm>=4.64.0
```

## Pipeline Completo de Comparación

```bash
# 1. Preparar datos (si aún no está hecho)
python prepare_data.py

# 2. Entrenar ambos modelos
python train.py --dataset blood           # ResNet18
python train_vgg16.py --dataset blood     # VGG16 Small

# 3. Generar comparación
python compare_models.py --dataset blood

# 4. (Opcional) Análisis XAI
python xai_explanations.py --dataset blood
python quantus_evaluation.py --dataset blood

# 5. (Opcional) Análisis en notebook
jupyter notebook "3. Quantus_eval.ipynb"
```

## Referencias

- **VGG Original**: Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", ICLR 2015
- **ResNet**: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- **MedMNIST**: Yang et al., "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification", 2021
- **Quantus**: Hedström et al., "Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations", JMLR 2023

## Contacto y Contribuciones

Este módulo forma parte del TFM de Laura Monné sobre clasificación de imágenes biomédicas y explicabilidad en Deep Learning.

Repositorio: https://github.com/LauraMonne/TFM_Laura_Monne
