# ResNet-19 para Clasificación de Imágenes Médicas con MedMNIST - TFM Laura Monne

Repositorio oficial: [https://github.com/LauraMonne/TFM_Laura_Monne](https://github.com/LauraMonne/TFM_Laura_Monne)

Este proyecto implementa una arquitectura **ResNet-19** personalizada para clasificación de imágenes médicas usando los datasets **BloodMNIST**, **RetinaMNIST** y **BreastMNIST** del repositorio **MedMNIST**.

## 📋 Descripción del Proyecto

El proyecto incluye:
- **Arquitectura ResNet-19**: Implementación personalizada con bloques residuales
- **Datasets MedMNIST**: BloodMNIST, RetinaMNIST y BreastMNIST
- **Data Augmentation**: Transformaciones para mejorar el rendimiento
- **Entrenamiento Completo**: Scripts de entrenamiento con validación y evaluación
- **Visualización de resultados**: métricas, gráficas y matriz de confusión.

## 🏗️ Arquitectura ResNet-19

La arquitectura implementada incluye:
- **Capa inicial**: Conv2d(7x7) + BatchNorm + ReLU + MaxPool
- **3 Capas residuales**: 
  - Layer 1: 2 bloques BasicBlock con 64 canales
  - Layer 2: 2 bloques BasicBlock con 128 canales  
  - Layer 3: 2 bloques BasicBlock con 256 canales
- **Capa final**: AdaptiveAvgPool + Linear(256 → 15 clases)
- **Total de parámetros**: ~5.6M parámetros entrenables

## 📊 Datasets Utilizados

| Dataset | Muestras Entrenamiento | Muestras Validación | Muestras Test | Clases | Canales |
|---------|----------------------|-------------------|---------------|--------|---------|
| BloodMNIST | 11,959 | 1,712 | 3,421 | 8 | RGB (3) |
| RetinaMNIST | 1,080 | 120 | 400 | 5 | RGB (3) |
| BreastMNIST | 546 | 78 | 156 | 2 | Escala de grises (1) |
| **Total Combinado** | **13,585** | **1,910** | **3,977** | **15** | **Mixto** |

## 🚀 Instalación y Uso

### Requisitos
```bash
torch>=2.0.0
torchvision>=0.15.0
medmnist>=2.1.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
tqdm>=4.64.0
tensorboard>=2.10.0
Pillow>=9.0.0
```

### Instalación
```bash
git clone https://github.com/tu-usuario/medmnist-resnet19.git
cd medmnist-resnet19
pip install -r requirements.txt
```

### Uso Rápido

1. **Preparar los datos**:
```bash
python prepare_data.py
```

2. **Probar la implementación** (3 épocas):
```bash
python quick_test.py
```

3. **Entrenamiento completo**:
```bash
python train.py --dataset retina --epochs 20 --batch-size 64 --lr 1e-3 --weight-decay 1e-4 --seed 42

```

## 📁 Estructura del Proyecto

```
medmnist_resnet19_project/
├── prepare_data.py          # Preparación y carga de datasets
├── resnet19.py             # Implementación de ResNet-19
├── train.py                # Script de entrenamiento completo
├── quick_test.py           # Prueba rápida (3 épocas)
├── data_utils.py           # Utilidades para manejo de datos
├── dataset_wrapper.py      # Wrapper para conversión de labels
├── requirements.txt        # Dependencias del proyecto
├── README.md              # Este archivo
├── data/                  # Datasets descargados
│   ├── bloodmnist.npz
│   ├── retinamnist.npz
│   └── breastmnist.npz
└── outputs/               # Resultados del entrenamiento
    ├── training_history.png
    ├── confusion_matrix.png
    ├── best_model.pth
    └── training_results.json
```

## 🔧 Características Técnicas

### Data Augmentation
- **Entrenamiento**: Redimensionamiento, flip horizontal, rotación, color jitter
- **Validación/Test**: Solo redimensionamiento y normalización
- **Manejo de canales**: Conversión automática de escala de grises a RGB

### Optimización
- **Optimizador**: AdamW con weight decay
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Patience de 10 épocas
- **Batch Size**: 32 (configurable)

### Métricas de Evaluación
- Accuracy por época
- Loss de entrenamiento y validación
- Matriz de confusión
- Reporte de clasificación detallado

## 📈 Resultados

En la prueba rápida (3 épocas):
- **Precisión de entrenamiento**: ~72.5%
- **Precisión de validación**: ~79.2%
- **Tiempo de entrenamiento**: ~1.2 horas (CPU)

## 🧠 Reproducibilidad
```` bash
from train import set_seed
set_seed(42)
````

## 🎯 Próximos Pasos

- [ ] Entrenamiento completo con más épocas
- [ ] Optimización de hiperparámetros
- [ ] Comparación con otras arquitecturas
- [ ] Análisis de errores por dataset
- [ ] Implementación de técnicas avanzadas (mixup, cutmix)



## 📚 Referencias

- [MedMNIST: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification](https://medmnist.com/)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👥 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📞 Contacto

Para preguntas o sugerencias, por favor abre un issue en el repositorio.

---

**Nota**: Este proyecto es parte de un trabajo académico sobre clasificación de imágenes médicas usando redes neuronales convolucionales.
