# 📊 Guía de Explicabilidad (XAI) - ResNet-18 MedMNIST

## 📋 Descripción

Este script implementa métodos de explicabilidad según la memoria del TFM, aplicando diferentes técnicas XAI y evaluándolas cuantitativamente con Quantus.

## 🔧 Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Verificar instalación

```bash
python -c "import grad_cam; import captum; import quantus; print('✅ Todas las librerías instaladas')"
```

## 🚀 Uso

### Ejecutar explicabilidad completa

```bash
python xai_explanations.py
```

### Configuración

El script está configurado para:
- Cargar el modelo desde `results/best_model.pth`
- Generar explicaciones para 20 muestras por defecto
- Guardar resultados en `outputs/`

### Cambiar número de muestras

Editar línea 466 en `xai_explanations.py`:
```python
num_samples = 20  # Cambiar a número deseado
```

## 📊 Métodos Implementados

### 1. Grad-CAM
- **Librería**: pytorch-grad-cam
- **Descripción**: Identifica regiones importantes usando gradientes de la última capa convolucional
- **Salida**: `outputs/gradcam/`

### 2. Grad-CAM++
- **Librería**: pytorch-grad-cam
- **Descripción**: Versión mejorada de Grad-CAM con mejor localización
- **Salida**: `outputs/gradcampp/`

### 3. Integrated Gradients (IG)
- **Librería**: Captum
- **Descripción**: Calcula contribución de píxeles a lo largo de un trayecto interpolado
- **Salida**: `outputs/integrated_gradients/`

### 4. Saliency Maps (Vanilla Saliency)
- **Librería**: Captum
- **Descripción**: Muestra píxeles con mayor impacto directo sobre la predicción
- **Salida**: `outputs/saliency/`

## 📈 Evaluación Cuantitativa (Quantus)

### Métricas Evaluadas

1. **Faithfulness (Fidelidad)**
   - Métrica: Faithfulness Correlation
   - Mide si la explicación refleja el comportamiento interno del modelo
   - Rango: [-1, 1] (mayor es mejor)

2. **Robustness (Robustez)**
   - Métrica: Average Sensitivity
   - Evalúa estabilidad ante perturbaciones leves
   - Rango: [0, ∞] (menor es mejor)

3. **Complexity (Complejidad)**
   - Métrica: Entropy
   - Estima simplicidad de la explicación
   - Rango: [0, ∞] (menor es mejor para interpretabilidad)

4. **Randomization (Aleatorización)**
   - Métrica: Randomization Test
   - Mide dependencia de la explicación respecto a semillas aleatorias
   - Rango: [-1, 1] (mayor es mejor)

5. **Localization (Localización)**
   - Métrica: Region Perturbation
   - Determina precisión espacial de la explicación
   - Rango: [0, 1] (mayor es mejor)

### Resultados

Los resultados de Quantus se guardan en:
- `outputs/quantus_evaluation.json`

Formato:
```json
{
  "gradcam": {
    "faithfulness": {"mean": 0.75, "std": 0.12},
    "robustness": {"mean": 0.15, "std": 0.05},
    "complexity": {"mean": 2.3, "std": 0.4},
    "randomization": {"mean": 0.82, "std": 0.08},
    "localization": {"mean": 0.68, "std": 0.15}
  },
  "integrated_gradients": {...},
  "saliency": {...}
}
```

## 📁 Estructura de Salida

```
outputs/
├── gradcam/                    # Mapas Grad-CAM
│   └── img_*_class_*.png
├── gradcampp/                  # Mapas Grad-CAM++
│   └── img_*_class_*.png
├── integrated_gradients/       # Mapas Integrated Gradients
│   └── img_*_class_*.png
├── saliency/                   # Mapas Saliency
│   └── img_*_class_*.png
├── explanations_results.json   # Metadatos de explicaciones
└── quantus_evaluation.json     # Resultados de evaluación cuantitativa
```

## 🐛 Solución de Problemas

### Error: "Modelo no encontrado"
- **Causa**: No se ha entrenado el modelo
- **Solución**: Ejecutar `python train.py` primero

### Error: "Grad-CAM no disponible"
- **Causa**: Librería no instalada
- **Solución**: `pip install grad-cam`

### Error: "Captum no disponible"
- **Causa**: Librería no instalada
- **Solución**: `pip install captum`

### Error: "Quantus no disponible"
- **Causa**: Librería no instalada
- **Solución**: `pip install quantus`

### Error en evaluación Quantus
- **Causa**: Puede ser por memoria insuficiente o formato de datos
- **Solución**: Reducir `num_samples` o verificar formato de imágenes

## 📚 Referencias

- [PyTorch Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)
- [Captum](https://captum.ai/)
- [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [Integrated Gradients Paper](https://arxiv.org/abs/1703.01365)

## 📝 Notas

- El modelo ResNet-18 adaptativo maneja automáticamente imágenes RGB y escala de grises
- Las explicaciones se generan para la clase predicha por el modelo
- La evaluación con Quantus puede tardar varios minutos según el número de muestras
- Se recomienda usar GPU para acelerar la generación de explicaciones

## 🔄 Próximos Pasos

1. Analizar resultados de Quantus para comparar métodos
2. Generar visualizaciones comparativas
3. Incorporar resultados en la memoria del TFM
4. Optimizar parámetros de evaluación según necesidades

