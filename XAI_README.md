# 📊 Guía de Explicabilidad (XAI) - ResNet-18 MedMNIST

## 📋 Descripción

Este script implementa los métodos de explicabilidad descritos en la memoria del TFM, generando mapas XAI (Grad-CAM, Grad-CAM++, Integrated Gradients y Saliency) y dejando preparados los artefactos necesarios para su evaluación cuantitativa con Quantus en un notebook independiente.

## 🔧 Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Verificar instalación

```bash
python -c "import pytorch_grad_cam; from captum.attr import IntegratedGradients, Saliency; import quantus; print('✅ Todas las librerías instaladas')"
```

## 🚀 Uso

### Ejecutar explicabilidad completa

```bash
python xai_explanations.py
```

### Configuración

El script está configurado para:

- Cargar el modelo desde `results/best_model.pth`
- Generar explicaciones para un máximo de **500 muestras** del conjunto de test,
  estratificadas por dataset:
  - 300 de BloodMNIST
  - 150 de RetinaMNIST
  - 50 de BreastMNIST
- Guardar resultados en `outputs/`

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

**Nota importante**: El script actual NO ejecuta la evaluación cuantitativa automáticamente. La función `evaluate_with_quantus_stub()` solo informa sobre la disponibilidad de Quantus. La evaluación cuantitativa debe realizarse en un notebook dedicado usando los mapas generados por este script.

### Métricas a Evaluar (en notebook separado)

Para evaluar los mapas generados, puedes usar Quantus en un notebook con las siguientes métricas:

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

### Cómo Evaluar con Quantus

1. Ejecutar este script para generar los mapas: `python xai_explanations.py`
2. Crear un notebook Jupyter para la evaluación cuantitativa
3. Cargar los mapas generados desde `outputs/`
4. Usar la librería Quantus para evaluar cada método según las 5 dimensiones

**Ejemplo de evaluación** (en notebook):
```python
import quantus
# Cargar mapas generados
# Evaluar con las métricas definidas
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
└── explanations_results.json   # Metadatos de explicaciones
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

### Error: "too many indices for tensor of dimension 1"
- **Causa**: Problema con el callback de Grad-CAM (ya corregido en versión actual)
- **Solución**: Asegúrate de tener la versión más reciente del script desde GitHub

### Error en evaluación Quantus
- **Nota**: La evaluación cuantitativa no se ejecuta automáticamente en este script
- **Solución**: Realizar la evaluación en un notebook dedicado usando los mapas generados

## 📚 Referencias

- [PyTorch Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)
- [Captum](https://captum.ai/)
- [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [Integrated Gradients Paper](https://arxiv.org/abs/1703.01365)

## 📝 Notas

- El modelo ResNet-18 adaptativo maneja automáticamente imágenes RGB y escala de grises
- Las explicaciones se generan para la clase predicha por el modelo
- **La evaluación cuantitativa con Quantus NO se ejecuta automáticamente** en este script
  - El script solo genera los mapas de explicabilidad
  - La evaluación cuantitativa debe hacerse en un notebook dedicado
- Se recomienda usar GPU para acelerar la generación de explicaciones
- El callback de Grad-CAM está corregido para manejar correctamente tensores 1D y 2D

## 🔄 Próximos Pasos

1. **Ejecutar el script**: `python xai_explanations.py` para generar mapas
2. **Crear notebook de evaluación**: Implementar evaluación cuantitativa con Quantus
3. Analizar resultados de Quantus para comparar métodos
4. Generar visualizaciones comparativas
5. Incorporar resultados en la memoria del TFM

