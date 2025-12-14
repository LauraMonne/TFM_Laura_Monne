# Guía de Explicabilidad (XAI) - ResNet-18 MedMNIST

## Descripción

Este documento describe el uso del script `xai_explanations.py`, encargado de generar explicaciones visuales post-hoc mediante distintos métodos de explicabilidad (XAI) sobre modelos **ResNet-18 entrenados de forma independiente por dataset**.

Las explicaciones generadas se utilizan para:
- el análisis cualitativo presentado en el Capítulo 4 de la memoria del TFM,
- servir como entrada a la evaluación cuantitativa con **Quantus**, descrita en `QUANTUS_README.md`.

## Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Verificar instalación

```bash
python -c "import pytorch_grad_cam; from captum.attr import IntegratedGradients, Saliency; print('Librerías XAI disponibles')"
```

## Uso

### Generar explicaciones XAI

```bash
python xai_explanations.py
```
El script genera explicaciones post-hoc para cada uno de los siguientes dominios biomédicos:
- BloodMNIST
- RetinaMNIST
- BreastMNIST

Cada dataset se analiza de forma independiente, en coherencia con el entrenamiento de tres modelos distintos.

### Configuración

El script está configurado para:

- Cargar automáticamente los checkpoints entrenados:
  - `results/best_model_blood.pth`
  - `results/best_model_retina.pth`
  - `results/best_model_breast.pth`
- Aplicar explicabilidad sobre un **subconjunto representativo del conjunto de tes**t por dataset
- Generar explicaciones para la **clase predicha por el modelo**
- Guardar mapas de atribución y metadatos estructurados en la carpeta `outputs/`

El número de muestras explicadas se controla para equilibrar:
- representatividad,
- coste computacional,
- coherencia con la evaluación cuantitativa posterior.

## Métodos Implementados

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

## Evaluación Cuantitativa (Quantus)

**Nota importante**: El script `xai_explanations.py` NO ejecuta la evaluación cuantitativa automáticamente. Para ello se ha añadido `quantus_evaluation.py`, descrito en `QUANTUS_README.md`.

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

## Estructura de Salida

```
outputs/
├── gradcam/
├── gradcampp/
├── integrated_gradients/
├── saliency/
└── explanations_results_<dataset>.json
```
Cada fichero JSON documenta:
- índice de la imagen,
- dataset de origen,
- clase real y clase predicha,
- rutas a los mapas generados por cada método XAI.

Esto garantiza trazabilidad completa entre imagen, predicción y explicación.



## Referencias

- Selvaraju et al., Grad-CAM, ICCV 2017
- Chattopadhay et al., Grad-CAM++, WACV 2018
- Sundararajan et al., Integrated Gradients, ICML 2017
- Captum: https://captum.ai/
- PyTorch Grad-CAM: https://github.com/jacobgil/pytorch-grad-cam
