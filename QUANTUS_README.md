# 📈 Evaluación cuantitativa de la explicabilidad (Quantus)

Este documento explica cómo ejecutar el script `quantus_evaluation.py` para medir la calidad de las explicaciones (Grad-CAM, Grad-CAM++, Integrated Gradients y Saliency) según las 5 dimensiones descritas en la memoria del TFM.

## ✅ Requisitos previos

- Haber entrenado el modelo (`python train.py`) y disponer de `results/best_model.pth`.
- Haber generado las explicaciones base si se necesita comparar visualmente (`python xai_explanations.py`).
- Tener instaladas las dependencias:
  ```bash
  pip install -r requirements.txt
  pip install quantus
  ```

## 🚀 Ejecución del script

Desde la raíz del proyecto:
```bash
python quantus_evaluation.py \
    --model_path results/best_model.pth \
    --data_dir ./data \
    --num_samples 30 \
    --methods gradcam integrated_gradients saliency
```

Parámetros principales:
| Flag | Descripción | Default |
|------|-------------|---------|
| `--num_samples` | Nº de imágenes del test utilizadas para generar atribuciones | 30 |
| `--methods` | Métodos XAI a evaluar (`gradcam`, `gradcampp`, `integrated_gradients`, `saliency`) | `gradcam ig saliency` |
| `--output` | Ruta del JSON con resultados | `outputs/quantus_metrics.json` |
| `--device` | `cuda` o `cpu` | Detectado automáticamente |

## 🧠 Qué calcula cada métrica

| Dimensión | Métrica (Quantus) | Descripción resumida |
|-----------|-------------------|----------------------|
| Fidelidad | `FaithfulnessCorrelation` | Correlación entre atribución y logits del modelo. |
| Robustez  | `AvgSensitivity` | Sensibilidad a perturbaciones leves en la entrada. |
| Complejidad | `Entropy` | Simplicidad / dispersión de la explicación. |
| Aleatorización | `ModelParameterRandomisation` | Comprueba dependencia respecto a pesos del modelo. |
| Localización | `RegionPerturbation` (proxy) | Evalúa qué ocurre al anular regiones de alta atribución. |

**Nota**: Si se dispone de máscaras anatómicas/ROI, se puede extender el script para usar la métrica `AttributionLocalisation` de Quantus con supervisión.

## 📁 Salida

El script genera `outputs/quantus_metrics.json` con el siguiente formato:
```json
{
  "gradcam": {
    "faithfulness": {"mean": 0.74, "std": 0.11},
    "robustness": {"mean": 0.18, "std": 0.05},
    "complexity": {"mean": 2.10, "std": 0.30},
    "randomization": {"mean": 0.80, "std": 0.07},
    "localization": {"mean": 0.62, "std": 0.12}
  },
  "integrated_gradients": {...},
  "saliency": {...}
}
```

Estos resultados pueden exportarse a tablas o gráficos para la memoria del TFM.

## 🛠️ Consejos prácticos

- Reducir `--num_samples` si la GPU/CPU no dispone de suficiente memoria.
- Usar `--methods gradcam gradcampp` para comparar ambas variantes.
- Si se ejecuta en CPU, considerar `--num_samples 10` para pruebas rápidas.
- Para análisis avanzados, trasladar el pipeline a un notebook y visualizar las distribuciones de cada métrica.

## 🔄 Flujo recomendado

1. Entrenar modelo (`train.py`).
2. Generar mapas (`xai_explanations.py`).
3. Ejecutar evaluación cuantitativa (`quantus_evaluation.py`).
4. Analizar `outputs/quantus_metrics.json` y resumir en la memoria.

Con este flujo se cumple la sección 3.8 de la memoria, aportando métricas objetivas de fidelidad, robustez, complejidad, aleatorización y localización para los métodos de explicabilidad. 

