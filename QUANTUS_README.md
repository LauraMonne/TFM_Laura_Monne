# Evaluación cuantitativa de la explicabilidad (Quantus)

Este documento explica cómo ejecutar el script `quantus_evaluation.py` para medir la calidad de las explicaciones (Grad-CAM, Grad-CAM++, Integrated Gradients y Saliency) sobre modelos entrenados de manera independiente por dataset.

## Requisitos previos

1. Haber entrenado el modelo (`python train.py`) y disponer de los checkpoints:
- results/best_model_blood.pth
- results/best_model_retina.pth
- results/best_model_breast.pth
2. Tener los datasets MedMNIST preparados en la carpeta data/ (BloodMNIST, RetinaMNIST y BreastMNIST).
3. Tener instaladas las dependencias del proyecto:
  ```bash
  pip install -r requirements.txt
  pip install quantus
  ```

## 🚀 Ejecución del script

La evaluación cuantitativa se realiza por dataset, de forma coherente con el entrenamiento de tres modelos independientes.
```bash
python quantus_evaluation.py \
    --dataset blood \
    --model_path results/best_model_blood.pth \
    --data_dir ./data \
    --num_samples 30

python quantus_evaluation.py \
    --dataset retina \
    --model_path results/best_model_retina.pth \
    --data_dir ./data \
    --num_samples 30

python quantus_evaluation.py \
    --dataset breast \
    --model_path results/best_model_breast.pth \
    --data_dir ./data \
    --num_samples 30
```

Parámetros principales:
| Flag | Descripción | Default |
|------|-------------|---------|
| `--dataset` | Dataset a evaluar (`blood`, `retina`, `breast`) | Obligatorio |
| `--model_path` | Ruta al checkpoint del modelo | Obligatorio |
| `--num_samples` | Nº de imágenes del test utilizadas para generar atribuciones | 30 |
| `--sample_strategy` | Muestreo del test: `first` (primeras N) o `reservoir` (aleatorio uniforme) | `reservoir` |
| `--seed` | Semilla para muestreo aleatorio | 42 |
| `--target` | Etiquetas objetivo para métricas: `pred` (predicha) o `true` (real) | `pred` |
| `--methods` | Métodos XAI (`gradcam`, `gradcampp`, `integrated_gradients`, `saliency`) | Todos |
| `--device` | `cuda` o `cpu` | Detectado automáticamente |

## Métricas de explicabilidad evaluadas

| Dimensión | Métrica (Quantus) | Descripción resumida |
|-----------|-------------------|----------------------|
| Fidelidad | `FaithfulnessCorrelation` | Correlación entre atribución y logits del modelo. |
| Robustez  | `AvgSensitivity` | Sensibilidad a perturbaciones leves en la entrada. |
| Complejidad | `Entropy` | Simplicidad / dispersión de la explicación. |
| Aleatorización | `ModelParameterRandomisation` | Comprueba dependencia respecto a pesos del modelo. |
| Localización | `RegionPerturbation` (proxy) | Evalúa qué ocurre al anular regiones de alta atribución. |

**Nota de configuración (rendimiento):** el script reduce el coste computacional con
`nr_runs=30` en Faithfulness, `regions_evaluation=30` en Localización y en Randomization
usa `skip_layers=True` (comparación solo original vs totalmente randomizado). Ajusta estos
valores en `quantus_evaluation.py` si necesitas mayor fidelidad estadística.

## Salida

El script genera `outputs/quantus_metrics_<dataset>.json`.

Estos ficheros son posteriormente procesados en el notebook `quantus_eval.ipynb` para generar:
- `quantus_table_raw_<dataset>.csv`
- `quantus_table_normalized_<dataset>.csv`
- `quantus_radar_<dataset>.png`

Estos resultados se utilizan directamente en el Capítulo 4 (Resultados) y se discuten en el Capítulo 5 (Discusión) del TFM.


## Flujo recomendado

1. Entrenar los modelos
```bash
python train.py --dataset blood
python train.py --dataset retina
python train.py --dataset breast
```
2. Generar explicaciones visuales (`xai_explanations.py`).
3. Ejecutar la evaluación cuantitativa con Quantus (por dataset).
4. Analizar los resultados en `notebooks/quantus_eval.ipynb`.

Con este flujo se cumple la sección 3.8 de la memoria, aportando métricas objetivas de fidelidad, robustez, complejidad, aleatorización y localización para los métodos de explicabilidad. 
