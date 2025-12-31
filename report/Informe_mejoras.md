# Informe de sesión — Issues y cambios realizados

Fecha: 2025-12-31

## 1) Problemas detectados

1. **Modelos en Git LFS no descargados**
   - Los archivos `results/best_model_*.pth` eran *punteros LFS* y no pesos reales.
   - Esto provocaba errores al cargar con PyTorch (`_pickle.UnpicklingError: invalid load key, 'v'.`).

2. **Métrica de Randomization (MPRT) inválida**
   - `build_explain_func` ignoraba el modelo pasado por Quantus (model randomizado) y usaba siempre el explainer del modelo original.
   - Resultado: la métrica quedaba casi constante (~1.0) y el eje de aleatorización era inservible.

3. **Desalineación de objetivos (targets)**
   - Las atribuciones se calculaban con la clase *predicha* y las métricas se evaluaban con la clase *real* (`y_batch`).
   - Esto dañaba la **faithfulness** (valores negativos/ruidosos) y hacía incoherentes las métricas.

4. **Faithfulness agregada en un único valor**
   - Se devolvía un único score agregado, perdiendo distribución por muestra (inestable para pocos samples).

5. **Notebook y ruta absoluta**:

Esto es particular al servidor. 

   - `3. Quantus_eval.ipynb` esperaba `/home/TFM_Laura_Monne`; en el entorno actual el repo estaba en `/workspace/TFM_Laura_Monne`.
   - Esto bloqueaba la ejecución

6. **Radar/Pentágono “extraño”**
   - La visualización no usaba el *radar_factory* del tutorial de Quantus.
   - Se mezclaron normalizaciones y orden de métricas distinto, generando una forma poco “pentagonal”.

7. **Tiempo de cómputo excesivo**
   - La evaluación Quantus tardaba demasiado (especialmente Randomization y Localisation).

---

## 2) Cambios implementados

### 2.1 Código

- **`xai_explanations.py`**
  - `XAIExplainer` ahora admite `create_dirs=False` para evitar crear carpetas al usarlo dentro de Quantus.

- **`quantus_evaluation.py`**
  - Nuevos argumentos CLI:
    - `--sample_strategy` (`first`/`reservoir`)
    - `--seed`
    - `--target` (`pred`/`true`)
  - **Muestreo reservoir** para obtener un subconjunto uniforme del test.
  - **Alineación de objetivos**: si `--target pred`, todas las métricas usan la clase predicha.
  - **MPRT corregida**: `explain_func` ahora usa el modelo que Quantus pasa (randomizado) mediante un explainer cacheado por modelo.
  - **Faithfulness sin agregación** (`return_aggregate=False`) y reducción de coste (`nr_runs=30`).
  - **Localisation más ligera** (`regions_evaluation=30`).
  - **Randomization optimizada** (`skip_layers=True`, `return_last_correlation=True`).
  - Se añade metadata en el JSON de salida.

- **`QUANTUS_README.md`**
  - Documentación de los nuevos flags y de las reducciones de coste para Quantus.

### 2.2 Notebook

- **`3. Quantus_eval.ipynb`**
  - Lectura de metadata de JSON y resumen en tabla.
  - Tabla de medias, desviaciones (`std`) y normalizadas.
  - Normalización con dos modos (`RADAR_STYLE = "quantus_rank"` o `scaled`).
  - **Radar estilo Quantus**: se incorpora `radar_factory` y ranking como en el tutorial oficial.
  - Orden de métricas alineado con el tutorial: Faithfulness → Localisation → Complexity → Randomisation → Robustness.

### 2.3 Entorno y ejecución

- Se instaló **Git LFS** y se hizo `git lfs pull` para descargar los pesos reales.
- Se creó un **symlink** para compatibilidad de rutas:
  - `/home/TFM_Laura_Monne -> /workspace/TFM_Laura_Monne`
- Se ejecutó el pipeline completo de Quantus y se regeneraron resultados y figuras.

---

## 3) Resultados regenerados

Archivos actualizados en `outputs/`:

- `quantus_metrics_blood.json`
- `quantus_metrics_retina.json`
- `quantus_metrics_breast.json`
- `quantus_radar_blood.png`
- `quantus_radar_retina.png`
- `quantus_radar_breast.png`
- `quantus_table_raw_*.csv`
- `quantus_table_std_*.csv`
- `quantus_table_normalized_*.csv`

---

## 4) Estado actual

- El radar ahora es un **pentágono real** (estilo Quantus), con ranking y polígonos regulares.
- La métrica de **randomization** ya no es constante y aporta información.
- Las métricas están **alineadas con el target elegido** (`pred` por defecto).
- Los tiempos de cómputo son **razonables**.

---

## 5) Recomendaciones siguientes (si se desea)

- Subir `num_samples` a 100–200 para menor varianza en métricas.
- Comparar `--target true` vs `pred` para discusión metodológica.
- Incluir en la memoria una nota explícita de **ranking vs escala** en el radar.

