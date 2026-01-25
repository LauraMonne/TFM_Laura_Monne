# Guía de Uso: VGG16 Small para Clasificación y Comparación

Esta guía describe cómo usar el modelo VGG16 Small implementado y compararlo con ResNet18.

## Contenido

1. [Preparación del entorno](#1-preparación-del-entorno)
2. [Entrenamiento de modelos](#2-entrenamiento-de-modelos)
3. [Comparación de modelos](#3-comparación-de-modelos)
4. [Análisis de explicabilidad](#4-análisis-de-explicabilidad)
5. [Solución de problemas](#5-solución-de-problemas)

---

## 1. Preparación del entorno

### Instalación de dependencias

```bash
# Clonar el repositorio (si aún no lo has hecho)
git clone https://github.com/LauraMonne/TFM_Laura_Monne.git
cd TFM_Laura_Monne

# Instalar dependencias
pip install -r requirements.txt
```

### Preparar los datos

```bash
# Descargar y preparar datasets MedMNIST
python prepare_data.py
```

Este comando:
- Descarga BloodMNIST, RetinaMNIST y BreastMNIST
- Aplica transformaciones (resize a 224×224, normalización)
- Guarda metadatos en `dataset_info.json`
- Genera visualización en `dataset_samples.png`

**Nota:** Los datasets son grandes (~1.5GB cada uno). La descarga puede tardar varios minutos.

---

## 2. Entrenamiento de modelos

### 2.1 Entrenamiento de VGG16 Small

#### Entrenar en BloodMNIST (8 clases)

```bash
python train_vgg16.py --dataset blood
```

**Configuración por defecto:**
- Batch size: 64
- Épocas: 50 (mínimo 10 recomendado)
- Learning rate: 1e-3
- Optimizador: Adam
- Early stopping: 10 épocas

**Salida:**
- `results/best_model_vgg16_blood.pth` - Mejor checkpoint
- `results/training_results_vgg16_blood.json` - Métricas y configuración
- `results/training_history_vgg16_blood.png` - Curvas de entrenamiento
- `results/confusion_matrix_vgg16_blood.png` - Matriz de confusión

#### Entrenar en RetinaMNIST (5 clases)

```bash
python train_vgg16.py --dataset retina
```

**Configuración adaptada:**
- Batch size: 32 (dataset pequeño)
- Épocas: 50
- Learning rate: 1e-4
- Focal Loss activado (clases desbalanceadas)
- Early stopping: 10 épocas

#### Entrenar en BreastMNIST (2 clases)

```bash
python train_vgg16.py --dataset breast
```

**Configuración adaptada:**
- Batch size: 32
- Épocas: 50
- Learning rate: 5e-4
- Focal Loss activado
- Early stopping: 12 épocas

#### Opciones adicionales

```bash
# Especificar número de épocas personalizado
python train_vgg16.py --dataset blood --epochs 100

# Ver ayuda
python train_vgg16.py --help
```

### 2.2 Entrenamiento de ResNet18 (para comparación)

Si aún no has entrenado ResNet18, hazlo con:

```bash
python train.py --dataset blood
python train.py --dataset retina
python train.py --dataset breast
```

---

## 3. Comparación de modelos

Una vez entrenados ambos modelos, puedes compararlos:

### 3.1 Comparación completa

```bash
# Comparar en BloodMNIST
python compare_models.py --dataset blood

# Comparar en RetinaMNIST
python compare_models.py --dataset retina

# Comparar en BreastMNIST
python compare_models.py --dataset breast
```

### 3.2 Salidas de la comparación

El script genera:

1. **Reporte JSON** (`results/comparison_report_{dataset}.json`):
   ```json
   {
     "dataset": "blood",
     "num_classes": 8,
     "architecture_comparison": {
       "vgg16": {"total": 3700000, "trainable": 3700000},
       "resnet18": {"total": 11000000, "trainable": 11000000}
     },
     "training_comparison": {
       "vgg16": {"test_acc": 92.5, "test_loss": 0.25},
       "resnet18": {"test_acc": 94.2, "test_loss": 0.21}
     }
   }
   ```

2. **Gráficas comparativas** (`results/comparison_vgg16_resnet18_{dataset}.png`):
   - Pérdida en entrenamiento y validación
   - Accuracy en entrenamiento y validación
   - Comparación lado a lado

### 3.3 Interpretar resultados

**Métricas clave:**

| Aspecto | VGG16 Small | ResNet18 | Ventaja |
|---------|-------------|----------|---------|
| Parámetros | ~3.7M | ~11M | VGG16 (66% menos) |
| Tiempo entrenamiento | Menor | Mayor | VGG16 |
| Accuracy | Variable | Variable | Depende del dataset |
| Complejidad | Simple | Media | VGG16 (más interpretable) |

---

## 4. Análisis de explicabilidad

### 4.1 Generar explicaciones XAI

```bash
# Generar explicaciones para ambos modelos
python xai_explanations.py --dataset blood
```

**Nota:** Asegúrate de que el script `xai_explanations.py` soporte VGG16. Si no, puede necesitar adaptación.

### 4.2 Evaluar con Quantus

```bash
# Evaluar explicabilidad
python quantus_evaluation.py --dataset blood
```

### 4.3 Métricas de explicabilidad a comparar

1. **Fidelidad**:
   - Faithfulness Correlation
   - Pixel Flipping

2. **Robustez**:
   - Local Lipschitz Estimate
   - Max-Sensitivity

3. **Localización**:
   - Pointing Game
   - Attribution Localization

4. **Complejidad**:
   - Sparseness
   - Complexity

### 4.4 Análisis en notebooks

```bash
# Abrir notebook de análisis
jupyter notebook "3. Quantus_eval.ipynb"
```

En el notebook, puedes:
- Visualizar mapas de calor XAI
- Comparar explicaciones entre modelos
- Analizar métricas de Quantus
- Generar gráficas para el TFM

---

## 5. Solución de problemas

### Error: "No module named 'torch'"

```bash
pip install torch torchvision
```

### Error: "No space left on device"

```bash
# Limpiar archivos temporales
rm -rf /tmp/*

# Limpiar caché de pip
pip cache purge

# Verificar espacio
df -h
```

### Error: "Automatic download failed" (MedMNIST)

Si la descarga automática falla:

```bash
# Usar script de descarga manual
python download_medmnist_manual.py

# O descargar manualmente desde:
# https://zenodo.org/records/10519652
# Colocar archivos .npz en ./data/
```

### Error: "Model checkpoint not found"

Asegúrate de entrenar el modelo primero:

```bash
# Entrenar VGG16
python train_vgg16.py --dataset blood

# Verificar que se creó el checkpoint
ls -lh results/best_model_vgg16_blood.pth
```

### Entrenamientos muy lentos

```bash
# Verificar si CUDA está disponible
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Si no hay GPU, reduce batch_size y épocas
python train_vgg16.py --dataset blood --epochs 20
```

### Validar estructura del modelo sin entrenar

```bash
# Ejecutar pruebas de estructura
python test_vgg16.py
```

---

## Flujo de trabajo completo

### Paso 1: Configuración inicial

```bash
# Clonar repo e instalar dependencias
git clone https://github.com/LauraMonne/TFM_Laura_Monne.git
cd TFM_Laura_Monne
pip install -r requirements.txt

# Preparar datos
python prepare_data.py
```

### Paso 2: Entrenar modelos

```bash
# Entrenar ResNet18 (si no está hecho)
python train.py --dataset blood

# Entrenar VGG16 Small
python train_vgg16.py --dataset blood

# Repetir para otros datasets si es necesario
```

### Paso 3: Comparar

```bash
# Generar comparación
python compare_models.py --dataset blood

# Revisar resultados
cat results/comparison_report_blood.json
```

### Paso 4: Analizar explicabilidad

```bash
# Generar explicaciones
python xai_explanations.py --dataset blood

# Evaluar con Quantus
python quantus_evaluation.py --dataset blood

# Analizar en notebook
jupyter notebook "3. Quantus_eval.ipynb"
```

---

## Resultados esperados

### Tiempo de ejecución (estimado)

| Tarea | BloodMNIST | RetinaMNIST | BreastMNIST |
|-------|-----------|-------------|-------------|
| Preparación datos | 10-20 min | 10-20 min | 10-20 min |
| Entrenamiento VGG16 (50 épocas) | 15-30 min | 10-20 min | 10-20 min |
| Comparación | 1-2 min | 1-2 min | 1-2 min |
| Explicabilidad | 5-10 min | 5-10 min | 5-10 min |

**Nota:** Los tiempos dependen del hardware (GPU/CPU).

### Métricas esperadas (orientativas)

| Dataset | Modelo | Test Accuracy |
|---------|--------|---------------|
| BloodMNIST | ResNet18 | 93-96% |
| BloodMNIST | VGG16 Small | 90-94% |
| RetinaMNIST | ResNet18 | 50-60% |
| RetinaMNIST | VGG16 Small | 48-58% |
| BreastMNIST | ResNet18 | 85-90% |
| BreastMNIST | VGG16 Small | 83-88% |

---

## Referencias

- Documentación completa de VGG16: [VGG16_README.md](VGG16_README.md)
- Documentación XAI: [XAI_README.md](XAI_README.md)
- README principal: [README.md](README.md)

---

## Contacto

Para dudas o problemas, consulta:
- Issues del repositorio: https://github.com/LauraMonne/TFM_Laura_Monne/issues
- Documentación del proyecto

---

**Última actualización:** 2026-01-03
