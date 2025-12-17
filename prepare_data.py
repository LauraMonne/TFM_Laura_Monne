"""
Preparación de datasets MedMNIST para ResNet-18.

En la versión actual del proyecto se trabaja **exclusivamente con tres modelos
independientes**:
- BloodMNIST  (8 clases)
- RetinaMNIST (5 clases)
- BreastMNIST (2 clases)

Este módulo se encarga de:
- Descargar y preparar **cada dataset por separado** (redimensionado a 224x224 y
  conversión consistente a 3 canales RGB).
- Proporcionar metadatos de cada dataset (`get_dataset_info`, `load_datasets`),
  que son usados por `train.py`, `xai_explanations.py` y `quantus_evaluation.py`.
"""

from __future__ import annotations

import os
import json
from typing import Dict, Sequence, Tuple

import torch
from torch.utils.data import ConcatDataset
import numpy as np
import medmnist
from medmnist import INFO
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset_wrapper import MedMNISTWrapper

# Define el orden canónico para calcular offsets de clase de forma consistente.
CANONICAL_ORDER = ["bloodmnist", "retinamnist", "breastmnist"]

# -----------------------
# Config y metadatos base
# -----------------------
# Obtiene información base de los datasets.
# Devuelve un diccionario con metadatos de cada dataset: clase, info, tarea, canales originales/input y número de clases.
def get_dataset_info() -> Dict[str, dict]:
    return {
        "bloodmnist": {
            "class": medmnist.BloodMNIST,
            "info": INFO["bloodmnist"],
            "task": "multi-class",
            "original_channels": 3,
            "input_channels": 3,
            "n_classes": 8,
        },
        "retinamnist": {
            "class": medmnist.RetinaMNIST,
            "info": INFO["retinamnist"],
            "task": "multi-class",
            "original_channels": 3,
            "input_channels": 3,
            "n_classes": 5,
        },
        "breastmnist": {
            "class": medmnist.BreastMNIST,
            "info": INFO["breastmnist"],
            "task": "binary-class",
            "original_channels": 1,
            "input_channels": 3,
            "n_classes": 2,
        },
    }

# -----------------------
# Transforms
# -----------------------
# Define las medias y desviaciones estándar para la normalización a 3 canales.
_RGB_MEAN = [0.485, 0.456, 0.406]
_RGB_STD  = [0.229, 0.224, 0.225]
# Define las transformaciones para el entrenamiento.
# Aplica redimensionamiento, flip horizontal, rotación, color jitter, conversión a RGB y normalización.
def _make_train_transform(target_size: int, original_channels: int) -> transforms.Compose:
    to_rgb = []
    if original_channels == 1:
        to_rgb.append(transforms.Grayscale(num_output_channels=3))

    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2 if original_channels != 1 else 0.0,
            hue=0.1 if original_channels != 1 else 0.0
        ),
        *to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=_RGB_MEAN, std=_RGB_STD),
    ])
# Define las transformaciones para la evaluación.
# Aplica redimensionamiento, conversión a RGB y normalización.
def _make_eval_transform(target_size: int, original_channels: int) -> transforms.Compose:
    to_rgb = []
    if original_channels == 1:
        to_rgb.append(transforms.Grayscale(num_output_channels=3))
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        *to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=_RGB_MEAN, std=_RGB_STD),
    ])
# Crea las transformaciones para el entrenamiento y la evaluación.
def create_transforms(target_size: int = 224, original_channels: int = 3) -> Tuple[transforms.Compose, transforms.Compose]:
    return _make_train_transform(target_size, original_channels), _make_eval_transform(target_size, original_channels)

# -----------------------
# Carga por dataset
# -----------------------
# Carga los datasets de MedMNIST.
# Devuelve un diccionario con los datasets cargados: train, val, test y metadatos.
def load_datasets(data_dir: str = "./data", target_size: int = 224) -> Dict[str, dict]:
    print("Cargando datasets de MedMNIST...")
    meta_all = get_dataset_info()
    os.makedirs(data_dir, exist_ok=True)
    # Inicializa el diccionario de resultados.
    result: Dict[str, dict] = {}
    for name in CANONICAL_ORDER:
        meta = meta_all[name]
        print(f"\nCargando {name.upper()}...")
        # Crea las transformaciones para el entrenamiento y la evaluación.
        train_tf, eval_tf = create_transforms(target_size, meta["original_channels"])
        cls = meta["class"]
        # Carga los datasets de entrenamiento, validación y test.
        train_ds = cls(split="train", transform=train_tf, download=True, root=data_dir)
        val_ds   = cls(split="val",   transform=eval_tf, download=True, root=data_dir)
        test_ds  = cls(split="test",  transform=eval_tf, download=True, root=data_dir)
        # Agrega los datasets al diccionario de resultados.
        result[name] = {
            "train": train_ds,
            "val":   val_ds,
            "test":  test_ds,
            "meta":  meta,
        }
        # Imprime información sobre los datasets cargados.
        print(f"  - Entrenamiento: {len(train_ds)}")
        print(f"  - Validación:    {len(val_ds)}")
        print(f"  - Test:          {len(test_ds)}")
        print(f"  - Clases:        {meta['n_classes']}")
        print(f"  - Canales (orig):{meta['original_channels']} -> input: {meta['input_channels']}")
    # Devuelve el diccionario de resultados.    
    return result

# -----------------------
# Visualización
# -----------------------

def _denormalize_tensor(img_tensor: torch.Tensor, mean=_RGB_MEAN, std=_RGB_STD) -> torch.Tensor:
    """
    Desnormaliza un tensor de imagen normalizado con ImageNet stats.
    Convierte de (C, H, W) a (C, H, W) en rango [0, 1].
    """
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]  # Si es batch, toma el primero
    
    img = img_tensor.clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        img[c] = img[c] * s + m
    
    return torch.clamp(img, 0.0, 1.0)

def create_visualization_transforms(target_size: int = 224) -> transforms.Compose:
    """
    Crea transformaciones para visualización (sin normalización).
    Útil para mostrar imágenes antes de aplicar normalización ImageNet.
    """
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor()
    ])

def visualize_samples(datasets: Dict[str, dict], num_samples: int = 5) -> None:
    """
    Visualiza muestras de los datasets.
    Las imágenes normalizadas se desnormalizan automáticamente para visualización correcta.
    """
    rows = len(datasets)
    cols = num_samples
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i, name in enumerate(CANONICAL_ORDER):
        if name not in datasets: 
            continue
        dct = datasets[name]
        ds = dct["train"]
        meta = dct["meta"]
        original_channels = meta["original_channels"]
        
        for j in range(min(cols, len(ds))):
            img, label = ds[j]
            
            # Si es tensor, puede estar normalizado
            if isinstance(img, torch.Tensor):
                # Desnormalizar si parece estar normalizado (valores fuera de [0,1])
                if img.min() < 0 or img.max() > 1.5:
                    img = _denormalize_tensor(img)
                
                # Convertir a formato de visualización
                if img.shape[0] == 1:
                    # Grayscale (1 canal)
                    vis = img.squeeze(0).numpy()
                    axes[i, j].imshow(vis, cmap="gray", vmin=0, vmax=1)
                elif img.shape[0] == 3:
                    # RGB (3 canales)
                    vis = img.permute(1, 2, 0).numpy()
                    vis = np.clip(vis, 0.0, 1.0)  # Asegurar rango [0,1]
                    axes[i, j].imshow(vis)
                else:
                    # Fallback: mostrar primer canal
                    vis = img[0].numpy()
                    axes[i, j].imshow(vis, cmap="gray", vmin=0, vmax=1)
            else:
                # PIL Image u otro formato
                axes[i, j].imshow(img)
            
            # Título con información del dataset y clase
            dataset_display_name = name.replace("mnist", "").upper()
            axes[i, j].set_title(f"{dataset_display_name}\nClase: {int(label) if hasattr(label,'__int__') else label}", 
                                fontsize=10)
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig("dataset_samples.png", dpi=150, bbox_inches="tight")
    print("✅ Visualización guardada en 'dataset_samples.png'")
    plt.show()

# -----------------------
# Persistencia de metadatos
# -----------------------
# Guarda la información de los datasets.
# Guarda la información de los datasets en un archivo JSON.
def save_dataset_info(datasets: Dict[str, dict], filename: str = "dataset_info.json") -> None:
    info_dict = {}
    for name in CANONICAL_ORDER:
        if name not in datasets: 
            continue
        dct = datasets[name]
        meta = dct["meta"]
        info_dict[name] = {
            "train_samples": len(dct["train"]),
            "val_samples":   len(dct["val"]),
            "test_samples":  len(dct["test"]),
            "n_classes":     meta["n_classes"],
            "original_channels": meta["original_channels"],
            "input_channels":    meta["input_channels"],
            "image_size": 28,
            "task": meta["task"],
        }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(info_dict, f, indent=2, ensure_ascii=False)

    print(f"Información de datasets guardada en {filename}")

# -----------------------
# Main
# -----------------------
# Main function que prepara los datos y (opcionalmente) visualiza / guarda info.
def main():
    data_dir = "./data"
    target_size = 224

    print("=== PREPARACIÓN DE DATOS MEDMNIST ===")
    print(f"Tamaño objetivo de imagen: {target_size}x{target_size}")
    print(f"Directorio de datos: {data_dir}")

    os.makedirs(data_dir, exist_ok=True)

    # Carga TODOS los datasets por separado. Estos serán utilizados después
    # por `train.py` para crear DataLoaders específicos por dataset.
    datasets = load_datasets(data_dir=data_dir, target_size=target_size)


    print("\nGenerando visualización de muestras...")
    visualize_samples(datasets)

    save_dataset_info(datasets)

    print("\n=== PREPARACIÓN COMPLETADA ===")
    print("Los datos están listos para el entrenamiento con ResNet-18")

if __name__ == "__main__":
    main()
"""
Resumen
El script prepare_data.py prepara los datasets MedMNIST para ResNet-18:
1. Configuración: define metadatos y orden canónico de los datasets.
2. Transformaciones:
- Entrenamiento: redimensiona, aumentos (flip, rotación, ColorJitter), conversión a RGB si es necesario, normalización.
- Evaluación: redimensiona, conversión a RGB si es necesario, normalización.
3. Carga: descarga y carga BloodMNIST, RetinaMNIST y BreastMNIST (train/val/test) por separado.
4. Visualización: genera una figura con muestras de cada dataset.
5. Persistencia: guarda metadatos en JSON.

Resultado: datasets listos para entrenar tres modelos ResNet-18 independientes (blood, retina, breast) con imágenes de 224x224 y 3 canales.
"""