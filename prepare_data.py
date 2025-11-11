"""
Preparación de datasets MedMNIST para ResNet-18.
Incluye BloodMNIST, RetinaMNIST y BreastMNIST con redimensionamiento a 224x224
y conversión consistente a 3 canales (RGB). Al combinar datasets se aplican
offsets de clase para obtener un espacio global de 15 clases (8+5+2).
"""

from __future__ import annotations

import os
import json
from typing import Dict, Sequence, Tuple

import torch
from torch.utils.data import ConcatDataset
import medmnist
from medmnist import INFO
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset_wrapper import MedMNISTWrapper

# Orden canónico para offsets estables
CANONICAL_ORDER = ["bloodmnist", "retinamnist", "breastmnist"]

# -----------------------
# Config y metadatos base
# -----------------------

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

_RGB_MEAN = [0.485, 0.456, 0.406]
_RGB_STD  = [0.229, 0.224, 0.225]

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

def create_transforms(target_size: int = 224, original_channels: int = 3) -> Tuple[transforms.Compose, transforms.Compose]:
    return _make_train_transform(target_size, original_channels), _make_eval_transform(target_size, original_channels)

# -----------------------
# Carga por dataset
# -----------------------

def load_datasets(data_dir: str = "./data", target_size: int = 224) -> Dict[str, dict]:
    print("Cargando datasets de MedMNIST...")
    meta_all = get_dataset_info()
    os.makedirs(data_dir, exist_ok=True)

    result: Dict[str, dict] = {}
    for name in CANONICAL_ORDER:
        meta = meta_all[name]
        print(f"\nCargando {name.upper()}...")

        train_tf, eval_tf = create_transforms(target_size, meta["original_channels"])
        cls = meta["class"]

        train_ds = cls(split="train", transform=train_tf, download=True, root=data_dir)
        val_ds   = cls(split="val",   transform=eval_tf, download=True, root=data_dir)
        test_ds  = cls(split="test",  transform=eval_tf, download=True, root=data_dir)

        result[name] = {
            "train": train_ds,
            "val":   val_ds,
            "test":  test_ds,
            "meta":  meta,
        }

        print(f"  - Entrenamiento: {len(train_ds)}")
        print(f"  - Validación:    {len(val_ds)}")
        print(f"  - Test:          {len(test_ds)}")
        print(f"  - Clases:        {meta['n_classes']}")
        print(f"  - Canales (orig):{meta['original_channels']} -> input: {meta['input_channels']}")

    return result

# -----------------------
# Combinación con offsets
# -----------------------

def _compute_offsets(names: Sequence[str], meta_all: Dict[str, dict]) -> Dict[str, int]:
    offsets = {}
    acc = 0
    for name in names:
        offsets[name] = acc
        acc += int(meta_all[name]["n_classes"])
    return offsets

def create_combined_dataset(
    datasets: Dict[str, dict] | Sequence[str],
    split: str = "train",
    data_dir: str = "./data",
    target_size: int = 224,
    apply_offsets: bool = True,
) -> ConcatDataset:
    if isinstance(datasets, dict):
        # Fuerza orden canónico para offsets estables
        names = [n for n in CANONICAL_ORDER if n in datasets.keys()]
        loaded = datasets
    else:
        # Si llega una lista, también se respeta el orden canónico
        names = [n for n in CANONICAL_ORDER if n in datasets]
        loaded = load_datasets(data_dir=data_dir, target_size=target_size)

    meta_all = get_dataset_info()
    offsets = _compute_offsets(names, meta_all) if apply_offsets else {n: 0 for n in names}

    parts = []
    for name in names:
        if name not in loaded:
            raise KeyError(f"Dataset '{name}' no está cargado.")
        base_ds = loaded[name][split]
        wrapped = MedMNISTWrapper(base_ds, class_offset=offsets[name], dataset_name=name)
        parts.append(wrapped)

    return ConcatDataset(parts)

# -----------------------
# Visualización
# -----------------------

def create_visualization_transforms(target_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor()
    ])

def visualize_samples(datasets: Dict[str, dict], num_samples: int = 5) -> None:
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
        for j in range(min(cols, len(ds))):
            img, label = ds[j]
            if isinstance(img, torch.Tensor):
                if img.shape[0] == 1:
                    vis = img.squeeze(0).numpy()
                    axes[i, j].imshow(vis, cmap="gray")
                else:
                    axes[i, j].imshow(img.permute(1, 2, 0).numpy())
            else:
                axes[i, j].imshow(img)
            axes[i, j].set_title(f"{name}\nClase: {int(label) if hasattr(label,'__int__') else label}")
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig("dataset_samples.png", dpi=150, bbox_inches="tight")
    plt.show()

# -----------------------
# Persistencia de metadatos
# -----------------------

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

def main():
    data_dir = "./data"
    target_size = 224

    print("=== PREPARACIÓN DE DATOS MEDMNIST ===")
    print(f"Tamaño objetivo de imagen: {target_size}x{target_size}")
    print(f"Directorio de datos: {data_dir}")

    os.makedirs(data_dir, exist_ok=True)

    datasets = load_datasets(data_dir=data_dir, target_size=target_size)

    print("\nCreando datasets combinados con offsets de clase...")
    combined_train = create_combined_dataset(datasets, split="train", apply_offsets=True)
    combined_val   = create_combined_dataset(datasets, split="val",   apply_offsets=True)
    combined_test  = create_combined_dataset(datasets, split="test",  apply_offsets=True)

    print(f"Train combinado: {len(combined_train)} muestras")
    print(f"Val   combinado: {len(combined_val)} muestras")
    print(f"Test  combinado: {len(combined_test)} muestras")

    print("\nGenerando visualización de muestras...")
    visualize_samples(datasets)

    save_dataset_info(datasets)

    print("\n=== PREPARACIÓN COMPLETADA ===")
    print("Los datos están listos para el entrenamiento con ResNet-18")

if __name__ == "__main__":
    main()
