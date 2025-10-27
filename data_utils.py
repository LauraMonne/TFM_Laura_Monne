"""
Utilidades de datos:
- Collate function robusta para unificar a 3 canales (float32, [0,1]).
- Creación de DataLoaders con opciones de reproducibilidad.
"""

from __future__ import annotations
from typing import Any, List, Sequence, Tuple, Optional

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import functional as F


# -----------------------
# Conversión a tensor 3 canales
# -----------------------

def _to_3ch_tensor(img: Any) -> torch.Tensor:
    """
    Convierte una imagen a tensor CHW (3, H, W) float32 en [0, 1].
    Soporta: PIL.Image, np.ndarray (H,W[,C]), torch.Tensor (CHW o HWC).
    - Si 1 canal -> repite a 3 canales.
    - Si HWC -> permuta a CHW.
    """
    # PIL
    if isinstance(img, Image.Image):
        t = F.pil_to_tensor(img.convert("RGB"))            # uint8, (3,H,W)
        t = F.convert_image_dtype(t, dtype=torch.float32)  # -> float32 [0,1]
        return t

    # NumPy
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:             # (H,W)
            t = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)
        elif arr.ndim == 3:
            # (H,W,C) o (C,H,W)
            if arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[-1]:
                t = torch.from_numpy(arr)           # (C,H,W)
            else:
                t = torch.from_numpy(arr).permute(2, 0, 1)  # (C,H,W)
        else:
            raise ValueError(f"NumPy con ndim={arr.ndim} no soportado.")

        t = t.contiguous()
        t = t.float().div_(255.0) if t.dtype == torch.uint8 else t.to(torch.float32)

        if t.ndim != 3:
            raise ValueError("Se esperaba tensor 3D (C,H,W) tras la conversión.")
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        elif t.shape[0] != 3:
            raise ValueError(f"Número de canales no soportado: {t.shape[0]}")
        return t

    # Torch tensor
    if isinstance(img, torch.Tensor):
        t = img
        if t.ndim == 2:                # (H,W)
            t = t.unsqueeze(0)         # (1,H,W)
        elif t.ndim == 3:
            # Puede ser (H,W,C) -> permutar
            if t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):
                t = t.permute(2, 0, 1)
        else:
            raise ValueError(f"Torch tensor con ndim={t.ndim} no soportado.")

        t = t.float().div_(255.0) if t.dtype == torch.uint8 else t.to(torch.float32)

        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        elif t.shape[0] != 3:
            raise ValueError(f"Número de canales no soportado: {t.shape[0]}")
        return t

    raise TypeError(f"Tipo de imagen no soportado: {type(img)}")


# -----------------------
# Collate function
# -----------------------

def custom_collate_fn(batch: Sequence[Tuple[Any, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate que:
      - Acepta imágenes en PIL/np.ndarray/torch.Tensor.
      - Normaliza a tensor float32 (3, H, W) en [0,1].
      - Devuelve (batch_images, batch_labels).
    """
    images, labels = zip(*batch)
    processed: List[torch.Tensor] = [_to_3ch_tensor(img) for img in images]
    images_tensor = torch.stack(processed, dim=0)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    return images_tensor, labels_tensor


# -----------------------
# Reproducibilidad para DataLoader
# -----------------------

def _seed_worker(worker_id: int) -> None:
    """Semillas deterministas por worker."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# -----------------------
# DataLoaders
# -----------------------

def create_data_loaders_fixed(
    datasets: Sequence[str],
    batch_size: int = 32,
    num_workers: Optional[int] = 0,
    seed: Optional[int] = None,
    drop_last_train: bool = True,
    shuffle_train: bool = True,
):
    """
    Crea DataLoaders de train/val/test combinando datasets de MedMNIST,
    aplicando un wrapper de etiquetas y usando un collate que normaliza a 3 canales.

    Args:
        datasets: Lista de nombres de datasets, p.ej. ["retina", "blood", "breast"].
        batch_size: Tamaño de batch.
        num_workers: Workers del DataLoader. 0 por defecto (estable en Windows).
        seed: Semilla global (activa reproducibilidad del muestreo).
        drop_last_train: Descarta último batch incompleto en train.
        shuffle_train: Barajar train.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Importes locales para evitar ciclos
    from prepare_data import create_combined_dataset
    from dataset_wrapper import MedMNISTWrapper

    # Datasets combinados crudos
    train_raw = create_combined_dataset(datasets, "train")
    val_raw   = create_combined_dataset(datasets, "val")
    test_raw  = create_combined_dataset(datasets, "test")

    # Wrap
    train_ds = MedMNISTWrapper(train_raw)
    val_ds   = MedMNISTWrapper(val_raw)
    test_ds  = MedMNISTWrapper(test_raw)

    # Reproducibilidad
    generator = None
    worker_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_fn = _seed_worker
        # Semillas mínimas por si el usuario no las puso en otra parte
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    pin_memory = torch.cuda.is_available()
    persistent = bool(num_workers) and (num_workers or 0) > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=(num_workers or 0),
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        collate_fn=custom_collate_fn,
        worker_init_fn=worker_fn,
        generator=generator,
        persistent_workers=persistent,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=(num_workers or 0),
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=custom_collate_fn,
        worker_init_fn=worker_fn,
        generator=generator,
        persistent_workers=persistent,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=(num_workers or 0),
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=custom_collate_fn,
        worker_init_fn=worker_fn,
        generator=generator,
        persistent_workers=persistent,
    )

    print(
        f"DataLoaders creados:\n"
        f"  - Train: {len(train_ds)} muestras, {len(train_loader)} batches\n"
        f"  - Val:   {len(val_ds)} muestras, {len(val_loader)} batches\n"
        f"  - Test:  {len(test_ds)} muestras, {len(test_loader)} batches"
    )

    return train_loader, val_loader, test_loader
