"""
Utilidades de datos:
- Collate function robusta para unificar a 3 canales (float32, [0,1]).
- Creación de DataLoaders con opciones de reproducibilidad.
"""
# Importaciones, ipos, aleatoriedad, NumPy, PyTorch, DataLoader, PIL y transforms.
from __future__ import annotations
from typing import Any, List, Sequence, Tuple, Optional

import random
import numpy as np
import torch
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
# Convierte una imagen a tensor CHW (3, H, W) float32 en [0, 1].
# Soporta: PIL.Image, np.ndarray (H,W[,C]), torch.Tensor (CHW o HWC).
# Si 1 canal -> repite a 3 canales.
# Si HWC -> permuta a CHW
# Si hay 1 canal, repite a 3; si no es 1 ni 3, lanza error
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
# Convierte un tensor a tensor CHW (3, H, W) float32 en [0, 1].
# Soporta: torch.Tensor (CHW o HWC).
# Si 1 canal -> repite a 3 canales.
# Si HWC -> permuta a CHW
# Si hay 1 canal, repite a 3; si no es 1 ni 3, lanza error
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
# Collate function que:
# - Acepta imágenes en PIL/np.ndarray/torch.Tensor.
# - Normaliza a tensor float32 (3, H, W) en [0,1].
# - Devuelve (batch_images, batch_labels).
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
# Semillas deterministas por worker.
# Semilla basada en la semilla inicial de PyTorch.
def _seed_worker(worker_id: int) -> None:
    """Semillas deterministas por worker."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


    """
    Resumen
    El script data_utils.py proporciona:
    1. Normalización de imágenes: _to_3ch_tensor convierte PIL/NumPy/Tensor a tensor (3, H, W) float32 en [0, 1], repitiendo canales si es necesario.
    2. Collate function: custom_collate_fn agrupa batches normalizando todas las imágenes.
    3. Reproducibilidad: _seed_worker y configuración de generadores para resultados deterministas.
    Útil para trabajar con datasets médicos (MedMNIST) que pueden tener formatos variados, asegurando que todas las imágenes lleguen al modelo en el mismo formato (3 canales, float32, [0, 1]).
    """
