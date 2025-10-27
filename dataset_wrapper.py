"""
Wrapper para datasets MedMNIST que:
- Convierte labels a enteros escalares de forma robusta.
- Soporta offset de clases para combinaciones de datasets.
- Permite (opcional) aplicar transform/target_transform y devolver el índice.
"""

from __future__ import annotations
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _to_scalar_int(y: Any) -> int:
    """Convierte y a un entero escalar robustamente."""
    # Torch tensor
    if isinstance(y, torch.Tensor):
        if y.numel() == 1:
            return int(y.item())
        # Caso común MedMNIST: tensor shape (1,)
        if y.dim() == 1 and len(y) == 1:
            return int(y[0].item())
        # Si llega one-hot u otra forma, intenta argmax
        if y.dim() >= 1:
            return int(torch.as_tensor(y).argmax().item())
        raise ValueError(f"Etiqueta torch no convertible a escalar: shape={tuple(y.shape)}")

    # NumPy array
    if isinstance(y, np.ndarray):
        if y.size == 1:
            return int(y.reshape(-1)[0].item())
        # Si parece one-hot
        if y.ndim >= 1:
            return int(np.asarray(y).argmax())
        raise ValueError(f"Etiqueta numpy no convertible a escalar: shape={y.shape}")

    # Lista/tupla
    if isinstance(y, (list, tuple)):
        if len(y) == 1:
            return int(y[0])
        # Si parece one-hot
        try:
            arr = np.asarray(y)
            if arr.ndim >= 1:
                return int(arr.argmax())
        except Exception:
            pass
        raise ValueError(f"Etiqueta list/tuple no convertible a escalar: {y}")

    # Escalar “normal”
    try:
        return int(y)
    except Exception as e:
        raise TypeError(f"Tipo de etiqueta no soportado: {type(y)}") from e


class MedMNISTWrapper(Dataset):
    """
    Wrapper que normaliza las etiquetas a `int` y permite aplicar un desplazamiento (offset)
    para mapear a un espacio global cuando se combinan varios datasets.

    Args:
        dataset: dataset base de MedMNIST (o compatible con __getitem__ -> (img, label)).
        class_offset: offset opcional que se suma a la etiqueta (por defecto 0).
        transform: transformación opcional a aplicar sobre la imagen.
        target_transform: transformación opcional a aplicar sobre la etiqueta (tras offset).
        return_index: si True, __getitem__ devuelve (img, label, idx).
        dataset_name: nombre opcional del dataset (para logs/depuración).
    """

    def __init__(
        self,
        dataset: Dataset,
        class_offset: int = 0,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        return_index: bool = False,
        dataset_name: Optional[str] = None,
    ) -> None:
        self.dataset = dataset
        self._class_offset = int(class_offset or 0)
        self.transform = transform
        self.target_transform = target_transform
        self.return_index = bool(return_index)
        self._dataset_name = dataset_name

        # Intenta inferir n_clases si existe el atributo
        self._n_classes = None
        for attr in ("n_classes", "classes", "NUM_CLASSES"):
            if hasattr(dataset, attr):
                try:
                    val = getattr(dataset, attr)
                    self._n_classes = int(val) if isinstance(val, (int, np.integer)) else int(len(val))
                    break
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def n_classes(self) -> Optional[int]:
        return self._n_classes

    @property
    def class_offset(self) -> int:
        return self._class_offset

    @property
    def dataset_name(self) -> Optional[str]:
        return self._dataset_name

    def __getitem__(self, idx: int) -> Tuple[Any, int] | Tuple[Any, int, int]:
        img, label = self.dataset[idx]

        # Transforms sobre imagen (si procede). Tu collate ya homogeneiza canales.
        if self.transform is not None:
            img = self.transform(img)

        # Normaliza etiqueta a escalar int
        label_int = _to_scalar_int(label)

        # Aplica offset global si lo has configurado
        if self._class_offset:
            label_int = label_int + self._class_offset

        # target_transform al final
        if self.target_transform is not None:
            label_int = self.target_transform(label_int)

        if self.return_index:
            return img, label_int, idx
        return img, label_int
