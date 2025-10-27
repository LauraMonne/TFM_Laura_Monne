"""
Script de prueba rápida para ResNet-19 con MedMNIST.
Entrena pocas épocas para verificar que todo el pipeline funciona.
Incluye reproducibilidad básica.
"""

from __future__ import annotations
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Módulos propios
from prepare_data import load_datasets, get_dataset_info
from resnet19 import create_model
from data_utils import create_data_loaders_fixed


def set_seed(seed: int = 42) -> None:
    """Reproducibilidad básica (Python/NumPy/PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Para determinismo (algo más lento):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _infer_num_classes_from_meta() -> int:
    """Suma n_clases de blood(8) + retina(5) + breast(2) = 15."""
    meta = get_dataset_info()
    return int(meta["bloodmnist"]["n_classes"]
               + meta["retinamnist"]["n_classes"]
               + meta["breastmnist"]["n_classes"])


def quick_train_test() -> torch.nn.Module:
    """Prueba rápida del entrenamiento con pocas épocas."""
    print("=== PRUEBA RÁPIDA DE ENTRENAMIENTO RESNET-19 ===")

    # Config mínima para smoke test
    config = {
        "batch_size": 16,     # pequeño para ir rápido
        "epochs": 3,          # 3 épocas
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "num_workers": 2,     # en Windows, si da guerra, pon 0
        "seed": 42,
        "target_size": 224,
        "datasets": ["bloodmnist", "retinamnist", "breastmnist"],
    }

    # Reproducibilidad
    set_seed(config["seed"])

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Cargar datasets (ya con transforms a 3 canales)
    print("\nCargando datasets...")
    datasets = load_datasets(data_dir="./data", target_size=config["target_size"])

    # DataLoaders (usa el collate que homogeneiza a (3,H,W) y seed para sampling)
    train_loader, val_loader, _ = create_data_loaders_fixed(
        datasets=datasets,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        seed=config["seed"],
    )

    # Modelo
    num_classes = _infer_num_classes_from_meta()
    print(f"\nCreando modelo ResNet-19 (num_classes={num_classes})...")
    model = create_model(num_classes=num_classes).to(device)

    # Entrenamiento rápido
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=config["learning_rate"],
                            weight_decay=config["weight_decay"])

    print(f"\nIniciando entrenamiento de prueba ({config['epochs']} épocas)...")
    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        print(f"\nÉpoca {epoch}/{config['epochs']}")

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc="Entrenando")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            preds = output.argmax(dim=1)
            train_total += target.size(0)
            train_correct += (preds == target).sum().item()

            if batch_idx % 50 == 0 and train_total > 0:
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{100.0 * train_correct / train_total:.2f}%"
                })

        train_loss = train_loss_sum / max(1, len(train_loader))
        train_acc = 100.0 * train_correct / max(1, train_total)

        # --- Val ---
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validando")
            for data, target in pbar:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = model(data)
                loss = criterion(output, target)

                val_loss_sum += loss.item()
                preds = output.argmax(dim=1)
                val_total += target.size(0)
                val_correct += (preds == target).sum().item()

                if val_total > 0:
                    pbar.set_postfix({
                        "Loss": f"{loss.item():.4f}",
                        "Acc": f"{100.0 * val_correct / val_total:.2f}%"
                    })

        val_loss = val_loss_sum / max(1, len(val_loader))
        val_acc = 100.0 * val_correct / max(1, val_total)

        print(f"Entrenamiento - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Validación   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    elapsed = time.time() - start_time
    print(f"\nPrueba completada en {elapsed:.2f} s")
    print("La implementación funciona correctamente.")

    return model


if __name__ == "__main__":
    _ = quick_train_test()
