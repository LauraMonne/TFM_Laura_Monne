"""
Script de prueba rápida para ResNet-18 con MedMNIST.
Entrena pocas épocas para verificar que todo el pipeline funciona.
Incluye reproducibilidad básica.

Uso:
    python quick_test.py --dataset blood    # BloodMNIST (8 clases)
    python quick_test.py --dataset retina   # RetinaMNIST (5 clases)
    python quick_test.py --dataset breast   # BreastMNIST (2 clases)
"""

from __future__ import annotations
import argparse
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Módulos propios
from prepare_data import load_datasets, get_dataset_info
from resnet18 import create_model, set_seed
from train import create_data_loaders

# Infiere el número de clases según el dataset seleccionado.
def _infer_num_classes_from_meta(dataset_name: str) -> int:
    """
    Devuelve el número de clases según el dataset:
    - blood: 8 clases
    - retina: 5 clases
    - breast: 2 clases
    """
    meta = get_dataset_info()
    name_map = {
        "blood": "bloodmnist",
        "retina": "retinamnist",
        "breast": "breastmnist",
    }
    if dataset_name not in name_map:
        raise ValueError(f"Dataset desconocido: {dataset_name}. Usa: blood, retina o breast.")
    med_name = name_map[dataset_name]
    return int(meta[med_name]["n_classes"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prueba rápida de ResNet-18 en MedMNIST (por dataset individual)."
    )
    parser.add_argument(
        "--dataset",
        default="blood",
        choices=["blood", "retina", "breast"],
        help="Dataset a probar: blood (8 clases), retina (5 clases) o breast (2 clases).",
    )
    return parser.parse_args()

# Prueba rápida del entrenamiento.
# Define configuración mínima para una prueba rápida: batch pequeño, 3 épocas, hiperparámetros básicos.
def quick_train_test(dataset_name: str = "blood") -> torch.nn.Module:
    """
    Prueba rápida del entrenamiento con pocas épocas.
    
    Args:
        dataset_name: "blood", "retina" o "breast"
    """
    print(f"=== PRUEBA RÁPIDA DE ENTRENAMIENTO RESNET-18 ({dataset_name.upper()}) ===")

    # Config mínima para smoke test
    config = {
        "batch_size": 16,     # pequeño para ir rápido
        "epochs": 3,          # 3 épocas
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "num_workers": 2,
        "seed": 42,
        "target_size": 224,
    }

    # Reproducibilidad
    set_seed(config["seed"])

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Cargar datasets (ya con transforms a 3 canales)
    print(f"\nCargando dataset {dataset_name}...")
    datasets = load_datasets(data_dir="./data", target_size=config["target_size"])

    # Número de clases según dataset
    num_classes = _infer_num_classes_from_meta(dataset_name)
    print(f"Número de clases: {num_classes}")

    # DataLoaders (usa create_data_loaders de train.py para filtrar por dataset)
    train_loader, val_loader, _, _ = create_data_loaders(
        datasets=datasets,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        num_classes=num_classes,
        dataset_name=dataset_name,
    )

    # Modelo
    print(f"\nCreando modelo ResNet-18 (num_classes={num_classes})...")
    model = create_model(num_classes=num_classes).to(device)
# Define pérdida (CrossEntropyLoss) y optimizador (AdamW), e inicia el cronómetro.  
    # Entrenamiento rápido
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=config["learning_rate"],
                            weight_decay=config["weight_decay"])

    print(f"\nIniciando entrenamiento de prueba ({config['epochs']} épocas)...")
    start_time = time.time()
# Entrena por épocas.
# Imprime el progreso de entrenamiento y validación.
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
# Valida por épocas.
# Imprime el progreso de validación.
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
# Imprime los resultados de entrenamiento y validación.
        print(f"Entrenamiento - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Validación   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    elapsed = time.time() - start_time
# Imprime el tiempo de ejecución.
    print(f"\nPrueba completada en {elapsed:.2f} s")
    print("La implementación funciona correctamente.")
# Devuelve el modelo.
    return model

# Ejecuta la prueba rápida cuando se ejecuta el script directamente.
if __name__ == "__main__":
    args = parse_args()
    _ = quick_train_test(dataset_name=args.dataset)
"""
Resumen
El script quick_test.py realiza una prueba rápida de entrenamiento con ResNet-18 por dataset individual:
1. Argumentos: --dataset (blood, retina, breast).
2. Configuración: batch 16, 3 épocas, hiperparámetros básicos.
3. Reproducibilidad: fija semillas para Python, NumPy y PyTorch.
4. Dispositivo: verifica si hay GPU disponible.
5. Carga: carga solo el dataset seleccionado (8, 5 o 2 clases según corresponda).
6. Modelo: crea ResNet-18 con el número de clases correcto.
7. Entrenamiento: por 3 épocas, imprime pérdida y precisión por época.
8. Validación: por 3 épocas, imprime pérdida y precisión por época.
9. Resultados: imprime tiempo de ejecución y mensaje de éxito.

Útil para verificar que el pipeline funciona antes de un entrenamiento completo. Es un "smoke test" que detecta errores básicos rápidamente.
"""