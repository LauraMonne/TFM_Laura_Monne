"""
Entrenamiento ResNet-18 con MedMNIST (mejorado).

Ahora soporta entrenamiento **por dataset**:
- `python train.py --dataset blood`   → BloodMNIST (8 clases)
- `python train.py --dataset retina`  → RetinaMNIST (5 clases)
- `python train.py --dataset breast`  → BreastMNIST (2 clases)

Guarda checkpoints separados:
- `results/best_model_blood.pth`
- `results/best_model_retina.pth`
- `results/best_model_breast.pth`
"""

import argparse
import os
import time
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

# Propios
from prepare_data import load_datasets, create_combined_dataset, get_dataset_info
from resnet18 import create_model, set_seed

# AMP
from torch.cuda.amp import autocast, GradScaler

# ----------------------------- Utilidades -----------------------------
# Calcula pesos de clase inversos a la frecuencia sobre un dataset con etiquetas [0..num_classes-1].
# Devuelve un tensor con los pesos de clase y los conteos de cada clase.
def compute_class_weights(dataset, num_classes: int):
    """
    Calcula pesos de clase inversos a la frecuencia sobre un dataset con etiquetas [0..num_classes-1].
    """
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        y = int(y.item()) if isinstance(y, torch.Tensor) else int(y)
        labels.append(y)

    labels = np.array(labels)
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.clip(counts, 1, None)
    inv = 1.0 / counts
    inv = inv * (num_classes / inv.sum())  # normalizar
    return inv.astype(np.float32), counts

# Early stopping para detener el entrenamiento si la pérdida de validación no mejora después de un número de épocas.
# Restaura el mejor modelo si se activa.
class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_state = None
# Early stopping.
# Si la pérdida de validación no mejora después de un número de épocas, se detiene el entrenamiento.
# Restaura el mejor modelo si se activa.
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_state is not None:
                model.load_state_dict(self.best_state, strict=True)
            return True
        return False

# ----------------------------- Trainer -----------------------------
# Entrenador que maneja el entrenamiento, validación y evaluación.
# Inicializa el entrenador con el modelo, DataLoaders, dispositivo, configuración y pesos de clase (opcional).
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, config, class_weights=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        w = torch.tensor(class_weights, device=device, dtype=torch.float32) if (config.get('use_class_weights', True) and class_weights is not None) else None
        self.criterion = nn.CrossEntropyLoss(weight=w)

        self.early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            min_delta=1e-4,
            restore_best_weights=True
        )
        self.scaler = GradScaler()
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    # Entrena una época.
    # Imprime el progreso de entrenamiento.
    # Devuelve la pérdida y la precisión de entrenamiento.
    # Usa tqdm para mostrar el progreso de entrenamiento.

    def train_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc='Entrenando', leave=False)
        for data, target in pbar:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            gc_norm = float(self.config.get('grad_clip_norm', 0.0) or 0.0)
            if gc_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gc_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.0 * correct / max(1,total):.2f}%'})

        return running_loss / max(1, len(self.train_loader)), 100.0 * correct / max(1, total)
# Valida una época.
# Imprime el progreso de validación.
# Devuelve la pérdida y la precisión de validación.
# Usa tqdm para mostrar el progreso de validación.
    def validate_epoch(self):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validando', leave=False)
            for data, target in pbar:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.0 * correct / max(1,total):.2f}%'})

        return running_loss / max(1, len(self.val_loader)), 100.0 * correct / max(1, total)
# Entrena el modelo.
# Imprime el progreso de entrenamiento.
# Devuelve el historial de entrenamiento.
# Usa tqdm para mostrar el progreso de entrenamiento.
    def train(self):
        print("Iniciando entrenamiento...")
        start = time.time()
        for epoch in range(self.config['epochs']):
            print(f"\nÉpoca {epoch+1}/{self.config['epochs']}")
            print("-" * 50)

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            self.scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Entrenamiento - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Validación   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            if val_loss < self.best_val_loss - 1e-6:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.save_model("results/best_model.pth")
                print(f"✔️ Nuevo mejor modelo (val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%)")

            if self.early_stopping(val_loss, self.model):
                print(f"⏹️ Early stopping activado en época {epoch+1}")
                break

        elapsed = (time.time() - start) / 60.0
        print(f"\nEntrenamiento completado en {elapsed:.2f} minutos")
        print(f"Mejor val_loss: {self.best_val_loss:.4f} | Mejor val_acc: {self.best_val_acc:.2f}%")
        return self.history

# Evalúa el modelo.
# Imprime el progreso de evaluación.
# Devuelve el historial de evaluación.
# Usa tqdm para mostrar el progreso de evaluación.
    def evaluate(self):
        print("\nEvaluando en conjunto de test...")
        self.model.eval()
        all_preds, all_targets = [], []
        test_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Evaluando', leave=False)
            for data, target in pbar:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)

                test_loss += loss.item()
                _, pred = output.max(1)
                total += target.size(0)
                correct += pred.eq(target).sum().item()
                all_preds.extend(pred.detach().cpu().numpy())
                all_targets.extend(target.detach().cpu().numpy())
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.0 * correct / max(1,total):.2f}%'})

        test_loss /= max(1, len(self.test_loader))
        test_acc = 100.0 * correct / max(1, total)

        print(f"\nResultados en Test:\nLoss: {test_loss:.4f}\nAccuracy: {test_acc:.2f}%")
        print(classification_report(all_targets, all_preds, digits=4))

        self._plot_confusion_matrix(all_targets, all_preds)

        # Sufijo por dataset (para no pisar ficheros entre modelos)
        suffix = ""
        if "dataset_name" in self.config and self.config["dataset_name"] != "combined":
            suffix = f"_{self.config['dataset_name']}"

        os.makedirs("results", exist_ok=True)
        np.savez(
            f"results/preds_test{suffix}.npz",
            y_true=np.array(all_targets),
            y_pred=np.array(all_preds),
        )

        return {'test_loss': test_loss, 'test_acc': test_acc}
# Genera y guarda la matriz de confusión.
    def _plot_confusion_matrix(self, y_true, y_pred):
        os.makedirs("results", exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Matriz de Confusión (Test)")
        plt.xlabel("Predicción")
        plt.ylabel("Verdadero")
        plt.tight_layout()

        suffix = ""
        if "dataset_name" in self.config and self.config["dataset_name"] != "combined":
            suffix = f"_{self.config['dataset_name']}"
        plt.savefig(f"results/confusion_matrix{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close()
# Guarda checkpoint con estado del modelo, optimizador, configuración y historial.
    def save_model(self, filename):
        os.makedirs("results", exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, filename)
# Genera y guarda gráficas de pérdida y precisión por época.
    def plot_history(self):
        os.makedirs("results", exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(self.history["train_loss"], label="Entrenamiento")
        ax1.plot(self.history["val_loss"], label="Validación")
        ax1.set_title("Pérdida por Época")
        ax1.legend()
        ax1.grid(True)
        ax2.plot(self.history["train_acc"], label="Entrenamiento")
        ax2.plot(self.history["val_acc"], label="Validación")
        ax2.set_title("Precisión por Época")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()

        suffix = ""
        if "dataset_name" in self.config and self.config["dataset_name"] != "combined":
            suffix = f"_{self.config['dataset_name']}"
        plt.savefig(f"results/training_history{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close()

"""
DataLoaders para entrenamiento/validación/test.
Permiten entrenar:
- Un modelo combinado (15 clases)    → dataset_name='combined'
- Un modelo por dataset específico   → dataset_name in {'blood','retina','breast'}
"""


def create_data_loaders(
    datasets,
    batch_size: int = 64,
    num_workers: int = 4,
    num_classes: int = 15,
    dataset_name: str = "combined",
):
    """
    Crea los DataLoaders para entrenamiento/validación/test.

    - Si dataset_name == 'combined': mezcla Blood+Retina+Breast (15 clases, con offsets).
    - Si dataset_name == 'blood'   : solo BloodMNIST  (8 clases).
    - Si dataset_name == 'retina'  : solo RetinaMNIST (5 clases).
    - Si dataset_name == 'breast'  : solo BreastMNIST (2 clases).
    """
    persistent = num_workers > 0

    if dataset_name == "combined":
        train_dataset = create_combined_dataset(datasets, split="train")
        val_dataset = create_combined_dataset(datasets, split="val")
        test_dataset = create_combined_dataset(datasets, split="test")
    else:
        name_map = {
            "blood": "bloodmnist",
            "retina": "retinamnist",
            "breast": "breastmnist",
        }
        if dataset_name not in name_map:
            raise ValueError(f"Dataset desconocido: {dataset_name}")
        med_name = name_map[dataset_name]
        if med_name not in datasets:
            raise KeyError(f"'{med_name}' no está cargado en datasets.")

        # Usamos create_combined_dataset pero pasando solo un dataset
        sub = {med_name: datasets[med_name]}
        train_dataset = create_combined_dataset(sub, split="train", apply_offsets=False)
        val_dataset = create_combined_dataset(sub, split="val", apply_offsets=False)
        test_dataset = create_combined_dataset(sub, split="test", apply_offsets=False)

    # Pesos de clase para la loss
    class_weights_vec, counts = compute_class_weights(train_dataset, num_classes=num_classes)
    print("Distribución de clases en train:", counts.tolist())
    print(
        "Pesos de clase (inv. freq normalizados):",
        [float(f"{w:.4f}") for w in class_weights_vec],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader, class_weights_vec


# ----------------------------- MAIN -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento ResNet-18 en MedMNIST (por dataset o combinado)."
    )
    parser.add_argument(
        "--dataset",
        default="combined",
        choices=["combined", "blood", "retina", "breast"],
        help="Qué dataset entrenar: combined (15 clases), blood, retina o breast.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    # Mapa dataset -> nombre interno MedMNIST
    name_map = {
        "combined": "combined",
        "blood": "bloodmnist",
        "retina": "retinamnist",
        "breast": "breastmnist",
    }

    meta_all = get_dataset_info()
    if args.dataset == "combined":
        # 8 + 5 + 2
        num_classes = (
            meta_all["bloodmnist"]["n_classes"]
            + meta_all["retinamnist"]["n_classes"]
            + meta_all["breastmnist"]["n_classes"]
        )
    else:
        med_name = name_map[args.dataset]
        num_classes = int(meta_all[med_name]["n_classes"])

    suffix = "" if args.dataset == "combined" else f"_{args.dataset}"
    best_model_path = f"results/best_model{suffix}.pth"
    training_results_path = f"results/training_results{suffix}.json"

    config = {
        "batch_size": 64,
        "epochs": 120,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "early_stopping_patience": 12,
        "num_workers": 4,
        "use_class_weights": True,
        "grad_clip_norm": 1.0,
        "num_classes": num_classes,
        "dataset_name": args.dataset,
        "best_model_path": best_model_path,
    }

    print("=== ENTRENAMIENTO RESNET-18 (AMP, val/época, class weights, grad clip) ===")
    print("Dataset:", args.dataset, f"({num_classes} clases)")
    print(json.dumps(config, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Cargar datasets (siempre los tres; el filtrado se hace en create_data_loaders)
    datasets = load_datasets("./data", target_size=224)

    # DataLoaders según el dataset elegido
    train_loader, val_loader, test_loader, class_weights_vec = create_data_loaders(
        datasets,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        num_classes=config["num_classes"],
        dataset_name=args.dataset,
    )

    # Modelo con el número de clases correcto
    model = create_model(num_classes=config["num_classes"])

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        config,
        class_weights=class_weights_vec if config["use_class_weights"] else None,
    )

    history = trainer.train()

    # Cargar SIEMPRE el mejor checkpoint antes de evaluar.
    best_ckpt = torch.load(best_model_path, map_location=device)
    trainer.model.load_state_dict(best_ckpt["model_state_dict"])

    results = trainer.evaluate()
    trainer.plot_history()

    os.makedirs("results", exist_ok=True)
    with open(training_results_path, "w") as f:
        json.dump({"config": config, "history": history, "results": results}, f, indent=2)

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print(f"Precisión final en test: {results['test_acc']:.2f}%")
    print("Resultados guardados en carpeta 'results/'")


if __name__ == "__main__":
    main()
"""
Resumen
El script train.py entrena un modelo ResNet-18 con MedMNIST:
1. Mixed precision (AMP): acelera el entrenamiento.
2. Pesos de clase: balancea clases desbalanceadas.
3. Gradient clipping: estabiliza el entrenamiento.
4. Early stopping: evita sobreentrenamiento.
5. ReduceLROnPlateau: ajusta el learning rate.
6. Validación cada época: seguimiento continuo.
7. Guardado del mejor modelo: evalúa siempre el mejor checkpoint.
8. Evaluación completa: reporte de clasificación, matriz de confusión y guardado de predicciones.
Incluye visualización de curvas y guardado de resultados para análisis posterior.
"""