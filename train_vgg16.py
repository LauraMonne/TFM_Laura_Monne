"""
Entrenamiento VGG16 Small con MedMNIST.

Soporta entrenamiento **por dataset**:
- `python train_vgg16.py --dataset blood`   → BloodMNIST (8 clases)
- `python train_vgg16.py --dataset retina`  → RetinaMNIST (5 clases)
- `python train_vgg16.py --dataset breast`  → BreastMNIST (2 clases)

Guarda checkpoints separados:
- `results/best_model_vgg16_blood.pth`
- `results/best_model_vgg16_retina.pth`
- `results/best_model_vgg16_breast.pth`
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
from prepare_data import load_datasets, get_dataset_info
from dataset_wrapper import MedMNISTWrapper
from vgg16 import create_model, set_seed

# AMP
from torch.cuda.amp import autocast, GradScaler

# ----------------------------- Utilidades -----------------------------

def compute_class_weights(dataset, num_classes: int):
    """
    Calcula pesos de clase inversos a la frecuencia sobre un dataset.
    
    Args:
        dataset: Dataset de PyTorch con tuplas (imagen, label)
        num_classes: Número de clases esperadas
    
    Returns:
        tuple: (pesos_clase, conteos) como arrays de numpy
            - pesos_clase: Array de pesos normalizados (sum=num_classes)
            - conteos: Conteo de muestras por clase
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

# Early stopping para detener el entrenamiento si la pérdida de validación no mejora
class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_state = None

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

# Early stopping basado en val_acc
class EarlyStoppingAcc:
    def __init__(self, patience=7, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_acc = None
        self.counter = 0
        self.best_state = None
    
    def __call__(self, val_acc, model):
        if self.best_acc is None:
            self.best_acc = val_acc
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False

        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
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
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, config, class_weights=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        # Usar Focal Loss si está configurado
        if config.get('use_focal_loss', False):
            try:
                from focal_loss import FocalLoss
                w = torch.tensor(class_weights, device=device, dtype=torch.float32) if (config.get('use_class_weights', True) and class_weights is not None) else None
                focal_gamma = config.get('focal_gamma', 2.0)
                self.criterion = FocalLoss(alpha=w, gamma=focal_gamma, reduction='mean')
                print("✅ Usando Focal Loss para manejar clases desbalanceadas")
            except ImportError:
                print("⚠️  No se pudo importar FocalLoss, usando CrossEntropyLoss")
                w = torch.tensor(class_weights, device=device, dtype=torch.float32) if (config.get('use_class_weights', True) and class_weights is not None) else None
                self.criterion = nn.CrossEntropyLoss(weight=w)
        else:
            w = torch.tensor(class_weights, device=device, dtype=torch.float32) if (config.get('use_class_weights', True) and class_weights is not None) else None
            self.criterion = nn.CrossEntropyLoss(weight=w)

        # Early stopping
        use_acc_for_early_stop = config.get('use_acc_for_best_model', False)
        if use_acc_for_early_stop:
            self.early_stopping = EarlyStoppingAcc(
                patience=config['early_stopping_patience'],
                min_delta=1e-4,
                restore_best_weights=True
            )
        else:
            self.early_stopping = EarlyStopping(
                patience=config['early_stopping_patience'],
                min_delta=1e-4,
                restore_best_weights=True
            )
        
        self.scaler = GradScaler()
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

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
            gc_norm = self.config.get('grad_clip_norm', 0.0)
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

            # Guardar mejor modelo
            use_acc_for_best = self.config.get('use_acc_for_best_model', False)
            if use_acc_for_best:
                if val_acc > self.best_val_acc + 1e-6:
                    self.best_val_loss = val_loss
                    self.best_val_acc = val_acc
                    model_path = self.config.get('best_model_path', 'results/best_model_vgg16.pth')
                    self.save_model(model_path)
                    print(f"✔️ Nuevo mejor modelo (val_acc={val_acc:.2f}%, val_loss={val_loss:.4f})")
            else:
                if val_loss < self.best_val_loss - 1e-6:
                    self.best_val_loss = val_loss
                    self.best_val_acc = val_acc
                    model_path = self.config.get('best_model_path', 'results/best_model_vgg16.pth')
                    self.save_model(model_path)
                    print(f"✔️ Nuevo mejor modelo (val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%)")

            # Early stopping
            use_acc_for_early_stop = self.config.get('use_acc_for_best_model', False)
            if use_acc_for_early_stop:
                should_stop = self.early_stopping(val_acc, self.model)
            else:
                should_stop = self.early_stopping(val_loss, self.model)
            
            if should_stop:
                print(f"⏹️ Early stopping activado en época {epoch+1}")
                break

        elapsed = (time.time() - start) / 60.0
        print(f"\nEntrenamiento completado en {elapsed:.2f} minutos")
        print(f"Mejor val_loss: {self.best_val_loss:.4f} | Mejor val_acc: {self.best_val_acc:.2f}%")
        return self.history

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

        # Sufijo por dataset
        suffix = ""
        if "dataset_name" in self.config and self.config["dataset_name"] != "combined":
            suffix = f"_{self.config['dataset_name']}"

        os.makedirs("results", exist_ok=True)
        np.savez(
            f"results/preds_test_vgg16{suffix}.npz",
            y_true=np.array(all_targets),
            y_pred=np.array(all_preds),
        )

        return {'test_loss': test_loss, 'test_acc': test_acc}

    def _plot_confusion_matrix(self, y_true, y_pred):
        os.makedirs("results", exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        
        suffix = ""
        if "dataset_name" in self.config and self.config["dataset_name"] != "combined":
            suffix = f"_{self.config['dataset_name']}"
        
        # Matriz de confusión absoluta
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Matriz de Confusión VGG16 (Test)")
        plt.xlabel("Predicción")
        plt.ylabel("Verdadero")
        plt.tight_layout()
        plt.savefig(f"results/confusion_matrix_vgg16{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        # Matriz de confusión normalizada
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
        plt.title("Matriz de Confusión Normalizada VGG16 (Test)")
        plt.xlabel("Predicción")
        plt.ylabel("Verdadero")
        plt.tight_layout()
        plt.savefig(f"results/confusion_matrix_normalized_vgg16{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close()

    def save_model(self, filename):
        os.makedirs("results", exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, filename)

    def plot_history(self):
        os.makedirs("results", exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(self.history["train_loss"], label="Entrenamiento")
        ax1.plot(self.history["val_loss"], label="Validación")
        ax1.set_title("Pérdida por Época - VGG16")
        ax1.legend()
        ax1.grid(True)
        ax2.plot(self.history["train_acc"], label="Entrenamiento")
        ax2.plot(self.history["val_acc"], label="Validación")
        ax2.set_title("Precisión por Época - VGG16")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()

        suffix = ""
        if "dataset_name" in self.config and self.config["dataset_name"] != "combined":
            suffix = f"_{self.config['dataset_name']}"
        plt.savefig(f"results/training_history_vgg16{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close()

def create_data_loaders(
    datasets,
    batch_size: int = 64,
    num_workers: int = 4,
    num_classes: int = 8,
    dataset_name: str = "blood",
):
    """Crea los DataLoaders para entrenamiento/validación/test."""
    persistent = num_workers > 0

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

    base_train = datasets[med_name]["train"]
    base_val = datasets[med_name]["val"]
    base_test = datasets[med_name]["test"]

    train_dataset = MedMNISTWrapper(base_train, class_offset=0, dataset_name=med_name)
    val_dataset = MedMNISTWrapper(base_val, class_offset=0, dataset_name=med_name)
    test_dataset = MedMNISTWrapper(base_test, class_offset=0, dataset_name=med_name)

    class_weights_vec, counts = compute_class_weights(train_dataset, num_classes=num_classes)
    print("Distribución de clases en train:", counts.tolist())
    print("Pesos de clase:", [float(f"{w:.4f}") for w in class_weights_vec])

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = persistent
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, class_weights_vec


# ----------------------------- MAIN -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento VGG16 Small en MedMNIST (un modelo por dataset)."
    )
    parser.add_argument(
        "--dataset",
        default="blood",
        choices=["blood", "retina", "breast"],
        help="Qué dataset entrenar: blood (8 clases), retina (5 clases) o breast (2 clases).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Número de épocas de entrenamiento (por defecto depende del dataset)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    name_map = {
        "blood": "bloodmnist",
        "retina": "retinamnist",
        "breast": "breastmnist",
    }

    meta_all = get_dataset_info()
    med_name = name_map[args.dataset]
    num_classes = int(meta_all[med_name]["n_classes"])

    suffix = f"_{args.dataset}"
    best_model_path = f"results/best_model_vgg16{suffix}.pth"
    training_results_path = f"results/training_results_vgg16{suffix}.json"

    # Configuración base para VGG16
    # Similar a ResNet18 pero ajustada para VGG16
    if args.dataset == "retina":
        config = {
            "batch_size": 32,
            "epochs": args.epochs if args.epochs else 50,  # Mínimo 10, recomendado más
            "learning_rate": 1e-4,
            "weight_decay": 2e-4,
            "early_stopping_patience": 10,
            "num_workers": 4,
            "use_class_weights": True,
            "use_focal_loss": True,
            "focal_gamma": 2.0,
            "use_acc_for_best_model": True,
            "grad_clip_norm": 1.0,
            "num_classes": num_classes,
            "dataset_name": args.dataset,
            "best_model_path": best_model_path,
        }
    elif args.dataset == "blood":
        config = {
            "batch_size": 64,
            "epochs": args.epochs if args.epochs else 50,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "early_stopping_patience": 10,
            "num_workers": 4,
            "use_class_weights": True,
            "grad_clip_norm": 1.0,
            "num_classes": num_classes,
            "dataset_name": args.dataset,
            "best_model_path": best_model_path,
        }
    else:  # breast
        config = {
            "batch_size": 32,
            "epochs": args.epochs if args.epochs else 50,
            "learning_rate": 5e-4,
            "weight_decay": 2e-4,
            "early_stopping_patience": 12,
            "num_workers": 4,
            "use_class_weights": True,
            "use_focal_loss": True,
            "focal_gamma": 2.0,
            "use_acc_for_best_model": False,
            "grad_clip_norm": 1.0,
            "num_classes": num_classes,
            "dataset_name": args.dataset,
            "best_model_path": best_model_path,
        }

    print("=== ENTRENAMIENTO VGG16 SMALL ===")
    print("Dataset:", args.dataset, f"({num_classes} clases)")
    print(json.dumps(config, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Cargar datasets
    datasets = load_datasets("./data", target_size=224)

    # DataLoaders
    train_loader, val_loader, test_loader, class_weights_vec = create_data_loaders(
        datasets,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        num_classes=config["num_classes"],
        dataset_name=args.dataset,
    )

    # Crear modelo VGG16 Small
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

    # Cargar mejor checkpoint antes de evaluar
    if not os.path.exists(best_model_path):
        print(f"⚠️ Advertencia: No se encontró el checkpoint '{best_model_path}'.")
        print("   Se evaluará con el modelo final del entrenamiento.")
    else:
        best_ckpt = torch.load(best_model_path, map_location=device)
        trainer.model.load_state_dict(best_ckpt["model_state_dict"])
        print(f"✓ Modelo cargado desde '{best_model_path}'")

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
