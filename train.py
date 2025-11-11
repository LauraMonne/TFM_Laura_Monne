"""
Entrenamiento ResNet-18 con MedMNIST (mejorado)
Mejoras clave:
- AMP (mixed precision)
- Validación cada época
- Gradient clipping (con AMP)
- SOLO class weights (sin WeightedRandomSampler para no sobre-corregir)
- Siempre evalúa el mejor checkpoint (best_model.pth)
"""

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
from prepare_data import load_datasets, create_combined_dataset
from resnet18 import create_model, set_seed

# AMP
from torch.cuda.amp import autocast, GradScaler

# ----------------------------- Utilidades -----------------------------
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

# ----------------------------- Trainer -----------------------------
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

        os.makedirs("results", exist_ok=True)
        np.savez("results/preds_test.npz", y_true=np.array(all_targets), y_pred=np.array(all_preds))

        return {'test_loss': test_loss, 'test_acc': test_acc}

    def _plot_confusion_matrix(self, y_true, y_pred):
        os.makedirs("results", exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión (Test)')
        plt.xlabel('Predicción'); plt.ylabel('Verdadero')
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
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
        ax1.plot(self.history['train_loss'], label='Entrenamiento')
        ax1.plot(self.history['val_loss'], label='Validación')
        ax1.set_title('Pérdida por Época'); ax1.legend(); ax1.grid(True)
        ax2.plot(self.history['train_acc'], label='Entrenamiento')
        ax2.plot(self.history['val_acc'], label='Validación')
        ax2.set_title('Precisión por Época'); ax2.legend(); ax2.grid(True)
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=150, bbox_inches='tight')
        plt.close()

# ----------------------------- DataLoaders -----------------------------
def create_data_loaders(datasets, batch_size=64, num_workers=4, num_classes=15):
    """
    Crea los datasets combinados y DataLoaders.
    (Sin sampler ponderado: usamos únicamente class weights en la loss)
    """
    persistent = num_workers > 0

    train_dataset = create_combined_dataset(datasets, split='train')
    val_dataset   = create_combined_dataset(datasets, split='val')
    test_dataset  = create_combined_dataset(datasets, split='test')

    # Pesos de clase para la loss
    class_weights_vec, counts = compute_class_weights(train_dataset, num_classes=num_classes)
    print("Distribución de clases en train:", counts.tolist())
    print("Pesos de clase (inv. freq normalizados):", [float(f"{w:.4f}") for w in class_weights_vec])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2,
        persistent_workers=persistent
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2,
        persistent_workers=persistent
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2,
        persistent_workers=persistent
    )

    return train_loader, val_loader, test_loader, class_weights_vec

# ----------------------------- MAIN -----------------------------
def main():
    set_seed(42)
    config = {
        'batch_size': 64,
        'epochs': 120,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'early_stopping_patience': 12,
        'num_workers': 4,
        'use_class_weights': True,
        'grad_clip_norm': 1.0,
        'num_classes': 15
    }

    print("=== ENTRENAMIENTO RESNET-18 (AMP, val/época, class weights, grad clip) ===")
    print(json.dumps(config, indent=2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    datasets = load_datasets('./data', target_size=224)

    train_loader, val_loader, test_loader, class_weights_vec = create_data_loaders(
        datasets,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        num_classes=config['num_classes']
    )

    model = create_model(num_classes=config['num_classes'])
    trainer = Trainer(
        model, train_loader, val_loader, test_loader, device, config,
        class_weights=class_weights_vec if config['use_class_weights'] else None
    )

    history = trainer.train()

    # Cargar SIEMPRE el mejor checkpoint antes de evaluar
    best_ckpt = torch.load('results/best_model.pth', map_location=device)
    trainer.model.load_state_dict(best_ckpt['model_state_dict'])

    results = trainer.evaluate()
    trainer.plot_history()

    os.makedirs('results', exist_ok=True)
    with open('results/training_results.json', 'w') as f:
        json.dump({'config': config, 'history': history, 'results': results}, f, indent=2)

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print(f"Precisión final en test: {results['test_acc']:.2f}%")
    print("Resultados guardados en carpeta 'results/'")

if __name__ == "__main__":
    main()
