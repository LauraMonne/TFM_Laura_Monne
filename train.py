"""
Entrenamiento ResNet-18 con MedMNIST (GPU + AMP)
Mejoras:
- Validación en CADA época (early stopping más fino)
- Ponderación por clase para mitigar desbalanceo
- Determinismo real (CUDNN benchmark desactivado)
- Guardado de y_prob (probabilidades) y misclasificaciones
- Reporte por clase en CSV + CM normalizada
- Grad clipping opcional
"""

import os
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Módulos propios
from prepare_data import load_datasets, create_combined_dataset
from resnet18 import create_model, set_seed

# AMP
from torch.cuda.amp import autocast, GradScaler


# ---------------------------
# Early Stopping
# ---------------------------
class EarlyStopping:
    """Evita overfitting con paciencia sobre la pérdida de validación."""
    def __init__(self, patience=7, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict()
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
            return False


# ---------------------------
# Utilidades
# ---------------------------
def compute_class_weights(train_dataset, num_classes):
    """
    Calcula pesos de clase inversamente proporcionales a la frecuencia.
    Intenta usar atributos comunes; si no, itera una vez por el dataset.
    """
    # 1) intentamos train_dataset.targets o .labels
    targets = None
    for attr in ["targets", "labels", "y", "ys"]:
        if hasattr(train_dataset, attr):
            arr = getattr(train_dataset, attr)
            if isinstance(arr, (list, np.ndarray)) or torch.is_tensor(arr):
                targets = np.array(arr) if not torch.is_tensor(arr) else arr.cpu().numpy()
                break

    # 2) fallback: iterar
    if targets is None:
        idxs = []
        for i in range(len(train_dataset)):
            _, y = train_dataset[i]
            idxs.append(int(y))
        targets = np.array(idxs)

    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0  # evitar divisiones por 0
    weights = counts.sum() / (counts * len(counts))  # inversa proporcional normalizada
    return torch.tensor(weights, dtype=torch.float32)


def accuracy_from_logits(logits, targets):
    _, pred = logits.max(1)
    correct = pred.eq(targets).sum().item()
    total = targets.size(0)
    return correct, total


# ---------------------------
# Trainer
# ---------------------------
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, config, class_weights=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2  # más reactivo
        )

        if class_weights is not None:
            class_weights = class_weights.to(device, non_blocking=True)
            print("Pesos de clase:", class_weights.detach().cpu().numpy().round(3).tolist())
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            min_delta=0.0005
        )

        self.grad_clip = config.get("grad_clip_norm", None)
        self.scaler = GradScaler(enabled=(device.type == 'cuda'))
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc='Entrenando')

        for data, target in pbar:
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(self.device.type == 'cuda')):
                output = self.model(data)
                loss = self.criterion(output, target)

            self.scaler.scale(loss).backward()

            # grad clipping opcional
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            c, t = accuracy_from_logits(output, target)
            correct += c
            total += t
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})

        return running_loss / len(self.train_loader), 100. * correct / total

    def validate_epoch(self):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validando')
            for data, target in pbar:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                with autocast(enabled=(self.device.type == 'cuda')):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                running_loss += loss.item()
                c, t = accuracy_from_logits(output, target)
                correct += c
                total += t
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})

        return running_loss / len(self.val_loader), 100. * correct / total

    def train(self):
        print("Iniciando entrenamiento...")
        start_time = time.time()
        best_val_loss = math.inf
        best_val_acc = 0.0

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

            # Guardamos por mejor val_loss (más estable)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                self.save_model("results/best_model.pth")
                print(f"✔️ Nuevo mejor modelo (val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%)")

            if self.early_stopping(val_loss, self.model):
                print(f"⏹️ Early stopping activado en época {epoch+1}")
                break

        total_time = (time.time() - start_time) / 60
        print(f"\nEntrenamiento completado en {total_time:.2f} minutos")
        print(f"Mejor val_loss: {best_val_loss:.4f} | Mejor val_acc: {best_val_acc:.2f}%")
        return self.history

    def evaluate(self):
        print("\nEvaluando en conjunto de test...")
        self.model.eval()
        all_predictions, all_targets = [], []
        all_probs = []

        test_loss, correct, total = 0.0, 0, 0
        softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Evaluando')
            for data, target in pbar:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                with autocast(enabled=(self.device.type == 'cuda')):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                probs = softmax(output).detach().cpu().numpy()
                all_probs.append(probs)

                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})

        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total

        print(f"\nResultados en Test:\nLoss: {test_loss:.4f}\nAccuracy: {test_acc:.2f}%")
        print(classification_report(all_targets, all_predictions, digits=4))

        # Artefactos
        os.makedirs("results", exist_ok=True)

        # Guardar predicciones y probabilidades
        all_probs = np.concatenate(all_probs, axis=0)
        np.savez("results/preds_test.npz", y_true=np.array(all_targets), y_pred=np.array(all_predictions))
        np.save("results/y_prob_test.npy", all_probs)

        # Reporte por clase en CSV
        from sklearn.metrics import precision_recall_fscore_support
        labels_sorted = sorted(set(all_targets))
        prec, rec, f1, support = precision_recall_fscore_support(
            all_targets, all_predictions, labels=labels_sorted, zero_division=0
        )
        import csv
        with open("results/class_report_test.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["clase", "precision", "recall", "f1", "soporte"])
            for i, c in enumerate(labels_sorted):
                writer.writerow([c, round(float(prec[i]),4), round(float(rec[i]),4),
                                 round(float(f1[i]),4), int(support[i])])

        # Matrices de confusión (cruda y normalizada)
        self._plot_confusion_matrix(all_targets, all_predictions, normalized=False, out="results/confusion_matrix.png")
        self._plot_confusion_matrix(all_targets, all_predictions, normalized=True,  out="results/confusion_matrix_norm.png")

        return {'test_loss': test_loss, 'test_acc': test_acc}

    def _plot_confusion_matrix(self, y_true, y_pred, normalized=False, out="results/confusion_matrix.png"):
        cm = confusion_matrix(y_true, y_pred, normalize=("true" if normalized else None))
        plt.figure(figsize=(10, 8))
        fmt = ".2f" if normalized else "d"
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues')
        plt.title('Matriz de Confusión ' + ('(Normalizada)' if normalized else '(Test)'))
        plt.xlabel('Predicción'); plt.ylabel('Verdadero')
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches='tight')
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
        ax1.plot(self.history['train_loss'], label='Train')
        ax1.plot(self.history['val_loss'], label='Val')
        ax1.set_title('Pérdida por Época'); ax1.legend(); ax1.grid(True)
        ax2.plot(self.history['train_acc'], label='Train')
        ax2.plot(self.history['val_acc'], label='Val')
        ax2.set_title('Precisión por Época'); ax2.legend(); ax2.grid(True)
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=150, bbox_inches='tight')
        plt.close()


# ---------------------------
# DataLoaders
# ---------------------------
def create_data_loaders(datasets, batch_size=32, num_workers=2):
    """Optimizado para GPU"""
    persistent = num_workers > 0
    train_dataset = create_combined_dataset(datasets, 'train')
    val_dataset   = create_combined_dataset(datasets, 'val')
    test_dataset  = create_combined_dataset(datasets, 'test')

    # Nota: prefetch_factor solo tiene efecto si num_workers > 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              prefetch_factor=(2 if num_workers > 0 else None),
                              persistent_workers=persistent)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            prefetch_factor=(2 if num_workers > 0 else None),
                            persistent_workers=persistent)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             prefetch_factor=(2 if num_workers > 0 else None),
                             persistent_workers=persistent)
    print(f"\nDataLoaders creados:")
    print(f"  - Train: {len(train_dataset)} muestras, {len(train_loader)} batches")
    print(f"  - Val:   {len(val_dataset)} muestras, {len(val_loader)} batches")
    print(f"  - Test:  {len(test_dataset)} muestras, {len(test_loader)} batches")
    return train_loader, val_loader, test_loader, train_dataset


# ---------------------------
# MAIN
# ---------------------------
def main():
    # Determinismo real (sobre-escribe lo de set_seed si fuera necesario)
    set_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # <- importante para determinismo

    config = {
        'batch_size': 64,
        'epochs': 120,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'early_stopping_patience': 12,
        'num_workers': 4,
        'use_class_weights': True,     # <- activar pesos de clase
        'grad_clip_norm': 1.0          # <- clipping suave, o None para desactivar
    }

    print("=== ENTRENAMIENTO RESNET-18 (AMP, val cada época, class weights) ===")
    print(json.dumps(config, indent=2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    datasets = load_datasets('./data', target_size=224)
    train_loader, val_loader, test_loader, train_dataset = create_data_loaders(
        datasets, config['batch_size'], config['num_workers']
    )

    num_classes = 15
    class_weights = None
    if config['use_class_weights']:
        class_weights = compute_class_weights(train_dataset, num_classes=num_classes)

    model = create_model(num_classes=num_classes)
    trainer = Trainer(model, train_loader, val_loader, test_loader, device, config, class_weights=class_weights)

    history = trainer.train()
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
