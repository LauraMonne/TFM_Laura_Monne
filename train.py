"""
Entrenamiento ResNet-19 con MedMNIST (optimizado para GPU y Colab)
- Mixed Precision (AMP)
- Validación cada 3 épocas
- DataLoader optimizado para GPU
- Grad Clipping y checkpoints best/last
"""

import os
import time
import json
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

# ------------------------------------------------------------
# Módulos propios
# ------------------------------------------------------------
from prepare_data import load_datasets, create_combined_dataset
from resnet19 import create_model, set_seed

# ------------------------------------------------------------
# Mixed precision
# ------------------------------------------------------------
from torch.cuda.amp import autocast, GradScaler


# ============================================================
# EARLY STOPPING
# ============================================================
class EarlyStopping:
    """Evita overfitting con paciencia"""
    def __init__(self, patience=4, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


# ============================================================
# TRAINER
# ============================================================
class Trainer:
    """Clase principal de entrenamiento"""
    def __init__(self, model, train_loader, val_loader, test_loader, device, config):
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
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            min_delta=0.001
        )

        # Mixed Precision
        self.use_amp = (device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)

        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # ------------------------------------------------------------
    def train_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc='Entrenando')

        for data, target in pbar:
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                output = self.model(data)
                loss = self.criterion(output, target)

            # Grad Clipping
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})

        return running_loss / len(self.train_loader), 100. * correct / total

    # ------------------------------------------------------------
    def validate_epoch(self):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validando')
            for data, target in pbar:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                with autocast(enabled=self.use_amp):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})

        return running_loss / len(self.val_loader), 100. * correct / total

    # ------------------------------------------------------------
    def train(self):
        print("Iniciando entrenamiento...")
        start_time = time.time()
        best_val_acc = 0

        for epoch in range(self.config['epochs']):
            print(f"\nÉpoca {epoch+1}/{self.config['epochs']}")
            print("-" * 50)

            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validar cada 3 épocas
            if (epoch + 1) % 3 == 0 or (epoch + 1) == self.config['epochs']:
                val_loss, val_acc = self.validate_epoch()
                self.scheduler.step(val_loss)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                print(f"Validación   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model("results/best_model.pth")
                    print(f"✅ Nuevo mejor modelo guardado (Acc: {val_acc:.2f}%)")

                if self.early_stopping(val_loss, self.model):
                    print(f"⏹️ Early stopping activado en época {epoch+1}")
                    break
            else:
                self.history['val_loss'].append(None)
                self.history['val_acc'].append(None)

            # Guardar checkpoint de última época
            self.save_model("results/last_model.pth")

            print(f"Entrenamiento - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

        total_time = (time.time() - start_time) / 60
        print(f"\nEntrenamiento completado en {total_time:.2f} minutos")
        print(f"Mejor precisión de validación: {best_val_acc:.2f}%")
        return self.history

    # ------------------------------------------------------------
    def evaluate(self):
        print("\nEvaluando en conjunto de test...")
        self.model.eval()
        all_predictions, all_targets = [], []
        test_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Evaluando')
            for data, target in pbar:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                with autocast(enabled=self.use_amp):
                    output = self.model(data)
                    loss = self.criterion(output, target)

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
        print(classification_report(all_targets, all_predictions))
        plot_confusion_matrix(all_targets, all_predictions)
        return {'test_loss': test_loss, 'test_acc': test_acc}

    # ------------------------------------------------------------
    def save_model(self, filename):
        os.makedirs("results", exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, filename)

    # ------------------------------------------------------------
    def plot_history(self):
        os.makedirs("results", exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(self.history['train_loss'], label='Entrenamiento')
        val_loss = [v if v is not None else float('nan') for v in self.history['val_loss']]
        ax1.plot(val_loss, label='Validación')
        ax1.legend(); ax1.grid(True)
        ax2.plot(self.history['train_acc'], label='Entrenamiento')
        val_acc = [v if v is not None else float('nan') for v in self.history['val_acc']]
        ax2.plot(val_acc, label='Validación')
        ax2.legend(); ax2.grid(True)
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=150)
        plt.show()


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================
def plot_confusion_matrix(y_true, y_pred):
    os.makedirs("results", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción'); plt.ylabel('Verdadero')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# DATALOADER OPTIMIZADO
# ============================================================
def create_data_loaders(datasets, batch_size=32, num_workers=2):
    persistent = num_workers > 0
    train_dataset = create_combined_dataset(datasets, 'train')
    val_dataset = create_combined_dataset(datasets, 'val')
    test_dataset = create_combined_dataset(datasets, 'test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              prefetch_factor=2, persistent_workers=persistent)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            prefetch_factor=2, persistent_workers=persistent)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             prefetch_factor=2, persistent_workers=persistent)
    print(f"\nDataLoaders creados:")
    print(f"  - Train: {len(train_dataset)} muestras, {len(train_loader)} batches")
    print(f"  - Val:   {len(val_dataset)} muestras, {len(val_loader)} batches")
    print(f"  - Test:  {len(test_dataset)} muestras, {len(test_loader)} batches")
    return train_loader, val_loader, test_loader


# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(42)
    config = {
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'early_stopping_patience': 4,  # ajustado a validación cada 3 épocas
        'num_workers': 2
    }

    print("=== ENTRENAMIENTO RESNET-19 CON MEDMNIST (AMP y val cada 3 epocas) ===")
    print(json.dumps(config, indent=2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    datasets = load_datasets('./data', target_size=224)
    train_loader, val_loader, test_loader = create_data_loaders(datasets, config['batch_size'], config['num_workers'])

    model = create_model(num_classes=15)
    trainer = Trainer(model, train_loader, val_loader, test_loader, device, config)

    history = trainer.train()
    results = trainer.evaluate()
    trainer.plot_history()

    with open('results/training_results.json', 'w') as f:
        json.dump({'config': config, 'history': history, 'results': results}, f, indent=2)

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print(f"Precisión final en test: {results['test_acc']:.2f}%")
    print("Resultados guardados en carpeta 'results/'")


if __name__ == "__main__":
    main()
