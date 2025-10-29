"""
Script de entrenamiento para ResNet-19 con datasets MedMNIST
Incluye entrenamiento, validación y evaluación con reproducibilidad y guardado estructurado.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Importar nuestros módulos
from prepare_data import load_datasets, create_combined_dataset
from resnet19 import create_model, set_seed


# ------------------------------------------------------------
# CLASE EARLY STOPPING
# ------------------------------------------------------------
class EarlyStopping:
    """Early stopping para evitar overfitting"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Guarda los mejores pesos del modelo"""
        self.best_weights = model.state_dict().copy()


# ------------------------------------------------------------
# CLASE TRAINER
# ------------------------------------------------------------
class Trainer:
    """Clase para manejar el entrenamiento del modelo"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        self.model.to(device)
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            min_delta=0.001
        )
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def train_epoch(self):
        """Entrena el modelo por una época"""
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(self.train_loader, desc='Entrenando')
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
        
        return running_loss / len(self.train_loader), 100. * correct / total
    
    def validate_epoch(self):
        """Valida el modelo por una época"""
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validando')
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
        
        return running_loss / len(self.val_loader), 100. * correct / total
    
    def train(self):
        """Entrena el modelo completo"""
        print("Iniciando entrenamiento...")
        start_time = time.time()
        best_val_acc = 0
        
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
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model("results/best_model.pth")
                print(f"Nuevo mejor modelo guardado (Acc: {val_acc:.2f}%)")
            
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping activado en época {epoch+1}")
                break
        
        total_time = (time.time() - start_time) / 60
        print(f"\nEntrenamiento completado en {total_time:.2f} minutos")
        print(f"Mejor precisión de validación: {best_val_acc:.2f}%")
        return self.history
    
    def evaluate(self):
        """Evalúa el modelo en el conjunto de test"""
        print("\nEvaluando en conjunto de test...")
        self.model.eval()
        all_predictions, all_targets = [], []
        test_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Evaluando')
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
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
        
        print(f"\nResultados en Test:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_acc:.2f}%")
        print("\nReporte de clasificación:")
        print(classification_report(all_targets, all_predictions))
        
        plot_confusion_matrix(all_targets, all_predictions)
        
        return {'test_loss': test_loss, 'test_acc': test_acc}
    
    def save_model(self, filename):
        os.makedirs("results", exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, filename)
    
    def plot_history(self):
        """Grafica el historial de entrenamiento"""
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
        plt.show()


# ------------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------------
def create_data_loaders(datasets, batch_size=32, num_workers=4):
    """Crea los data loaders"""
    train_dataset = create_combined_dataset(datasets, 'train')
    val_dataset = create_combined_dataset(datasets, 'val')
    test_dataset = create_combined_dataset(datasets, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"\nDataLoaders creados:")
    print(f"  - Train: {len(train_dataset)} muestras, {len(train_loader)} batches")
    print(f"  - Val:   {len(val_dataset)} muestras, {len(val_loader)} batches")
    print(f"  - Test:  {len(test_dataset)} muestras, {len(test_loader)} batches")
    return train_loader, val_loader, test_loader


def plot_confusion_matrix(y_true, y_pred):
    """Grafica la matriz de confusión"""
    os.makedirs("results", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción'); plt.ylabel('Verdadero')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    """Función principal de entrenamiento"""
    set_seed(42)
    
    config = {
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'early_stopping_patience': 10,
        'num_workers': 4
    }
    
    print("=== ENTRENAMIENTO RESNET-19 CON MEDMNIST ===")
    print(json.dumps(config, indent=2))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
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
    trainer, results = main()