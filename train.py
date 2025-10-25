"""
Script de entrenamiento para ResNet-19 con datasets MedMNIST
Incluye entrenamiento, validación y evaluación
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
from resnet19 import create_model

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

class Trainer:
    """Clase para manejar el entrenamiento del modelo"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        # Mover modelo al dispositivo
        self.model.to(device)
        
        # Configurar optimizador y scheduler
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3,
            verbose=True
        )
        
        # Función de pérdida
        self.criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            min_delta=0.001
        )
        
        # Historial de entrenamiento
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def train_epoch(self):
        """Entrena el modelo por una época"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Entrenando')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Estadísticas
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Actualizar barra de progreso
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Valida el modelo por una época"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
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
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Entrena el modelo completo"""
        print("Iniciando entrenamiento...")
        print(f"Dispositivo: {self.device}")
        print(f"Épocas: {self.config['epochs']}")
        print(f"Tamaño de batch: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        
        start_time = time.time()
        best_val_acc = 0
        
        for epoch in range(self.config['epochs']):
            print(f"\nÉpoca {epoch+1}/{self.config['epochs']}")
            print("-" * 50)
            
            # Entrenar
            train_loss, train_acc = self.train_epoch()
            
            # Validar
            val_loss, val_acc = self.validate_epoch()
            
            # Actualizar scheduler
            self.scheduler.step(val_loss)
            
            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Mostrar resultados
            print(f"Entrenamiento - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Validación   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Guardar mejor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f"best_model_epoch_{epoch+1}.pth")
                print(f"Nuevo mejor modelo guardado (Acc: {val_acc:.2f}%)")
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping activado en época {epoch+1}")
                break
        
        training_time = time.time() - start_time
        print(f"\nEntrenamiento completado en {training_time/60:.2f} minutos")
        print(f"Mejor precisión de validación: {best_val_acc:.2f}%")
        
        return self.history
    
    def evaluate(self):
        """Evalúa el modelo en el conjunto de test"""
        print("\nEvaluando en conjunto de test...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        test_loss = 0.0
        correct = 0
        total = 0
        
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
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total
        
        print(f"\nResultados en Test:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_acc:.2f}%")
        
        return {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def save_model(self, filename):
        """Guarda el modelo"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, filename)
    
    def load_model(self, filename):
        """Carga el modelo"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.history = checkpoint['history']
    
    def plot_history(self):
        """Grafica el historial de entrenamiento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history['train_loss'], label='Entrenamiento')
        ax1.plot(self.history['val_loss'], label='Validación')
        ax1.set_title('Pérdida por Época')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.history['train_acc'], label='Entrenamiento')
        ax2.plot(self.history['val_acc'], label='Validación')
        ax2.set_title('Precisión por Época')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Precisión (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

def create_data_loaders(datasets, batch_size=32, num_workers=4):
    """Crea los data loaders para entrenamiento, validación y test"""
    
    # Crear datasets combinados
    train_dataset = create_combined_dataset(datasets, 'train')
    val_dataset = create_combined_dataset(datasets, 'val')
    test_dataset = create_combined_dataset(datasets, 'test')
    
    # Crear data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Data loaders creados:")
    print(f"  - Entrenamiento: {len(train_dataset)} muestras, {len(train_loader)} batches")
    print(f"  - Validación: {len(val_dataset)} muestras, {len(val_loader)} batches")
    print(f"  - Test: {len(test_dataset)} muestras, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Grafica la matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Función principal de entrenamiento"""
    
    # Configuración
    config = {
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'early_stopping_patience': 10,
        'num_workers': 4
    }
    
    print("=== ENTRENAMIENTO RESNET-19 CON MEDMNIST ===")
    print(f"Configuración: {json.dumps(config, indent=2)}")
    
    # Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    # Cargar datos
    print("\nCargando datasets...")
    datasets = load_datasets('./data', target_size=224)
    
    # Crear data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        datasets, 
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Crear modelo
    print("\nCreando modelo ResNet-19...")
    model = create_model(num_classes=15)
    
    # Crear trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, device, config)
    
    # Entrenar
    history = trainer.train()
    
    # Evaluar
    results = trainer.evaluate()
    
    # Graficar resultados
    trainer.plot_history()
    
    # Guardar resultados
    results_dict = {
        'config': config,
        'history': history,
        'test_results': {
            'test_loss': results['test_loss'],
            'test_acc': results['test_acc']
        }
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print(f"Precisión final en test: {results['test_acc']:.2f}%")
    print("Resultados guardados en 'training_results.json'")
    print("Historial de entrenamiento guardado en 'training_history.png'")
    
    return trainer, results

if __name__ == "__main__":
    trainer, results = main()
