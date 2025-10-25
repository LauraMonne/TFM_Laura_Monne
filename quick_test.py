"""
Script de prueba rápida para ResNet-19 con MedMNIST
Entrenamiento con pocas épocas para verificar que todo funciona
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

# Importar nuestros módulos
from prepare_data import load_datasets
from resnet19 import create_model
from data_utils import create_data_loaders_fixed

def quick_train_test():
    """Prueba rápida del entrenamiento con pocas épocas"""
    
    print("=== PRUEBA RAPIDA DE ENTRENAMIENTO RESNET-19 ===")
    
    # Configuración mínima para prueba
    config = {
        'batch_size': 16,  # Batch más pequeño para prueba
        'epochs': 3,       # Solo 3 épocas para prueba rápida
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'num_workers': 2
    }
    
    # Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    # Cargar datos
    print("\nCargando datasets...")
    datasets = load_datasets('./data', target_size=224)
    
    # Crear data loaders
    train_loader, val_loader, _ = create_data_loaders_fixed(datasets, batch_size=config['batch_size'])
    
    # Crear modelo
    print("\nCreando modelo ResNet-19...")
    model = create_model(num_classes=15)
    model.to(device)
    
    # Configurar entrenamiento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    print(f"\nIniciando entrenamiento de prueba ({config['epochs']} épocas)...")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        print(f"\nÉpoca {epoch+1}/{config['epochs']}")
        
        # Entrenamiento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc='Entrenando')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 50 == 0:  # Mostrar progreso cada 50 batches
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validación
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validando')
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f"Entrenamiento - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Validación   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
    
    training_time = time.time() - start_time
    print(f"\nPrueba completada en {training_time:.2f} segundos")
    print("La implementación funciona correctamente!")
    
    return model

if __name__ == "__main__":
    model = quick_train_test()
