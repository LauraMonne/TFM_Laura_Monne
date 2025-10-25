"""
Script para preparar los datasets de MedMNIST para entrenamiento con ResNet-19
Incluye BloodMNIST, RetinaMNIST y BreastMNIST con redimensionamiento a 224x224
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
import medmnist
from medmnist import INFO, Evaluator
import matplotlib.pyplot as plt
from torchvision import transforms
import json

def get_dataset_info():
    """Obtiene información de los datasets de MedMNIST"""
    datasets = {
        'bloodmnist': {
            'class': medmnist.BloodMNIST,
            'info': INFO['bloodmnist'],
            'task': 'multi-class',
            'n_channels': 3,
            'n_classes': 8
        },
        'retinamnist': {
            'class': medmnist.RetinaMNIST,
            'info': INFO['retinamnist'],
            'task': 'multi-class',
            'n_channels': 3,
            'n_classes': 5
        },
        'breastmnist': {
            'class': medmnist.BreastMNIST,
            'info': INFO['breastmnist'],
            'task': 'binary-class',
            'n_channels': 1,
            'n_classes': 2
        }
    }
    return datasets

def create_transforms(target_size=224, n_channels=3):
    """Crea las transformaciones para los datos"""
    
    if n_channels == 1:
        # Para imágenes en escala de grises (BreastMNIST)
        mean = [0.5]
        std = [0.5]
        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
    else:
        # Para imágenes RGB (BloodMNIST, RetinaMNIST)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    
    # Transformaciones para entrenamiento (con data augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        color_jitter,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Transformaciones para validación/test (sin data augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform

def load_datasets(data_dir='./data', target_size=224):
    """Carga los datasets de MedMNIST"""
    
    print("Cargando datasets de MedMNIST...")
    datasets_info = get_dataset_info()
    
    datasets = {}
    
    for dataset_name, info in datasets_info.items():
        print(f"\nCargando {dataset_name.upper()}...")
        
        # Crear directorio para el dataset
        dataset_path = os.path.join(data_dir, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        
        # Crear transformaciones específicas para este dataset
        train_transform, val_transform = create_transforms(target_size, info['n_channels'])
        
        # Cargar dataset
        dataset_class = info['class']
        
        train_dataset = dataset_class(
            split='train', 
            transform=train_transform,
            download=True,
            root=data_dir
        )
        
        val_dataset = dataset_class(
            split='val', 
            transform=val_transform,
            download=True,
            root=data_dir
        )
        
        test_dataset = dataset_class(
            split='test', 
            transform=val_transform,
            download=True,
            root=data_dir
        )
        
        datasets[dataset_name] = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'info': info
        }
        
        print(f"  - Entrenamiento: {len(train_dataset)} muestras")
        print(f"  - Validación: {len(val_dataset)} muestras")
        print(f"  - Test: {len(test_dataset)} muestras")
        print(f"  - Clases: {info['n_classes']}")
        print(f"  - Canales: {info['n_channels']}")
    
    return datasets

def create_combined_dataset(datasets, split='train'):
    """Combina todos los datasets en uno solo"""
    combined_datasets = []
    
    for dataset_name, dataset_dict in datasets.items():
        combined_datasets.append(dataset_dict[split])
    
    return ConcatDataset(combined_datasets)

def create_visualization_transforms(target_size=224):
    """Crea transformaciones para visualización (sin normalización)"""
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor()
    ])

def visualize_samples(datasets, num_samples=5):
    """Visualiza muestras de cada dataset"""
    
    fig, axes = plt.subplots(len(datasets), num_samples, figsize=(15, 3*len(datasets)))
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (dataset_name, dataset_dict) in enumerate(datasets.items()):
        # Crear dataset temporal para visualización sin normalización
        info = dataset_dict['info']
        dataset_class = info['class']
        
        # Crear transformación de visualización
        vis_transform = create_visualization_transforms()
        
        # Cargar dataset temporal para visualización
        vis_dataset = dataset_class(
            split='train', 
            transform=vis_transform,
            download=False,
            root='./data'
        )
        
        for j in range(min(num_samples, len(vis_dataset))):
            img, label = vis_dataset[j]
            
            # Convertir tensor a numpy para visualización
            if img.shape[0] == 1:  # Imagen en escala de grises
                img_np = img.squeeze(0).numpy()
                axes[i, j].imshow(img_np, cmap='gray')
            else:  # Imagen RGB
                img_np = img.permute(1, 2, 0).numpy()
                axes[i, j].imshow(img_np)
            
            axes[i, j].set_title(f'{dataset_name}\nClase: {label}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

def save_dataset_info(datasets, filename='dataset_info.json'):
    """Guarda información de los datasets"""
    info_dict = {}
    
    for dataset_name, dataset_dict in datasets.items():
        info_dict[dataset_name] = {
            'train_samples': len(dataset_dict['train']),
            'val_samples': len(dataset_dict['val']),
            'test_samples': len(dataset_dict['test']),
            'n_classes': dataset_dict['info']['n_classes'],
            'n_channels': dataset_dict['info']['n_channels'],
            'task': dataset_dict['info']['task']
        }
    
    with open(filename, 'w') as f:
        json.dump(info_dict, f, indent=2)
    
    print(f"Información de datasets guardada en {filename}")

def main():
    """Función principal para preparar los datos"""
    
    # Configuración
    data_dir = './data'
    target_size = 224
    
    print("=== PREPARACIÓN DE DATOS MEDMNIST ===")
    print(f"Tamaño objetivo de imagen: {target_size}x{target_size}")
    print(f"Directorio de datos: {data_dir}")
    
    # Crear directorio de datos
    os.makedirs(data_dir, exist_ok=True)
    
    # Cargar datasets
    datasets = load_datasets(data_dir, target_size)
    
    # Crear datasets combinados
    print("\nCreando datasets combinados...")
    combined_train = create_combined_dataset(datasets, 'train')
    combined_val = create_combined_dataset(datasets, 'val')
    combined_test = create_combined_dataset(datasets, 'test')
    
    print(f"Dataset combinado de entrenamiento: {len(combined_train)} muestras")
    print(f"Dataset combinado de validación: {len(combined_val)} muestras")
    print(f"Dataset combinado de test: {len(combined_test)} muestras")
    
    # Visualizar muestras
    print("\nGenerando visualización de muestras...")
    visualize_samples(datasets)
    
    # Guardar información
    save_dataset_info(datasets)
    
    print("\n=== PREPARACIÓN COMPLETADA ===")
    print("Los datos están listos para el entrenamiento con ResNet-19")
    
    return datasets, {
        'train': combined_train,
        'val': combined_val,
        'test': combined_test
    }

if __name__ == "__main__":
    datasets, combined_datasets = main()
