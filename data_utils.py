"""
Collate function personalizado para manejar diferentes números de canales
"""

import torch
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    """
    Collate function personalizado que maneja diferentes números de canales
    Convierte todas las imágenes a 3 canales para compatibilidad
    """
    images, labels = zip(*batch)
    
    # Convertir todas las imágenes a 3 canales
    processed_images = []
    for img in images:
        if img.shape[0] == 1:  # Escala de grises
            # Duplicar el canal para crear 3 canales
            img_3ch = img.repeat(3, 1, 1)
        else:  # Ya es RGB
            img_3ch = img
        processed_images.append(img_3ch)
    
    # Stack las imágenes y labels
    images_tensor = torch.stack(processed_images, 0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return images_tensor, labels_tensor

def create_data_loaders_fixed(datasets, batch_size=32):
    """Crea data loaders con collate function personalizado"""
    
    from prepare_data import create_combined_dataset
    from dataset_wrapper import MedMNISTWrapper
    
    # Crear datasets combinados
    train_dataset_raw = create_combined_dataset(datasets, 'train')
    val_dataset_raw = create_combined_dataset(datasets, 'val')
    test_dataset_raw = create_combined_dataset(datasets, 'test')
    
    # Aplicar wrapper
    train_dataset = MedMNISTWrapper(train_dataset_raw)
    val_dataset = MedMNISTWrapper(val_dataset_raw)
    test_dataset = MedMNISTWrapper(test_dataset_raw)
    
    # Crear data loaders con collate function personalizado
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    print(f"Data loaders creados:")
    print(f"  - Entrenamiento: {len(train_dataset)} muestras, {len(train_loader)} batches")
    print(f"  - Validación: {len(val_dataset)} muestras, {len(val_loader)} batches")
    print(f"  - Test: {len(test_dataset)} muestras, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader
