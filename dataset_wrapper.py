"""
Wrapper para datasets MedMNIST que convierte labels a escalares
"""

import torch
from torch.utils.data import Dataset

class MedMNISTWrapper(Dataset):
    """Wrapper que convierte labels de MedMNIST a escalares"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Convertir label a escalar si es necesario
        if hasattr(label, 'item'):
            label = label.item()
        elif isinstance(label, (list, tuple)) and len(label) == 1:
            label = label[0]
        elif isinstance(label, torch.Tensor):
            label = label.item()
        
        return img, label
