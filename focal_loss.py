"""
Implementación de Focal Loss para manejar clases desbalanceadas.

Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Útil para datasets con clases muy desbalanceadas como RetinaMNIST.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss para manejar clases desbalanceadas.
    
    Args:
        alpha: Tensor de pesos por clase (opcional). Si es None, usa pesos uniformes.
        gamma: Factor de focusing (default: 2.0). Valores más altos enfocan más en ejemplos difíciles.
        reduction: 'mean', 'sum', o 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calcular Cross Entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Calcular probabilidad de la clase correcta
        pt = torch.exp(-ce_loss)
        
        # Calcular Focal Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

