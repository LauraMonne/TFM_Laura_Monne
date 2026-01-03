"""
Implementación de VGG16 pequeño para clasificación de imágenes médicas
Basado en la arquitectura VGG estándar pero con filtros reducidos
Incluye Batch Normalization para mejorar la estabilidad durante el entrenamiento
"""

import torch
import torch.nn as nn

# Reproductibilidad y rendimiento
def set_seed(seed=42):
    """
    Fija las semillas aleatorias para reproducibilidad.
    
    Args:
        seed: Semilla para generadores aleatorios
    
    Note:
        torch.backends.cudnn.benchmark=True optimiza CUDNN para modelos con
        tamaño de entrada fijo, mejorando el rendimiento en GPU.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optimización CUDNN para tamaños de entrada fijos (mejor rendimiento en GPU)
    torch.backends.cudnn.benchmark = True


class VGG16Small(nn.Module):
    """
    VGG16 reducido con Batch Normalization.
    
    Arquitectura:
    - Bloque 1: 2 capas conv (32 filtros) + MaxPool
    - Bloque 2: 2 capas conv (64 filtros) + MaxPool
    - Bloque 3: 3 capas conv (128 filtros) + MaxPool
    - Bloque 4: 3 capas conv (256 filtros) + MaxPool
    - Bloque 5: 3 capas conv (256 filtros) + MaxPool
    - Capas densas: 512 -> 256 -> num_classes
    
    En comparación con VGG16 original (64->128->256->512->512),
    esta versión usa (32->64->128->256->256) para reducir parámetros.
    """
    
    def __init__(self, num_classes=15, input_channels=3):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Bloque 1: 2 capas conv con 32 filtros
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Bloque 2: 2 capas conv con 64 filtros
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Bloque 3: 3 capas conv con 128 filtros
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Bloque 4: 3 capas conv con 256 filtros
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Bloque 5: 3 capas conv con 256 filtros (reducido de 512 en VGG16 original)
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Adaptive pooling para hacer el modelo flexible al tamaño de entrada
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Capas densas (fully connected)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa los pesos de la red usando el método de inicialización de Kaiming."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass del modelo."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG16SmallAdaptive(nn.Module):
    """
    Modelo VGG16 pequeño adaptativo que soporta tanto RGB como escala de grises.
    Contiene dos modelos: uno para RGB (3 canales) y otro para escala de grises (1 canal).
    """
    
    def __init__(self, num_classes=15):
        super().__init__()
        self.rgb_model = VGG16Small(num_classes=num_classes, input_channels=3)
        self.gray_model = VGG16Small(num_classes=num_classes, input_channels=1)
    
    def forward(self, x):
        """Selecciona el modelo según el número de canales de entrada."""
        if x.shape[1] == 3:
            return self.rgb_model(x)
        elif x.shape[1] == 1:
            return self.gray_model(x)
        else:
            raise ValueError(f"Canales no soportados: {x.shape[1]} (esperado 1 o 3)")


def create_model(num_classes=15, pretrained=False, freeze_backbone=False):
    """
    Crea modelo VGG16 pequeño.
    
    Args:
        num_classes: Número de clases de salida
        pretrained: Si True, usa pesos pre-entrenados (no implementado para VGG16Small)
        freeze_backbone: Si True y pretrained=True, congela las capas convolucionales
    
    Returns:
        Modelo VGG16Small adaptativo
    """
    if pretrained:
        print("⚠️  Advertencia: Pre-entrenamiento no disponible para VGG16Small")
        print("   Se creará modelo sin pre-entrenar")
    
    # Usar nuestro modelo personalizado
    model = VGG16SmallAdaptive(num_classes=num_classes)
    
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Modelo VGG16 Small creado:")
    print(f"  - Parámetros totales: {total:,}")
    print(f"  - Parámetros entrenables: {train:,}")
    print(f"  - Número de clases: {num_classes}")
    
    return model


if __name__ == "__main__":
    print("Probando modelo VGG16 Small...")
    set_seed(42)
    m = create_model(num_classes=15)
    m.eval()
    
    with torch.no_grad():
        for shape in [(1, 3, 224, 224), (1, 1, 224, 224)]:
            x = torch.randn(shape)
            y = m(x)
            print(f"{shape} -> {y.shape}")
            assert y.shape == (shape[0], 15), f"Expected shape (1, 15), got {y.shape}"
    
    print("✅ Test de forma OK")
    
    # Comparar número de parámetros con ResNet18
    print("\n📊 Comparación de parámetros:")
    print(f"   VGG16 Small: {total:,} parámetros")
    print(f"   ResNet18 típico: ~11,000,000 parámetros")
    print(f"   Reducción: ~{(1 - total/11000000)*100:.1f}%")
