"""
Implementación de ResNet-18 para clasificación de imágenes médicas
Basado en la arquitectura ResNet estándar con bloques residuales
Incluye versión adaptativa y función de prueba.
"""

import torch
import torch.nn as nn
import torchvision.models as models

# Reproductibilidad y rendimiento
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Rendimiento en GPU:
    torch.backends.cudnn.benchmark = True

class BasicBlock(nn.Module):
    """Bloque básico de ResNet con conexión residual."""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=15, input_channels=3):
        super().__init__()
        self.in_channels = 64
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
# Construye 4 capas con BasicBlock (64→128→256→512 canales), cada una con 2 bloques. Pooling adaptativo y capa lineal final.
        self.layer1 = self._make_layer(BasicBlock, 64,  2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512, num_classes)
# Inicialización de pesos.
# Inicialización: capa inicial 7x7 (stride 2), BatchNorm, MaxPooling.
# 4 capas residuales: 2 bloques BasicBlock con 64 canales, 2 bloques BasicBlock con 128 canales, 2 bloques BasicBlock con 256 canales, 2 bloques BasicBlock con 512 canales.
# Capa final: AdaptiveAvgPool2d (1, 1), Linear (512 → num_classes).
        self._initialize_weights()

# Creación de capas residuales.
# Se crea una capa residual con el bloque básico y se repite para construir la red.
# out_channels es el número de canales de salida, blocks es el número de bloques,
# stride es el stride de la primera convolución.
# downsample es la conexión residual.
# El bloque básico se repite en las capas residuales para construir la red.
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
# Se inicializan los pesos de la red usando el método de inicialización de Kaiming.

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
# Forward: capa inicial, pooling, 4 capas residuales, pooling adaptativo, aplanado y salida lineal.
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
# Contiene dos modelos: uno para RGB (3 canales) y otro para escala de grises (1 canal).
# Forward: si es RGB, se usa el modelo RGB, si es escala de grises, se usa el modelo escala de grises.
class ResNet18Adaptive(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.rgb_model  = ResNet18(num_classes=num_classes, input_channels=3)
        self.gray_model = ResNet18(num_classes=num_classes, input_channels=1)
# Selecciona el modelo según el número de canales de entrada.
    def forward(self, x):
        if x.shape[1] == 3:
            return self.rgb_model(x)
        elif x.shape[1] == 1:
            return self.gray_model(x)
        else:
            raise ValueError(f"Canales no soportados: {x.shape[1]} (esperado 1 o 3)")
# Crea el modelo ResNet-18 adaptativo,  cuenta parámetros e imprime estadísticas.
def create_model(num_classes=15, pretrained=False, freeze_backbone=False):
    """
    Crea modelo ResNet-18.
    
    Args:
        num_classes: Número de clases de salida
        pretrained: Si True, usa pesos pre-entrenados de ImageNet (torchvision)
        freeze_backbone: Si True y pretrained=True, congela las capas convolucionales
    """
    if pretrained:
        # Usar ResNet-18 de torchvision con pesos pre-entrenados de ImageNet
        print("📥 Cargando ResNet-18 pre-entrenado en ImageNet...")
        resnet18_pretrained = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Reemplazar la capa final para nuestro número de clases
        # Añadir Dropout más agresivo para reducir sobreajuste
        num_features = resnet18_pretrained.fc.in_features
        dropout_rate = 0.7 if not freeze_backbone else 0.5  # Más dropout si entrenamos todo
        resnet18_pretrained.fc = nn.Sequential(
            nn.Dropout(dropout_rate),  # Dropout más agresivo (0.7) para reducir sobreajuste
            nn.Linear(num_features, num_classes)
        )
        
        # Congelar capas del backbone si se solicita (útil para fine-tuning)
        if freeze_backbone:
            print("🔒 Congelando capas del backbone (solo se entrenará la capa final)...")
            # Congelar todas las capas excepto la final
            for name, param in resnet18_pretrained.named_parameters():
                if 'fc' not in name:  # Congelar todo excepto la capa final
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            # Contar parámetros entrenables
            trainable = sum(p.numel() for p in resnet18_pretrained.parameters() if p.requires_grad)
            total = sum(p.numel() for p in resnet18_pretrained.parameters())
            print(f"   Parámetros entrenables: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        
        model = resnet18_pretrained
        print("✅ Modelo pre-entrenado cargado correctamente")
    else:
        # Usar nuestro modelo personalizado (sin pre-entrenamiento)
        model = ResNet18Adaptive(num_classes=num_classes)
    
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Modelo ResNet-18 creado:")
    print(f"  - Parámetros totales: {total:,}")
    print(f"  - Parámetros entrenables: {train:,}")
    print(f"  - Número de clases: {num_classes}")
    if pretrained:
        print(f"  - Pre-entrenado: ImageNet")
    return model
# Prueba el modelo con imágenes RGB y en escala de grises, verificando que la salida tenga la forma esperada (batch_size, 15).
# Si la salida no tiene la forma esperada, lanza un error.
if __name__ == "__main__":
    print("Probando modelo ResNet-18...")
    set_seed(42)
    m = create_model(num_classes=15)
    m.eval()
    with torch.no_grad():
        for shape in [(1,3,224,224),(1,1,224,224)]:
            x = torch.randn(shape)
            y = m(x)
            print(shape, "->", y.shape)
            assert y.shape == (shape[0], 15)
    print("✅ Test de forma OK")
