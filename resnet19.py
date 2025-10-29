"""
Implementación de ResNet-19 para clasificación de imágenes médicas
Basado en la arquitectura ResNet con bloques residuales
Incluye versión adaptativa y función de prueba.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Reproductibilidad

def set_seed(seed=42):
    """Fija las semillas para reproducibilidad"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Bloque básico de ResNet con conexiones residuales

class BasicBlock(nn.Module):
    """Bloque básico de ResNet con conexiones residuales"""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Arquitectura ResNet-19 personalizada para MedMNIST

class ResNet19(nn.Module):
    """Arquitectura ResNet-19 personalizada para MedMNIST"""

    def __init__(self, num_classes=15, input_channels=3):
        """
        Args:
            num_classes: Número total de clases (8 + 5 + 2 = 15)
            input_channels: Número de canales de entrada (3 para RGB, 1 para escala de grises)
        """
        super(ResNet19, self).__init__()

        self.in_channels = 64
        self.relu = nn.ReLU(inplace=True)

        # Capa inicial
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Bloques residuales
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)

        # Capas finales
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        # Inicialización de pesos
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """Crea una capa de bloques residuales"""
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Inicializa los pesos de la red"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass de la red"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Modelo adaptativo

class ResNet19Adaptive(nn.Module):
    """ResNet-19 adaptativo que maneja diferentes números de canales de entrada"""

    def __init__(self, num_classes=15):
        super(ResNet19Adaptive, self).__init__()
        self.rgb_model = ResNet19(num_classes=num_classes, input_channels=3)
        self.gray_model = ResNet19(num_classes=num_classes, input_channels=1)

    def forward(self, x):
        """Forward pass adaptativo según el número de canales"""
        if x.shape[1] == 3:
            return self.rgb_model(x)
        elif x.shape[1] == 1:
            return self.gray_model(x)
        else:
            raise ValueError(
                f"Número de canales no soportado: {x.shape[1]}. "
                "Solo se admiten 1 (grises) o 3 (RGB)."
            )

# Función para crear el modelo
def create_model(num_classes=15, pretrained=False):
    """
    Crea el modelo ResNet-19
    Args:
        num_classes: Número de clases (15 para los 3 datasets combinados)
        pretrained: Si usar pesos pre-entrenados (no disponible para esta versión)
    """
    if pretrained:
        print("⚠️ Advertencia: ResNet-19 personalizado no tiene pesos preentrenados disponibles.")
        print("Usando inicialización aleatoria.")

    model = ResNet19Adaptive(num_classes=num_classes)

    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Modelo ResNet-19 creado:")
    print(f"  - Parámetros totales: {total_params:,}")
    print(f"  - Parámetros entrenables: {trainable_params:,}")
    print(f"  - Número de clases: {num_classes}")

    return model

# Función de prueba para verificar que el modelo funciona correctamente

def test_model():
    """Prueba de funcionamiento básico del modelo"""
    print("Probando modelo ResNet-19...")
    set_seed(42)

    model = create_model(num_classes=15)
    model.eval()

    # Casos de prueba: diferentes tamaños y canales
    test_cases = [
        (1, 3, 224, 224),
        (1, 1, 224, 224),
        (4, 3, 224, 224),
        (4, 1, 224, 224),
    ]

    with torch.no_grad():
        for i, shape in enumerate(test_cases, 1):
            x = torch.randn(shape)
            output = model(x)
            print(f"Test {i}: Input {shape} -> Output {output.shape}")
            assert output.shape == (shape[0], 15), f"Error en test {i}"

    print("✅ Todos los tests pasaron correctamente!")
    return model

# Ejecutar la función de prueba

if __name__ == "__main__":
    model = test_model()
    print("\nArquitectura del modelo:")
    print(model)