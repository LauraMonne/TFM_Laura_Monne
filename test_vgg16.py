"""
Script de prueba para validar la implementación de VGG16 Small.
Este script verifica la estructura del modelo sin necesidad de entrenar.

Uso:
    python test_vgg16.py
"""

import sys

def test_vgg16_structure():
    """Prueba la estructura del modelo VGG16 Small."""
    print("="*60)
    print("PRUEBA DE ESTRUCTURA VGG16 SMALL")
    print("="*60)
    
    try:
        import torch
        print("✓ PyTorch importado correctamente")
    except ImportError:
        print("✗ PyTorch no disponible. Instala con: pip install torch")
        return False
    
    try:
        from vgg16 import VGG16Small, VGG16SmallAdaptive, create_model
        print("✓ Módulo vgg16 importado correctamente")
    except ImportError as e:
        print(f"✗ Error al importar vgg16: {e}")
        return False
    
    # Test 1: Crear modelo simple
    print("\n--- Test 1: Crear modelo VGG16Small ---")
    try:
        model = VGG16Small(num_classes=8, input_channels=3)
        print(f"✓ Modelo VGG16Small creado (8 clases, 3 canales)")
        
        # Contar parámetros
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Parámetros totales: {total_params:,}")
        print(f"  - Parámetros entrenables: {trainable_params:,}")
    except Exception as e:
        print(f"✗ Error al crear VGG16Small: {e}")
        return False
    
    # Test 2: Crear modelo adaptativo
    print("\n--- Test 2: Crear modelo VGG16SmallAdaptive ---")
    try:
        model_adaptive = VGG16SmallAdaptive(num_classes=5)
        print(f"✓ Modelo VGG16SmallAdaptive creado (5 clases)")
        
        # Contar parámetros
        total_params = sum(p.numel() for p in model_adaptive.parameters())
        trainable_params = sum(p.numel() for p in model_adaptive.parameters() if p.requires_grad)
        print(f"  - Parámetros totales: {total_params:,}")
        print(f"  - Parámetros entrenables: {trainable_params:,}")
    except Exception as e:
        print(f"✗ Error al crear VGG16SmallAdaptive: {e}")
        return False
    
    # Test 3: Crear con función create_model
    print("\n--- Test 3: Crear con create_model() ---")
    try:
        model = create_model(num_classes=2)
        print(f"✓ Modelo creado con create_model (2 clases)")
    except Exception as e:
        print(f"✗ Error al crear modelo con create_model: {e}")
        return False
    
    # Test 4: Verificar estructura de capas
    print("\n--- Test 4: Verificar estructura de capas ---")
    try:
        model = VGG16Small(num_classes=8, input_channels=3)
        
        # Verificar bloques convolucionales
        assert hasattr(model, 'block1'), "Falta block1"
        assert hasattr(model, 'block2'), "Falta block2"
        assert hasattr(model, 'block3'), "Falta block3"
        assert hasattr(model, 'block4'), "Falta block4"
        assert hasattr(model, 'block5'), "Falta block5"
        print("✓ Todos los bloques convolucionales presentes")
        
        # Verificar avgpool y classifier
        assert hasattr(model, 'avgpool'), "Falta avgpool"
        assert hasattr(model, 'classifier'), "Falta classifier"
        print("✓ Avgpool y classifier presentes")
        
        # Verificar número de capas en classifier
        assert len(model.classifier) == 7, f"Classifier debería tener 7 capas, tiene {len(model.classifier)}"
        print("✓ Classifier tiene el número correcto de capas")
        
    except AssertionError as e:
        print(f"✗ Error de estructura: {e}")
        return False
    except Exception as e:
        print(f"✗ Error inesperado: {e}")
        return False
    
    # Test 5: Verificar forward pass (sin ejecutar, solo estructura)
    print("\n--- Test 5: Verificar método forward ---")
    try:
        model = VGG16Small(num_classes=8, input_channels=3)
        model.eval()
        
        # Crear tensor de prueba pequeño para verificar dimensiones
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            
            # Verificar shape de salida
            expected_shape = (1, 8)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            print(f"✓ Forward pass correcto: {x.shape} -> {output.shape}")
            
    except Exception as e:
        print(f"✗ Error en forward pass: {e}")
        return False
    
    # Test 6: Verificar modelo adaptativo con diferentes canales
    print("\n--- Test 6: Modelo adaptativo con diferentes canales ---")
    try:
        model = VGG16SmallAdaptive(num_classes=8)
        model.eval()
        
        with torch.no_grad():
            # Test con RGB (3 canales)
            x_rgb = torch.randn(2, 3, 224, 224)
            output_rgb = model(x_rgb)
            assert output_rgb.shape == (2, 8), f"Expected (2, 8), got {output_rgb.shape}"
            print(f"✓ RGB (3 canales): {x_rgb.shape} -> {output_rgb.shape}")
            
            # Test con escala de grises (1 canal)
            x_gray = torch.randn(2, 1, 224, 224)
            output_gray = model(x_gray)
            assert output_gray.shape == (2, 8), f"Expected (2, 8), got {output_gray.shape}"
            print(f"✓ Grayscale (1 canal): {x_gray.shape} -> {output_gray.shape}")
            
    except Exception as e:
        print(f"✗ Error en modelo adaptativo: {e}")
        return False
    
    # Test 7: Comparar con ResNet18
    print("\n--- Test 7: Comparación con ResNet18 ---")
    try:
        from resnet18 import create_model as create_resnet18
        
        vgg16_model = create_model(num_classes=8)
        resnet18_model = create_resnet18(num_classes=8)
        
        vgg16_params = sum(p.numel() for p in vgg16_model.parameters())
        resnet18_params = sum(p.numel() for p in resnet18_model.parameters())
        
        reduction = ((resnet18_params - vgg16_params) / resnet18_params) * 100
        
        print(f"  VGG16 Small:  {vgg16_params:>12,} parámetros")
        print(f"  ResNet18:     {resnet18_params:>12,} parámetros")
        print(f"  Reducción:    {reduction:>11.1f}%")
        
        # VGG16 Small debería tener menos parámetros
        assert vgg16_params < resnet18_params, "VGG16 Small debería tener menos parámetros que ResNet18"
        print("✓ VGG16 Small tiene menos parámetros que ResNet18")
        
    except ImportError:
        print("⚠ ResNet18 no disponible para comparación")
    except AssertionError as e:
        print(f"✗ Error de aserción: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ Error inesperado en comparación: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✅ TODAS LAS PRUEBAS PASARON")
    print("="*60)
    print("\nEl modelo VGG16 Small está correctamente implementado y listo para entrenar.")
    print("\nPróximos pasos:")
    print("  1. Entrenar: python train_vgg16.py --dataset blood")
    print("  2. Comparar: python compare_models.py --dataset blood")
    print("\n")
    
    return True


def main():
    """Función principal."""
    success = test_vgg16_structure()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
