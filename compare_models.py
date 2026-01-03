"""
Script de comparación entre modelos VGG16 Small y ResNet18.

Este script compara:
1. Métricas de clasificación (accuracy, loss)
2. Número de parámetros
3. Estructura de modelos
4. Preparación para comparaciones de explicabilidad

Uso:
    python compare_models.py --dataset blood
    python compare_models.py --dataset retina
    python compare_models.py --dataset breast
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch

from vgg16 import create_model as create_vgg16
from resnet18 import create_model as create_resnet18


def load_training_results(model_name: str, dataset: str):
    """Carga los resultados de entrenamiento de un modelo."""
    results_path = f"results/training_results_{model_name}_{dataset}.json"
    
    if not os.path.exists(results_path):
        print(f"⚠️  No se encontró {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    return data


def count_parameters(model):
    """Cuenta el número de parámetros de un modelo."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compare_architectures(num_classes: int, dataset: str):
    """Compara las arquitecturas de VGG16 y ResNet18."""
    print(f"\n{'='*60}")
    print(f"COMPARACIÓN DE ARQUITECTURAS - {dataset.upper()}")
    print(f"{'='*60}\n")
    
    # Crear modelos
    print("Creando VGG16 Small...")
    vgg16_model = create_vgg16(num_classes=num_classes)
    vgg16_total, vgg16_trainable = count_parameters(vgg16_model)
    
    print("\nCreando ResNet18...")
    resnet18_model = create_resnet18(num_classes=num_classes)
    resnet18_total, resnet18_trainable = count_parameters(resnet18_model)
    
    # Comparación de parámetros
    print("\n" + "="*60)
    print("PARÁMETROS DEL MODELO")
    print("="*60)
    print(f"{'Modelo':<20} {'Total':>15} {'Entrenables':>15}")
    print("-"*60)
    print(f"{'VGG16 Small':<20} {vgg16_total:>15,} {vgg16_trainable:>15,}")
    print(f"{'ResNet18':<20} {resnet18_total:>15,} {resnet18_trainable:>15,}")
    print("-"*60)
    
    # Diferencia
    diff = resnet18_total - vgg16_total
    diff_percent = (diff / resnet18_total) * 100
    print(f"\nDiferencia: {abs(diff):,} parámetros")
    if diff > 0:
        print(f"VGG16 tiene {diff_percent:.1f}% menos parámetros que ResNet18")
    else:
        print(f"VGG16 tiene {abs(diff_percent):.1f}% más parámetros que ResNet18")
    
    return {
        'vgg16': {'total': vgg16_total, 'trainable': vgg16_trainable},
        'resnet18': {'total': resnet18_total, 'trainable': resnet18_trainable}
    }


def compare_training_results(dataset: str):
    """Compara los resultados de entrenamiento de ambos modelos."""
    print(f"\n{'='*60}")
    print(f"COMPARACIÓN DE RESULTADOS DE ENTRENAMIENTO - {dataset.upper()}")
    print(f"{'='*60}\n")
    
    # Cargar resultados
    vgg16_results = load_training_results("vgg16", dataset)
    resnet18_results = load_training_results("resnet18", dataset)  
    
    # Intentar cargar con nombre alternativo si no existe
    if resnet18_results is None:
        resnet18_results = load_training_results("", dataset)
    
    if vgg16_results is None and resnet18_results is None:
        print("❌ No se encontraron resultados de entrenamiento para ningún modelo")
        return None
    
    comparison = {}
    
    # Comparar métricas de test
    print("MÉTRICAS EN TEST")
    print("-"*60)
    print(f"{'Métrica':<25} {'VGG16 Small':<20} {'ResNet18':<20}")
    print("-"*60)
    
    if vgg16_results and 'results' in vgg16_results:
        vgg16_test_acc = vgg16_results['results'].get('test_acc', 'N/A')
        vgg16_test_loss = vgg16_results['results'].get('test_loss', 'N/A')
        comparison['vgg16'] = {
            'test_acc': vgg16_test_acc,
            'test_loss': vgg16_test_loss
        }
    else:
        vgg16_test_acc = 'No disponible'
        vgg16_test_loss = 'No disponible'
        comparison['vgg16'] = None
    
    if resnet18_results and 'results' in resnet18_results:
        resnet18_test_acc = resnet18_results['results'].get('test_acc', 'N/A')
        resnet18_test_loss = resnet18_results['results'].get('test_loss', 'N/A')
        comparison['resnet18'] = {
            'test_acc': resnet18_test_acc,
            'test_loss': resnet18_test_loss
        }
    else:
        resnet18_test_acc = 'No disponible'
        resnet18_test_loss = 'No disponible'
        comparison['resnet18'] = None
    
    # Formatear para imprimir
    vgg16_acc_str = f"{vgg16_test_acc:.2f}%" if isinstance(vgg16_test_acc, (int, float)) else vgg16_test_acc
    resnet18_acc_str = f"{resnet18_test_acc:.2f}%" if isinstance(resnet18_test_acc, (int, float)) else resnet18_test_acc
    vgg16_loss_str = f"{vgg16_test_loss:.4f}" if isinstance(vgg16_test_loss, (int, float)) else vgg16_test_loss
    resnet18_loss_str = f"{resnet18_test_loss:.4f}" if isinstance(resnet18_test_loss, (int, float)) else resnet18_test_loss
    
    print(f"{'Test Accuracy':<25} {vgg16_acc_str:<20} {resnet18_acc_str:<20}")
    print(f"{'Test Loss':<25} {vgg16_loss_str:<20} {resnet18_loss_str:<20}")
    print("-"*60)
    
    # Calcular diferencia si ambos disponibles
    if isinstance(vgg16_test_acc, (int, float)) and isinstance(resnet18_test_acc, (int, float)):
        acc_diff = vgg16_test_acc - resnet18_test_acc
        print(f"\nDiferencia en Accuracy: {acc_diff:+.2f}%")
        if acc_diff > 0:
            print(f"→ VGG16 supera a ResNet18 por {acc_diff:.2f}%")
        elif acc_diff < 0:
            print(f"→ ResNet18 supera a VGG16 por {abs(acc_diff):.2f}%")
        else:
            print(f"→ Ambos modelos tienen el mismo accuracy")
    
    return comparison


def plot_training_curves(dataset: str):
    """Genera gráficas comparativas de las curvas de entrenamiento."""
    print(f"\n{'='*60}")
    print(f"GENERANDO GRÁFICAS COMPARATIVAS - {dataset.upper()}")
    print(f"{'='*60}\n")
    
    vgg16_results = load_training_results("vgg16", dataset)
    resnet18_results = load_training_results("resnet18", dataset)
    
    if resnet18_results is None:
        resnet18_results = load_training_results("", dataset)
    
    if vgg16_results is None and resnet18_results is None:
        print("❌ No se pueden generar gráficas sin datos de entrenamiento")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss de entrenamiento
    ax = axes[0, 0]
    if vgg16_results and 'history' in vgg16_results:
        ax.plot(vgg16_results['history']['train_loss'], label='VGG16 Small', linewidth=2)
    if resnet18_results and 'history' in resnet18_results:
        ax.plot(resnet18_results['history']['train_loss'], label='ResNet18', linewidth=2)
    ax.set_title('Pérdida en Entrenamiento', fontsize=12, fontweight='bold')
    ax.set_xlabel('Época')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss de validación
    ax = axes[0, 1]
    if vgg16_results and 'history' in vgg16_results:
        ax.plot(vgg16_results['history']['val_loss'], label='VGG16 Small', linewidth=2)
    if resnet18_results and 'history' in resnet18_results:
        ax.plot(resnet18_results['history']['val_loss'], label='ResNet18', linewidth=2)
    ax.set_title('Pérdida en Validación', fontsize=12, fontweight='bold')
    ax.set_xlabel('Época')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy de entrenamiento
    ax = axes[1, 0]
    if vgg16_results and 'history' in vgg16_results:
        ax.plot(vgg16_results['history']['train_acc'], label='VGG16 Small', linewidth=2)
    if resnet18_results and 'history' in resnet18_results:
        ax.plot(resnet18_results['history']['train_acc'], label='ResNet18', linewidth=2)
    ax.set_title('Accuracy en Entrenamiento', fontsize=12, fontweight='bold')
    ax.set_xlabel('Época')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy de validación
    ax = axes[1, 1]
    if vgg16_results and 'history' in vgg16_results:
        ax.plot(vgg16_results['history']['val_acc'], label='VGG16 Small', linewidth=2)
    if resnet18_results and 'history' in resnet18_results:
        ax.plot(resnet18_results['history']['val_acc'], label='ResNet18', linewidth=2)
    ax.set_title('Accuracy en Validación', fontsize=12, fontweight='bold')
    ax.set_xlabel('Época')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Comparación VGG16 vs ResNet18 - {dataset.upper()}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = f"results/comparison_vgg16_resnet18_{dataset}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Gráfica guardada en: {output_path}")
    plt.close()


def generate_comparison_report(dataset: str, num_classes: int):
    """Genera un reporte completo de comparación."""
    print(f"\n{'='*60}")
    print(f"REPORTE DE COMPARACIÓN - {dataset.upper()}")
    print(f"{'='*60}\n")
    
    report = {
        'dataset': dataset,
        'num_classes': num_classes,
        'architecture_comparison': None,
        'training_comparison': None,
        'timestamp': None
    }
    
    # Comparación de arquitecturas
    arch_comp = compare_architectures(num_classes, dataset)
    report['architecture_comparison'] = arch_comp
    
    # Comparación de resultados
    training_comp = compare_training_results(dataset)
    report['training_comparison'] = training_comp
    
    # Generar gráficas
    plot_training_curves(dataset)
    
    # Guardar reporte
    import datetime
    report['timestamp'] = datetime.datetime.now().isoformat()
    
    output_path = f"results/comparison_report_{dataset}.json"
    os.makedirs("results", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Reporte completo guardado en: {output_path}")
    
    # Resumen para explicabilidad
    print(f"\n{'='*60}")
    print("PREPARACIÓN PARA ANÁLISIS DE EXPLICABILIDAD")
    print(f"{'='*60}\n")
    print("Para comparar métricas de explicabilidad:")
    print(f"1. Asegúrate de tener modelos entrenados para {dataset}")
    print(f"   - results/best_model_vgg16_{dataset}.pth")
    print(f"   - results/best_model_{dataset}.pth")
    print("2. Genera explicaciones con: python xai_explanations.py")
    print("3. Evalúa con Quantus: python quantus_evaluation.py --dataset", dataset)
    print("4. Compara resultados en el notebook: 3. Quantus_eval.ipynb")


def main():
    parser = argparse.ArgumentParser(
        description="Comparación entre modelos VGG16 Small y ResNet18"
    )
    parser.add_argument(
        "--dataset",
        default="blood",
        choices=["blood", "retina", "breast"],
        help="Dataset para comparar: blood, retina o breast"
    )
    args = parser.parse_args()
    
    # Mapeo de número de clases
    num_classes_map = {
        "blood": 8,
        "retina": 5,
        "breast": 2
    }
    
    num_classes = num_classes_map[args.dataset]
    
    # Generar reporte completo
    generate_comparison_report(args.dataset, num_classes)
    
    print(f"\n{'='*60}")
    print("COMPARACIÓN COMPLETADA")
    print(f"{'='*60}\n")
    print(f"Revisa los archivos generados en la carpeta 'results/':")
    print(f"  - comparison_report_{args.dataset}.json")
    print(f"  - comparison_vgg16_resnet18_{args.dataset}.png")


if __name__ == "__main__":
    main()
