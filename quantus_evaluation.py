"""
Evaluación cuantitativa de explicabilidad usando Quantus.

Mide 5 dimensiones para Grad-CAM, Grad-CAM++ (opcional),
Integrated Gradients y Saliency:
- Fidelidad (FaithfulnessCorrelation)
- Robustez (AvgSensitivity)
- Complejidad (Complexity/Entropy)
- Aleatorización (ModelParameterRandomisation/MPRT)
- Localización (RegionPerturbation como aproximación de Localization Ratio)

Uso:
    python quantus_evaluation.py --num_samples 30 --methods gradcam integrated_gradients saliency
"""

import argparse
import json
import os
import traceback
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

try:
    import quantus
except ImportError as exc:
    raise SystemExit(
        "❌ quantus no está instalado. Ejecuta: pip install quantus"
    ) from exc

from prepare_data import load_datasets
from data_utils import create_data_loaders_fixed
from xai_explanations import (
    XAIExplainer,
    load_trained_model,
)


# ============================================================
#  Argumentos de línea de comandos
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluación cuantitativa de XAI con Quantus."
    )
    parser.add_argument(
        "--model_path",
        default="results/best_model.pth",
        help="Ruta al checkpoint entrenado.",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Directorio con los datasets MedMNIST.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="Número de muestras del set de test a evaluar.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size al recorrer el conjunto de test.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu o cuda.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["gradcam", "integrated_gradients", "saliency"],
        help="Métodos XAI a evaluar.",
    )
    parser.add_argument(
        "--output",
        default="outputs/quantus_metrics.json",
        help="Ruta de guardado para los resultados.",
    )
    return parser.parse_args()


# ============================================================
#  Utilidades de datos
# ============================================================

def collect_samples(test_loader, num_samples, device):
    """Recopila x_batch, y_batch del conjunto de test."""
    xs, ys = [], []
    with torch.no_grad():
        for idx, (data, target) in enumerate(tqdm(test_loader, desc="Recolectando muestras")):
            if idx >= num_samples:
                break
            xs.append(data)
            ys.append(target)
    if not xs:
        raise RuntimeError("No se encontraron muestras. Ajusta num_samples.")
    x_batch = torch.cat(xs, dim=0).to(device)
    y_batch = torch.cat(ys, dim=0).to(device)
    return x_batch, y_batch


def hwc(tensor_batch: torch.Tensor) -> np.ndarray:
    """Convierte lote BCHW -> BHWC para Quantus."""
    np_batch = tensor_batch.detach().cpu().numpy()
    if np_batch.ndim == 4:
        # Si está en formato BCHW (batch, channels, height, width)
        if np_batch.shape[1] in (1, 3) and np_batch.shape[-1] not in (1, 3):
            np_batch = np.transpose(np_batch, (0, 2, 3, 1))  # BCHW -> BHWC
        # Si ya está en BHWC, no hacer nada
    return np_batch


# ============================================================
#  Atribuciones XAI (reutiliza XAIExplainer)
# ============================================================

def expand_heatmap_to_channels(heatmap: np.ndarray, channels: int) -> torch.Tensor:
    """Expande un heatmap HxW a CxHxW repitiendo por canal."""
    if heatmap.ndim != 2:
        raise ValueError("El heatmap debe ser 2D.")
    tensor = torch.tensor(heatmap, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).repeat(channels, 1, 1)  # Resultado: (C, H, W)
    return tensor


def compute_attributions(
    explainer: XAIExplainer,
    x_batch: torch.Tensor,
    preds: torch.Tensor,
    method: str,
) -> torch.Tensor:
    """Genera atribuciones para todo el batch usando el método especificado."""
    attributions = []
    for idx in tqdm(range(len(x_batch)), desc=f"Atribuciones {method}"):
        sample = x_batch[idx : idx + 1]
        target_class = int(preds[idx].item())
        try:
            if method == "gradcam":
                result = explainer.generate_gradcam(
                    sample, target_class, save_path=None
                )
                if result is None:
                    raise RuntimeError("Grad-CAM retornó None")
                _, heatmap = result
                attr = expand_heatmap_to_channels(heatmap, sample.shape[1])
            elif method == "gradcampp":
                result = explainer.generate_gradcampp(
                    sample, target_class, save_path=None
                )
                if result is None:
                    raise RuntimeError("Grad-CAM++ retornó None")
                _, heatmap = result
                attr = expand_heatmap_to_channels(heatmap, sample.shape[1])
            elif method == "integrated_gradients":
                result = explainer.generate_integrated_gradients(
                    sample, target_class, save_path=None
                )
                if result is None:
                    raise RuntimeError("IG retornó None")
                attr = result[1][0].detach().cpu()
            elif method == "saliency":
                result = explainer.generate_saliency_map(
                    sample, target_class, save_path=None
                )
                if result is None:
                    raise RuntimeError("Saliency retornó None")
                attr = result[1][0].detach().cpu()
            else:
                raise ValueError(f"Método desconocido: {method}")
        except Exception as err:  # noqa: BLE001
            print(f"⚠️ Error generando atribución para muestra {idx}: {err}")
            attr = torch.zeros_like(sample[0].cpu())
        attributions.append(attr)
    return torch.stack(attributions, dim=0)


# ============================================================
#  Evaluación con Quantus
# ============================================================

class ModelWrapper(torch.nn.Module):
    """Wrapper explícito para asegurar compatibilidad con Quantus."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def __getattr__(self, name):
        # Delegar atributos al modelo subyacente
        return getattr(self.model, name)


def build_predict_fn(model: torch.nn.Module, device: torch.device):
    """Devuelve una función predict_func compatible con Quantus."""

    def predict(x_np: np.ndarray):
        model.eval()
        with torch.no_grad():
            if not isinstance(x_np, np.ndarray):
                raise TypeError("predict_func espera un np.ndarray")
            tensor = torch.tensor(x_np, dtype=torch.float32)
            if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(0, 3, 1, 2)
            tensor = tensor.to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            return probs.detach().cpu().numpy()

    return predict


def evaluate_metric(metric, model, wrapped_model, x_batch_np, y_batch_np, a_batch_np, device):
    """Evalúa una métrica de Quantus pasando el modelo directamente."""
    # Intentar múltiples formas de pasar el modelo (sin predict_func)
    methods_to_try = [
        # Método 1: Modelo envuelto directamente
        lambda: metric(
            model=wrapped_model,
            x_batch=x_batch_np,
            y_batch=y_batch_np,
            a_batch=a_batch_np,
            device=device,
        ),
        # Método 2: Modelo original directamente
        lambda: metric(
            model=model,
            x_batch=x_batch_np,
            y_batch=y_batch_np,
            a_batch=a_batch_np,
            device=device,
        ),
    ]
    
    last_error = None
    for method in methods_to_try:
        try:
            scores = method()
            return float(np.nanmean(scores)), float(np.nanstd(scores)), scores
        except Exception as e:
            last_error = e
            continue
    
    # Si todos fallan, lanzar el último error
    raise last_error


def evaluate_methods(model, explainer, x_batch, y_batch, methods):
    device = next(model.parameters()).device
    model.eval()  # Asegurar que el modelo está en modo evaluación
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    
    # Crear un wrapper explícito del modelo para Quantus
    # Esto asegura que Quantus reconozca correctamente el tipo
    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()

    # Convertimos a numpy BHWC para Quantus
    x_batch_np = hwc(x_batch)
    y_batch_np = y_batch.detach().cpu().numpy()
    
    # Asegurar que x_batch_np esté en formato BHWC
    # Si está en BCHW, convertir a BHWC
    if x_batch_np.ndim == 4:
        if x_batch_np.shape[1] in (1, 3) and x_batch_np.shape[-1] not in (1, 3):
            x_batch_np = np.transpose(x_batch_np, (0, 2, 3, 1))

    with torch.no_grad():
        logits = model(x_batch)
        preds = logits.argmax(dim=1)

    # Usar MPRT en lugar de ModelParameterRandomisation (deprecated)
    try:
        RandomizationMetric = quantus.MPRT
    except AttributeError:
        RandomizationMetric = quantus.ModelParameterRandomisation
    
    # Inicializar métricas con parámetros compatibles
    # Usamos try/except para manejar diferentes versiones de Quantus
    # Nota: abs=True porque el warning indica que se debe aplicar operación absoluta
    metrics = {}
    
    # FaithfulnessCorrelation
    try:
        metrics["faithfulness"] = quantus.FaithfulnessCorrelation(
            abs=True,
            normalise=False,
        )
    except Exception:
        # Si falla, intentar sin parámetros
        metrics["faithfulness"] = quantus.FaithfulnessCorrelation()
    
    # AvgSensitivity
    try:
        metrics["robustness"] = quantus.AvgSensitivity(
            abs=True,
            normalise=False,
        )
    except Exception:
        metrics["robustness"] = quantus.AvgSensitivity()
    
    # Complexity (puede llamarse Entropy o Complexity según la versión)
    try:
        metrics["complexity"] = quantus.Complexity(
            abs=True,
            normalise=False,
        )
    except AttributeError:
        try:
            # Intentar con Entropy si Complexity no existe
            metrics["complexity"] = quantus.Entropy(
                abs=True,
                normalise=False,
            )
        except AttributeError:
            # Si tampoco existe Entropy, intentar sin parámetros
            try:
                metrics["complexity"] = quantus.Complexity()
            except AttributeError:
                metrics["complexity"] = quantus.Entropy()
    except Exception:
        # Si falla por otros motivos, intentar sin parámetros
        try:
            metrics["complexity"] = quantus.Complexity()
        except AttributeError:
            metrics["complexity"] = quantus.Entropy()
    
    # Randomization
    try:
        metrics["randomization"] = RandomizationMetric(
            abs=True,
            normalise=False,
        )
    except Exception:
        metrics["randomization"] = RandomizationMetric()
    
    # RegionPerturbation
    try:
        metrics["localization"] = quantus.RegionPerturbation(
            abs=True,
            normalise=False,
        )
    except Exception:
        metrics["localization"] = quantus.RegionPerturbation()

    for method in methods:
        print(f"\n=== Evaluando {method} ===")
        try:
            attr_batch = compute_attributions(explainer, x_batch, preds, method)
        except ValueError as err:
            print(f"⚠️ {err}. Saltando método.")
            continue

        a_batch_np = hwc(attr_batch)
        
        # Debug: verificar formas antes de evaluar
        # print(f"DEBUG: x_batch_np shape: {x_batch_np.shape}, a_batch_np shape: {a_batch_np.shape}")
        
        # Asegurar que las atribuciones tengan el mismo formato que el input
        # Ambos deben estar en BHWC (batch, height, width, channels)
        # Si las formas no coinciden, verificar y corregir
        if a_batch_np.shape != x_batch_np.shape:
            # Verificar si las atribuciones están en BCHW en lugar de BHWC
            if (a_batch_np.ndim == 4 and x_batch_np.ndim == 4 and 
                a_batch_np.shape[0] == x_batch_np.shape[0] and
                a_batch_np.shape[1] in (1, 3) and a_batch_np.shape[-1] not in (1, 3)):
                # Atribuciones están en BCHW, convertir a BHWC
                a_batch_np = np.transpose(a_batch_np, (0, 2, 3, 1))
        
        # Verificar que las formas coincidan después de la corrección
        if a_batch_np.shape != x_batch_np.shape:
            print(f"⚠️ Advertencia: Formas no coinciden - x_batch: {x_batch_np.shape}, a_batch: {a_batch_np.shape}")
        
        method_results = {}

        for metric_name, metric in metrics.items():
            print(f" -> {metric_name}")
            try:
                mean, std, scores = evaluate_metric(
                    metric, model, wrapped_model, x_batch_np, y_batch_np, a_batch_np, device
                )
                method_results[metric_name] = {
                    "mean": mean,
                    "std": std,
                    "scores": [float(s) for s in scores],
                }
                print(f"    {mean:.4f} ± {std:.4f}")
            except Exception as err:  # noqa: BLE001
                print(f"    ⚠️ Error en {metric_name}: {err}")
                traceback.print_exc()
                method_results[metric_name] = None

        results[method] = method_results

    return results


def save_results(results: Dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f"\n✅ Resultados guardados en {output_path}")


# ============================================================
#  main()
# ============================================================

def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("  EVALUACIÓN QUANTUS - RESNET18 XAI")
    print("=" * 60)
    print(f"Dispositivo: {device}")
    print(f"Métodos: {args.methods}")
    print(f"Muestras a evaluar: {args.num_samples}")

    model = load_trained_model(args.model_path, device)

    datasets = load_datasets(args.data_dir, target_size=224)
    _, _, test_loader = create_data_loaders_fixed(
        datasets=datasets,
        batch_size=args.batch_size,
        num_workers=0,
        seed=42,
    )

    explainer = XAIExplainer(model, device, num_classes=15)
    x_batch, y_batch = collect_samples(test_loader, args.num_samples, device)

    results = evaluate_methods(model, explainer, x_batch, y_batch, args.methods)
    save_results(results, args.output)


if __name__ == "__main__":
    main()

