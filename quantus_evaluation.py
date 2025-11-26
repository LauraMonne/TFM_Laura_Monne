"""
Evaluación cuantitativa de explicabilidad usando Quantus.

Mide 5 dimensiones para Grad-CAM, Grad-CAM++ (opcional),
Integrated Gradients y Saliency:
- Fidelidad (FaithfulnessCorrelation)
- Robustez (AvgSensitivity)
- Complejidad (Entropy)
- Aleatorización (ModelParameterRandomisation)
- Localización (RegionPerturbation como aproximación de Localization Ratio)

Uso:
    python quantus_evaluation.py --num_samples 30 --methods gradcam integrated_gradients saliency
"""

import argparse
import json
import os
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
    if np_batch.ndim == 4 and np_batch.shape[1] in (1, 3):
        np_batch = np.transpose(np_batch, (0, 2, 3, 1))
    return np_batch


# ============================================================
#  Atribuciones XAI (reutiliza XAIExplainer)
# ============================================================

def expand_heatmap_to_channels(heatmap: np.ndarray, channels: int) -> torch.Tensor:
    """Expande un heatmap HxW a CxHxW repitiendo por canal."""
    if heatmap.ndim != 2:
        raise ValueError("El heatmap debe ser 2D.")
    tensor = torch.tensor(heatmap, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).repeat(channels, 1, 1)
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

def evaluate_metric(metric, model, x_batch_np, y_batch_np, a_batch_np):
    """
    Envuelve la llamada a la métrica de Quantus.
    En Quantus >=0.3, la firma típica es:

        metric(
            model=model,         # torch.nn.Module
            x_batch=x_batch_np,  # np.ndarray
            y_batch=y_batch_np,  # np.ndarray
            a_batch=a_batch_np,  # np.ndarray
        )
    """
    scores = metric(
        model=model,
        x_batch=x_batch_np,
        y_batch=y_batch_np,
        a_batch=a_batch_np,
    )
    return float(np.nanmean(scores)), float(np.nanstd(scores)), scores


def evaluate_methods(model, explainer, x_batch, y_batch, methods):
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    # Convertimos a numpy BHWC para Quantus
    x_batch_np = hwc(x_batch)
    y_batch_np = y_batch.detach().cpu().numpy()

    with torch.no_grad():
        logits = model(x_batch)
        preds = logits.argmax(dim=1)

    # Definición de métricas Quantus (sin kwargs problemáticos)
    metrics = {
        "faithfulness": quantus.FaithfulnessCorrelation(),
        "robustness": quantus.AvgSensitivity(),
        "complexity": quantus.Entropy(),
        "randomization": quantus.ModelParameterRandomisation(),
        "localization": quantus.RegionPerturbation(),
    }

    for method in methods:
        print(f"\n=== Evaluando {method} ===")
        try:
            attr_batch = compute_attributions(explainer, x_batch, preds, method)
        except ValueError as err:
            print(f"⚠️ {err}. Saltando método.")
            continue

        a_batch_np = hwc(attr_batch)
        method_results = {}

        for metric_name, metric in metrics.items():
            print(f" -> {metric_name}")
            try:
                mean, std, scores = evaluate_metric(
                    metric, model, x_batch_np, y_batch_np, a_batch_np
                )
                method_results[metric_name] = {
                    "mean": mean,
                    "std": std,
                    "scores": [float(s) for s in scores],
                }
                print(f"    {mean:.4f} ± {std:.4f}")
            except Exception as err:  # noqa: BLE001
                print(f"    ⚠️ Error en {metric_name}: {err}")
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
