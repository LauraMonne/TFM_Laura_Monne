"""
Evaluación cuantitativa de explicabilidad usando Quantus.

Mide 5 dimensiones para varios métodos XAI (Grad-CAM, Grad-CAM++, IG, Saliency):
- Fidelidad      -> FaithfulnessCorrelation
- Robustez       -> AvgSensitivity
- Complejidad    -> Complexity (o Entropy)
- Aleatorización -> MPRT / ModelParameterRandomisation
- Localización   -> RegionPerturbation

Uso típico:
    python quantus_evaluation.py --num_samples 30 --methods gradcam integrated_gradients saliency
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

try:
    import quantus
except ImportError as exc:
    raise SystemExit(
        "quantus no está instalado. Ejecuta: pip install quantus"
    ) from exc

from prepare_data import load_datasets
from data_utils import create_data_loaders_fixed
from xai_explanations import XAIExplainer, load_trained_model


# ============================================================
#  Argumentos de línea de comandos
# ============================================================

# Construye el parser de argumentos.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación cuantitativa de XAI con Quantus (versión simplificada)."
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
        default=["gradcam", "gradcampp", "integrated_gradients", "saliency"],
        help="Métodos XAI a evaluar (gradcam, gradcampp, integrated_gradients, saliency).",
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

# Recopila muestras del conjunto de test.
# Devuelve un tensor BCHW (batch, channels, height, width).
# y un tensor BHWC (batch, height, width, channels).
def collect_samples(test_loader, num_samples: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Recopila x_batch, y_batch del conjunto de test."""
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
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

# Convierte un tensor BCHW a BHWC para Quantus.
def to_bhwc(tensor_batch: torch.Tensor) -> np.ndarray:
    """Convierte lote BCHW -> BHWC (formato común en Quantus)."""
    np_batch = tensor_batch.detach().cpu().numpy()  # (B, C, H, W)
    if np_batch.ndim == 4:
        np_batch = np.transpose(np_batch, (0, 2, 3, 1))  # BCHW -> BHWC
    return np_batch


# ============================================================
#  Atribuciones XAI (reutiliza XAIExplainer)
# ============================================================

# Expande un heatmap HxW a CxHxW repitiendo por canal.
def expand_heatmap_to_channels(heatmap: np.ndarray, channels: int) -> torch.Tensor:
    """Expande un heatmap HxW a CxHxW repitiendo por canal."""
    if heatmap.ndim != 2:
        raise ValueError("El heatmap debe ser 2D.")
    tensor = torch.tensor(heatmap, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).repeat(channels, 1, 1)  # (C, H, W)
    return tensor


def compute_attributions(
    explainer: XAIExplainer,
    x_batch: torch.Tensor,
    preds: torch.Tensor,
    method: str,
) -> torch.Tensor:
    """
    Genera atribuciones para todo el batch usando el método especificado.
    Devuelve un tensor BCHW (batch, channels, height, width).
    """
    attributions: List[torch.Tensor] = []
    for idx in tqdm(range(len(x_batch)), desc=f"Atribuciones {method}"):
        sample = x_batch[idx : idx + 1]
        target_class = int(preds[idx].item())
        try:
            if method == "gradcam":
                result = explainer.generate_gradcam(sample, target_class, save_path=None)
                if result is None:
                    raise RuntimeError("Grad-CAM retornó None")
                _, heatmap = result
                attr = expand_heatmap_to_channels(heatmap, sample.shape[1])
            elif method == "gradcampp":
                result = explainer.generate_gradcampp(sample, target_class, save_path=None)
                if result is None:
                    raise RuntimeError("Grad-CAM++ retornó None")
                _, heatmap = result
                attr = expand_heatmap_to_channels(heatmap, sample.shape[1])
            elif method == "integrated_gradients":
                result = explainer.generate_integrated_gradients(sample, target_class, save_path=None)
                if result is None:
                    raise RuntimeError("IG retornó None")
                attr = result[1][0].detach().cpu()  # (C, H, W)
            elif method == "saliency":
                result = explainer.generate_saliency_map(sample, target_class, save_path=None)
                if result is None:
                    raise RuntimeError("Saliency retornó None")
                attr = result[1][0].detach().cpu()  # (C, H, W)
            else:
                raise ValueError(f"Método desconocido: {method}")
        except Exception as err:
            print(f"⚠️ Error generando atribución para muestra {idx}: {err}")
            attr = torch.zeros_like(sample[0].cpu())
        attributions.append(attr)
    return torch.stack(attributions, dim=0)  # (B, C, H, W)


# ============================================================
#  Métricas de Quantus
# ============================================================

# Crea un conjunto estándar de métricas de Quantus.
# Devuelve un diccionario con las métricas.
def create_metrics() -> Dict[str, object]:
    """Crea un conjunto estándar de métricas de Quantus."""
    metrics: Dict[str, object] = {}

    # Fidelidad
    metrics["faithfulness"] = quantus.FaithfulnessCorrelation()

    # Robustez
    metrics["robustness"] = quantus.AvgSensitivity()

    # Complejidad o Entropy
    try:
        metrics["complexity"] = quantus.Complexity()
    except AttributeError:
        metrics["complexity"] = quantus.Entropy()

    # Aleatorización (MPRT o ModelParameterRandomisation)
    try:
        RandomizationMetric = quantus.MPRT
    except AttributeError:
        RandomizationMetric = quantus.ModelParameterRandomisation
    metrics["randomization"] = RandomizationMetric()

    # Localización
    metrics["localization"] = quantus.RegionPerturbation()

    return metrics

# Evalúa cada método XAI con varias métricas de Quantus.
# Devuelve un diccionario: results[method][metric] = {mean, std, scores}.
def evaluate_methods(
    model: torch.nn.Module,
    explainer: XAIExplainer,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    methods: list[str],
    device: torch.device,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evalúa cada método XAI con varias métricas de Quantus.
    Devuelve un diccionario: results[method][metric] = {mean, std, scores}.
    """
    model.eval()
    metrics = create_metrics()

    # Predicciones del modelo (para usar como clases objetivo)
    with torch.no_grad():
        logits = model(x_batch.to(device))
        preds = logits.argmax(dim=1)

    # Convertir datos a BHWC para Quantus
    x_bhwc = to_bhwc(x_batch)  # (B, H, W, C)
    y_np = y_batch.detach().cpu().numpy()

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for method in methods:
        print(f"\n=== Evaluando método XAI: {method} ===")

        # 1) Atribuciones BCHW -> BHWC
        attr_bchw = compute_attributions(explainer, x_batch, preds, method)
        attr_bhwc = to_bhwc(attr_bchw)  # (B, H, W, C)

        method_results: Dict[str, Dict[str, float]] = {}

        for metric_name, metric in metrics.items():
            print(f" -> Métrica: {metric_name}")
            try:
                # Llamada directa a Quantus (asume x_batch y a_batch en BHWC)
                scores = metric(
                    model=model,
                    x_batch=x_bhwc,
                    y_batch=y_np,
                    a_batch=attr_bhwc,
                    device=device,
                )

                scores = np.array(scores, dtype=float).flatten()
                mean = float(np.nanmean(scores))
                std = float(np.nanstd(scores))

                method_results[metric_name] = {
                    "mean": mean,
                    "std": std,
                    "scores": scores.tolist(),
                }
                print(f"    {mean:.4f} ± {std:.4f}")
            except Exception as err:
                print(f"    ⚠️ Error evaluando {metric_name} para {method}: {err}")
                method_results[metric_name] = None

        results[method] = method_results

    return results

# Guarda los resultados en un archivo JSON.
def save_results(results: Dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f"\n✅ Resultados guardados en {output_path}")


# ============================================================
#  main()
# ============================================================

# Función principal
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("  EVALUACIÓN QUANTUS - RESNET18 XAI (simplificada)")
    print("=" * 60)
    print(f"Dispositivo: {device}")
    print(f"Métodos: {args.methods}")
    print(f"Muestras a evaluar: {args.num_samples}")

    # Modelo entrenado
    model = load_trained_model(args.model_path, device)

    # Datos de test
    datasets = load_datasets(args.data_dir, target_size=224)
    _, _, test_loader = create_data_loaders_fixed(
        datasets=datasets,
        batch_size=args.batch_size,
        num_workers=0,
        seed=42,
    )

    # Muestreo de ejemplos de test
    x_batch, y_batch = collect_samples(test_loader, args.num_samples, device)

    # Explainer XAI (reutiliza la misma lógica que xai_explanations.py)
    explainer = XAIExplainer(model, device, num_classes=15)

    # Evaluación
    results = evaluate_methods(model, explainer, x_batch, y_batch, args.methods, device)
    save_results(results, args.output)


if __name__ == "__main__":
    main()

"""
Resumen
El script quantus_evaluation.py evalúa la explicabilidad de un modelo entrenado con varios métodos XAI (Grad-CAM, Grad-CAM++, Integrated Gradients y Saliency) usando las métricas de Quantus.
1. Argumentos: lee los parámetros de línea de comandos.
2. Datos: carga los datasets MedMNIST y crea un loader de test.
3. Muestreo: recoge un batch de muestras del conjunto de test.
4. Explainer: inicializa el objeto XAIExplainer.
5. Evaluación: llama a evaluate_methods() para cada método XAI.
6. Guarda: guarda los resultados en un archivo JSON.

Resultado: un archivo JSON con los resultados de la evaluación cuantitativa de la explicabilidad.
"""