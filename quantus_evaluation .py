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
import inspect
import json
import os
from typing import Dict, List, Callable

import numpy as np
import torch
from tqdm import tqdm

try:
    import quantus
except ImportError as exc:
    raise SystemExit(
        "quantus no está instalado. Ejecuta: pip install quantus"
    ) from exc

from prepare_data import load_datasets, get_dataset_info
from train import create_data_loaders
from xai_explanations import XAIExplainer, load_trained_model


# ============================================================
#  Argumentos de línea de comandos
# ============================================================

# Construye el parser de argumentos.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación cuantitativa de XAI con Quantus (por dataset individual)."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["blood", "retina", "breast"],
        help="Dataset a evaluar: blood (8 clases), retina (5 clases) o breast (2 clases).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Ruta al checkpoint entrenado. Si no se especifica, usa results/best_model_{dataset}.pth",
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
        "--sample_strategy",
        choices=["first", "reservoir"],
        default="reservoir",
        help="Estrategia de muestreo del test: first (primeras N) o reservoir (aleatorio uniforme).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para muestreo aleatorio (solo si sample_strategy=reservoir).",
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
        "--target",
        choices=["pred", "true"],
        default="pred",
        help="Clases objetivo para métricas XAI: pred (predicha por el modelo) o true (etiqueta real).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta de guardado para los resultados. Si no se especifica, usa outputs/quantus_metrics_{dataset}.json",
    )
    return parser.parse_args()


# ============================================================
#  Utilidades de datos
# ============================================================

# Recopila muestras del conjunto de test.
# Devuelve un tensor BCHW (batch, channels, height, width).
# y un tensor BHWC (batch, height, width, channels).
def collect_samples(
    test_loader,
    num_samples: int,
    device: torch.device,
    sample_strategy: str = "reservoir",
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recopila x_batch, y_batch del conjunto de test.

    - first: toma las primeras N muestras en orden.
    - reservoir: muestreo aleatorio uniforme sobre todo el test (requiere recorrerlo completo).
    """
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    seen = 0
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        for _, (data, target) in enumerate(tqdm(test_loader, desc="Recolectando muestras")):
            bsz = data.shape[0]
            for i in range(bsz):
                seen += 1
                sample_x = data[i : i + 1].cpu()
                sample_y = target[i : i + 1].cpu()

                if sample_strategy == "first":
                    if len(xs) < num_samples:
                        xs.append(sample_x)
                        ys.append(sample_y)
                    if len(xs) >= num_samples:
                        break
                else:
                    if len(xs) < num_samples:
                        xs.append(sample_x)
                        ys.append(sample_y)
                    else:
                        j = rng.integers(0, seen)
                        if j < num_samples:
                            xs[j] = sample_x
                            ys[j] = sample_y
            if sample_strategy == "first" and len(xs) >= num_samples:
                break

    if not xs:
        raise RuntimeError("No se encontraron muestras. Ajusta num_samples.")

    x_batch = torch.cat(xs, dim=0).to(device)
    y_batch = torch.cat(ys, dim=0).to(device).view(-1)
    return x_batch, y_batch

# Convierte un tensor BCHW a BHWC para Quantus.
def to_numpy_bchw(tensor_batch: torch.Tensor) -> np.ndarray:
    """Convierte un tensor BCHW a NumPy BCHW (sin cambiar el orden de ejes)."""
    return tensor_batch.detach().cpu().numpy()


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
    targets: torch.Tensor,
    method: str,
) -> torch.Tensor:
    """
    Genera atribuciones para todo el batch usando el método especificado.
    Devuelve un tensor BCHW (batch, channels, height, width).
    """
    attributions: List[torch.Tensor] = []
    for idx in tqdm(range(len(x_batch)), desc=f"Atribuciones {method}"):
        sample = x_batch[idx : idx + 1]
        target_class = int(targets[idx].item())
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
#  explain_func para métricas que lo requieren (robustness, randomization)
# ============================================================


def build_explain_func(
    method: str,
    device: torch.device,
    num_classes: int,
    dataset: str,
) -> Callable:
    """
    Construye una explain_func compatible con Quantus.
    Firma esperada: explain_func(model, inputs, targets, **kwargs) -> np.ndarray
    """
    explainer_cache: Dict[int, XAIExplainer] = {}

    def get_explainer(model: torch.nn.Module) -> XAIExplainer:
        key = id(model)
        if key not in explainer_cache:
            explainer_cache[key] = XAIExplainer(
                model,
                device=device,
                num_classes=num_classes,
                dataset=dataset,
                create_dirs=False,
            )
        return explainer_cache[key]

    def explain_func(model, inputs, targets, **kwargs):
        explainer = get_explainer(model)
        # inputs puede venir como np.ndarray o torch.Tensor, BCHW o BHWC
        if isinstance(inputs, np.ndarray):
            x = torch.tensor(inputs, dtype=torch.float32)
        else:
            x = inputs

        if x.ndim == 4 and x.shape[-1] in (1, 3):  # BHWC -> BCHW
            x = x.permute(0, 3, 1, 2)

        x = x.to(device)

        if isinstance(targets, np.ndarray):
            y = torch.tensor(targets, dtype=torch.long, device=device)
        else:
            y = targets.to(device)

        attributions: List[torch.Tensor] = []
        for i in range(len(x)):
            sample = x[i : i + 1]
            target_class = int(y[i].item())
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
                    attr = result[1][0].detach().cpu()
                elif method == "saliency":
                    result = explainer.generate_saliency_map(sample, target_class, save_path=None)
                    if result is None:
                        raise RuntimeError("Saliency retornó None")
                    attr = result[1][0].detach().cpu()
                else:
                    raise ValueError(f"Método desconocido: {method}")
            except Exception as err:
                print(f"⚠️ Error en explain_func para muestra {i}: {err}")
                attr = torch.zeros_like(sample[0].cpu())
            attributions.append(attr)

        # Devuelve BCHW como NumPy
        return torch.stack(attributions, dim=0).detach().cpu().numpy()

    return explain_func


# ============================================================
#  Métricas de Quantus
# ============================================================

# Inicializa métricas de Quantus con kwargs compatibles (según firma).
def _init_metric(metric_cls, **kwargs):
    try:
        sig = inspect.signature(metric_cls)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return metric_cls(**filtered)
    except Exception:
        return metric_cls()

# Crea un conjunto estándar de métricas de Quantus.
# Devuelve un diccionario con las métricas.
def create_metrics() -> Dict[str, object]:
    """Crea un conjunto estándar de métricas de Quantus."""
    metrics: Dict[str, object] = {}

    # Fidelidad
    metrics["faithfulness"] = _init_metric(
        quantus.FaithfulnessCorrelation,
        return_aggregate=False,
        aggregate_func=None,
        nr_runs=30,  # reducir coste computacional
        subset_size=224,
        disable_warnings=True,
    )

    # Robustez
    # Configuración para evitar valores inf/nan:
    # - return_nan_when_prediction_changes=True: devuelve nan en lugar de inf cuando la predicción cambia
    # - nr_samples=30: reduce el número de muestras para evitar problemas numéricos (default=200)
    # - lower_bound=0.02: ruido mínimo muy pequeño para evitar cambios de predicción
    # - upper_bound=0.15: ruido máximo pequeño para mantener predicciones estables
    # - abs=True: usa valores absolutos para evitar problemas con signos
    # - normalise=True: normaliza las explicaciones para estabilidad numérica
    # - similarity_func: usar correlación en lugar de distancia euclidiana (más robusta)
    metrics["robustness"] = quantus.AvgSensitivity(
        nr_samples=30,  # Reducir muestras para evitar problemas numéricos
        abs=True,  # Usar valores absolutos
        normalise=True,  # Normalizar para estabilidad
        lower_bound=0.02,  # Ruido mínimo muy pequeño para evitar cambios de predicción
        upper_bound=0.15,  # Ruido máximo pequeño para mantener predicciones estables
        return_nan_when_prediction_changes=True,  # Devolver nan en lugar de inf
        disable_warnings=True,  # Desactivar warnings para limpieza
    )

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
    metrics["randomization"] = _init_metric(
        RandomizationMetric,
        skip_layers=True,  # solo comparar modelo original vs totalmente randomizado (mucho más rápido)
        return_last_correlation=True,
        layer_order="top_down",
        seed=42,
        disable_warnings=True,
    )

    # Localización
    metrics["localization"] = _init_metric(
        quantus.RegionPerturbation,
        regions_evaluation=30,  # reducir coste computacional
        disable_warnings=True,
    )

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
    target_mode: str,
    dataset: str,
    num_classes: int,
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

    if target_mode == "pred":
        targets = preds
    else:
        targets = y_batch.view(-1)

    # Convertir datos a NumPy (manteniendo BCHW)
    x_np = to_numpy_bchw(x_batch)  # (B, C, H, W)
    y_np = targets.detach().cpu().numpy()

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for method in methods:
        print(f"\n=== Evaluando método XAI: {method} ===")

        # 1) Atribuciones (mantenemos BCHW para coincidir con x_np)
        attr_bchw = compute_attributions(explainer, x_batch, targets, method)
        attr_np = to_numpy_bchw(attr_bchw)  # (B, C, H, W)

        method_results: Dict[str, Dict[str, float]] = {}

        # explain_func para métricas que lo requieren (robustness, randomization)
        explain_fn = build_explain_func(method, device, num_classes=num_classes, dataset=dataset)

        for metric_name, metric in metrics.items():
            print(f" -> Métrica: {metric_name}")
            try:
                # Robustez y aleatorización: usar explain_func (Quantus calcula a_batch internamente)
                if metric_name in {"robustness", "randomization"}:
                    scores = metric(
                        model=model,
                        x_batch=x_np,
                        y_batch=y_np,
                        explain_func=explain_fn,
                        device=device,
                    )
                else:
                    # Resto de métricas: usar atribuciones precomputadas (a_batch)
                    scores = metric(
                        model=model,
                        x_batch=x_np,
                        y_batch=y_np,
                        a_batch=attr_np,
                        device=device,
                    )

                # Algunas métricas (p.ej. randomization) pueden devolver un dict.
                if isinstance(scores, dict):
                    # Caso típico: clave 'scores'
                    if "scores" in scores:
                        raw_scores = scores["scores"]
                    else:
                        # Buscar el primer valor que parezca una colección numérica
                        raw_scores = None
                        for v in scores.values():
                            if isinstance(v, (list, tuple, np.ndarray)):
                                raw_scores = v
                                break
                        if raw_scores is None:
                            raise TypeError(
                                f"Formato de salida de métrica '{metric_name}' no soportado: claves={list(scores.keys())}"
                            )
                else:
                    raw_scores = scores

                raw_scores = np.array(raw_scores, dtype=float).flatten()
                
                # Filtrar inf y nan antes de calcular estadísticas
                valid_scores = raw_scores[np.isfinite(raw_scores)]  # isfinite = no inf y no nan
                
                if len(valid_scores) == 0:
                    # Si todos los valores son inf/nan, usar None
                    mean = None
                    std = None
                    print(f"    ⚠️  Todos los valores son inf/nan, usando None")
                elif len(valid_scores) < len(raw_scores):
                    # Si hay algunos valores válidos, calcular solo con ellos
                    mean = float(np.mean(valid_scores))
                    std = float(np.std(valid_scores))
                    invalid_count = len(raw_scores) - len(valid_scores)
                    print(f"    ⚠️  {invalid_count}/{len(raw_scores)} valores inválidos (inf/nan) filtrados")
                else:
                    # Todos los valores son válidos
                    mean = float(np.mean(valid_scores))
                    std = float(np.std(valid_scores))
                
                # Convertir inf y nan a None para JSON
                mean_json = None if (mean is None or (mean is not None and (np.isinf(mean) or np.isnan(mean)))) else mean
                std_json = None if (std is None or (std is not None and (np.isinf(std) or np.isnan(std)))) else std
                
                # Convertir scores: inf -> None, nan -> None
                scores_list = []
                for s in raw_scores:
                    if np.isinf(s) or np.isnan(s):
                        scores_list.append(None)
                    else:
                        scores_list.append(float(s))

                method_results[metric_name] = {
                    "mean": mean_json,
                    "std": std_json,
                    "scores": scores_list,
                }
                
                # Print con manejo de inf/nan - mostrar valores reales o advertencia
                if mean_json is None:
                    if len(valid_scores) == 0:
                        print(f"    None (todos los valores son inf/nan)")
                    else:
                        print(f"    None (filtrados {len(raw_scores) - len(valid_scores)}/{len(raw_scores)} valores inválidos)")
                elif std_json is None:
                    print(f"    {mean:.4f} ± None")
                else:
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
    print("  EVALUACIÓN QUANTUS - RESNET18 XAI")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Dispositivo: {device}")
    print(f"Métodos: {args.methods}")
    print(f"Muestras a evaluar: {args.num_samples}")
    print(f"Estrategia muestreo: {args.sample_strategy} (seed={args.seed})")
    print(f"Target de métricas: {args.target}")

    # Determinar número de clases según dataset
    meta_all = get_dataset_info()
    name_map = {"blood": "bloodmnist", "retina": "retinamnist", "breast": "breastmnist"}
    med_name = name_map[args.dataset]
    num_classes = int(meta_all[med_name]["n_classes"])

    # Determinar ruta del modelo
    if args.model_path is None:
        model_path = f"results/best_model_{args.dataset}.pth"
    else:
        model_path = args.model_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se ha encontrado {model_path}. "
            f"Ejecuta primero: python train.py --dataset {args.dataset}"
        )

    # Modelo entrenado
    model = load_trained_model(model_path, device, num_classes=num_classes)

    # Datos de test
    datasets = load_datasets(args.data_dir, target_size=224)
    _, _, test_loader, _ = create_data_loaders(
        datasets=datasets,
        batch_size=args.batch_size,
        num_workers=0,
        num_classes=num_classes,
        dataset_name=args.dataset,
    )

    # Muestreo de ejemplos de test
    x_batch, y_batch = collect_samples(
        test_loader,
        args.num_samples,
        device,
        sample_strategy=args.sample_strategy,
        seed=args.seed,
    )

    # Explainer XAI
    explainer = XAIExplainer(
        model,
        device,
        num_classes=num_classes,
        dataset=args.dataset,
        create_dirs=False,
    )

    # Evaluación
    results = evaluate_methods(
        model,
        explainer,
        x_batch,
        y_batch,
        args.methods,
        device,
        target_mode=args.target,
        dataset=args.dataset,
        num_classes=num_classes,
    )
    
    # Añadir metadata al resultado
    results["metadata"] = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "num_samples": args.num_samples,
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "target": args.target,
        "methods": args.methods,
    }
    
    # Determinar ruta de salida
    if args.output is None:
        output_path = f"outputs/quantus_metrics_{args.dataset}.json"
    else:
        output_path = args.output
    
    save_results(results, output_path)
    print(f"\n✅ Resultados guardados en: {output_path}")


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
