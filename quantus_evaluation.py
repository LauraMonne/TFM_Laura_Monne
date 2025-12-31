"""
Evaluación cuantitativa de explicabilidad usando Quantus - VERSIÓN CORREGIDA

Mide 5 dimensiones para varios métodos XAI (Grad-CAM, Grad-CAM++, IG, Saliency):
- Fidelidad      -> FaithfulnessCorrelation
- Robustez       -> AvgSensitivity
- Complejidad    -> Complexity (o Entropy)
- Aleatorización -> MPRT / ModelParameterRandomisation (con métrica alternativa mejorada)
- Localización   -> RegionPerturbation
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from typing import Dict, List, Callable, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import quantus
except ImportError as exc:
    raise SystemExit(
        "quantus no está instalado. Ejecuta: pip install quantus"
    ) from exc

from prepare_data import load_datasets, get_dataset_info
from train import create_data_loaders
from xai_explanations import XAIExplainer, load_trained_model

# Importar Captum para recrear explicadores
try:
    from captum.attr import IntegratedGradients, Saliency
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    IntegratedGradients = None
    Saliency = None


# ============================================================
#  Reproducibilidad
# ============================================================

def set_global_seed(seed: int) -> None:
    """Establece la semilla global para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
#  Argumentos de línea de comandos
# ============================================================

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
        type=str,
        default=None,
        help="Ruta de guardado para los resultados. Si no se especifica, usa outputs/quantus_metrics_{dataset}.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed global para reproducibilidad.",
    )
    return parser.parse_args()


# ============================================================
#  Utilidades de datos
# ============================================================

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


def to_numpy_bchw(tensor_batch: torch.Tensor) -> np.ndarray:
    """Convierte un tensor BCHW a NumPy BCHW."""
    return tensor_batch.detach().cpu().numpy()


# ============================================================
#  Sanitización / normalización de atribuciones
# ============================================================

def sanitize_attribution(attr: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Sanitiza y normaliza atribuciones para mejorar estabilidad numérica.
    """
    attr = attr.to(dtype=torch.float32)
    attr = torch.nan_to_num(attr, nan=0.0, posinf=0.0, neginf=0.0)
    
    if torch.all(attr == 0):
        return attr
    
    if attr.ndim == 3:  # (C, H, W)
        mn = torch.min(attr)
        mx = torch.max(attr)
        if (mx - mn).abs() < eps:
            return torch.zeros_like(attr)
        attr = (attr - mn) / (mx - mn + eps)
    elif attr.ndim == 4:  # (B, C, H, W)
        for b in range(attr.shape[0]):
            sample = attr[b]
            mn = torch.min(sample)
            mx = torch.max(sample)
            if (mx - mn).abs() < eps:
                attr[b] = torch.zeros_like(sample)
            else:
                attr[b] = (sample - mn) / (mx - mn + eps)
    
    return attr


# ============================================================
#  Atribuciones XAI
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
    """Genera atribuciones para todo el batch. Devuelve tensor BCHW."""
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
                attr = result[1][0].detach().cpu()
            elif method == "saliency":
                result = explainer.generate_saliency_map(sample, target_class, save_path=None)
                if result is None:
                    raise RuntimeError("Saliency retornó None")
                attr = result[1][0].detach().cpu()
            else:
                raise ValueError(f"Método desconocido: {method}")
            
            attr = sanitize_attribution(attr)
        except Exception as err:
            print(f"⚠️ Error generando atribución para muestra {idx}: {err}")
            attr = torch.zeros_like(sample[0].cpu())
        attributions.append(attr)
    return torch.stack(attributions, dim=0)


# ============================================================
#  explain_func para Quantus - VERSIÓN MEJORADA
# ============================================================

def build_explain_func(
    base_explainer: XAIExplainer,
    method: str,
    device: torch.device,
    model_override=None,  # NUEVO: permite pasar un modelo específico
) -> Callable:
    """Construye explain_func compatible con Quantus - VERSIÓN CORREGIDA."""

    def explain_func(model, inputs, targets, **kwargs):
        # CORRECCIÓN CRÍTICA: Usar model_override si está disponible
        actual_model = model_override if model_override is not None else model
        
        # Crear un explainer temporal con el modelo correcto
        num_classes = base_explainer.num_classes
        dataset_name = base_explainer.dataset
        
        # IMPORTANTE: Crear explainer temporal que use el modelo correcto
        temp_explainer = XAIExplainer(actual_model, device, num_classes, dataset_name)
        
        # Forzar actualización del modelo en métodos Captum
        if CAPTUM_AVAILABLE:
            try:
                if IntegratedGradients is not None:
                    temp_explainer.ig = IntegratedGradients(actual_model)
                if Saliency is not None:
                    temp_explainer.saliency = Saliency(actual_model)
            except Exception as e:
                print(f"⚠️ Error reinicializando Captum: {e}")
        
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
                    result = temp_explainer.generate_gradcam(sample, target_class, save_path=None)
                    if result is None:
                        raise RuntimeError("Grad-CAM retornó None")
                    _, heatmap = result
                    attr = expand_heatmap_to_channels(heatmap, sample.shape[1])
                elif method == "gradcampp":
                    result = temp_explainer.generate_gradcampp(sample, target_class, save_path=None)
                    if result is None:
                        raise RuntimeError("Grad-CAM++ retornó None")
                    _, heatmap = result
                    attr = expand_heatmap_to_channels(heatmap, sample.shape[1])
                elif method == "integrated_gradients":
                    result = temp_explainer.generate_integrated_gradients(sample, target_class, save_path=None)
                    if result is None:
                        raise RuntimeError("IG retornó None")
                    attr = result[1][0].detach().cpu()
                elif method == "saliency":
                    result = temp_explainer.generate_saliency_map(sample, target_class, save_path=None)
                    if result is None:
                        raise RuntimeError("Saliency retornó None")
                    attr = result[1][0].detach().cpu()
                else:
                    raise ValueError(f"Método desconocido: {method}")
                
                attr = sanitize_attribution(attr)
            except Exception as err:
                print(f"⚠️ Error en explain_func para muestra {i}: {err}")
                attr = torch.zeros_like(sample[0].cpu())
            attributions.append(attr)

        return torch.stack(attributions, dim=0).detach().cpu().numpy()

    return explain_func


# ============================================================
#  Métrica alternativa de Randomization (MEJORADA Y CORREGIDA)
# ============================================================

def create_alternative_randomization_metric(base_explainer, method, device):
    """
    Métrica alternativa mejorada - CORREGIDA para usar modelos correctos.
    
    INTERPRETACIÓN CORRECTA:
    - Score ALTO (cercano a 1.0) = explicaciones MUY DIFERENTES = BUENO
    - Score BAJO (cercano a 0.0) = explicaciones SIMILARES = MALO
    """
    
    class ImprovedRandomizationMetric:
        def __init__(self):
            self.base_explainer = base_explainer
            self.method = method
            self.device = device
        
        def __call__(self, model, x_batch, y_batch, explain_func, **kwargs):
            scores = []
            
            print("       🔄 Generando explicaciones con modelo entrenado...")
            # CORRECCIÓN: Crear explain_func con el modelo original
            explain_func_original = build_explain_func(
                self.base_explainer, 
                self.method, 
                self.device,
                model_override=model  # Usar el modelo entrenado
            )
            expl_original = explain_func_original(model, x_batch, y_batch)
            
            print("       🎲 Creando modelo con parámetros aleatorizados...")
            model_randomized = copy.deepcopy(model)
            
            # Aleatorizar TODOS los parámetros con mayor varianza
            with torch.no_grad():
                for name, param in model_randomized.named_parameters():
                    if len(param.shape) >= 2:
                        torch.nn.init.normal_(param, mean=0.0, std=1.0)
                    else:
                        torch.nn.init.normal_(param, mean=0.0, std=0.5)
            
            model_randomized.eval()
            
            print("       🔄 Generando explicaciones con modelo aleatorizado...")
            # CORRECCIÓN: Crear explain_func con el modelo aleatorizado
            explain_func_random = build_explain_func(
                self.base_explainer,
                self.method,
                self.device,
                model_override=model_randomized  # Usar el modelo aleatorizado
            )
            expl_randomized = explain_func_random(model_randomized, x_batch, y_batch)
            
            print("       📊 Calculando diferencias entre explicaciones...")
            
            # Verificar que las explicaciones son diferentes
            diff_check = np.abs(expl_original - expl_randomized).mean()
            print(f"       ℹ️  Diferencia promedio absoluta: {diff_check:.6f}")
            
            # Calcular score por muestra
            for i in range(len(x_batch)):
                exp_orig = np.array(expl_original[i])
                exp_rand = np.array(expl_randomized[i])
                
                exp_orig_flat = exp_orig.flatten()
                exp_rand_flat = exp_rand.flatten()
                
                min_len = min(len(exp_orig_flat), len(exp_rand_flat))
                exp_orig_flat = exp_orig_flat[:min_len]
                exp_rand_flat = exp_rand_flat[:min_len]
                
                if HAS_SCIPY:
                    try:
                        correlation, _ = spearmanr(exp_orig_flat, exp_rand_flat)
                        if np.isnan(correlation):
                            correlation = 0.0
                    except Exception:
                        correlation = 0.0
                    
                    # Score alto = baja correlación = bueno
                    score = 1.0 - (correlation + 1.0) / 2.0
                    
                else:
                    # Fallback con distancia euclidiana
                    combined = np.concatenate([exp_orig_flat, exp_rand_flat])
                    if combined.max() - combined.min() > 1e-8:
                        exp_orig_norm = (exp_orig_flat - combined.min()) / (combined.max() - combined.min())
                        exp_rand_norm = (exp_rand_flat - combined.min()) / (combined.max() - combined.min())
                    else:
                        exp_orig_norm = exp_orig_flat
                        exp_rand_norm = exp_rand_flat
                    
                    distance = np.linalg.norm(exp_orig_norm - exp_rand_norm)
                    score = min(distance / np.sqrt(2), 1.0)
                
                scores.append(score)
            
            del model_randomized
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return np.array(scores)
    
    return ImprovedRandomizationMetric()


# ============================================================
#  Métricas de Quantus - VERSIÓN MEJORADA
# ============================================================

def create_metrics(explainer, method, device) -> Dict[str, object]:
    """Crea métricas de Quantus con configuración optimizada."""
    metrics: Dict[str, object] = {}

    # Fidelidad
    metrics["faithfulness"] = quantus.FaithfulnessCorrelation()

    # Robustez
    metrics["robustness"] = quantus.AvgSensitivity(
        nr_samples=30,
        abs=True,
        normalise=True,
        lower_bound=0.02,
        upper_bound=0.15,
        return_nan_when_prediction_changes=True,
        disable_warnings=True,
    )

    # Complejidad
    try:
        metrics["complexity"] = quantus.Complexity()
    except AttributeError:
        metrics["complexity"] = quantus.Entropy()

    # Aleatorización - CORRECCIÓN: Pasar explainer, method y device
    print("📊 Usando métrica alternativa mejorada para randomization")
    print("   (Compara modelo entrenado vs. modelo aleatorizado)")
    metrics["randomization"] = create_alternative_randomization_metric(
        explainer, method, device
    )

    # Localización
    metrics["localization"] = quantus.RegionPerturbation()

    return metrics


# ============================================================
#  Evaluación de métodos XAI - VERSIÓN CORREGIDA
# ============================================================

def evaluate_methods(
    model: torch.nn.Module,
    explainer: XAIExplainer,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    methods: list[str],
    device: torch.device,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Evalúa cada método XAI con métricas de Quantus."""
    model.eval()

    with torch.no_grad():
        logits = model(x_batch.to(device))
        preds = logits.argmax(dim=1)

    x_np = to_numpy_bchw(x_batch)
    y_np = y_batch.detach().cpu().numpy()

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for method in methods:
        print(f"\n=== Evaluando método XAI: {method} ===")

        attr_bchw = compute_attributions(explainer, x_batch, preds, method)
        attr_np = to_numpy_bchw(attr_bchw)

        method_results: Dict[str, Dict[str, float]] = {}

        # CORRECCIÓN: Crear métricas específicas para este método
        metrics = create_metrics(explainer, method, device)
        explain_fn = build_explain_func(explainer, method, device)

        for metric_name, metric in metrics.items():
            print(f" -> Métrica: {metric_name}")
            try:
                if metric_name in {"robustness", "randomization"}:
                    if metric_name == "randomization":
                        print(f"    🔍 Calculando randomization para {method}...")
                        print(f"       Comparando modelo entrenado vs. aleatorizado...")
                    
                    scores = metric(
                        model=model,
                        x_batch=x_np,
                        y_batch=y_np,
                        explain_func=explain_fn,
                        device=device,
                    )
                else:
                    scores = metric(
                        model=model,
                        x_batch=x_np,
                        y_batch=y_np,
                        a_batch=attr_np,
                        device=device,
                    )

                if isinstance(scores, dict):
                    if "scores" in scores:
                        raw_scores = scores["scores"]
                    else:
                        raw_scores = None
                        for v in scores.values():
                            if isinstance(v, (list, tuple, np.ndarray)):
                                raw_scores = v
                                break
                        if raw_scores is None:
                            raise TypeError(f"Formato no soportado para '{metric_name}'")
                else:
                    raw_scores = scores

                raw_scores = np.array(raw_scores, dtype=float).flatten()
                valid_scores = raw_scores[np.isfinite(raw_scores)]
                
                if len(valid_scores) == 0:
                    mean = None
                    std = None
                    print(f"    ⚠️  Todos los valores son inf/nan")
                elif len(valid_scores) < len(raw_scores):
                    mean = float(np.mean(valid_scores))
                    std = float(np.std(valid_scores))
                    invalid_count = len(raw_scores) - len(valid_scores)
                    print(f"    ⚠️  {invalid_count}/{len(raw_scores)} valores inválidos filtrados")
                else:
                    mean = float(np.mean(valid_scores))
                    std = float(np.std(valid_scores))
                
                mean_json = None if (mean is None or np.isinf(mean) or np.isnan(mean)) else mean
                std_json = None if (std is None or np.isinf(std) or np.isnan(std)) else std
                
                scores_list = [
                    None if (np.isinf(s) or np.isnan(s)) else float(s)
                    for s in raw_scores
                ]

                method_results[metric_name] = {
                    "mean": mean_json,
                    "std": std_json,
                    "scores": scores_list,
                }
                
                if mean_json is None:
                    print(f"    None (valores inválidos)")
                elif std_json is None:
                    print(f"    {mean:.4f} ± None")
                else:
                    print(f"    {mean:.4f} ± {std:.4f}")
                    
            except Exception as err:
                print(f"    ⚠️ Error evaluando {metric_name} para {method}: {err}")
                import traceback
                traceback.print_exc()
                method_results[metric_name] = None

        results[method] = method_results

    return results


# ============================================================
#  Guardar resultados
# ============================================================

def save_results(results: Dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f"\n✅ Resultados guardados en {output_path}")


# ============================================================
#  main()
# ============================================================

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    
    set_global_seed(args.seed)

    print("=" * 60)
    print("  EVALUACIÓN QUANTUS - RESNET18 XAI (VERSIÓN CORREGIDA)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Dispositivo: {device}")
    print(f"Métodos: {args.methods}")
    print(f"Muestras a evaluar: {args.num_samples}")
    print(f"Seed: {args.seed}")

    meta_all = get_dataset_info()
    name_map = {"blood": "bloodmnist", "retina": "retinamnist", "breast": "breastmnist"}
    med_name = name_map[args.dataset]
    num_classes = int(meta_all[med_name]["n_classes"])

    if args.model_path is None:
        model_path = f"results/best_model_{args.dataset}.pth"
    else:
        model_path = args.model_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se ha encontrado {model_path}. "
            f"Ejecuta primero: python train.py --dataset {args.dataset}"
        )

    model = load_trained_model(model_path, device, num_classes=num_classes)

    datasets = load_datasets(args.data_dir, target_size=224)
    _, _, test_loader, _ = create_data_loaders(
        datasets=datasets,
        batch_size=args.batch_size,
        num_workers=0,
        num_classes=num_classes,
        dataset_name=args.dataset,
    )

    x_batch, y_batch = collect_samples(test_loader, args.num_samples, device)

    explainer = XAIExplainer(model, device, num_classes=num_classes, dataset=args.dataset)

    results = evaluate_methods(model, explainer, x_batch, y_batch, args.methods, device)
    
    results["metadata"] = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "num_samples": args.num_samples,
        "methods": args.methods,
    }
    
    if args.output is None:
        output_path = f"outputs/quantus_metrics_{args.dataset}.json"
    else:
        output_path = args.output
    
    save_results(results, output_path)
    print(f"\n✅ Resultados guardados en: {output_path}")


if __name__ == "__main__":
    main()
