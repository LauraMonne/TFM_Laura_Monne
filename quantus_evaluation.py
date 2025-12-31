"""
Evaluación cuantitativa de explicabilidad usando Quantus.

Mide 5 dimensiones para varios métodos XAI (Grad-CAM, Grad-CAM++, IG, Saliency):
- Fidelidad      -> FaithfulnessCorrelation
- Robustez       -> AvgSensitivity
- Complejidad    -> Complexity (o Entropy)
- Aleatorización -> MPRT / ModelParameterRandomisation
- Localización   -> RegionPerturbation

Uso típico:
    python quantus_evaluation.py --dataset retina --num_samples 100 --seed 123
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Callable, Tuple

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
#  Reproducibilidad
# ============================================================

def set_global_seed(seed: int) -> None:
    """Establece la semilla global para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinismo (puede reducir rendimiento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
def to_numpy_bchw(tensor_batch: torch.Tensor) -> np.ndarray:
    """Convierte un tensor BCHW a NumPy BCHW (sin cambiar el orden de ejes)."""
    return tensor_batch.detach().cpu().numpy()


# ============================================================
#  Sanitización / normalización de atribuciones
# ============================================================

def sanitize_attribution(attr: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Sanitiza y normaliza atribuciones para mejorar estabilidad numérica.
    
    - Fuerza float32 para consistencia
    - Reemplaza nan/inf por 0
    - Normaliza por muestra a [0,1] (min-max) para evitar mapas constantes raros
    
    Args:
        attr: Tensor de atribuciones (C, H, W) o (B, C, H, W)
        eps: Tolerancia para detectar mapas constantes
        
    Returns:
        Tensor sanitizado y normalizado
    """
    attr = attr.to(dtype=torch.float32)
    attr = torch.nan_to_num(attr, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Si todo es cero (mapa vacío), devolvemos tal cual
    if torch.all(attr == 0):
        return attr
    
    # Normalización min-max por tensor (C,H,W) o (B,C,H,W)
    if attr.ndim == 3:  # (C, H, W)
        mn = torch.min(attr)
        mx = torch.max(attr)
        if (mx - mn).abs() < eps:
            # Mapa constante -> lo dejamos a ceros
            return torch.zeros_like(attr)
        attr = (attr - mn) / (mx - mn + eps)
    elif attr.ndim == 4:  # (B, C, H, W) - normalizar por muestra
        for b in range(attr.shape[0]):
            sample = attr[b]
            mn = torch.min(sample)
            mx = torch.max(sample)
            if (mx - mn).abs() < eps:
                attr[b] = torch.zeros_like(sample)
            else:
                attr[b] = (sample - mn) / (mx - mn + eps)
    else:
        # Formato no soportado, devolver tal cual
        return attr
    
    return attr


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
            
            # Sanitizar atribución antes de añadirla
            attr = sanitize_attribution(attr)
        except Exception as err:
            print(f"⚠️ Error generando atribución para muestra {idx}: {err}")
            attr = torch.zeros_like(sample[0].cpu())
        attributions.append(attr)
    return torch.stack(attributions, dim=0)  # (B, C, H, W)


# ============================================================
#  explain_func para métricas que lo requieren (robustness, randomization)
# ============================================================


def build_explain_func(
    explainer: XAIExplainer,
    method: str,
    device: torch.device,
) -> Callable:
    """
    Construye una explain_func compatible con Quantus.
    Firma esperada: explain_func(model, inputs, targets, **kwargs) -> np.ndarray
    """

    def explain_func(model, inputs, targets, **kwargs):
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
                
                # Sanitizar atribución antes de añadirla
                attr = sanitize_attribution(attr)
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

# Crea un conjunto estándar de métricas de Quantus.
# Devuelve un diccionario con las métricas.
def create_metrics() -> Dict[str, object]:
    """Crea un conjunto estándar de métricas de Quantus."""
    metrics: Dict[str, object] = {}

    # Fidelidad
    metrics["faithfulness"] = quantus.FaithfulnessCorrelation()

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
    # MPRT mide cómo cambian las explicaciones cuando se aleatorizan los parámetros del modelo.
    # Valores cercanos a 1.0 indican que las explicaciones no cambian (malo - explicaciones no son sensibles).
    # Valores cercanos a 0.0 indican que las explicaciones cambian significativamente (bueno - explicaciones son sensibles).
    # 
    # PROBLEMA: Si todos los métodos obtienen ~1.0, puede ser que:
    # 1. La métrica no esté aleatorizando correctamente los parámetros
    # 2. Las explicaciones realmente no cambian cuando se aleatorizan parámetros (problema de los métodos XAI)
    # 3. Hay un bug en cómo se está usando la métrica
    #
    # SOLUCIÓN: Probar diferentes configuraciones y métricas alternativas
    try:
        RandomizationMetric = quantus.MPRT
        print("📊 Usando métrica MPRT para randomization")
    except AttributeError:
        RandomizationMetric = quantus.ModelParameterRandomisation
        print("📊 Usando métrica ModelParameterRandomisation para randomization")
    
    # Intentar configurar con parámetros que ayuden a diferenciar métodos
    randomization_metric = None
    config_attempts = [
        # Configuración 1: Con función de similitud explícita (correlación de Spearman)
        {
            "name": "correlación Spearman + normalización",
            "params": {
                "similarity_func": quantus.similarity_func.correlation_spearman,
                "normalise": True,
                "disable_warnings": True,
            }
        },
        # Configuración 2: Con correlación de Pearson
        {
            "name": "correlación Pearson + normalización",
            "params": {
                "similarity_func": quantus.similarity_func.correlation_pearson,
                "normalise": True,
                "disable_warnings": True,
            }
        },
        # Configuración 3: Solo normalización
        {
            "name": "solo normalización",
            "params": {
                "normalise": True,
                "disable_warnings": True,
            }
        },
        # Configuración 4: Por defecto
        {
            "name": "por defecto",
            "params": {}
        }
    ]
    
    for attempt in config_attempts:
        try:
            randomization_metric = RandomizationMetric(**attempt["params"])
            print(f"   ✓ Configuración exitosa: {attempt['name']}")
            break
        except (TypeError, AttributeError, KeyError) as e:
            continue
    
    if randomization_metric is None:
        # Si todas las configuraciones fallan, usar la más básica
        print("   ⚠️  Todas las configuraciones fallaron, usando configuración mínima")
        randomization_metric = RandomizationMetric()
    
    metrics["randomization"] = randomization_metric

    # Localización
    metrics["localization"] = quantus.RegionPerturbation()

    # NOTA: Si MPRT sigue dando valores constantes (~1.0) para todos los métodos,
    # se puede considerar usar métricas alternativas como:
    # - quantus.RandomLogit: Mide la distancia entre explicación original y una clase aleatoria
    # - Una métrica personalizada que compare explicaciones con diferentes seeds
    # Sin embargo, MPRT es la métrica estándar para randomization, así que primero
    # intentamos con las configuraciones mejoradas arriba.

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

    # Convertir datos a NumPy (manteniendo BCHW)
    x_np = to_numpy_bchw(x_batch)  # (B, C, H, W)
    y_np = y_batch.detach().cpu().numpy()

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for method in methods:
        print(f"\n=== Evaluando método XAI: {method} ===")

        # 1) Atribuciones (mantenemos BCHW para coincidir con x_np)
        attr_bchw = compute_attributions(explainer, x_batch, preds, method)
        attr_np = to_numpy_bchw(attr_bchw)  # (B, C, H, W)

        method_results: Dict[str, Dict[str, float]] = {}

        # explain_func para métricas que lo requieren (robustness, randomization)
        explain_fn = build_explain_func(explainer, method, device)

        for metric_name, metric in metrics.items():
            print(f" -> Métrica: {metric_name}")
            try:
                # Robustez y aleatorización: usar explain_func (Quantus calcula a_batch internamente)
                if metric_name in {"robustness", "randomization"}:
                    # Logging adicional para randomization
                    if metric_name == "randomization":
                        print(f"    🔍 Calculando randomization (MPRT) para {method}...")
                        print(f"       Esto puede tardar ya que aleatoriza parámetros del modelo.")
                    
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
                
                # Detección especial para randomization: verificar si todos los valores son constantes
                if metric_name == "randomization" and len(raw_scores) > 0:
                    valid_for_check = raw_scores[np.isfinite(raw_scores)]
                    if len(valid_for_check) > 0:
                        # Verificar si todos los valores son muy cercanos a 1.0 (constantes)
                        all_near_one = np.all(np.abs(valid_for_check - 1.0) < 0.01)
                        if all_near_one:
                            print(f"    ⚠️  ADVERTENCIA: Todos los valores de randomization están cerca de 1.0")
                            print(f"       Esto indica que las explicaciones no cambian cuando se aleatorizan los parámetros.")
                            print(f"       Posibles causas:")
                            print(f"       1. Los métodos XAI no son sensibles a la aleatorización de parámetros")
                            print(f"       2. La métrica MPRT no está funcionando correctamente")
                            print(f"       3. El modelo es demasiado robusto a la aleatorización")
                
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
    
    # Establecer semilla para reproducibilidad
    set_global_seed(args.seed)

    print("=" * 60)
    print("  EVALUACIÓN QUANTUS - RESNET18 XAI")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Dispositivo: {device}")
    print(f"Métodos: {args.methods}")
    print(f"Muestras a evaluar: {args.num_samples}")
    print(f"Seed: {args.seed}")

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
    x_batch, y_batch = collect_samples(test_loader, args.num_samples, device)

    # Explainer XAI
    explainer = XAIExplainer(model, device, num_classes=num_classes)

    # Evaluación
    results = evaluate_methods(model, explainer, x_batch, y_batch, args.methods, device)
    
    # Añadir metadata al resultado
    results["metadata"] = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "num_samples": args.num_samples,
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