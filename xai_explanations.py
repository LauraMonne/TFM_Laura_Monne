"""
Script de Explicabilidad (XAI) para ResNet-18 - MedMNIST

Métodos implementados:
- Grad-CAM y Grad-CAM++
- Integrated Gradients (IG)
- Saliency Maps (Vanilla Saliency)

Genera:
- Mapas PNG en outputs/
- Metadatos en outputs/explanations_results.json

"""

import os
import json
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ====== Librerías externas de XAI ======
# Librerías externas de XAI: pytorch-grad-cam, captum y quantus.

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    print("pytorch-grad-cam no disponible. Instala con: pip install grad-cam")

try:
    from captum.attr import IntegratedGradients, Saliency
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("captum no disponible. Instala con: pip install captum")

try:
    import quantus 
    QUANTUS_AVAILABLE = True
except ImportError:
    QUANTUS_AVAILABLE = False
    print("quantus no disponible. Instala con: pip install quantus")

# ====== Módulos propios del repo ======
import argparse
from prepare_data import load_datasets, get_dataset_info
from resnet18 import create_model
from train import create_data_loaders


# ============================================================
#  Utilidades de directorios e imagen
# ============================================================
# Crea las carpetas de salida si no existen, organizadas por dataset.
def _ensure_dirs(dataset: str):
    """
    Crea las carpetas de salida organizadas por dataset.
    
    Args:
        dataset: "blood", "retina" o "breast"
    """
    base_dir = f"outputs/{dataset}"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f"{base_dir}/gradcam", exist_ok=True)
    os.makedirs(f"{base_dir}/gradcampp", exist_ok=True)
    os.makedirs(f"{base_dir}/integrated_gradients", exist_ok=True)
    os.makedirs(f"{base_dir}/saliency", exist_ok=True)

# Desnormaliza una imagen.
# Convierte un tensor (1, C, H, W) o (C, H, W) a una imagen numpy (H, W, 3) en rango [0,1].

def _denormalize_image(
    img_tensor: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Desnormaliza un tensor (1, C, H, W) o (C, H, W) a una imagen numpy (H, W, 3)
    en rango [0,1], lista para show_cam_on_image.
    """
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.clone().detach().cpu()
    else:
        img = torch.tensor(img_tensor).clone().detach()

    if img.dim() == 4:
        img = img[0]  # tomar la primera imagen del batch

    # Si es 1 canal, lo expandimos a 3 canales
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)

    for c, (m, s) in enumerate(zip(mean, std)):
        img[c].mul_(s).add_(m)

    img = torch.clamp(img, 0.0, 1.0)
    img_np = img.permute(1, 2, 0).numpy()  # (H, W, C)
    return img_np.astype(np.float32)


# ============================================================
#  Clase principal de explicabilidad
# ============================================================
# Inicializa el modelo, las capas target y los métodos Captum si están disponibles.

class XAIExplainer:
    def __init__(self, model: nn.Module, device: torch.device, num_classes: int = 15, dataset: str = "blood"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.num_classes = num_classes
        self.dataset = dataset
        _ensure_dirs(dataset)

        # Métodos Captum
        if CAPTUM_AVAILABLE:
            try:
                self.ig = IntegratedGradients(self.model)
                self.saliency = Saliency(self.model)
            except Exception as e:
                print(f"Error inicializando Captum: {e}")
                self.ig = None
                self.saliency = None
        else:
            self.ig = None
            self.saliency = None

    # ----------------- Grad-CAM helpers -----------------
    # Obtiene el modelo base y las capas target para Grad-CAM.
    def _get_gradcam_target(self, input_tensor: torch.Tensor):
        """
        Devuelve (modelo_base, [layer_target]) para Grad-CAM.
        Tiene en cuenta si el modelo tiene atributos rgb_model/gray_model.
        """
        c = input_tensor.shape[1]

        # Caso modelo adaptativo con rgb_model / gray_model
        if hasattr(self.model, "rgb_model") and c == 3:
            return self.model.rgb_model, [self.model.rgb_model.layer4]
        if hasattr(self.model, "gray_model") and c == 1:
            return self.model.gray_model, [self.model.gray_model.layer4]

        # Caso ResNet-18 estándar
        if hasattr(self.model, "layer4"):
            return self.model, [self.model.layer4]

        # Fallback
        return self.model, [self.model]

    def _class_target_fn(self, cls_idx: int):
        """
        Devuelve un callable que extrae la logit de la clase cls_idx del output del modelo.
        Maneja outputs 1D o 2D para compatibilidad con distintas versiones de pytorch-grad-cam.
        """
        class ClassTarget:
            def __init__(self, idx: int):
                self.idx = idx

            def __call__(self, model_output: torch.Tensor):
                if model_output.dim() == 1:
                    # (num_classes,)
                    return model_output[self.idx]
                elif model_output.dim() == 2:
                    # (batch, num_classes)
                    return model_output[:, self.idx]
                else:
                    flat = model_output.flatten()
                    if flat.numel() > self.idx:
                        return flat[self.idx]
                    return flat[0]

        return ClassTarget(cls_idx)
# Genera Grad-CAM y devuelve (overlay_img, heatmap_2d)
    def generate_gradcam(self, input_tensor: torch.Tensor, target_class: int, save_path: str | None):
        """
        Genera Grad-CAM y devuelve (overlay_img, heatmap_2d)
        """
        if not GRAD_CAM_AVAILABLE:
            return None

        try:
            input_tensor = input_tensor.to(self.device)
            grad_model, target_layers = self._get_gradcam_target(input_tensor)

            cam = GradCAM(
                model=grad_model,
                target_layers=target_layers,
            )

            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[self._class_target_fn(target_class)],
                eigen_smooth=False,
                aug_smooth=False,
            )  # (1, H, W)

            rgb_img = _denormalize_image(input_tensor)
            overlay = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)

            if save_path is not None:
                plt.imsave(save_path, overlay)

            return overlay, grayscale_cam[0]

        except Exception as e:
            print(f"Error en Grad-CAM: {e}")
            return None
# Genera Grad-CAM++ y devuelve (overlay_img, heatmap_2d)
    def generate_gradcampp(self, input_tensor: torch.Tensor, target_class: int, save_path: str | None):
        """
        Genera Grad-CAM++ y devuelve (overlay_img, heatmap_2d)
        """
        if not GRAD_CAM_AVAILABLE:
            return None

        try:
            input_tensor = input_tensor.to(self.device)
            grad_model, target_layers = self._get_gradcam_target(input_tensor)

            campp = GradCAMPlusPlus(
                model=grad_model,
                target_layers=target_layers,
            )

            grayscale_cam = campp(
                input_tensor=input_tensor,
                targets=[self._class_target_fn(target_class)],
                eigen_smooth=False,
                aug_smooth=False,
            )

            rgb_img = _denormalize_image(input_tensor)
            overlay = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)

            if save_path is not None:
                plt.imsave(save_path, overlay)

            return overlay, grayscale_cam[0]

        except Exception as e:
            print(f"Error en Grad-CAM++: {e}")
            return None

    # ----------------- Integrated Gradients -----------------
    # Genera mapas de Integrated Gradients.
    # Devuelve (attr_hwc_normalizada, tensor_attrib_original)
    def generate_integrated_gradients(self, input_tensor: torch.Tensor, target_class: int, save_path: str | None):
        """
        Genera mapas de Integrated Gradients.
        Devuelve (attr_hwc_normalizada, tensor_attrib_original)
        """
        if self.ig is None:
            return None

        try:
            input_tensor = input_tensor.to(self.device)
            baseline = torch.zeros_like(input_tensor)

            attributions = self.ig.attribute(
                input_tensor,
                baseline,
                target=target_class,
                n_steps=50,
            )  # (1, C, H, W)

            attr = attributions[0].detach().cpu().numpy()  # (C, H, W)
            attr = np.abs(attr)

            # Mapa escalar
            attr_scalar = attr.mean(axis=0)  # (H, W)
            attr_scalar /= (attr_scalar.max() + 1e-8)

            rgb_img = _denormalize_image(input_tensor)

            if save_path is not None:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(rgb_img)
                axes[0].set_title("Imagen original")
                axes[0].axis("off")
                axes[1].imshow(attr_scalar, cmap="magma")
                axes[1].set_title("Integrated Gradients (escala)")
                axes[1].axis("off")
                heatmap = plt.cm.magma(attr_scalar)[..., :3]
                overlay = 0.6 * rgb_img + 0.4 * heatmap
                overlay = np.clip(overlay, 0.0, 1.0)
                axes[2].imshow(overlay)
                axes[2].set_title("Superposición")
                axes[2].axis("off")
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()

            attr_hwc = np.repeat(attr_scalar[..., None], 3, axis=2)
            return attr_hwc, attributions

        except Exception as e:
            print(f"Error en Integrated Gradients: {e}")
            return None

    # ----------------- Saliency Map -----------------
    # Genera mapas de saliencia (Vanilla Saliency).
    # Devuelve (saliency_hwc_normalizada, tensor_attrib_original)
    def generate_saliency_map(self, input_tensor: torch.Tensor, target_class: int, save_path: str | None):
        """
        Genera mapas de saliencia (Vanilla Saliency).
        Devuelve (saliency_hwc_normalizada, tensor_attrib_original)
        """
        if self.saliency is None:
            return None

        try:
            input_tensor = input_tensor.to(self.device)
            input_tensor.requires_grad = True

            attributions = self.saliency.attribute(
                input_tensor,
                target=target_class,
            )  # (1, C, H, W)

            attr = attributions[0].detach().cpu().numpy()
            attr = np.abs(attr)

            attr_scalar = attr.mean(axis=0)
            attr_scalar /= (attr_scalar.max() + 1e-8)

            rgb_img = _denormalize_image(input_tensor)

            if save_path is not None:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(rgb_img)
                axes[0].set_title("Imagen original")
                axes[0].axis("off")
                axes[1].imshow(attr_scalar, cmap="hot")
                axes[1].set_title("Saliency (escala)")
                axes[1].axis("off")
                heatmap = plt.cm.hot(attr_scalar)[..., :3]
                overlay = 0.6 * rgb_img + 0.4 * heatmap
                overlay = np.clip(overlay, 0.0, 1.0)
                axes[2].imshow(overlay)
                axes[2].set_title("Superposición")
                axes[2].axis("off")
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()

            saliency_hwc = np.repeat(attr_scalar[..., None], 3, axis=2)
            return saliency_hwc, attributions

        except Exception as e:
            print(f"Error en Saliency Map: {e}")
            return None

    # ----------------- Wrapper: generar todo para una imagen -----------------
    # Genera todas las explicaciones para una imagen: ejecuta Grad-CAM, Grad-CAM++, Integrated Gradients y Saliency, guarda los mapas y devuelve un diccionario con rutas y estados.
    def generate_all_explanations(self, input_tensor: torch.Tensor, pred_class: int, image_idx: int) -> Dict:
        """
        Genera Grad-CAM, Grad-CAM++, IG y Saliency para una sola imagen.
        Devuelve un diccionario con rutas y estados.
        """
        results = {}

        # Base path para las imágenes según el dataset
        base_path = f"outputs/{self.dataset}"

        # Grad-CAM
        if GRAD_CAM_AVAILABLE:
            gradcam_path = f"{base_path}/gradcam/img_{image_idx}_class_{pred_class}.png"
            res_gc = self.generate_gradcam(input_tensor, pred_class, gradcam_path)
            results["gradcam"] = (
                {"path": gradcam_path, "status": "success"} if res_gc is not None
                else {"status": "error"}
            )
        else:
            results["gradcam"] = {"status": "not_available"}

        # Grad-CAM++
        if GRAD_CAM_AVAILABLE:
            gradcampp_path = f"{base_path}/gradcampp/img_{image_idx}_class_{pred_class}.png"
            res_gcpp = self.generate_gradcampp(input_tensor, pred_class, gradcampp_path)
            results["gradcampp"] = (
                {"path": gradcampp_path, "status": "success"} if res_gcpp is not None
                else {"status": "error"}
            )
        else:
            results["gradcampp"] = {"status": "not_available"}

        # Integrated Gradients
        if self.ig is not None:
            ig_path = f"{base_path}/integrated_gradients/img_{image_idx}_class_{pred_class}.png"
            res_ig = self.generate_integrated_gradients(input_tensor, pred_class, ig_path)
            results["integrated_gradients"] = (
                {"path": ig_path, "status": "success"} if res_ig is not None
                else {"status": "error"}
            )
        else:
            results["integrated_gradients"] = {"status": "not_available"}

        # Saliency
        if self.saliency is not None:
            sal_path = f"{base_path}/saliency/img_{image_idx}_class_{pred_class}.png"
            res_sal = self.generate_saliency_map(input_tensor, pred_class, sal_path)
            results["saliency"] = (
                {"path": sal_path, "status": "success"} if res_sal is not None
                else {"status": "error"}
            )
        else:
            results["saliency"] = {"status": "not_available"}

        return results


# ============================================================
#  Carga del modelo entrenado
# ============================================================
def load_trained_model(model_path: str, device: torch.device, num_classes: int = 15) -> nn.Module:
    print(f"Cargando modelo desde {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Intentar obtener la configuración del checkpoint
    use_pretrained = False
    freeze_backbone = False
    
    if "config" in checkpoint:
        config = checkpoint["config"]
        use_pretrained = config.get("use_pretrained", False)
        freeze_backbone = config.get("freeze_backbone", False)
        print(f"  Configuración detectada desde checkpoint: pretrained={use_pretrained}, freeze_backbone={freeze_backbone}")
    else:
        # Si no hay config, intentar detectar el tipo de modelo por las claves del state_dict
        state_dict_keys = list(checkpoint["model_state_dict"].keys())
        print(f"  No hay 'config' en checkpoint. Detectando tipo de modelo por claves del state_dict...")
        print(f"  Primeras 5 claves: {state_dict_keys[:5]}")
        
        # Si las claves empiezan con "rgb_model." o "gray_model.", es ResNet18Adaptive
        if any(k.startswith("rgb_model.") or k.startswith("gray_model.") for k in state_dict_keys):
            print("  Detectado: ResNet18Adaptive (sin pre-entrenamiento)")
            use_pretrained = False
        # Si las claves empiezan con "conv1." o "layer1.", es ResNet-18 estándar de torchvision
        elif any(k.startswith("conv1.") or k.startswith("layer1.") for k in state_dict_keys):
            print("  Detectado: ResNet-18 estándar (probablemente pre-entrenado)")
            use_pretrained = True
            # Si tiene "fc.1.weight" en lugar de "fc.weight", tiene Dropout (fine-tuning)
            if any("fc.1.weight" in k for k in state_dict_keys):
                print("  Detectado: fc tiene Dropout (fine-tuning activado)")
                freeze_backbone = True
        else:
            print("  ⚠️ No se pudo detectar el tipo de modelo. Usando ResNet18Adaptive por defecto.")
            use_pretrained = False
    
    # Crear el modelo con la configuración correcta
    print(f"  Creando modelo con: pretrained={use_pretrained}, freeze_backbone={freeze_backbone}, num_classes={num_classes}")
    model = create_model(
        num_classes=num_classes,
        pretrained=use_pretrained,
        freeze_backbone=freeze_backbone
    )
    
    # Cargar el state_dict
    # Nota: cuando freeze_backbone=True, algunas capas pueden estar congeladas,
    # pero el state_dict contiene todos los parámetros, así que cargamos todo
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print("  ✓ State dict cargado correctamente con strict=True")
    except RuntimeError as e:
        print(f"  ⚠️ Error al cargar con strict=True")
        print(f"  Error: {str(e)[:200]}...")  # Mostrar solo los primeros 200 caracteres
        print("  Intentando cargar con strict=False (ignorando claves no coincidentes)...")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing_keys:
            print(f"  ⚠️ Claves faltantes ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"  ⚠️ Claves inesperadas ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
    
    model.to(device)
    model.eval()
    print("Modelo cargado correctamente.")
    return model


# ============================================================
#  Evaluación cuantitativa
# ============================================================

def evaluate_with_quantus():
    """
    Stub MUY simple para evitar errores con versiones de Quantus.
    Deja constancia en consola de que la evaluación cuantitativa
    puede hacerse con Quantus, pero no se fuerza aquí.
    """
    if not QUANTUS_AVAILABLE:
        print("\nQuantus no está instalado: se omite la evaluación cuantitativa automática.")
        return

    print(
        "\nQuantus está instalado, pero por compatibilidad de versiones "
        "la evaluación cuantitativa detallada no se ejecuta automáticamente "
        "en este script."
    )


# ============================================================
#  parse_args()
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera explicaciones XAI para ResNet-18 en MedMNIST (por dataset individual)."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["blood", "retina", "breast"],
        help="Dataset a usar: blood (8 clases), retina (5 clases) o breast (2 clases).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Ruta al checkpoint del modelo. Si no se especifica, se usa results/best_model_{dataset}.pth",
    )
    return parser.parse_args()


# ============================================================
#  main()
# ============================================================
def main():
    args = parse_args()
    
    print("=" * 60)
    print("EXPLICABILIDAD (XAI) - ResNet-18 MedMNIST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

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

    print(f"Dataset: {args.dataset} ({num_classes} clases)")
    print(f"Modelo: {model_path}")

    # Nº máximo de explicaciones por dataset (límite alto para datasets individuales)
    total_max = 500
    if args.dataset == "blood":
        max_blood = total_max
        max_retina = 0
        max_breast = 0
    elif args.dataset == "retina":
        max_blood = 0
        max_retina = total_max
        max_breast = 0
    else:  # breast
        max_blood = 0
        max_retina = 0
        max_breast = total_max

    # ----- Cargar modelo entrenado -----
    model = load_trained_model(model_path, device, num_classes=num_classes)

    # ----- Cargar datos de test -----
    print("\nCargando datos de test...")
    datasets = load_datasets("./data", target_size=224)
    
    # Usar create_data_loaders de train.py para filtrar por dataset
    _, _, test_loader, _ = create_data_loaders(
        datasets=datasets,
        batch_size=1,
        num_workers=0,
        num_classes=num_classes,
        dataset_name=args.dataset,
    )
    
    # Para datasets individuales, todas las clases están en el rango [0, num_classes)
    if args.dataset == "blood":
        BLOOD_RANGE = range(0, num_classes)
        RETINA_RANGE = range(0)
        BREAST_RANGE = range(0)
    elif args.dataset == "retina":
        BLOOD_RANGE = range(0)
        RETINA_RANGE = range(0, num_classes)
        BREAST_RANGE = range(0)
    else:  # breast
        BLOOD_RANGE = range(0)
        RETINA_RANGE = range(0)
        BREAST_RANGE = range(0, num_classes)

    explainer = XAIExplainer(model, device, num_classes=num_classes, dataset=args.dataset)

    # ----- Generar explicaciones -----
    print(f"\nGenerando explicaciones para dataset {args.dataset} (hasta {total_max} muestras)...")
    
    from tqdm import tqdm

    count_blood = 0
    count_retina = 0
    count_breast = 0

    all_results: List[Dict] = []
    global_idx = 0  # índice para nombrar los ficheros de salida

    for data, target in tqdm(test_loader, desc="Generando explicaciones"):
        true_class = int(target.item())

        # Para datasets individuales, todas las muestras pertenecen al mismo dataset
        bucket = args.dataset
        
        # Verificar límite total según el dataset
        if args.dataset == "blood":
            if count_blood >= max_blood:
                break
        elif args.dataset == "retina":
            if count_retina >= max_retina:
                break
        elif args.dataset == "breast":
            if count_breast >= max_breast:
                break

        data = data.to(device)

        with torch.no_grad():
            logits = model(data)
            pred_class = int(logits.argmax(dim=1).item())

        res = explainer.generate_all_explanations(
            input_tensor=data,
            pred_class=pred_class,
            image_idx=global_idx,  # usamos índice global para nombrar imágenes
        )

        all_results.append(
            {
                "global_idx": global_idx,
                "bucket": bucket,
                "true_class": true_class,
                "pred_class": pred_class,
                "methods": res,
            }
        )

        global_idx += 1
        if bucket == "blood":
            count_blood += 1
        elif bucket == "retina":
            count_retina += 1
        elif bucket == "breast":
            count_breast += 1

    # ----- Guardar metadatos -----
    print("\nGuardando metadatos de explicabilidad...")
    os.makedirs("outputs", exist_ok=True)
    # Guardar un JSON por dataset, para no pisar resultados entre blood/retina/breast.
    json_path = f"outputs/explanations_results_{args.dataset}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    total_count = count_blood + count_retina + count_breast
    print(
        f"\nResumen de muestras explicadas:"
        f"\n  {args.dataset.upper()}MNIST: {total_count}"
        f"\n  TOTAL: {total_count}"
        f"\n  JSON:  {json_path}"
    )

    # ----- Evaluación cuantitativa (stub) -----
    evaluate_with_quantus()

    print("\n" + "=" * 60)
    print("EXPLICABILIDAD COMPLETADA")
    print("=" * 60)
    print(f"Mapas guardados en 'outputs/{args.dataset}/':")
    print(f"  - Grad-CAM:             outputs/{args.dataset}/gradcam/")
    print(f"  - Grad-CAM++:           outputs/{args.dataset}/gradcampp/")
    print(f"  - Integrated Gradients: outputs/{args.dataset}/integrated_gradients/")
    print(f"  - Saliency Maps:        outputs/{args.dataset}/saliency/")
    print("Metadatos:")
    print(f"  - outputs/explanations_results_{args.dataset}.json")


if __name__ == "__main__":
    main()
"""
Resumen
Explicaciones XAI para ResNet-18:
1. Métodos implementados:
- Grad-CAM: mapas de activación con gradientes
- Grad-CAM++: versión mejorada de Grad-CAM
- Integrated Gradients: atribuciones integradas desde un baseline
- Saliency Maps: mapas de saliencia basados en gradientes
2. Proceso:
- Carga el modelo entrenado
- Carga datos de test
- Genera todas las explicaciones para cada imagen
- Guarda mapas PNG en directorios organizados
- Guarda metadatos en JSON
3. Salidas:
- Mapas PNG en outputs/ (subdirectorios por método)
- Metadatos en outputs/explanations_results.json
Útil para analizar qué regiones de las imágenes son más importantes para las predicciones del modelo.

"""
