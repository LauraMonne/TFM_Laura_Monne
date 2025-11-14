"""
Script de Explicabilidad (XAI) para ResNet-18 - MedMNIST

Implementa métodos de explicabilidad según la memoria del TFM:
- Grad-CAM y Grad-CAM++
- Integrated Gradients (IG)
- Saliency Maps (Vanilla Saliency)

Genera:
- Mapas PNG en outputs/
- Metadatos en outputs/explanations_results.json

La parte de evaluación cuantitativa con Quantus queda preparada pero
simplificada para evitar problemas de versiones. No se fuerza el uso
de Quantus si la API no es compatible.
"""

import os
import json
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ====== Librerías externas de XAI ======
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    print("⚠️ pytorch-grad-cam no disponible. Instala con: pip install grad-cam")

try:
    from captum.attr import IntegratedGradients, Saliency
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("⚠️ captum no disponible. Instala con: pip install captum")

try:
    import quantus
    QUANTUS_AVAILABLE = True
except ImportError:
    QUANTUS_AVAILABLE = False
    print("⚠️ quantus no disponible. Instala con: pip install quantus")

# ====== Módulos propios del repo ======
from prepare_data import load_datasets
from resnet18 import create_model
from data_utils import create_data_loaders_fixed


# ============================================================
#  Utilidades
# ============================================================

def _ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/gradcam", exist_ok=True)
    os.makedirs("outputs/gradcampp", exist_ok=True)
    os.makedirs("outputs/integrated_gradients", exist_ok=True)
    os.makedirs("outputs/saliency", exist_ok=True)


def _denormalize_image(
    img_tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
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

class XAIExplainer:
    def __init__(self, model: nn.Module, device: torch.device, num_classes: int = 15):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.num_classes = num_classes

        _ensure_dirs()

        # Métodos Captum (si están disponibles)
        if CAPTUM_AVAILABLE:
            try:
                self.ig = IntegratedGradients(self.model)
                self.saliency = Saliency(self.model)
            except Exception as e:
                print(f"⚠️ Error inicializando Captum: {e}")
                self.ig = None
                self.saliency = None
        else:
            self.ig = None
            self.saliency = None

    # ----------------- Grad-CAM helpers -----------------

    def _get_gradcam_target(self, input_tensor):
        """
        Devuelve (modelo_base, [layer_target]) para Grad-CAM.
        Tiene en cuenta si el modelo tiene atributos rgb_model/gray_model.
        """
        # input_tensor: (1, C, H, W)
        c = input_tensor.shape[1]

        # Caso modelo adaptativo con rgb_model / gray_model
        if hasattr(self.model, "rgb_model") and c == 3:
            return self.model.rgb_model, [self.model.rgb_model.layer4]
        if hasattr(self.model, "gray_model") and c == 1:
            return self.model.gray_model, [self.model.gray_model.layer4]

        # Caso ResNet-18 estándar
        if hasattr(self.model, "layer4"):
            return self.model, [self.model.layer4]

        # Fallback (no debería ocurrir)
        return self.model, [self.model]

    def generate_gradcam(self, input_tensor, target_class: int, save_path: str | None):
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

            class ClassTarget:
                def __init__(self, cls_idx):
                    self.cls_idx = cls_idx

                def __call__(self, model_output):
                    return model_output[:, self.cls_idx]

            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[ClassTarget(target_class)],
                eigen_smooth=False,
                aug_smooth=False,
            )  # (1, H, W)

            # Imagen base desnormalizada
            rgb_img = _denormalize_image(input_tensor)
            overlay = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)

            if save_path is not None:
                plt.imsave(save_path, overlay)

            return overlay, grayscale_cam[0]
        except Exception as e:
            print(f"⚠️ Error en Grad-CAM: {e}")
            return None

    def generate_gradcampp(self, input_tensor, target_class: int, save_path: str | None):
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

            class ClassTarget:
                def __init__(self, cls_idx):
                    self.cls_idx = cls_idx

                def __call__(self, model_output):
                    return model_output[:, self.cls_idx]

            grayscale_cam = campp(
                input_tensor=input_tensor,
                targets=[ClassTarget(target_class)],
                eigen_smooth=False,
                aug_smooth=False,
            )

            rgb_img = _denormalize_image(input_tensor)
            overlay = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)

            if save_path is not None:
                plt.imsave(save_path, overlay)

            return overlay, grayscale_cam[0]
        except Exception as e:
            print(f"⚠️ Error en Grad-CAM++: {e}")
            return None

    # ----------------- Integrated Gradients -----------------

    def generate_integrated_gradients(self, input_tensor, target_class: int, save_path: str | None):
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

            # Pasamos a mapa escalar (H, W) para usar colormap sin problemas
            attr_scalar = attr.mean(axis=0)  # (H, W)
            attr_scalar /= (attr_scalar.max() + 1e-8)

            # Imagen base
            rgb_img = _denormalize_image(input_tensor)  # (H, W, 3)

            if save_path is not None:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                axes[0].imshow(rgb_img)
                axes[0].set_title("Imagen original")
                axes[0].axis("off")

                axes[1].imshow(attr_scalar, cmap="magma")
                axes[1].set_title("Integrated Gradients (escala)")
                axes[1].axis("off")

                heatmap = plt.cm.magma(attr_scalar)[..., :3]  # (H, W, 3)
                overlay = 0.6 * rgb_img + 0.4 * heatmap
                overlay = np.clip(overlay, 0.0, 1.0)

                axes[2].imshow(overlay)
                axes[2].set_title("Superposición")
                axes[2].axis("off")

                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()

            # También devolvemos versión HWC 3 canales para coherencia
            attr_hwc = np.repeat(attr_scalar[..., None], 3, axis=2)  # (H,W,3)
            return attr_hwc, attributions
        except Exception as e:
            print(f"⚠️ Error en Integrated Gradients: {e}")
            return None

    # ----------------- Saliency Map -----------------

    def generate_saliency_map(self, input_tensor, target_class: int, save_path: str | None):
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
            print(f"⚠️ Error en Saliency Map: {e}")
            return None

    # ----------------- Wrapper: generar todo para una imagen -----------------

    def generate_all_explanations(self, input_tensor, pred_class: int, image_idx: int):
        """
        Genera Grad-CAM, Grad-CAM++, IG y Saliency para una sola imagen.
        Devuelve un diccionario con rutas y estados.
        """
        results = {}

        # Grad-CAM
        if GRAD_CAM_AVAILABLE:
            gradcam_path = f"outputs/gradcam/img_{image_idx}_class_{pred_class}.png"
            res_gc = self.generate_gradcam(input_tensor, pred_class, gradcam_path)
            if res_gc is not None:
                results["gradcam"] = {
                    "path": gradcam_path,
                    "status": "success",
                }
            else:
                results["gradcam"] = {"status": "error"}
        else:
            results["gradcam"] = {"status": "not_available"}

        # Grad-CAM++
        if GRAD_CAM_AVAILABLE:
            gradcampp_path = f"outputs/gradcampp/img_{image_idx}_class_{pred_class}.png"
            res_gcpp = self.generate_gradcampp(input_tensor, pred_class, gradcampp_path)
            if res_gcpp is not None:
                results["gradcampp"] = {
                    "path": gradcampp_path,
                    "status": "success",
                }
            else:
                results["gradcampp"] = {"status": "error"}
        else:
            results["gradcampp"] = {"status": "not_available"}

        # Integrated Gradients
        if self.ig is not None:
            ig_path = f"outputs/integrated_gradients/img_{image_idx}_class_{pred_class}.png"
            res_ig = self.generate_integrated_gradients(input_tensor, pred_class, ig_path)
            if res_ig is not None:
                results["integrated_gradients"] = {
                    "path": ig_path,
                    "status": "success",
                }
            else:
                results["integrated_gradients"] = {"status": "error"}
        else:
            results["integrated_gradients"] = {"status": "not_available"}

        # Saliency
        if self.saliency is not None:
            sal_path = f"outputs/saliency/img_{image_idx}_class_{pred_class}.png"
            res_sal = self.generate_saliency_map(input_tensor, pred_class, sal_path)
            if res_sal is not None:
                results["saliency"] = {
                    "path": sal_path,
                    "status": "success",
                }
            else:
                results["saliency"] = {"status": "error"}
        else:
            results["saliency"] = {"status": "not_available"}

        return results


# ============================================================
#  Carga del modelo entrenado
# ============================================================

def load_trained_model(model_path: str, device: torch.device, num_classes: int = 15):
    print(f"Cargando modelo desde {model_path}...")
    model = create_model(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("✅ Modelo cargado correctamente.")
    return model


# ============================================================
#  Evaluación cuantitativa (stub seguro)
# ============================================================

def evaluate_with_quantus_stub():
    """
    Stub MUY simple para evitar errores con versiones de Quantus.
    Deja constancia en consola de que la evaluación cuantitativa
    puede hacerse con Quantus, pero no se fuerza aquí para no romper
    la ejecución del pipeline.
    """
    if not QUANTUS_AVAILABLE:
        print("\nℹ️ Quantus no está instalado: se omite la evaluación cuantitativa automática.")
        return None

    # Si está instalado pero la API cambia, evitamos errores:
    print(
        "\nℹ️ Quantus está instalado, pero por compatibilidad de versiones "
        "la evaluación cuantitativa detallada no se ejecuta automáticamente "
        "en este script. Los mapas generados pueden evaluarse con Quantus "
        "en un notebook dedicado."
    )
    return None


# ============================================================
#  main()
# ============================================================

def main():
    print("=" * 60)
    print("EXPLICABILIDAD (XAI) - ResNet-18 MedMNIST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    model_path = "results/best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ No se ha encontrado {model_path}. "
            f"Ejecuta primero: python train.py"
        )

    num_classes = 15
    num_samples = 20  # número de imágenes de test a explicar

    # ----- Cargar modelo entrenado -----
    model = load_trained_model(model_path, device, num_classes=num_classes)

    # ----- Cargar datos de test -----
    print("\nCargando datos de test...")
    datasets = load_datasets("./data", target_size=224)
    train_loader, val_loader, test_loader = create_data_loaders_fixed(
        datasets=datasets,
        batch_size=1,
        num_workers=0,
        seed=42,
    )
    del train_loader, val_loader  # solo usamos test

    explainer = XAIExplainer(model, device, num_classes=num_classes)

    # ----- Generar explicaciones -----
    print(f"\nGenerando explicaciones para {num_samples} muestras de test...")
    from tqdm import tqdm

    all_results = []

    for idx, (data, target) in enumerate(tqdm(test_loader, desc="Generando explicaciones")):
        if idx >= num_samples:
            break

        data = data.to(device)
        true_class = int(target.item())

        with torch.no_grad():
            logits = model(data)
            pred_class = int(logits.argmax(dim=1).item())

        res = explainer.generate_all_explanations(
            input_tensor=data,
            pred_class=pred_class,
            image_idx=idx,
        )

        all_results.append(
            {
                "image_idx": idx,
                "true_class": true_class,
                "pred_class": pred_class,
                "methods": res,
            }
        )

    # ----- Guardar metadatos -----
    print("\nGuardando metadatos de explicabilidad...")
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/explanations_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ----- Evaluación cuantitativa (stub) -----
    evaluate_with_quantus_stub()

    print("\n" + "=" * 60)
    print("✅ EXPLICABILIDAD COMPLETADA")
    print("=" * 60)
    print("Mapas guardados en 'outputs/':")
    print("  - Grad-CAM:             outputs/gradcam/")
    print("  - Grad-CAM++:           outputs/gradcampp/")
    print("  - Integrated Gradients: outputs/integrated_gradients/")
    print("  - Saliency Maps:        outputs/saliency/")
    print("Metadatos:")
    print("  - outputs/explanations_results.json")


if __name__ == "__main__":
    main()
