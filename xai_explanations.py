"""
Script de Explicabilidad (XAI) para ResNet-18 entrenado sobre BloodMNIST,
RetinaMNIST y BreastMNIST (MedMNIST v2).

Implementa:
- Grad-CAM
- Grad-CAM++
- Integrated Gradients
- Saliency Maps

Genera:
- Mapas de atribución (PNG) en outputs/
- Metadatos en outputs/explanations_results.json

La evaluación cuantitativa con Quantus se recomienda hacerla en un notebook
dedicado, cargando los mapas generados por este script.
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

# -------------------------
# Dependencias XAI
# -------------------------
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

# -------------------------
# Módulos propios del repo
# -------------------------
from prepare_data import load_datasets
from data_utils import create_data_loaders_fixed
from resnet18 import create_model


# ============================================================
# Utilidades generales
# ============================================================

def ensure_dirs():
    """Crear estructura de carpetas de salida."""
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/gradcam", exist_ok=True)
    os.makedirs("outputs/gradcampp", exist_ok=True)
    os.makedirs("outputs/integrated_gradients", exist_ok=True)
    os.makedirs("outputs/saliency", exist_ok=True)


def load_trained_model(model_path: str, device: torch.device, num_classes: int = 15) -> nn.Module:
    """Cargar el modelo entrenado desde results/best_model.pth."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo en {model_path}. Ejecuta primero: python train.py"
        )

    print(f"Cargando modelo desde {model_path}...")
    model = create_model(num_classes=num_classes)

    checkpoint = torch.load(model_path, map_location=device)
    # train.py guarda {'model_state_dict': ...}
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    print("✅ Modelo cargado correctamente.\n")
    return model


def get_test_loader(batch_size: int = 1, num_workers: int = 0, seed: int = 42):
    """Cargar datasets MedMNIST y devolver solo el test_loader."""
    print("Cargando datos de test...")
    datasets = load_datasets("./data", target_size=224)

    train_loader, val_loader, test_loader = create_data_loaders_fixed(
        datasets=datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    return test_loader


# ============================================================
# Clase principal de explicabilidad
# ============================================================

class XAIExplainer:
    def __init__(self, model: nn.Module, device: torch.device, num_classes: int = 15):
        self.model = model
        self.device = device
        self.num_classes = num_classes

        self.model.eval()
        ensure_dirs()

        # Inicializar métodos de Captum si están disponibles
        if CAPTUM_AVAILABLE:
            self.ig = IntegratedGradients(self.model)
            self.saliency = Saliency(self.model)
        else:
            self.ig = None
            self.saliency = None

    # -------------------------
    # Utilidades internas
    # -------------------------

    def _denormalize(self, img_tensor: torch.Tensor,
                     mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)) -> np.ndarray:
        """
        Desnormaliza un tensor imagen 3xHxW a numpy HxWx3 en [0,1] (para visualización).
        """
        img = img_tensor.detach().cpu().clone()
        if img.dim() == 4:
            img = img[0]

        for c, (m, s) in enumerate(zip(mean, std)):
            img[c].mul_(s).add_(m)

        img = torch.clamp(img, 0.0, 1.0)
        img_np = img.permute(1, 2, 0).numpy()  # HWC
        return img_np

    def _get_target_layer(self):
        """
        Devuelve la última capa convolucional para Grad-CAM.
        Asumimos un modelo tipo torchvision.models.resnet18.
        """
        if hasattr(self.model, "layer4"):
            # Último bloque de layer4
            layer4 = self.model.layer4
            # Puede ser Sequential -> coger el último
            if isinstance(layer4, nn.Sequential):
                return [layer4[-1]]
            return [layer4]
        # Fallback: intentar usar la penúltima capa
        for name, module in reversed(self.model._modules.items()):
            if isinstance(module, nn.Sequential):
                return [module]
        raise RuntimeError("No se pudo encontrar una capa convolucional para Grad-CAM.")

    class _ClassTarget:
        """Target para Grad-CAM compatible con salidas 1D o 2D."""
        def __init__(self, class_idx: int):
            self.class_idx = class_idx

        def __call__(self, model_output):
            # model_output puede ser shape [C] o [B, C]
            if model_output.dim() == 1:
                return model_output[self.class_idx]
            else:
                return model_output[:, self.class_idx]

    # -------------------------
    # Métodos de explicabilidad
    # -------------------------

    def generate_gradcam(self, input_tensor: torch.Tensor,
                         target_class: int,
                         save_path: str | None = None):
        """Generar Grad-CAM y devolver (overlay, heatmap)."""
        if not GRAD_CAM_AVAILABLE:
            print("⚠️ Grad-CAM no disponible (faltan dependencias).")
            return None

        try:
            target_layers = self._get_target_layer()

            cam = GradCAM(
                model=self.model,
                target_layers=target_layers,
                reshape_transform=None
            )

            targets = [self._ClassTarget(target_class)]
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets,
            )  # shape: (N, H, W)

            heatmap = grayscale_cam[0]  # HxW

            rgb_img = self._denormalize(input_tensor)
            if rgb_img.ndim == 2:
                rgb_img = np.stack([rgb_img] * 3, axis=-1)

            overlay = show_cam_on_image(rgb_img.astype(np.float32), heatmap, use_rgb=True)

            if save_path is not None:
                plt.imsave(save_path, overlay)

            return overlay, heatmap

        except Exception as e:
            print(f"⚠️ Error en Grad-CAM: {e}")
            return None

    def generate_gradcampp(self, input_tensor: torch.Tensor,
                           target_class: int,
                           save_path: str | None = None):
        """Generar Grad-CAM++ y devolver (overlay, heatmap)."""
        if not GRAD_CAM_AVAILABLE:
            return None

        try:
            target_layers = self._get_target_layer()

            cam = GradCAMPlusPlus(
                model=self.model,
                target_layers=target_layers,
                reshape_transform=None
            )

            targets = [self._ClassTarget(target_class)]
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets,
            )

            heatmap = grayscale_cam[0]

            rgb_img = self._denormalize(input_tensor)
            if rgb_img.ndim == 2:
                rgb_img = np.stack([rgb_img] * 3, axis=-1)

            overlay = show_cam_on_image(rgb_img.astype(np.float32), heatmap, use_rgb=True)

            if save_path is not None:
                plt.imsave(save_path, overlay)

            return overlay, heatmap

        except Exception as e:
            print(f"⚠️ Error en Grad-CAM++: {e}")
            return None

    def generate_integrated_gradients(self, input_tensor: torch.Tensor,
                                      target_class: int,
                                      save_path: str | None = None):
        """Generar Integrated Gradients (mapa + superposición)."""
        if self.ig is None:
            return None

        try:
            x = input_tensor.clone().detach().to(self.device)
            x.requires_grad_(True)

            baseline = torch.zeros_like(x)

            attributions = self.ig.attribute(
                x,
                baseline,
                target=target_class,
                n_steps=50,
            )  # shape: (1, C, H, W)

            attr = attributions[0].detach().cpu().numpy()   # C,H,W
            attr = np.abs(attr)
            attr = attr / (attr.max() + 1e-8)

            # Para visualización, colapsamos canales: media sobre C
            if attr.ndim == 3:
                attr_vis = attr.mean(axis=0)  # H,W
            else:
                attr_vis = attr

            # Normalizar a [0,1]
            attr_vis = attr_vis / (attr_vis.max() + 1e-8)

            rgb_img = self._denormalize(input_tensor)
            if rgb_img.ndim == 2:
                rgb_img = np.stack([rgb_img] * 3, axis=-1)

            # Crear overlay tipo heatmap
            cmap = plt.cm.jet(attr_vis)[..., :3]  # H,W,3
            overlay = 0.6 * rgb_img + 0.4 * cmap
            overlay = np.clip(overlay, 0, 1)

            if save_path is not None:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(rgb_img)
                axes[0].set_title("Imagen original")
                axes[0].axis("off")

                axes[1].imshow(attr_vis, cmap="jet")
                axes[1].set_title("IG (mapa)")
                axes[1].axis("off")

                axes[2].imshow(overlay)
                axes[2].set_title("IG + overlay")
                axes[2].axis("off")

                plt.tight_layout()
                fig.savefig(save_path, dpi=150)
                plt.close(fig)

            return attr_vis, attributions

        except Exception as e:
            print(f"⚠️ Error en Integrated Gradients: {e}")
            return None

    def generate_saliency(self, input_tensor: torch.Tensor,
                          target_class: int,
                          save_path: str | None = None):
        """Generar Saliency Map (Vanilla)."""
        if self.saliency is None:
            return None

        try:
            x = input_tensor.clone().detach().to(self.device)
            x.requires_grad_(True)

            self.model.zero_grad()
            attributions = self.saliency.attribute(x, target=target_class)

            attr = attributions[0].detach().cpu().numpy()  # C,H,W
            attr = np.abs(attr)
            attr = attr / (attr.max() + 1e-8)

            # Colapsar canales
            if attr.ndim == 3:
                attr_vis = attr.mean(axis=0)  # H,W
            else:
                attr_vis = attr

            attr_vis = attr_vis / (attr_vis.max() + 1e-8)

            rgb_img = self._denormalize(input_tensor)
            if rgb_img.ndim == 2:
                rgb_img = np.stack([rgb_img] * 3, axis=-1)

            cmap = plt.cm.hot(attr_vis)[..., :3]
            overlay = 0.6 * rgb_img + 0.4 * cmap
            overlay = np.clip(overlay, 0, 1)

            if save_path is not None:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(rgb_img)
                axes[0].set_title("Imagen original")
                axes[0].axis("off")

                axes[1].imshow(attr_vis, cmap="hot")
                axes[1].set_title("Saliency (mapa)")
                axes[1].axis("off")

                axes[2].imshow(overlay)
                axes[2].set_title("Saliency + overlay")
                axes[2].axis("off")

                plt.tight_layout()
                fig.savefig(save_path, dpi=150)
                plt.close(fig)

            return attr_vis, attributions

        except Exception as e:
            print(f"⚠️ Error en Saliency: {e}")
            return None

    # -------------------------
    # Orquestador para una imagen
    # -------------------------

    def explain_sample(self, input_tensor: torch.Tensor,
                       true_class: int,
                       pred_class: int,
                       idx: int,
                       save_all: bool = True):
        """
        Genera todas las explicaciones para una sola imagen.
        Devuelve un diccionario con rutas a los archivos generados.
        """
        results = {}

        # Grad-CAM
        if GRAD_CAM_AVAILABLE:
            gradcam_path = f"outputs/gradcam/img_{idx:04d}_pred_{pred_class}.png" if save_all else None
            gc = self.generate_gradcam(input_tensor, pred_class, gradcam_path)
            if gc is not None:
                results["gradcam"] = {
                    "path": gradcam_path,
                    "status": "ok",
                }

            gradcampp_path = f"outputs/gradcampp/img_{idx:04d}_pred_{pred_class}.png" if save_all else None
            gcpp = self.generate_gradcampp(input_tensor, pred_class, gradcampp_path)
            if gcpp is not None:
                results["gradcampp"] = {
                    "path": gradcampp_path,
                    "status": "ok",
                }

        # Integrated Gradients
        if self.ig is not None:
            ig_path = f"outputs/integrated_gradients/img_{idx:04d}_pred_{pred_class}.png" if save_all else None
            ig = self.generate_integrated_gradients(input_tensor, pred_class, ig_path)
            if ig is not None:
                results["integrated_gradients"] = {
                    "path": ig_path,
                    "status": "ok",
                }

        # Saliency
        if self.saliency is not None:
            sal_path = f"outputs/saliency/img_{idx:04d}_pred_{pred_class}.png" if save_all else None
            sal = self.generate_saliency(input_tensor, pred_class, sal_path)
            if sal is not None:
                results["saliency"] = {
                    "path": sal_path,
                    "status": "ok",
                }

        return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("EXPLICABILIDAD (XAI) - ResNet-18 MedMNIST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    model_path = "results/best_model.pth"
    num_classes = 15
    num_samples = 20  # nº de imágenes de test a explicar

    # 1) Modelo
    model = load_trained_model(model_path, device, num_classes=num_classes)

    # 2) Datos (test)
    test_loader = get_test_loader(batch_size=1, num_workers=0, seed=42)

    # 3) Explainer
    explainer = XAIExplainer(model, device, num_classes=num_classes)

    # 4) Generación de explicaciones
    print(f"\nGenerando explicaciones para {num_samples} muestras de test...")
    from tqdm import tqdm

    all_meta = []

    for idx, (images, targets) in enumerate(tqdm(test_loader, desc="Generando explicaciones")):
        if idx >= num_samples:
            break

        images = images.to(device)
        true_class = int(targets.item())

        with torch.no_grad():
            logits = model(images)
            pred_class = int(logits.argmax(dim=1).item())

        sample_results = explainer.explain_sample(
            input_tensor=images,
            true_class=true_class,
            pred_class=pred_class,
            idx=idx,
            save_all=True,
        )

        all_meta.append(
            {
                "sample_idx": idx,
                "true_class": true_class,
                "pred_class": pred_class,
                "results": sample_results,
            }
        )

    # 5) Guardar metadatos
    print("\nGuardando metadatos de explicabilidad...")
    os.makedirs("outputs", exist_ok=True)
    meta_path = "outputs/explanations_results.json"
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"Metadatos guardados en: {meta_path}")

    print(
        "\nℹ️ Para la evaluación cuantitativa (Faithfulness, Robustness, etc.) "
        "se recomienda usar Quantus desde un notebook, cargando el modelo y "
        "los mapas generados en 'outputs/'."
    )

    print("\n" + "=" * 60)
    print("✅ EXPLICABILIDAD COMPLETADA")
    print("=" * 60)
    print("Mapas guardados en:")
    print("  - Grad-CAM:             outputs/gradcam/")
    print("  - Grad-CAM++:           outputs/gradcampp/")
    print("  - Integrated Gradients: outputs/integrated_gradients/")
    print("  - Saliency Maps:        outputs/saliency/")
    print("Metadatos:")
    print("  - outputs/explanations_results.json")


if __name__ == "__main__":
    main()
