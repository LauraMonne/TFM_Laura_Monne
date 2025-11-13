"""
Script de Explicabilidad (XAI) para ResNet-18
Implementa métodos de explicabilidad según la memoria del TFM:
- Grad-CAM y Grad-CAM++
- Integrated Gradients (IG)
- Saliency Maps (Vanilla Saliency)

Evaluación cuantitativa con Quantus:
- Fidelidad (Faithfulness Correlation)
- Robustez (Average Sensitivity)
- Complejidad (Entropy)
- Aleatorización (Randomization Test)
- Localización (Attribution Localization Ratio)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Librerías de explicabilidad
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    print("⚠️ pytorch-grad-cam no disponible. Instala con: pip install grad-cam")

try:
    from captum.attr import IntegratedGradients, Saliency, visualization as viz
    from captum.attr import visualization as viz
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

# Módulos propios
from prepare_data import load_datasets, create_combined_dataset
from resnet18 import create_model, set_seed
from data_utils import create_data_loaders_fixed


class ResNet18Wrapper(nn.Module):
    """Wrapper para ResNet18Adaptive que permite usar Grad-CAM"""
    def __init__(self, adaptive_model):
        super().__init__()
        self.adaptive_model = adaptive_model
        # Determinar qué modelo usar basado en la primera imagen
        # Por defecto usamos rgb_model (3 canales)
        self.base_model = adaptive_model.rgb_model
        
    def forward(self, x):
        return self.adaptive_model(x)


class XAIExplainer:
    """Clase principal para explicabilidad"""
    
    def __init__(self, model, device, num_classes=15):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.model.eval()
        
        # Directorios de salida
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("outputs/gradcam", exist_ok=True)
        os.makedirs("outputs/gradcampp", exist_ok=True)
        os.makedirs("outputs/integrated_gradients", exist_ok=True)
        os.makedirs("outputs/saliency", exist_ok=True)
        os.makedirs("outputs/overlay", exist_ok=True)
        
        # Inicializar métodos
        self._init_gradcam()
        self._init_captum_methods()
        
    def _init_gradcam(self):
        """Inicializar Grad-CAM - No se inicializa estáticamente, se crea dinámicamente"""
        # Grad-CAM se inicializa dinámicamente en generate_gradcam() y generate_gradcampp()
        # porque el modelo puede variar según los canales de entrada
        pass
    
    def _init_captum_methods(self):
        """Inicializar métodos de Captum"""
        if not CAPTUM_AVAILABLE:
            self.ig = None
            self.saliency = None
            return
            
        try:
            self.ig = IntegratedGradients(self.model)
            self.saliency = Saliency(self.model)
        except Exception as e:
            print(f"⚠️ Error inicializando Captum: {e}")
            self.ig = None
            self.saliency = None
    
    def _denormalize_image(self, img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """Desnormalizar imagen para visualización"""
        if isinstance(img_tensor, torch.Tensor):
            img = img_tensor.clone()
        else:
            img = torch.tensor(img_tensor)
            
        if img.dim() == 4:
            img = img[0]
            
        # Desnormalizar
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
            
        # Clamp a [0, 1]
        img = torch.clamp(img, 0, 1)
        
        # Convertir a numpy
        img_np = img.permute(1, 2, 0).cpu().numpy()
        
        return img_np
    
    def _get_model_output(self, input_tensor, target_class=None):
        """Obtener salida del modelo para una clase específica"""
        with torch.no_grad():
            output = self.model(input_tensor)
            if target_class is not None:
                return output[0, target_class].item()
            return output
    
    def _get_gradcam_model(self, input_tensor):
        """Obtener el modelo y capa objetivo para Grad-CAM"""
        if hasattr(self.model, 'rgb_model') and input_tensor.shape[1] == 3:
            return self.model.rgb_model, [self.model.rgb_model.layer4]
        elif hasattr(self.model, 'gray_model') and input_tensor.shape[1] == 1:
            return self.model.gray_model, [self.model.gray_model.layer4]
        else:
            return self.model, [self.model.layer4]
    
    def generate_gradcam(self, input_tensor, target_class, save_path=None):
        """Generar mapa de activación Grad-CAM"""
        if not GRAD_CAM_AVAILABLE:
            return None
            
        try:
            # Obtener modelo y capa objetivo
            grad_model, target_layers = self._get_gradcam_model(input_tensor)
            
            # Crear Grad-CAM
            gradcam = GradCAM(
                model=grad_model,
                target_layers=target_layers,
                use_cuda=self.device.type == 'cuda'
            )
            
            # Clase callback para la clase objetivo
            class ClassTarget:
                def __init__(self, target_class):
                    self.target_class = target_class
                
                def __call__(self, model_output):
                    return model_output[:, self.target_class]
            
            # Generar mapa de activación
            grayscale_cam = gradcam(
                input_tensor=input_tensor,
                targets=[ClassTarget(target_class)],
                eigen_smooth=False,
                aug_smooth=False
            )
            
            # Obtener imagen original (desnormalizada)
            rgb_img = self._denormalize_image(input_tensor)
            
            # Si es escala de grises, convertir a RGB
            if len(rgb_img.shape) == 2 or rgb_img.shape[2] == 1:
                if len(rgb_img.shape) == 2:
                    rgb_img = np.stack([rgb_img] * 3, axis=-1)
                else:
                    rgb_img = np.repeat(rgb_img, 3, axis=2)
            
            # Superponer mapa de activación
            cam_image = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
            
            if save_path:
                plt.imsave(save_path, cam_image)
            
            return cam_image, grayscale_cam[0]
        except Exception as e:
            print(f"⚠️ Error en Grad-CAM: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_gradcampp(self, input_tensor, target_class, save_path=None):
        """Generar mapa de activación Grad-CAM++"""
        if not GRAD_CAM_AVAILABLE:
            return None
            
        try:
            # Obtener modelo y capa objetivo
            grad_model, target_layers = self._get_gradcam_model(input_tensor)
            
            # Crear Grad-CAM++
            gradcampp = GradCAMPlusPlus(
                model=grad_model,
                target_layers=target_layers,
                use_cuda=self.device.type == 'cuda'
            )
            
            # Clase callback para la clase objetivo
            class ClassTarget:
                def __init__(self, target_class):
                    self.target_class = target_class
                
                def __call__(self, model_output):
                    return model_output[:, self.target_class]
            
            # Generar mapa de activación
            grayscale_cam = gradcampp(
                input_tensor=input_tensor,
                targets=[ClassTarget(target_class)],
                eigen_smooth=False,
                aug_smooth=False
            )
            
            # Obtener imagen original
            rgb_img = self._denormalize_image(input_tensor)
            if len(rgb_img.shape) == 2 or rgb_img.shape[2] == 1:
                if len(rgb_img.shape) == 2:
                    rgb_img = np.stack([rgb_img] * 3, axis=-1)
                else:
                    rgb_img = np.repeat(rgb_img, 3, axis=2)
            
            # Superponer mapa de activación
            cam_image = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
            
            if save_path:
                plt.imsave(save_path, cam_image)
            
            return cam_image, grayscale_cam[0]
        except Exception as e:
            print(f"⚠️ Error en Grad-CAM++: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_integrated_gradients(self, input_tensor, target_class, save_path=None):
        """Generar atribuciones con Integrated Gradients"""
        if self.ig is None:
            return None
            
        try:
            # Baseline (imagen negra)
            baseline = torch.zeros_like(input_tensor)
            
            # Calcular atribuciones
            attributions = self.ig.attribute(
                input_tensor,
                baseline,
                target=target_class,
                n_steps=50
            )
            
            # Convertir a numpy para visualización
            attributions_np = attributions[0].cpu().detach().permute(1, 2, 0).numpy()
            attributions_np = np.abs(attributions_np)
            attributions_np = attributions_np / (attributions_np.max() + 1e-8)  # Normalizar
            
            # Obtener imagen original
            rgb_img = self._denormalize_image(input_tensor)
            
            # Visualizar
            if save_path:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(rgb_img)
                axes[0].set_title('Imagen Original')
                axes[0].axis('off')
                
                axes[1].imshow(attributions_np)
                axes[1].set_title('Integrated Gradients')
                axes[1].axis('off')
                
                # Superposición
                overlay = rgb_img.copy()
                heatmap = plt.cm.jet(attributions_np)[:, :, :3]
                overlay = 0.6 * overlay + 0.4 * heatmap
                axes[2].imshow(overlay)
                axes[2].set_title('Superposición')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            return attributions_np, attributions
        except Exception as e:
            print(f"⚠️ Error en Integrated Gradients: {e}")
            return None
    
    def generate_saliency_map(self, input_tensor, target_class, save_path=None):
        """Generar mapa de saliencia (Vanilla Saliency)"""
        if self.saliency is None:
            return None
            
        try:
            # Calcular saliencia
            attributions = self.saliency.attribute(
                input_tensor,
                target=target_class
            )
            
            # Convertir a numpy
            saliency_np = attributions[0].cpu().detach().permute(1, 2, 0).numpy()
            saliency_np = np.abs(saliency_np)
            saliency_np = saliency_np / (saliency_np.max() + 1e-8)
            
            # Obtener imagen original
            rgb_img = self._denormalize_image(input_tensor)
            
            # Visualizar
            if save_path:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(rgb_img)
                axes[0].set_title('Imagen Original')
                axes[0].axis('off')
                
                axes[1].imshow(saliency_np, cmap='hot')
                axes[1].set_title('Saliency Map')
                axes[1].axis('off')
                
                # Superposición
                overlay = rgb_img.copy()
                heatmap = plt.cm.hot(saliency_np)[:, :, :3]
                overlay = 0.6 * overlay + 0.4 * heatmap
                axes[2].imshow(overlay)
                axes[2].set_title('Superposición')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            return saliency_np, attributions
        except Exception as e:
            print(f"⚠️ Error en Saliency Map: {e}")
            return None
    
    def generate_all_explanations(self, input_tensor, target_class, image_idx, save_all=True):
        """Generar todas las explicaciones para una imagen"""
        results = {}
        
        # Grad-CAM
        if GRAD_CAM_AVAILABLE:
            gradcam_path = f"outputs/gradcam/img_{image_idx}_class_{target_class}.png" if save_all else None
            gradcam_result = self.generate_gradcam(input_tensor, target_class, gradcam_path)
            if gradcam_result:
                # Guardar solo metadatos, no los arrays completos (para evitar archivos JSON muy grandes)
                results['gradcam'] = {
                    'path': gradcam_path,
                    'status': 'success'
                }
        
        # Grad-CAM++
        if GRAD_CAM_AVAILABLE:
            gradcampp_path = f"outputs/gradcampp/img_{image_idx}_class_{target_class}.png" if save_all else None
            gradcampp_result = self.generate_gradcampp(input_tensor, target_class, gradcampp_path)
            if gradcampp_result:
                results['gradcampp'] = {
                    'path': gradcampp_path,
                    'status': 'success'
                }
        
        # Integrated Gradients
        if CAPTUM_AVAILABLE and self.ig:
            ig_path = f"outputs/integrated_gradients/img_{image_idx}_class_{target_class}.png" if save_all else None
            ig_result = self.generate_integrated_gradients(input_tensor, target_class, ig_path)
            if ig_result:
                results['integrated_gradients'] = {
                    'path': ig_path,
                    'status': 'success'
                }
        
        # Saliency Map
        if CAPTUM_AVAILABLE and self.saliency:
            saliency_path = f"outputs/saliency/img_{image_idx}_class_{target_class}.png" if save_all else None
            saliency_result = self.generate_saliency_map(input_tensor, target_class, saliency_path)
            if saliency_result:
                results['saliency'] = {
                    'path': saliency_path,
                    'status': 'success'
                }
        
        return results


def load_trained_model(model_path, device, num_classes=15):
    """Cargar modelo entrenado"""
    print(f"Cargando modelo desde {model_path}...")
    
    # Crear modelo
    model = create_model(num_classes=num_classes)
    
    # Cargar pesos
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ Modelo cargado correctamente")
    return model


def evaluate_with_quantus(model, explainer, data_loader, device, num_samples=50):
    """Evaluar explicaciones con Quantus según las 5 dimensiones de la memoria"""
    if not QUANTUS_AVAILABLE:
        print("⚠️ Quantus no disponible. Saltando evaluación cuantitativa.")
        return None
    
    print(f"\nEvaluando con Quantus ({num_samples} muestras)...")
    print("Métricas: Faithfulness, Robustness, Complexity, Randomization, Localization")
    
    # Preparar datos
    model.eval()
    x_batch = []
    y_batch = []
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if idx >= num_samples:
                break
            x_batch.append(data.to(device))
            y_batch.append(target.to(device))
    
    if len(x_batch) == 0:
        print("⚠️ No hay datos para evaluar")
        return None
    
    # Convertir a tensores
    x_batch = torch.cat(x_batch, dim=0)
    y_batch = torch.cat(y_batch, dim=0)
    
    # Función para obtener predicciones (compatible con Quantus)
    def predict_fn(x):
        """Función de predicción compatible con Quantus
        Quantus espera formato (batch, height, width, channels)
        pero el modelo espera (batch, channels, height, width)
        """
        model.eval()
        # Convertir numpy a tensor si es necesario
        if isinstance(x, np.ndarray):
            # Si está en formato HWC (Quantus), convertir a CHW (PyTorch)
            if len(x.shape) == 4 and x.shape[-1] in [1, 3]:
                # Formato HWC -> CHW
                x = np.transpose(x, (0, 3, 1, 2))
            x = torch.from_numpy(x).float().to(device)
        with torch.no_grad():
            output = model(x)
            return output.cpu().detach().numpy()
    
    # Función para obtener atribuciones (para cada método)
    def get_attributions(method_name):
        """Obtener atribuciones para un método específico"""
        attributions = []
        for i in range(len(x_batch)):
            x = x_batch[i:i+1]
            y = y_batch[i].item()
            
            # Obtener predicción
            with torch.no_grad():
                output = model(x)
                pred_class = output.argmax(dim=1).item()
            
            try:
                if method_name == 'gradcam':
                    result = explainer.generate_gradcam(x, pred_class, save_path=None)
                    if result:
                        # Convertir heatmap a formato de atribución
                        attr = result[1]  # heatmap
                        # Redimensionar a tamaño de entrada
                        attr_resized = cv2.resize(attr, (x.shape[3], x.shape[2]))
                        # Expandir dimensiones para coincidir con entrada
                        attr_expanded = np.expand_dims(attr_resized, axis=0)
                        attr_expanded = np.repeat(attr_expanded, x.shape[1], axis=0)
                        attributions.append(torch.tensor(attr_expanded, dtype=torch.float32))
                elif method_name == 'integrated_gradients':
                    if explainer.ig:
                        baseline = torch.zeros_like(x)
                        attr = explainer.ig.attribute(x, baseline, target=pred_class, n_steps=50)
                        attributions.append(attr[0].cpu())
                elif method_name == 'saliency':
                    if explainer.saliency:
                        attr = explainer.saliency.attribute(x, target=pred_class)
                        attributions.append(attr[0].cpu())
            except Exception as e:
                print(f"⚠️ Error generando atribuciones para {method_name}, muestra {i}: {e}")
                attributions.append(torch.zeros_like(x[0].cpu()))
        
        if len(attributions) > 0:
            return torch.stack(attributions, dim=0)
        return None
    
    # Evaluar cada método
    methods = ['gradcam', 'integrated_gradients', 'saliency']
    results = {}
    
    for method in methods:
        print(f"\nEvaluando {method}...")
        
        try:
            # Obtener atribuciones
            a_batch = get_attributions(method)
            if a_batch is None:
                print(f"⚠️ No se pudieron obtener atribuciones para {method}")
                continue
            
            # Convertir a numpy para Quantus
            # Quantus espera formato (batch, height, width, channels)
            x_batch_np = x_batch.cpu().detach().numpy()
            # Convertir de CHW a HWC para Quantus
            if x_batch_np.shape[1] in [1, 3]:  # Si está en formato CHW
                x_batch_np = np.transpose(x_batch_np, (0, 2, 3, 1))
            
            y_batch_np = y_batch.cpu().detach().numpy()
            a_batch_np = a_batch.cpu().detach().numpy()
            # Convertir atribuciones de CHW a HWC también
            if len(a_batch_np.shape) == 4 and a_batch_np.shape[1] in [1, 3]:
                a_batch_np = np.transpose(a_batch_np, (0, 2, 3, 1))
            
            method_results = {}
            
            # 1. Faithfulness Correlation
            try:
                faithfulness = quantus.FaithfulnessCorrelation(
                    nr_samples=10,
                    perturb_baseline='black',
                    similarity_func=quantus.similarity_func.correlation_spearman
                )
                faithfulness_scores = faithfulness(
                    model=predict_fn,
                    x_batch=x_batch_np,
                    y_batch=y_batch_np,
                    a_batch=a_batch_np,
                    device=device
                )
                method_results['faithfulness'] = {
                    'mean': float(np.nanmean(faithfulness_scores)),
                    'std': float(np.nanstd(faithfulness_scores)),
                    'scores': faithfulness_scores.tolist()
                }
                print(f"  Faithfulness: {np.nanmean(faithfulness_scores):.4f} ± {np.nanstd(faithfulness_scores):.4f}")
            except Exception as e:
                print(f"  ⚠️ Error en Faithfulness: {e}")
                method_results['faithfulness'] = None
            
            # 2. Average Sensitivity (Robustness)
            try:
                sensitivity = quantus.AvgSensitivity(
                    nr_samples=10,
                    perturb_baseline='black',
                    lower_bound=0.2
                )
                sensitivity_scores = sensitivity(
                    model=predict_fn,
                    x_batch=x_batch_np,
                    y_batch=y_batch_np,
                    a_batch=a_batch_np,
                    device=device
                )
                method_results['robustness'] = {
                    'mean': float(np.nanmean(sensitivity_scores)),
                    'std': float(np.nanstd(sensitivity_scores)),
                    'scores': sensitivity_scores.tolist()
                }
                print(f"  Robustness (Avg Sensitivity): {np.nanmean(sensitivity_scores):.4f} ± {np.nanstd(sensitivity_scores):.4f}")
            except Exception as e:
                print(f"  ⚠️ Error en Robustness: {e}")
                method_results['robustness'] = None
            
            # 3. Entropy (Complexity)
            try:
                entropy = quantus.Entropy()
                entropy_scores = entropy(
                    model=predict_fn,
                    x_batch=x_batch_np,
                    y_batch=y_batch_np,
                    a_batch=a_batch_np,
                    device=device
                )
                method_results['complexity'] = {
                    'mean': float(np.nanmean(entropy_scores)),
                    'std': float(np.nanstd(entropy_scores)),
                    'scores': entropy_scores.tolist()
                }
                print(f"  Complexity (Entropy): {np.nanmean(entropy_scores):.4f} ± {np.nanstd(entropy_scores):.4f}")
            except Exception as e:
                print(f"  ⚠️ Error en Complexity: {e}")
                method_results['complexity'] = None
            
            # 4. Randomization Test
            try:
                randomization = quantus.ModelParameterRandomisation(
                    layer_order='top_down',
                    similarity_func=quantus.similarity_func.correlation_pearson
                )
                randomization_scores = randomization(
                    model=predict_fn,
                    x_batch=x_batch_np,
                    y_batch=y_batch_np,
                    a_batch=a_batch_np,
                    device=device
                )
                method_results['randomization'] = {
                    'mean': float(np.nanmean(randomization_scores)),
                    'std': float(np.nanstd(randomization_scores)),
                    'scores': randomization_scores.tolist()
                }
                print(f"  Randomization: {np.nanmean(randomization_scores):.4f} ± {np.nanstd(randomization_scores):.4f}")
            except Exception as e:
                print(f"  ⚠️ Error en Randomization: {e}")
                method_results['randomization'] = None
            
            # 5. Attribution Localization Ratio
            # Nota: Esta métrica requiere máscaras de regiones de interés
            # Por ahora, usamos una aproximación basada en la concentración de atribución
            try:
                # Usar RegionPerturbation como proxy para localización
                localization = quantus.RegionPerturbation(
                    patch_size=7,
                    regions_evaluation=100
                )
                localization_scores = localization(
                    model=predict_fn,
                    x_batch=x_batch_np,
                    y_batch=y_batch_np,
                    a_batch=a_batch_np,
                    device=device
                )
                method_results['localization'] = {
                    'mean': float(np.nanmean(localization_scores)),
                    'std': float(np.nanstd(localization_scores)),
                    'scores': localization_scores.tolist()
                }
                print(f"  Localization: {np.nanmean(localization_scores):.4f} ± {np.nanstd(localization_scores):.4f}")
            except Exception as e:
                print(f"  ⚠️ Error en Localization: {e}")
                method_results['localization'] = None
            
            results[method] = method_results
            
        except Exception as e:
            print(f"⚠️ Error evaluando {method}: {e}")
            import traceback
            traceback.print_exc()
            results[method] = None
    
    return results


def main():
    """Función principal"""
    print("=" * 60)
    print("EXPLICABILIDAD (XAI) - ResNet-18 MedMNIST")
    print("=" * 60)
    
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    model_path = "results/best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}. Ejecuta train.py primero.")
    
    num_classes = 15
    num_samples = 20  # Número de imágenes a explicar
    
    # Cargar modelo
    model = load_trained_model(model_path, device, num_classes)
    
    # Cargar datos de test
    print("\nCargando datos de test...")
    datasets = load_datasets('./data', target_size=224)
    test_loader = create_data_loaders_fixed(
        datasets=datasets,
        batch_size=1,
        num_workers=0,
        seed=42
    )[2]  # test_loader
    
    # Crear explainer
    explainer = XAIExplainer(model, device, num_classes)
    
    # Generar explicaciones para muestras
    print(f"\nGenerando explicaciones para {num_samples} muestras...")
    all_results = []
    
    for idx, (data, target) in enumerate(tqdm(test_loader, desc="Generando explicaciones")):
        if idx >= num_samples:
            break
        
        data = data.to(device)
        target_class = target.item()
        
        # Obtener predicción del modelo
        with torch.no_grad():
            output = model(data)
            pred_class = output.argmax(dim=1).item()
        
        # Generar todas las explicaciones
        results = explainer.generate_all_explanations(
            data,
            pred_class,
            idx,
            save_all=True
        )
        
        all_results.append({
            'image_idx': idx,
            'true_class': target_class,
            'pred_class': pred_class,
            'results': results
        })
    
    # Guardar resultados
    print("\nGuardando resultados...")
    with open('outputs/explanations_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Evaluación con Quantus (si está disponible)
    if QUANTUS_AVAILABLE:
        quantus_results = evaluate_with_quantus(explainer.model, explainer, test_loader, device, num_samples=min(num_samples, 20))
        if quantus_results:
            # Guardar resultados de Quantus
            with open('outputs/quantus_evaluation.json', 'w') as f:
                # Convertir resultados a formato JSON serializable
                json_results = {}
                for method, metrics in quantus_results.items():
                    if metrics:
                        json_results[method] = {}
                        for metric_name, metric_data in metrics.items():
                            if metric_data:
                                json_results[method][metric_name] = {
                                    'mean': metric_data.get('mean'),
                                    'std': metric_data.get('std')
                                }
                json.dump(json_results, f, indent=2)
            print("\n✅ Evaluación con Quantus completada")
        else:
            print("\n⚠️ No se pudieron obtener resultados de Quantus")
    
    print("\n" + "=" * 60)
    print("✅ EXPLICABILIDAD COMPLETADA")
    print("=" * 60)
    print(f"Resultados guardados en 'outputs/'")
    print(f"  - Grad-CAM: outputs/gradcam/")
    print(f"  - Grad-CAM++: outputs/gradcampp/")
    print(f"  - Integrated Gradients: outputs/integrated_gradients/")
    print(f"  - Saliency Maps: outputs/saliency/")
    print(f"  - Metadatos: outputs/explanations_results.json")


if __name__ == "__main__":
    main()

