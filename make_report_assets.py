# make_report_assets.py
"""
Genera activos para la memoria (figuras y tablas) a partir de 'results/'.
- Carga training_results.json (historial, métricas globales)
- Carga preds_test.npz (y_true, y_pred)
- Produce:
  - results/fig_training_curves.png
  - results/fig_confusion_matrix.png
  - results/table_classification_report.csv
  - results/table_classification_report.tex
  - results/table_summary.json
"""
# Importaciones.
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
# Directorio de resultados.
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
# Archivos de resultados.
JR = os.path.join(RESULTS_DIR, "training_results.json")
NPZ = os.path.join(RESULTS_DIR, "preds_test.npz")
# Verifica que exista training_results.json, lo carga y extrae history, results y config.
if not os.path.exists(JR):
    raise FileNotFoundError(f"No se encuentra {JR}. Ejecuta el entrenamiento para generarlo.")
# Carga el archivo training_results.json.
with open(JR, "r") as f:
    payload = json.load(f)
# Extrae history, results y config.
history = payload.get("history", {})
results = payload.get("results", {})
config  = payload.get("config", {})
# Verifica que exista preds_test.npz, lo carga y extrae y_true y y_pred.
if not os.path.exists(NPZ):
    raise FileNotFoundError(f"No se encuentra {NPZ}. Vuelve a ejecutar la evaluación para generarlo.")
# Carga el archivo preds_test.npz.
npz = np.load(NPZ)
y_true = npz["y_true"]
y_pred = npz["y_pred"]
# Extrae train_loss, val_loss, train_acc y val_acc.Genera curvas de entrenamiento.
# ----- Curvas de entrenamiento -----
train_loss = history.get("train_loss", [])
val_loss   = history.get("val_loss", [])
train_acc  = history.get("train_acc", [])
val_acc    = history.get("val_acc", [])

# Validación cada 3 épocas → relleno con NaN para dibujar
val_loss = [v if v is not None else np.nan for v in val_loss]
val_acc  = [v if v is not None else np.nan for v in val_acc]
# Crea las figuras de las curvas de entrenamiento.
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.xlabel("Época"); plt.ylabel("Loss"); plt.title("Pérdida por época"); plt.grid(True); plt.legend()
# Crea la figura de la precisión por época.
plt.subplot(1,2,2)
plt.plot(train_acc, label="Train Acc")
plt.plot(val_acc, label="Val Acc")
plt.xlabel("Época"); plt.ylabel("Accuracy (%)"); plt.title("Precisión por época"); plt.grid(True); plt.legend()
# Guarda la figura de las curvas de entrenamiento.
plt.tight_layout()
out_curves = os.path.join(RESULTS_DIR, "fig_training_curves.png")
plt.savefig(out_curves, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Guardado: {out_curves}")
# Calcula la matriz de confusión, la visualiza con un heatmap y guarda la imagen.

# ----- Matriz de confusión -----
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión (Test)")
plt.xlabel("Predicción"); plt.ylabel("Verdadero")
out_cm = os.path.join(RESULTS_DIR, "fig_confusion_matrix.png")
plt.tight_layout()
plt.savefig(out_cm, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Guardado: {out_cm}")
# Genera el reporte de clasificación como diccionario, lo convierte a DataFrame, lo transpone y guarda CSV.
# ----- Classification report (CSV y LaTeX) -----
report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report_dict).T
out_csv = os.path.join(RESULTS_DIR, "table_classification_report.csv")
df_report.to_csv(out_csv)
print(f"[OK] Guardado: {out_csv}")
# Genera el reporte de clasificación como LaTeX, lo guarda en un archivo.
out_tex = os.path.join(RESULTS_DIR, "table_classification_report.tex")
with open(out_tex, "w", encoding="utf-8") as f:
    f.write(df_report.to_latex(float_format="%.3f", index=True))
print(f"[OK] Guardado: {out_tex}")
# Calcula el mejor accuracy de validación, lo guarda en un diccionario.
# ----- Resumen global -----
best_val_acc = None
if len(val_acc):
    numeric_vals = [x for x in val_acc if not (isinstance(x, float) and np.isnan(x))]
    if numeric_vals:
        best_val_acc = float(np.max(numeric_vals))
# Crea el resumen global, lo guarda en un diccionario.
summary = {
    "config": config,
    "test_accuracy": results.get("test_acc", None),
    "test_loss": results.get("test_loss", None),
    "best_val_accuracy": best_val_acc,
    "epochs_run": len(train_loss)
}
out_json = os.path.join(RESULTS_DIR, "table_summary.json")
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[OK] Guardado: {out_json}")
# Imprime un resumen de los archivos generados.
print("\nHecho. Archivos en 'results/':")
print(" - fig_training_curves.png")
print(" - fig_confusion_matrix.png")
print(" - table_classification_report.csv")
print(" - table_classification_report.tex")
print(" - table_summary.json")
"""
Resumen
El script make_report_assets.py genera activos para la memoria del TFM:
1. Carga de datos:
- training_results.json (historial y métricas)
- preds_test.npz (etiquetas y predicciones)
2. Genera 5 archivos:
- fig_training_curves.png: curvas de pérdida y precisión
- fig_confusion_matrix.png: matriz de confusión
- table_classification_report.csv: reporte en CSV
- table_classification_report.tex: reporte en LaTeX
- table_summary.json: resumen con configuración y métricas clave

Útil para automatizar la generación de figuras y tablas para la memoria a partir de los resultados del entrenamiento.
"""