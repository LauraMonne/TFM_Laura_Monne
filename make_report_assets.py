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

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

JR = os.path.join(RESULTS_DIR, "training_results.json")
NPZ = os.path.join(RESULTS_DIR, "preds_test.npz")

if not os.path.exists(JR):
    raise FileNotFoundError(f"No se encuentra {JR}. Ejecuta el entrenamiento para generarlo.")

with open(JR, "r") as f:
    payload = json.load(f)

history = payload.get("history", {})
results = payload.get("results", {})
config  = payload.get("config", {})

if not os.path.exists(NPZ):
    raise FileNotFoundError(f"No se encuentra {NPZ}. Vuelve a ejecutar la evaluación para generarlo.")

npz = np.load(NPZ)
y_true = npz["y_true"]
y_pred = npz["y_pred"]

# ----- Curvas de entrenamiento -----
train_loss = history.get("train_loss", [])
val_loss   = history.get("val_loss", [])
train_acc  = history.get("train_acc", [])
val_acc    = history.get("val_acc", [])

# Validación cada 3 épocas → relleno con NaN para dibujar
val_loss = [v if v is not None else np.nan for v in val_loss]
val_acc  = [v if v is not None else np.nan for v in val_acc]

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.xlabel("Época"); plt.ylabel("Loss"); plt.title("Pérdida por época"); plt.grid(True); plt.legend()

plt.subplot(1,2,2)
plt.plot(train_acc, label="Train Acc")
plt.plot(val_acc, label="Val Acc")
plt.xlabel("Época"); plt.ylabel("Accuracy (%)"); plt.title("Precisión por época"); plt.grid(True); plt.legend()

plt.tight_layout()
out_curves = os.path.join(RESULTS_DIR, "fig_training_curves.png")
plt.savefig(out_curves, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Guardado: {out_curves}")

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

# ----- Classification report (CSV y LaTeX) -----
report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report_dict).T
out_csv = os.path.join(RESULTS_DIR, "table_classification_report.csv")
df_report.to_csv(out_csv)
print(f"[OK] Guardado: {out_csv}")

out_tex = os.path.join(RESULTS_DIR, "table_classification_report.tex")
with open(out_tex, "w", encoding="utf-8") as f:
    f.write(df_report.to_latex(float_format="%.3f", index=True))
print(f"[OK] Guardado: {out_tex}")

# ----- Resumen global -----
best_val_acc = None
if len(val_acc):
    numeric_vals = [x for x in val_acc if not (isinstance(x, float) and np.isnan(x))]
    if numeric_vals:
        best_val_acc = float(np.max(numeric_vals))

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

print("\nHecho. Archivos en 'results/':")
print(" - fig_training_curves.png")
print(" - fig_confusion_matrix.png")
print(" - table_classification_report.csv")
print(" - table_classification_report.tex")
print(" - table_summary.json")
