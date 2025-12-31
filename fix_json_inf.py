"""
Script para limpiar archivos JSON que contienen valores Infinity o NaN
(que no son válidos en JSON estándar).
"""

import json
import sys
import os

def clean_json_value(value):
    """Convierte inf/nan a None (null en JSON)"""
    if isinstance(value, float):
        if value == float('inf') or value == float('-inf'):
            return None
        if value != value:  # NaN check
            return None
    elif isinstance(value, dict):
        return {k: clean_json_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [clean_json_value(item) for item in value]
    return value

def fix_json_file(filepath):
    """Lee, limpia y guarda un archivo JSON"""
    print(f"Leyendo {filepath}...")
    
    # Leer como texto primero para detectar "Infinity"
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'Infinity' in content or 'NaN' in content or 'nan' in content:
        print(f"  ADVERTENCIA: Detectado valores Infinity/NaN en {filepath}")
        
        # Reemplazar Infinity/NaN en el texto antes de parsear
        content = content.replace('Infinity', 'null')
        content = content.replace('NaN', 'null')
        content = content.replace('nan', 'null')
        
        # Parsear JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"  ERROR: Error parseando JSON despues de reemplazo: {e}")
            return False
        
        # Limpiar recursivamente
        data = clean_json_value(data)
        
        # Guardar
        backup_path = filepath + '.backup'
        if not os.path.exists(backup_path):
            os.rename(filepath, backup_path)
            print(f"  Backup guardado en {backup_path}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  OK: Archivo limpiado y guardado: {filepath}")
        return True
    else:
        print(f"  OK: Archivo ya esta limpio: {filepath}")
        return False

if __name__ == "__main__":
    files_to_fix = [
        "outputs/quantus_metrics_retina.json",
        "outputs/quantus_metrics_blood.json",
        "outputs/quantus_metrics_breast.json",
    ]
    
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            fix_json_file(filepath)
        else:
            print(f"  ADVERTENCIA: Archivo no encontrado: {filepath}")

