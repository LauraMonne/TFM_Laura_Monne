"""
Script auxiliar para descargar manualmente los archivos .npz de MedMNIST en tamaño 224x224.

Este script es útil cuando la descarga automática falla debido a problemas de red
o archivos corruptos.

Uso:
    python download_medmnist_manual.py

Los archivos se descargarán en la carpeta ./data/
"""

import os
import hashlib
import urllib.request
from pathlib import Path
from tqdm import tqdm

# URLs de Zenodo para los archivos 224x224
DATASETS = {
    "bloodmnist_224.npz": {
        "url": "https://zenodo.org/records/10519652/files/bloodmnist_224.npz?download=1",
        "md5": "b718ff6835fcbdb22ba9eacccd7b2601",
        "size_mb": 1540,  # Aproximado
    },
    "retinamnist_224.npz": {
        "url": "https://zenodo.org/records/10519652/files/retinamnist_224.npz?download=1",
        "md5": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",  # Actualizar con MD5 real si está disponible
        "size_mb": 1500,  # Aproximado
    },
    "breastmnist_224.npz": {
        "url": "https://zenodo.org/records/10519652/files/breastmnist_224.npz?download=1",
        "md5": "x1y2z3a4b5c6d7e8f9g0h1i2j3k4l5m6",  # Actualizar con MD5 real si está disponible
        "size_mb": 1500,  # Aproximado
    },
}

def calculate_md5(filepath: str) -> str:
    """Calcula el MD5 de un archivo."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url: str, filepath: str, expected_md5: str = None) -> bool:
    """
    Descarga un archivo con barra de progreso.
    
    Returns:
        True si la descarga fue exitosa y el MD5 coincide (si se proporciona)
    """
    print(f"\n📥 Descargando: {os.path.basename(filepath)}")
    print(f"   URL: {url}")
    print(f"   Destino: {filepath}")
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        # Descargar con barra de progreso
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r   Progreso: {percent:.1f}% ({mb_downloaded:.1f}MB / {mb_total:.1f}MB)", end="")
        
        urllib.request.urlretrieve(url, filepath, reporthook=show_progress)
        print()  # Nueva línea después de la barra de progreso
        
        # Verificar MD5 si se proporciona
        if expected_md5:
            print(f"   Verificando MD5...")
            actual_md5 = calculate_md5(filepath)
            if actual_md5.lower() != expected_md5.lower():
                print(f"   ⚠️  ADVERTENCIA: MD5 no coincide!")
                print(f"      Esperado: {expected_md5}")
                print(f"      Obtenido: {actual_md5}")
                print(f"      El archivo puede estar corrupto. Considera descargarlo de nuevo.")
                return False
            else:
                print(f"   ✅ MD5 verificado correctamente")
        
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"   ✅ Descarga completada: {file_size_mb:.1f}MB")
        return True
        
    except Exception as e:
        print(f"\n   ❌ Error durante la descarga: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"   🗑️  Archivo parcial eliminado")
        return False

def main():
    print("=" * 70)
    print("DESCARGA MANUAL DE MEDMNIST (224x224)")
    print("=" * 70)
    print("\nEste script descarga los archivos .npz de MedMNIST en tamaño 224x224")
    print("desde Zenodo. Útil cuando la descarga automática falla.\n")
    
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Verificar qué archivos ya existen
    existing_files = []
    missing_files = []
    
    for filename, info in DATASETS.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✅ {filename} ya existe ({file_size_mb:.1f}MB)")
            existing_files.append(filename)
        else:
            missing_files.append((filename, info))
    
    if not missing_files:
        print("\n🎉 Todos los archivos ya están descargados!")
        return
    
    print(f"\n📋 Archivos a descargar: {len(missing_files)}")
    total_size_mb = sum(info["size_mb"] for _, info in missing_files)
    print(f"📦 Tamaño total aproximado: {total_size_mb:.0f}MB (~{total_size_mb/1024:.1f}GB)")
    print(f"⏱️  Tiempo estimado: {total_size_mb/10:.0f} minutos (a 10MB/s)")
    
    respuesta = input("\n¿Continuar con la descarga? (s/n): ").strip().lower()
    if respuesta not in ['s', 'si', 'sí', 'y', 'yes']:
        print("Descarga cancelada.")
        return
    
    # Descargar archivos faltantes
    success_count = 0
    for filename, info in missing_files:
        filepath = os.path.join(data_dir, filename)
        if download_file(info["url"], filepath, info.get("md5")):
            success_count += 1
        else:
            print(f"\n❌ Error descargando {filename}. Puedes intentar descargarlo manualmente:")
            print(f"   1. Ve a: https://zenodo.org/records/10519652")
            print(f"   2. Busca: {filename}")
            print(f"   3. Descárgalo y colócalo en: {data_dir}/")
    
    print("\n" + "=" * 70)
    print(f"RESUMEN: {success_count}/{len(missing_files)} archivos descargados exitosamente")
    print("=" * 70)
    
    if success_count == len(missing_files):
        print("\n✅ Todos los archivos están listos. Puedes ejecutar:")
        print("   python prepare_data.py")
    else:
        print("\n⚠️  Algunos archivos fallaron. Revisa los errores arriba.")

if __name__ == "__main__":
    main()

