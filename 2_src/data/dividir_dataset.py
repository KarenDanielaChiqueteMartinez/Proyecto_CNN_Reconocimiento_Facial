"""
Script para dividir el dataset procesado en train/val/test.

Este script toma las imágenes procesadas y las divide en tres conjuntos:
- Train (70%): para entrenar el modelo
- Validation (15%): para ajustar hiperparámetros y evitar overfitting
- Test (15%): para evaluar el rendimiento final

"""

import json
import random
import shutil
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
PROCESSED_DIR = Path("1_dataset/processed")
SPLITS_DIR = Path("1_dataset/splits")
METADATA_DIR = Path("1_dataset/metadata")

# Proporciones de división
TRAIN_RATIO = 0.70   # 70% para entrenamiento
VAL_RATIO = 0.15     # 15% para validación
TEST_RATIO = 0.15    # 15% para test


def split_dataset():
    """
    Divide el dataset en train/val/test.
    
    Para cada persona:
    1. Obtiene todas sus imágenes
    2. Las mezcla aleatoriamente
    3. Divide según las proporciones definidas
    4. Copia a las carpetas correspondientes
    """
    # Crear directorios de splits
    train_dir = SPLITS_DIR / "train"
    val_dir = SPLITS_DIR / "val"
    test_dir = SPLITS_DIR / "test"
    
    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
    
    # Verificar que existe labels_map
    labels_map_path = METADATA_DIR / "labels_map.json"
    if not labels_map_path.exists():
        print(f"[ERROR] No se encontró {labels_map_path}")
        print("[INFO] Primero ejecuta procesar_imagenes.py")
        return
    
    with open(labels_map_path, 'r', encoding='utf-8') as f:
        labels_map = json.load(f)
    
    # Obtener carpetas de personas
    person_folders = sorted([d for d in PROCESSED_DIR.iterdir() if d.is_dir()])
    
    if len(person_folders) == 0:
        print(f"[ERROR] No se encontraron carpetas en {PROCESSED_DIR}")
        return
    
    print(f"[INFO] Dividiendo dataset de {len(person_folders)} personas...")
    
    # Contadores para estadísticas
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    # Procesar cada persona
    for person_folder in person_folders:
        person_name = person_folder.name
        
        # Obtener todas las imágenes de esta persona
        image_files = list(person_folder.glob("*.jpg"))
        image_files.extend(person_folder.glob("*.jpeg"))
        
        if len(image_files) == 0:
            print(f"[WARNING] No se encontraron imágenes en {person_name}")
            continue
        
        # Mezclar aleatoriamente para que la división sea imparcial
        random.shuffle(image_files)
        
        # Calcular cuántas imágenes van a cada conjunto
        n_total = len(image_files)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        n_test = n_total - n_train - n_val  # El resto va a test
        
        # Dividir la lista de imágenes
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Copiar archivos a cada split
        for split_name, files, split_dir in [
            ('train', train_files, train_dir),
            ('val', val_files, val_dir),
            ('test', test_files, test_dir)
        ]:
            # Crear carpeta para esta persona en el split
            person_split_dir = split_dir / person_name
            person_split_dir.mkdir(parents=True, exist_ok=True)
            
            # Copiar cada imagen
            for img_file in files:
                dest_path = person_split_dir / img_file.name
                shutil.copy2(img_file, dest_path)
            
            stats[split_name] += len(files)
            print(f"  {person_name}: {len(files)} imágenes -> {split_name}")
    
    # Guardar estadísticas de la división
    split_stats = {
        'train_images': stats['train'],
        'val_images': stats['val'],
        'test_images': stats['test'],
        'total_images': stats['train'] + stats['val'] + stats['test'],
        'train_ratio': f"{TRAIN_RATIO*100:.1f}%",
        'val_ratio': f"{VAL_RATIO*100:.1f}%",
        'test_ratio': f"{TEST_RATIO*100:.1f}%"
    }
    
    stats_path = METADATA_DIR / "split_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(split_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Estadísticas guardadas en {stats_path}")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("[SUCCESS] DIVISIÓN COMPLETADA")
    print("=" * 60)
    print(f"Train: {stats['train']} imágenes ({TRAIN_RATIO*100:.1f}%)")
    print(f"Val:   {stats['val']} imágenes ({VAL_RATIO*100:.1f}%)")
    print(f"Test:  {stats['test']} imágenes ({TEST_RATIO*100:.1f}%)")
    print(f"Total: {stats['train'] + stats['val'] + stats['test']} imágenes")
    print(f"\nSplits guardados en: {SPLITS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    split_dataset()
