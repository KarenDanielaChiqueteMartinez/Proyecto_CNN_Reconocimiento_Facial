"""
Script para procesar imágenes de raw/ y guardarlas en processed/.

Este script toma las imágenes originales, detecta los rostros,
los recorta, alinea y redimensiona a un tamaño estándar.

"""

import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from mtcnn import MTCNN

# =============================================================================
# CONFIGURACIÓN POR DEFECTO
# =============================================================================
RAW_DIR = Path("1_dataset/raw")
PROCESSED_DIR = Path("1_dataset/processed")
METADATA_DIR = Path("1_dataset/metadata")
TARGET_SIZE = (160, 160)  # Tamaño estándar para reconocimiento facial
MIN_FACE_SIZE = 40        # Tamaño mínimo de rostro a detectar


def detect_and_align_face(image_path, detector):
    """
    Detecta un rostro en la imagen, lo recorta y alinea.
    
    El proceso es:
    1. Detectar el rostro con MTCNN
    2. Obtener los puntos de los ojos
    3. Calcular el ángulo de inclinación
    4. Rotar para alinear los ojos horizontalmente
    5. Recortar y redimensionar
    
    Args:
        image_path: Ruta a la imagen original
        detector: Instancia de MTCNN
        
    Returns:
        Imagen del rostro procesada, o None si no se detectó
    """
    try:
        # Leer imagen con OpenCV
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # MTCNN necesita RGB, OpenCV usa BGR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detectar rostros
        detections = detector.detect_faces(img_rgb)
        
        if len(detections) == 0:
            return None
        
        # Tomar el rostro con mayor confianza
        best_face = max(detections, key=lambda x: x['confidence'])
        
        # Verificar que la confianza sea suficiente
        if best_face['confidence'] < 0.9:
            return None
        
        # Obtener coordenadas del bounding box
        x, y, w, h = best_face['box']
        
        # Verificar tamaño mínimo
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            return None
        
        # Obtener puntos clave (ojos, nariz, boca)
        keypoints = best_face['keypoints']
        
        # Agregar margen alrededor del rostro (30%)
        margin = 0.3
        x_margin = int(w * margin)
        y_margin = int(h * margin)
        
        # Calcular coordenadas del recorte con margen
        x1 = max(0, x - x_margin)
        y1 = max(0, y - y_margin)
        x2 = min(img_rgb.shape[1], x + w + x_margin)
        y2 = min(img_rgb.shape[0], y + h + y_margin)
        
        face_crop = img_rgb[y1:y2, x1:x2]
        
        # Alinear rostro usando la posición de los ojos
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        
        # Calcular ángulo de rotación necesario
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Rotar imagen para que los ojos queden horizontales
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        face_aligned = cv2.warpAffine(
            face_crop, rotation_matrix,
            (face_crop.shape[1], face_crop.shape[0]),
            flags=cv2.INTER_CUBIC
        )
        
        # Redimensionar al tamaño estándar
        face_resized = cv2.resize(
            face_aligned, TARGET_SIZE,
            interpolation=cv2.INTER_AREA
        )
        
        return face_resized
    
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None


def process_person_folder(person_name, raw_path, processed_path, detector, stats):
    """
    Procesa todas las imágenes de una persona.
    
    Args:
        person_name: Nombre de la persona (nombre de la carpeta)
        raw_path: Carpeta con imágenes originales
        processed_path: Carpeta donde guardar las procesadas
        detector: Instancia de MTCNN
        stats: Diccionario para guardar estadísticas
    """
    # Crear carpeta de destino
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Buscar todas las imágenes
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in raw_path.iterdir() if f.suffix in image_extensions]
    
    if len(image_files) == 0:
        print(f"[WARNING] No se encontraron imágenes en {raw_path}")
        return
    
    print(f"\n[INFO] Procesando {person_name} ({len(image_files)} imágenes)...")
    
    processed_count = 0
    failed_count = 0
    
    # Procesar cada imagen
    for img_file in tqdm(image_files, desc=f"  {person_name}"):
        face_img = detect_and_align_face(img_file, detector)
        
        if face_img is not None:
            # Generar nombre de salida con formato consistente
            output_name = f"{person_name}_{processed_count + 1:04d}.jpg"
            output_path = processed_path / output_name
            
            # Guardar imagen (convertir RGB a BGR para OpenCV)
            face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), face_bgr)
            processed_count += 1
        else:
            failed_count += 1
            print(f"  [WARNING] No se detectó rostro en: {img_file.name}")
    
    # Guardar estadísticas
    stats[person_name] = {
        'total_images': len(image_files),
        'processed': processed_count,
        'failed': failed_count,
        'success_rate': f"{(processed_count/len(image_files)*100):.1f}%"
    }
    
    print(f"  [OK] {processed_count} procesadas, {failed_count} fallidas")


def generate_labels_map(processed_dir):
    """
    Genera el archivo labels_map.json.
    
    Este archivo mapea índices numéricos a nombres de personas.
    Es necesario para que el modelo sepa qué número corresponde
    a cada persona durante el entrenamiento y la predicción.
    
    Args:
        processed_dir: Directorio con las carpetas de personas
        
    Returns:
        Diccionario {índice: nombre}
    """
    person_folders = sorted([d.name for d in processed_dir.iterdir() if d.is_dir()])
    labels_map = {idx: name for idx, name in enumerate(person_folders)}
    return labels_map


def main():
    """Función principal del script."""
    
    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description='Procesa imágenes de raw/ y las guarda en processed/'
    )
    parser.add_argument(
        '--raw-dir', type=str, default='1_dataset/raw',
        help='Directorio con imágenes originales'
    )
    parser.add_argument(
        '--processed-dir', type=str, default='1_dataset/processed',
        help='Directorio de salida'
    )
    parser.add_argument(
        '--target-size', type=int, nargs=2, default=[160, 160],
        help='Tamaño de salida (ancho alto)'
    )
    
    args = parser.parse_args()
    
    # Actualizar configuración global
    global RAW_DIR, PROCESSED_DIR, TARGET_SIZE
    RAW_DIR = Path(args.raw_dir)
    PROCESSED_DIR = Path(args.processed_dir)
    TARGET_SIZE = tuple(args.target_size)
    
    # Crear directorios necesarios
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Verificar que existe el directorio raw
    if not RAW_DIR.exists():
        print(f"[ERROR] No existe el directorio {RAW_DIR}")
        return
    
    # Inicializar detector
    print("[INFO] Inicializando detector de rostros (MTCNN)...")
    detector = MTCNN()
    
    # Obtener carpetas de personas
    person_folders = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    
    if len(person_folders) == 0:
        print(f"[ERROR] No se encontraron carpetas de personas en {RAW_DIR}")
        return
    
    print(f"\n[INFO] Encontradas {len(person_folders)} personas para procesar")
    
    # Procesar cada persona
    stats = {}
    for person_folder in person_folders:
        person_name = person_folder.name
        processed_path = PROCESSED_DIR / person_name
        process_person_folder(person_name, person_folder, processed_path, detector, stats)
    
    # Generar y guardar labels_map.json
    print("\n[INFO] Generando labels_map.json...")
    labels_map = generate_labels_map(PROCESSED_DIR)
    labels_map_path = METADATA_DIR / "labels_map.json"
    
    with open(labels_map_path, 'w', encoding='utf-8') as f:
        json.dump(labels_map, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Guardado en {labels_map_path}")
    
    # Generar y guardar dataset_stats.json
    print("\n[INFO] Generando dataset_stats.json...")
    
    total_images = sum(s['total_images'] for s in stats.values())
    total_processed = sum(s['processed'] for s in stats.values())
    total_failed = sum(s['failed'] for s in stats.values())
    
    dataset_stats = {
        'total_persons': len(stats),
        'total_raw_images': total_images,
        'total_processed_images': total_processed,
        'total_failed_images': total_failed,
        'overall_success_rate': f"{(total_processed/total_images*100):.1f}%",
        'target_size': TARGET_SIZE,
        'per_person_stats': stats
    }
    
    stats_path = METADATA_DIR / "dataset_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_stats, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Guardado en {stats_path}")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("[SUCCESS] PROCESAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Personas procesadas: {len(stats)}")
    print(f"Imágenes procesadas: {total_processed}/{total_images}")
    print(f"Tasa de éxito: {(total_processed/total_images*100):.1f}%")
    print(f"\nImágenes guardadas en: {PROCESSED_DIR}")
    print(f"Metadata guardada en: {METADATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
