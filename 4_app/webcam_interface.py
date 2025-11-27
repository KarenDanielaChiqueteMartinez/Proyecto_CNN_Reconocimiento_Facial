"""
Interfaz simplificada para usar con webcam.

Este script es un atajo rápido para iniciar el reconocimiento
facial sin tener que escribir argumentos. Solo ejecuta:
    python webcam_interface.py
"""

import sys
from pathlib import Path

# Agregar paths del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "2_src"))

from realtime.ui_visualization import FaceRecognitionApp


def main():
    """Inicia la aplicación de reconocimiento facial con webcam."""
    
    # Buscar el mejor modelo disponible
    models_dir = project_root / "3_models"
    model_files = list(models_dir.glob("*_best.h5"))
    
    if not model_files:
        print("[ERROR] No se encontró ningún modelo entrenado")
        print("[INFO] Primero debes entrenar un modelo ejecutando:")
        print("       python 2_src/training/train_transfer.py --model mobilenet")
        return
    
    # Usar el primer modelo encontrado
    model_path = model_files[0]
    print(f"[INFO] Usando modelo: {model_path.name}")
    
    # Crear y ejecutar aplicación
    app = FaceRecognitionApp(
        model_path=str(model_path),
        detection_method='haar'  # Haar es más rápido para tiempo real
    )
    
    print("\n" + "=" * 60)
    print("RECONOCIMIENTO FACIAL EN TIEMPO REAL")
    print("=" * 60)
    print("Controles:")
    print("  - Presiona 'q' para salir")
    print("  - Presiona 's' para guardar screenshot")
    print("=" * 60 + "\n")
    
    app.start(camera_index=0)


if __name__ == "__main__":
    main()
