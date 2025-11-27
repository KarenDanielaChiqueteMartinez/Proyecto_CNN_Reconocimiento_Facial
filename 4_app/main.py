"""
Aplicación principal para reconocimiento facial en tiempo real.

Este es el punto de entrada de la aplicación. Permite usar
la webcam, procesar videos o imágenes desde la línea de comandos.

Ejemplos de uso:
    # Webcam en vivo (por defecto)
    python main.py
    
    # Usar MTCNN (más preciso pero más lento)
    python main.py --detection mtcnn
    
    # Procesar una imagen
    python main.py --image foto.jpg
    
    # Procesar un video
    python main.py --video video.mp4 --output resultado.mp4
"""

import sys
import argparse
from pathlib import Path

# Agregar paths del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "2_src"))

from realtime.ui_visualization import FaceRecognitionApp


def parse_arguments():
    """
    Configura los argumentos de línea de comandos.
    
    Creo que es buena práctica usar argparse porque hace que
    el script sea más flexible sin tener que modificar el código.
    """
    parser = argparse.ArgumentParser(
        description='Aplicación de reconocimiento facial en tiempo real'
    )
    parser.add_argument(
        '--model', type=str,
        default='3_models/mobilenet_transfer_best.h5',
        help='Ruta al modelo entrenado'
    )
    parser.add_argument(
        '--labels', type=str, default=None,
        help='Ruta al labels_map.json'
    )
    parser.add_argument(
        '--detection', type=str, choices=['haar', 'mtcnn'],
        default='haar',
        help='Método de detección: haar (rápido) o mtcnn (preciso)'
    )
    parser.add_argument(
        '--camera', type=int, default=0,
        help='Índice de la cámara (default: 0)'
    )
    parser.add_argument(
        '--video', type=str, default=None,
        help='Procesar archivo de video en lugar de webcam'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Ruta para guardar video procesado'
    )
    parser.add_argument(
        '--image', type=str, default=None,
        help='Procesar una imagen estática'
    )
    
    return parser.parse_args()


def main():
    """Función principal de la aplicación."""
    
    args = parse_arguments()
    
    # Verificar que existe el modelo
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] No se encuentra el modelo en {model_path}")
        print("[INFO] Asegúrate de haber entrenado el modelo primero con:")
        print("       python 2_src/training/train_transfer.py --model mobilenet")
        return
    
    # Crear la aplicación
    app = FaceRecognitionApp(
        model_path=str(model_path),
        labels_map_path=args.labels,
        detection_method=args.detection
    )
    
    # Ejecutar según el modo seleccionado
    if args.image:
        # Modo: procesar imagen estática
        app.process_image(args.image, args.output)
    elif args.video:
        # Modo: procesar archivo de video
        app.process_video_file(args.video, args.output)
    else:
        # Modo: webcam en vivo (por defecto)
        app.start(camera_index=args.camera)


if __name__ == "__main__":
    main()
