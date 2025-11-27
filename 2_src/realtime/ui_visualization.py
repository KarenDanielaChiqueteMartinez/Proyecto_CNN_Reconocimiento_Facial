"""
Interfaz de visualización para reconocimiento facial en tiempo real.

- Usar la webcam en vivo
- Procesar archivos de video
- Procesar imágenes estáticas
"""

import sys
import cv2
from pathlib import Path

# Agregar paths del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "2_src"))

# Importar el reconocedor
try:
    from .detectar_y_clasificar import RealTimeFaceRecognizer
except ImportError:
    from realtime.detectar_y_clasificar import RealTimeFaceRecognizer


class FaceRecognitionApp:
    """
    Aplicación principal de reconocimiento facial.
    
    Esta clase maneja la interfaz gráfica y la captura de video,
    delegando el procesamiento al RealTimeFaceRecognizer.
    """
    
    def __init__(self, model_path, labels_map_path=None, detection_method='haar'):
        """
        Inicializa la aplicación.
        
        Args:
            model_path: Ruta al modelo entrenado (.h5)
            labels_map_path: Ruta al labels_map.json
            detection_method: 'haar' (rápido) o 'mtcnn' (preciso)
        """
        self.recognizer = RealTimeFaceRecognizer(
            model_path, labels_map_path, detection_method
        )
        self.cap = None  # Capturador de video
    
    def start(self, camera_index=0, window_name="Reconocimiento Facial"):
        """
        Inicia el reconocimiento facial con la webcam.
        
        Args:
            camera_index: Índice de la cámara (0 = cámara principal)
            window_name: Título de la ventana
        """
        # Abrir la cámara
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            print(f"[ERROR] No se pudo abrir la cámara {camera_index}")
            return
        
        print("[INFO] Presiona 'q' para salir")
        print("[INFO] Presiona 's' para guardar screenshot")
        
        frame_count = 0
        
        try:
            while True:
                # Leer frame de la cámara
                ret, frame = self.cap.read()
                
                if not ret:
                    print("[ERROR] No se pudo leer el frame")
                    break
                
                # Procesar frame (detectar y clasificar rostros)
                frame_processed = self.recognizer.process_frame(frame)
                
                # Mostrar FPS cada 30 frames
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    cv2.putText(
                        frame_processed, f"FPS: {fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2
                    )
                
                # Mostrar frame en ventana
                cv2.imshow(window_name, frame_processed)
                
                # Manejar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("[INFO] Cerrando aplicación...")
                    break
                elif key == ord('s'):
                    # Guardar screenshot
                    screenshot_path = Path("5_results") / f"screenshot_{frame_count}.jpg"
                    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(screenshot_path), frame_processed)
                    print(f"[INFO] Screenshot guardado en {screenshot_path}")
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupción del usuario")
        
        finally:
            self.cleanup()
    
    def process_video_file(self, video_path, output_path=None):
        """
        Procesa un archivo de video.
        
        Args:
            video_path: Ruta al video de entrada
            output_path: Ruta para guardar el video procesado (opcional)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"[ERROR] No se pudo abrir el video {video_path}")
            return
        
        # Obtener propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Video: {fps} FPS, {width}x{height}, {total_frames} frames")
        
        # Configurar writer para guardar video procesado
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Procesar frame
                frame_processed = self.recognizer.process_frame(frame)
                
                # Guardar si hay writer
                if writer:
                    writer.write(frame_processed)
                
                # Mostrar progreso
                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"[INFO] Progreso: {frame_count}/{total_frames} ({progress:.1f}%)")
                
                # Mostrar frame
                cv2.imshow("Procesando Video", frame_processed)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print(f"[INFO] Video procesado. {frame_count} frames")
    
    def process_image(self, image_path, output_path=None):
        """
        Procesa una imagen estática.
        
        Args:
            image_path: Ruta a la imagen
            output_path: Ruta para guardar resultado (opcional)
        """
        frame = cv2.imread(str(image_path))
        
        if frame is None:
            print(f"[ERROR] No se pudo cargar la imagen {image_path}")
            return
        
        # Procesar imagen
        frame_processed = self.recognizer.process_frame(frame)
        
        # Guardar si se especificó ruta
        if output_path:
            cv2.imwrite(str(output_path), frame_processed)
            print(f"[INFO] Imagen procesada guardada en {output_path}")
        
        # Mostrar resultado
        cv2.imshow("Imagen Procesada", frame_processed)
        print("[INFO] Presiona cualquier tecla para cerrar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def cleanup(self):
        """Libera recursos (cámara y ventanas)."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
