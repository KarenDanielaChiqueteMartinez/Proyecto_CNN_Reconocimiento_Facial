"""
Sistema de detección y clasificación facial en tiempo real.

Este módulo combina el detector de rostros con el modelo de
clasificación para identificar personas en video en vivo.
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from tensorflow import keras

# Agregar paths del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "2_src"))

# Importar módulos del proyecto
try:
    from .detector_rostro import FaceDetector
    from ..utils.config import IMAGE_SIZE, METADATA_DIR
except ImportError:
    from realtime.detector_rostro import FaceDetector
    from utils.config import IMAGE_SIZE, METADATA_DIR


class RealTimeFaceRecognizer:
    """
    Clasificador de rostros en tiempo real.
    
    Esta clase coordina todo el proceso:
    1. Detectar rostros en el frame
    2. Preprocesar cada rostro
    3. Clasificar con el modelo entrenado
    4. Dibujar resultados en el frame
    """
    
    def __init__(self, model_path, labels_map_path=None, detection_method='haar'):
        """
        Inicializa el reconocedor.
        
        Args:
            model_path: Ruta al archivo .h5 del modelo entrenado
            labels_map_path: Ruta al labels_map.json (opcional)
            detection_method: 'haar' o 'mtcnn'
        """
        # Cargar el modelo de clasificación
        print(f"[INFO] Cargando modelo desde {model_path}...")
        self.model = keras.models.load_model(str(model_path))
        print("[INFO] Modelo cargado exitosamente")
        
        # Cargar el mapeo de índices a nombres de personas
        if labels_map_path is None:
            labels_map_path = METADATA_DIR / "labels_map.json"
        
        with open(labels_map_path, 'r', encoding='utf-8') as f:
            labels_map = json.load(f)
        
        # Convertir a lista ordenada por índice
        self.class_names = [labels_map[str(i)] for i in range(len(labels_map))]
        print(f"[INFO] Clases cargadas: {self.class_names}")
        
        # Inicializar el detector de rostros
        self.face_detector = FaceDetector(method=detection_method)
        
        # Configuración
        self.target_size = IMAGE_SIZE
        self.confidence_threshold = 0.5  # Mínima confianza para mostrar nombre
    
    def preprocess_face(self, face_img):
        """
        Preprocesa una imagen de rostro para el modelo.
        
        El modelo espera:
        - Imagen en RGB (no BGR)
        - Tamaño 160x160
        - Valores normalizados entre 0 y 1
        - Batch dimension (1, 160, 160, 3)
        
        Args:
            face_img: Imagen del rostro en BGR
            
        Returns:
            Array listo para model.predict()
        """
        # Convertir de BGR a RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Redimensionar si es necesario
        if face_rgb.shape[:2] != self.target_size:
            face_rgb = cv2.resize(face_rgb, self.target_size, 
                                 interpolation=cv2.INTER_AREA)
        
        # Normalizar pixeles a [0, 1]
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # Agregar dimensión de batch
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def predict(self, face_img):
        """
        Predice la identidad de un rostro.
        
        Args:
            face_img: Imagen del rostro en BGR
            
        Returns:
            Tupla (nombre_persona, confianza) o (None, confianza) si
            la confianza es menor al umbral
        """
        # Preprocesar imagen
        face_batch = self.preprocess_face(face_img)
        
        # Obtener predicciones (probabilidades para cada clase)
        predictions = self.model.predict(face_batch, verbose=0)
        
        # Encontrar la clase con mayor probabilidad
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        # Verificar si supera el umbral de confianza
        if confidence < self.confidence_threshold:
            return None, confidence
        
        class_name = self.class_names[class_idx]
        return class_name, confidence
    
    def process_frame(self, frame):
        """
        Procesa un frame completo: detecta y clasifica todos los rostros.
        
        Args:
            frame: Frame de video en BGR
            
        Returns:
            Frame con bounding boxes y etiquetas dibujados
        """
        # Detectar todos los rostros
        faces = self.face_detector.detect_faces(frame)
        
        # Procesar cada rostro detectado
        for (x, y, w, h) in faces:
            # Extraer el rostro
            face_img = self.face_detector.extract_face(
                frame, (x, y, w, h), self.target_size
            )
            
            if face_img is None:
                continue
            
            # Clasificar el rostro
            name, confidence = self.predict(face_img)
            
            # Elegir color según si se reconoció o no
            # Verde = reconocido, Rojo = desconocido
            color = (0, 255, 0) if name else (0, 0, 255)
            
            # Dibujar rectángulo alrededor del rostro
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Preparar texto de la etiqueta
            if name:
                label = f"{name}: {confidence:.2f}"
            else:
                label = f"Desconocido: {confidence:.2f}"
            
            # Calcular posición del texto
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            label_y = max(y - 10, label_size[1] + 10)
            
            # Dibujar fondo para el texto (para que se lea mejor)
            cv2.rectangle(
                frame,
                (x, label_y - label_size[1] - 5),
                (x + label_size[0], label_y + 5),
                color, -1  # -1 = relleno sólido
            )
            
            # Dibujar el texto
            cv2.putText(
                frame, label, (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2  # Texto blanco
            )
        
        return frame
