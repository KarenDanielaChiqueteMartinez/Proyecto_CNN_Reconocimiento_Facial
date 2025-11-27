"""
Detector de rostros usando OpenCV o MTCNN.

Este módulo se encarga de encontrar rostros en imágenes o video.
Ofrece dos métodos:
- Haar Cascades: más rápido pero menos preciso
- MTCNN: más preciso pero más lento

Para tiempo real Haar es mejor opción, pero MTCNN
detecta mejor rostros en ángulos difíciles.
"""

import cv2
import numpy as np
from mtcnn import MTCNN


class FaceDetector:
    """
    Clase para detectar y extraer rostros de imágenes.
    
    Encapsula la lógica de detección para que sea fácil
    cambiar entre métodos sin modificar el resto del código.
    """
    
    def __init__(self, method='haar'):
        """
        Inicializa el detector.
        
        Args:
            method: 'haar' (rápido) o 'mtcnn' (preciso)
        """
        self.method = method
        
        if method == 'haar':
            # Haar Cascade viene incluido con OpenCV
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
        elif method == 'mtcnn':
            # MTCNN es una red neuronal especializada en detectar rostros
            self.detector = MTCNN()
        else:
            raise ValueError("Método debe ser 'haar' o 'mtcnn'")
    
    def detect_faces(self, frame):
        """
        Detecta todos los rostros en un frame.
        
        Args:
            frame: Imagen en formato BGR (como la lee OpenCV)
            
        Returns:
            Lista de tuplas (x, y, ancho, alto) con las coordenadas
            de cada rostro detectado
        """
        if self.method == 'haar':
            # Haar funciona mejor en escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,   # Cuánto reducir la imagen en cada escala
                minNeighbors=5,    # Mínimo de detecciones vecinas para confirmar
                minSize=(30, 30),  # Tamaño mínimo del rostro
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            return faces
        
        else:  # mtcnn
            # MTCNN necesita RGB (OpenCV usa BGR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_faces(rgb_frame)
            
            # Filtrar detecciones con baja confianza
            faces = []
            for detection in detections:
                if detection['confidence'] > 0.9:
                    x, y, w, h = detection['box']
                    faces.append((x, y, w, h))
            
            return faces
    
    def extract_face(self, frame, face_box, target_size=(160, 160)):
        """
        Extrae y preprocesa un rostro del frame.
        
        Recorta la región del rostro y la redimensiona al tamaño
        que espera nuestro modelo de clasificación.
        
        Args:
            frame: Imagen completa
            face_box: Tupla (x, y, ancho, alto) del rostro
            target_size: Tamaño de salida deseado
            
        Returns:
            Imagen del rostro recortada y redimensionada, o None si falla
        """
        x, y, w, h = face_box
        
        # Asegurar que las coordenadas están dentro del frame
        frame_height, frame_width = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame_width - x)
        h = min(h, frame_height - y)
        
        # Verificar que el recorte es válido
        if w <= 0 or h <= 0:
            return None
        
        # Recortar el rostro
        face_roi = frame[y:y+h, x:x+w]
        
        # Redimensionar al tamaño esperado por el modelo
        face_resized = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
        
        return face_resized
