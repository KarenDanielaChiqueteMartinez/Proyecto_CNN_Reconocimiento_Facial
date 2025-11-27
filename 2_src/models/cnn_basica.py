"""
Arquitectura CNN básica desde cero para reconocimiento facial.

"""

from tensorflow import keras
from tensorflow.keras import layers

# Importar configuración
try:
    from ..utils.config import INPUT_SHAPE, NUM_CLASSES
except ImportError:
    from utils.config import INPUT_SHAPE, NUM_CLASSES


def create_cnn_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Crear una CNN básica para reconocimiento facial.
    
    La arquitectura tiene 4 bloques convolucionales, cada uno con:
    - Capas Conv2D: extraen características de la imagen
    - BatchNormalization: estabiliza el entrenamiento
    - MaxPooling: reduce el tamaño y hace la red más robusta
    - Dropout: previene overfitting "apagando" neuronas aleatoriamente
    
    Al final hay capas Dense (fully connected) para la clasificación.
    
    Args:
        input_shape: Forma de entrada (alto, ancho, canales)
        num_classes: Número de personas a clasificar
        
    Returns:
        Modelo Keras (sin compilar)
    """
    model = keras.Sequential([
        # =====================================================================
        # BLOQUE 1: Extrae características básicas (bordes, texturas simples)
        # =====================================================================
        layers.Conv2D(32, (3, 3), activation='relu', 
                     input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),  # Reduce de 160x160 a 80x80
        layers.Dropout(0.25),
        
        # =====================================================================
        # BLOQUE 2: Características más complejas (partes del rostro)
        # =====================================================================
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),  # Reduce a 40x40
        layers.Dropout(0.25),
        
        # =====================================================================
        # BLOQUE 3: Características de alto nivel (combinaciones de partes)
        # =====================================================================
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),  # Reduce a 20x20
        layers.Dropout(0.25),
        
        # =====================================================================
        # BLOQUE 4: Características muy abstractas
        # =====================================================================
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),  # Reduce a 10x10
        layers.Dropout(0.25),
        
        # =====================================================================
        # CAPAS DE CLASIFICACIÓN
        # =====================================================================
        # GlobalAveragePooling reduce cada "mapa de características" a un número
        layers.GlobalAveragePooling2D(),
        
        # Capas densas para combinar las características y clasificar
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),  # Dropout más alto aquí para evitar overfitting
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Capa final: una neurona por persona, softmax da probabilidades
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compila el modelo con optimizador y función de pérdida.
    
    - Optimizador: cómo ajustar los pesos (Adam es muy usado)
    - Loss: cómo medir el error (categorical_crossentropy para clasificación)
    - Métricas: qué queremos monitorear durante el entrenamiento
    
    Args:
        model: Modelo Keras sin compilar
        learning_rate: Qué tan grandes son los pasos de aprendizaje
        
    Returns:
        Modelo compilado listo para entrenar
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    return model


def get_model_summary(model):
    """Imprime un resumen de la arquitectura del modelo."""
    model.summary()
    return model
