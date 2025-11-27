"""
Modelos con Transfer Learning para reconocimiento facial.

Transfer learning significa usar un modelo que ya fue entrenado en
millones de imágenes (ImageNet) y adaptarlo a nuestra tarea específica.
Esto funciona mejor que entrenar desde cero porque el modelo
ya "sabe" detectar características visuales generales.
"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

# Importar configuración
try:
    from ..utils.config import INPUT_SHAPE, NUM_CLASSES
except ImportError:
    from utils.config import INPUT_SHAPE, NUM_CLASSES


def create_mobilenetv2_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, 
                            freeze_base=False, fine_tune_layers=10):
    """
    Crea un modelo usando MobileNetV2 como base.
    
    MobileNetV2 es una red diseñada para ser ligera y rápida,
    ideal para aplicaciones en tiempo real como la nuestra.
    
    La estrategia es:
    1. Usar MobileNetV2 pre-entrenado como "extractor de características"
    2. Agregar nuestras propias capas de clasificación encima
    3. Opcionalmente, "fine-tunear" las últimas capas de la base
    
    Args:
        input_shape: Tamaño de las imágenes de entrada
        num_classes: Número de personas a clasificar
        freeze_base: Si True, no entrenar la base (solo nuestras capas)
        fine_tune_layers: Cuántas capas finales de la base entrenar
        
    Returns:
        Modelo Keras (sin compilar)
    """
    # Cargar MobileNetV2 sin la capa de clasificación original
    # (include_top=False quita las capas Dense del final)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',  # Pesos pre-entrenados en ImageNet
        pooling='avg'        # GlobalAveragePooling al final
    )
    
    # Configurar qué capas entrenar
    if freeze_base:
        # No entrenar ninguna capa de la base
        base_model.trainable = False
    else:
        # Fine-tuning: entrenar solo las últimas N capas
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
    
    # Construir el modelo completo agregando nuestras capas
    model = keras.Sequential([
        base_model,
        
        # Capas de clasificación (estas siempre se entrenan)
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Capa de salida
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ])
    
    return model


def create_efficientnet_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                             freeze_base=False, fine_tune_layers=10):
    """
    Crea un modelo usando EfficientNetB0 como base.
    
    EfficientNet es más preciso que MobileNet pero también más pesado.
    Es una buena opción si la velocidad no es crítica.
    
    Args:
        input_shape: Tamaño de las imágenes de entrada
        num_classes: Número de personas a clasificar
        freeze_base: Si True, congelar la base
        fine_tune_layers: Capas a fine-tunear
        
    Returns:
        Modelo Keras (sin compilar)
    """
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Configurar entrenamiento de la base
    if freeze_base:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
    
    # Agregar capas de clasificación
    model = keras.Sequential([
        base_model,
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compila el modelo para entrenamiento.
    
    Para transfer learning, usamos un learning rate más bajo que para
    entrenar desde cero, así no "destruimos" lo que el modelo ya aprendió.
    
    Args:
        model: Modelo Keras sin compilar
        learning_rate: Tasa de aprendizaje (0.0001 es común para fine-tuning)
        
    Returns:
        Modelo compilado
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    return model


def get_model_summary(model):
    """Imprime el resumen del modelo."""
    model.summary()
    return model
