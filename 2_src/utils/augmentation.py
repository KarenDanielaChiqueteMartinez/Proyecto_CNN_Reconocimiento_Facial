"""
Utilidades para Data Augmentation (aumento de datos).

Este módulo se encarga de crear los "generadores" que
alimentan imágenes al modelo durante el entrenamiento. El augmentation
aplica transformaciones aleatorias a las imágenes para que el modelo
aprenda a reconocer rostros en diferentes condiciones.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Importamos la configuración de augmentation
try:
    from .config import AUGMENTATION_CONFIG
except ImportError:
    from utils.config import AUGMENTATION_CONFIG


def get_train_datagen():
    """
    Crea un generador de datos CON augmentation para entrenamiento.
    
    La idea es que cada vez que el modelo ve una imagen,
    la ve con transformaciones aleatorias diferentes, así aprende
    a reconocer la misma persona en distintas condiciones.
    
    Returns:
        ImageDataGenerator configurado con augmentation
    """
    return ImageDataGenerator(
        rescale=1.0/255.0,  # Normalizar pixeles de [0-255] a [0-1]
        rotation_range=AUGMENTATION_CONFIG['rotation_range'],
        width_shift_range=AUGMENTATION_CONFIG['width_shift_range'],
        height_shift_range=AUGMENTATION_CONFIG['height_shift_range'],
        shear_range=AUGMENTATION_CONFIG['shear_range'],
        zoom_range=AUGMENTATION_CONFIG['zoom_range'],
        horizontal_flip=AUGMENTATION_CONFIG['horizontal_flip'],
        fill_mode=AUGMENTATION_CONFIG['fill_mode']
    )


def get_val_test_datagen():
    """
    Crea un generador de datos SIN augmentation para validación/test.

    
    Returns:
        ImageDataGenerator solo con normalización
    """
    return ImageDataGenerator(rescale=1.0/255.0)


def create_data_generators(train_dir, val_dir, test_dir=None, 
                          batch_size=32, target_size=(160, 160)):
    """
    Crea los tres generadores de datos (train, val, test) de una vez.
    
    Los generadores leen las imágenes de las carpetas y las organizan
    automáticamente por clase (cada subcarpeta es una clase/persona).
    
    Args:
        train_dir: Carpeta con datos de entrenamiento
        val_dir: Carpeta con datos de validación
        test_dir: Carpeta con datos de test (opcional)
        batch_size: Cuántas imágenes procesar a la vez
        target_size: Tamaño al que redimensionar las imágenes
        
    Returns:
        Tupla (train_gen, val_gen, test_gen)
    """
    # Generadores base
    train_datagen = get_train_datagen()
    val_test_datagen = get_val_test_datagen()
    
    # Generador de entrenamiento (con augmentation y shuffle)
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',  # One-hot encoding para las etiquetas
        shuffle=True               # Mezclar datos en cada época
    )
    
    # Generador de validación (sin augmentation ni shuffle)
    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # No mezclar para que las métricas sean consistentes
    )
    
    # Generador de test (opcional)
    test_gen = None
    if test_dir:
        test_gen = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
    
    return train_gen, val_gen, test_gen
