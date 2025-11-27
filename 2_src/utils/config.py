"""
Configuración global del proyecto de reconocimiento facial.

"""

from pathlib import Path

# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================
# Path(__file__) obtiene la ubicación de este archivo (config.py),
# y con .parent.parent.parent subimos 3 niveles hasta la raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Aquí definimos las carpetas principales del proyecto
DATA_DIR = PROJECT_ROOT / "1_dataset"
PROCESSED_DIR = DATA_DIR / "processed"      # Imágenes ya procesadas (rostros recortados)
SPLITS_DIR = DATA_DIR / "splits"            # Datos divididos en train/val/test
METADATA_DIR = DATA_DIR / "metadata"        # Archivos JSON con info del dataset
MODELS_DIR = PROJECT_ROOT / "3_models"      # Donde guardamos los modelos entrenados
RESULTS_DIR = PROJECT_ROOT / "5_results"    # Gráficas, reportes, etc.

# =============================================================================
# CONFIGURACIÓN DE IMÁGENES
# =============================================================================
# El tamaño 160x160 es común para reconocimiento facial.
# Todas las imágenes deben tener el mismo tamaño para que
# la red neuronal pueda procesarlas correctamente
IMAGE_SIZE = (160, 160)
INPUT_SHAPE = (*IMAGE_SIZE, 3)  # (160, 160, 3) - el 3 es por los canales RGB

# =============================================================================
# CONFIGURACIÓN DE ENTRENAMIENTO
# =============================================================================
# Batch size pequeño porque nuestro dataset es pequeño
# (si usamos batches muy grandes con pocos datos, el modelo no aprende bien)
BATCH_SIZE = 8

# Más épocas porque con datasets pequeños necesitamos más iteraciones
EPOCHS = 200

# Learning rate bajo para transfer learning - así no "destruimos" lo que
# el modelo ya aprendió de ImageNet
LEARNING_RATE = 0.0001

# Porcentaje de datos para validación
VALIDATION_SPLIT = 0.15

# =============================================================================
# CONFIGURACIÓN DEL MODELO
# =============================================================================
# Este valor se actualiza automáticamente cuando cargamos labels_map.json
NUM_CLASSES = 10

# =============================================================================
# DATA AUGMENTATION
# =============================================================================

AUGMENTATION_CONFIG = {
    'rotation_range': 30,           # Rotar hasta 30 grados
    'width_shift_range': 0.3,       # Mover horizontalmente hasta 30%
    'height_shift_range': 0.3,      # Mover verticalmente hasta 30%
    'shear_range': 0.3,             # Deformación tipo "cizalla"
    'zoom_range': 0.3,              # Zoom aleatorio
    'horizontal_flip': True,        # Voltear horizontalmente (como espejo)
    'fill_mode': 'nearest',         # Cómo rellenar pixeles vacíos
    'brightness_range': [0.7, 1.3]  # Variar el brillo
}

# =============================================================================
# CALLBACKS DE ENTRENAMIENTO
# =============================================================================
# Los callbacks son funciones que se ejecutan durante el entrenamiento.
# Early stopping detiene el entrenamiento si el modelo deja de mejorar
EARLY_STOPPING_PATIENCE = 20    # Esperar 20 épocas sin mejora antes de parar
REDUCE_LR_PATIENCE = 8          # Reducir learning rate si no mejora en 8 épocas
REDUCE_LR_FACTOR = 0.5          # Multiplicar LR por 0.5 cuando se reduce
MIN_LR = 1e-7                   # Learning rate mínimo permitido
