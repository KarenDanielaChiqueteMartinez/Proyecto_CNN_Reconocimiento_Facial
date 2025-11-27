"""
Utilidades para el entrenamiento de modelos.

Este módulo contiene funciones auxiliares que se usan durante
el entrenamiento, como crear callbacks y guardar el historial.
"""

import json
from tensorflow import keras

# Importar configuración
try:
    from ..utils.config import (
        MODELS_DIR, RESULTS_DIR, EARLY_STOPPING_PATIENCE,
        REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, MIN_LR
    )
except ImportError:
    from utils.config import (
        MODELS_DIR, RESULTS_DIR, EARLY_STOPPING_PATIENCE,
        REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, MIN_LR
    )


def create_callbacks(model_name, monitor='val_accuracy'):
    """
    Crea los callbacks para el entrenamiento.
    
    Los callbacks son funciones que Keras ejecuta automáticamente
    durante el entrenamiento. Entiendo que son muy útiles para:
    - Guardar el mejor modelo automáticamente
    - Detener el entrenamiento si ya no mejora
    - Ajustar el learning rate dinámicamente
    
    Args:
        model_name: Nombre del modelo (para los archivos)
        monitor: Métrica a monitorear (val_accuracy es buena opción)
        
    Returns:
        Lista de callbacks configurados
    """
    callbacks = []
    
    # Crear directorios si no existen
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # MODEL CHECKPOINT
    # =========================================================================
    # Guarda el modelo cada vez que mejora la métrica monitoreada.
    # Así no perdemos el mejor modelo si después empeora por overfitting.
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(MODELS_DIR / f"{model_name}_best.h5"),
        monitor=monitor,
        save_best_only=True,      # Solo guardar si es mejor que el anterior
        save_weights_only=False,  # Guardar modelo completo, no solo pesos
        mode='max',               # Queremos maximizar accuracy
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # =========================================================================
    # EARLY STOPPING
    # =========================================================================
    # Detiene el entrenamiento si la métrica no mejora después de N épocas.
    # Esto evita entrenar de más y perder tiempo.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,  # Al terminar, cargar los mejores pesos
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # =========================================================================
    # REDUCE LEARNING RATE ON PLATEAU
    # =========================================================================
    # Si el modelo deja de mejorar, reduce el learning rate.
    # A veces con pasos más pequeños el modelo puede seguir aprendiendo.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=REDUCE_LR_FACTOR,      # Multiplicar LR por este factor
        patience=REDUCE_LR_PATIENCE,  # Esperar N épocas antes de reducir
        min_lr=MIN_LR,                # No bajar de este valor
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # =========================================================================
    # CSV LOGGER
    # =========================================================================
    # Guarda las métricas de cada época en un archivo CSV.
    # Útil para analizar el entrenamiento después.
    csv_logger = keras.callbacks.CSVLogger(
        filename=str(RESULTS_DIR / f"{model_name}_training.log"),
        append=False
    )
    callbacks.append(csv_logger)
    
    return callbacks


def save_model_history(history, model_name):
    """
    Guarda el historial de entrenamiento en un archivo JSON.
    
    El historial contiene las métricas (loss, accuracy) de cada época.
    Lo guardamos para poder analizarlo o graficar después.
    
    Args:
        history: Objeto History retornado por model.fit()
        model_name: Nombre del modelo para el archivo
    """
    # Convertir valores numpy a float para que JSON pueda guardarlos
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]
    
    # Guardar como JSON
    history_path = RESULTS_DIR / f"{model_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"[INFO] Historial guardado en {history_path}")
