"""
Script para entrenar la CNN básica (desde cero).

"""

import sys
import json
from pathlib import Path

# Agregar el directorio del proyecto al path para poder importar los módulos
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "2_src"))

from models.cnn_basica import create_cnn_model, compile_model
from utils.augmentation import create_data_generators
from utils.config import (
    SPLITS_DIR, METADATA_DIR, BATCH_SIZE, EPOCHS, 
    LEARNING_RATE, IMAGE_SIZE
)
from training.utils_training import create_callbacks, save_model_history
from utils.evaluation import evaluate_model, plot_training_history


def main():
    """Función principal que ejecuta todo el entrenamiento."""
    
    print("=" * 60)
    print("ENTRENAMIENTO DE CNN BÁSICA")
    print("=" * 60)
    
    # =========================================================================
    # PASO 1: Cargar configuración del dataset
    # =========================================================================
    # El labels_map nos dice qué índice corresponde a cada persona
    labels_map_path = METADATA_DIR / "labels_map.json"
    with open(labels_map_path, 'r', encoding='utf-8') as f:
        labels_map = json.load(f)
    
    num_classes = len(labels_map)
    print(f"[INFO] Número de clases: {num_classes}")
    
    # =========================================================================
    # PASO 2: Crear generadores de datos
    # =========================================================================
    train_dir = SPLITS_DIR / "train"
    val_dir = SPLITS_DIR / "val"
    test_dir = SPLITS_DIR / "test"
    
    print("\n[INFO] Creando generadores de datos...")
    train_gen, val_gen, test_gen = create_data_generators(
        train_dir, val_dir, test_dir,
        batch_size=BATCH_SIZE,
        target_size=IMAGE_SIZE
    )
    
    # Mostrar información del dataset
    print(f"[INFO] Clases encontradas: {train_gen.class_indices}")
    print(f"[INFO] Imágenes de entrenamiento: {train_gen.samples}")
    print(f"[INFO] Imágenes de validación: {val_gen.samples}")
    print(f"[INFO] Imágenes de test: {test_gen.samples}")
    
    # Usar el número real de clases del generador
    actual_num_classes = len(train_gen.class_indices)
    
    # =========================================================================
    # PASO 3: Crear y compilar el modelo
    # =========================================================================
    print("\n[INFO] Creando modelo CNN básica...")
    model = create_cnn_model(
        input_shape=(*IMAGE_SIZE, 3), 
        num_classes=actual_num_classes
    )
    model = compile_model(model, learning_rate=LEARNING_RATE)
    model.summary()
    
    # =========================================================================
    # PASO 4: Configurar callbacks y entrenar
    # =========================================================================
    callbacks = create_callbacks("cnn_basica", monitor='val_accuracy')
    
    print("\n[INFO] Iniciando entrenamiento...")
    print(f"[INFO] Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"[INFO] Learning rate: {LEARNING_RATE}")
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # =========================================================================
    # PASO 5: Guardar resultados y evaluar
    # =========================================================================
    save_model_history(history, "cnn_basica")
    plot_training_history(history, "cnn_basica")
    
    # Crear labels_map con las clases reales del generador
    actual_labels_map = {
        str(i): name for i, name in enumerate(train_gen.class_indices.keys())
    }
    
    print("\n[INFO] Evaluando en conjunto de test...")
    results = evaluate_model(model, test_gen, actual_labels_map, "cnn_basica")
    
    # Guardar modelo final
    final_model_path = METADATA_DIR.parent.parent / "3_models" / "cnn_basica_final.h5"
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_model_path))
    print(f"[INFO] Modelo final guardado en {final_model_path}")
    
    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    print("\n" + "=" * 60)
    print("[SUCCESS] ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Accuracy en test: {results['accuracy']*100:.2f}%")
    print(f"Mejor val_accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
