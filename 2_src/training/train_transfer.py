"""
Script para entrenar modelo con Transfer Learning.

Este es el script principal de entrenamiento del proyecto.
Usa MobileNetV2 o EfficientNet pre-entrenados en ImageNet y
los adapta para reconocer las personas de nuestro dataset.
"""

import sys
import json
import argparse
from pathlib import Path

# Agregar paths del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "2_src"))

from models.transfer_learning import (
    create_mobilenetv2_model, 
    create_efficientnet_model,
    compile_model
)
from utils.augmentation import create_data_generators
from utils.config import (
    SPLITS_DIR, METADATA_DIR, BATCH_SIZE, EPOCHS, 
    LEARNING_RATE, IMAGE_SIZE
)
from training.utils_training import create_callbacks, save_model_history
from utils.evaluation import evaluate_model, plot_training_history


def parse_arguments():
    """
    Parsea los argumentos de línea de comandos.
    
    Esto permite ejecutar el script con diferentes configuraciones
    sin modificar el código. Por ejemplo:
        python train_transfer.py --model mobilenet --learning-rate 0.0001
    """
    parser = argparse.ArgumentParser(
        description='Entrenar modelo con Transfer Learning'
    )
    parser.add_argument(
        '--model', type=str, choices=['mobilenet', 'efficientnet'],
        default='mobilenet',
        help='Modelo base: mobilenet (rápido) o efficientnet (preciso)'
    )
    parser.add_argument(
        '--freeze-base', action='store_true',
        help='Congelar la base (solo entrenar capas nuevas)'
    )
    parser.add_argument(
        '--fine-tune-layers', type=int, default=10,
        help='Número de capas finales a fine-tunear (default: 10)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=LEARNING_RATE,
        help=f'Tasa de aprendizaje (default: {LEARNING_RATE})'
    )
    return parser.parse_args()


def main():
    """Función principal de entrenamiento."""
    
    args = parse_arguments()
    
    print("=" * 60)
    print(f"ENTRENAMIENTO CON TRANSFER LEARNING ({args.model.upper()})")
    print("=" * 60)
    
    # =========================================================================
    # PASO 1: Cargar configuración
    # =========================================================================
    labels_map_path = METADATA_DIR / "labels_map.json"
    with open(labels_map_path, 'r', encoding='utf-8') as f:
        labels_map = json.load(f)
    
    print(f"[INFO] Número de clases: {len(labels_map)}")
    
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
    
    print(f"[INFO] Clases encontradas: {train_gen.class_indices}")
    print(f"[INFO] Imágenes de entrenamiento: {train_gen.samples}")
    print(f"[INFO] Imágenes de validación: {val_gen.samples}")
    print(f"[INFO] Imágenes de test: {test_gen.samples}")
    
    actual_num_classes = len(train_gen.class_indices)
    print(f"[INFO] Número real de clases: {actual_num_classes}")
    
    # =========================================================================
    # PASO 3: Crear modelo según la opción elegida
    # =========================================================================
    model_name = f"{args.model}_transfer"
    print(f"\n[INFO] Creando modelo {args.model}...")
    
    # Seleccionar función de creación según el modelo elegido
    if args.model == 'mobilenet':
        model = create_mobilenetv2_model(
            input_shape=(*IMAGE_SIZE, 3),
            num_classes=actual_num_classes,
            freeze_base=args.freeze_base,
            fine_tune_layers=args.fine_tune_layers
        )
    else:
        model = create_efficientnet_model(
            input_shape=(*IMAGE_SIZE, 3),
            num_classes=actual_num_classes,
            freeze_base=args.freeze_base,
            fine_tune_layers=args.fine_tune_layers
        )
    
    model = compile_model(model, learning_rate=args.learning_rate)
    model.summary()
    
    # =========================================================================
    # PASO 4: Entrenar
    # =========================================================================
    callbacks = create_callbacks(model_name, monitor='val_accuracy')
    
    print("\n[INFO] Iniciando entrenamiento...")
    print(f"[INFO] Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"[INFO] Learning rate: {args.learning_rate}")
    print(f"[INFO] Freeze base: {args.freeze_base}")
    if not args.freeze_base:
        print(f"[INFO] Fine-tune layers: {args.fine_tune_layers}")
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # =========================================================================
    # PASO 5: Guardar y evaluar
    # =========================================================================
    save_model_history(history, model_name)
    plot_training_history(history, model_name)
    
    actual_labels_map = {
        str(i): name for i, name in enumerate(train_gen.class_indices.keys())
    }
    
    print("\n[INFO] Evaluando en conjunto de test...")
    results = evaluate_model(model, test_gen, actual_labels_map, model_name)
    
    # Guardar modelo final
    final_model_path = METADATA_DIR.parent.parent / "3_models" / f"{model_name}_final.h5"
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_model_path))
    print(f"[INFO] Modelo final guardado en {final_model_path}")
    
    # =========================================================================
    # RESUMEN
    # =========================================================================
    print("\n" + "=" * 60)
    print("[SUCCESS] ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Accuracy en test: {results['accuracy']*100:.2f}%")
    print(f"Mejor val_accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
