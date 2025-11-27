"""
Utilidades para evaluar modelos entrenados.

Este módulo genera reportes de clasificación, matrices de confusión
y gráficas del entrenamiento. Creo que es importante para entender
qué tan bien funciona nuestro modelo y dónde se equivoca.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Importar rutas de configuración
try:
    from ..utils.config import RESULTS_DIR
except ImportError:
    from utils.config import RESULTS_DIR


def evaluate_model(model, test_generator, labels_map, model_name):
    """
    Evalúa el modelo con el conjunto de test y genera reportes.
    
    Entiendo que esta función:
    1. Hace predicciones con el modelo
    2. Compara con las etiquetas reales
    3. Calcula métricas como accuracy, precision, recall
    4. Genera una matriz de confusión
    
    Args:
        model: Modelo Keras ya entrenado
        test_generator: Generador con datos de test
        labels_map: Diccionario {índice: nombre_persona}
        model_name: Nombre para guardar los archivos
        
    Returns:
        Diccionario con accuracy, reporte y matriz de confusión
    """
    print("[INFO] Evaluando modelo...")
    
    # Hacer predicciones (el modelo devuelve probabilidades para cada clase)
    predictions = model.predict(test_generator, verbose=1)
    
    # Convertir probabilidades a clase predicha (la de mayor probabilidad)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Obtener las clases verdaderas del generador
    true_classes = test_generator.classes
    
    # Lista de nombres de personas en orden
    class_names = [labels_map[str(i)] for i in range(len(labels_map))]
    
    # Calcular accuracy (porcentaje de aciertos)
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"\n[INFO] Accuracy en test: {accuracy*100:.2f}%")
    
    # Generar reporte detallado con precision, recall, f1-score por clase
    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        output_dict=True  # Para poder guardarlo como JSON
    )
    
    # Guardar reporte en JSON
    report_path = RESULTS_DIR / "reportes" / f"{model_name}_classification_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Reporte guardado en {report_path}")
    
    # Generar y guardar matriz de confusión
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    _plot_confusion_matrix(conf_matrix, class_names, model_name)
    
    return {
        'accuracy': float(accuracy),
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist()
    }


def _plot_confusion_matrix(conf_matrix, class_names, model_name):
    """
    Genera y guarda una imagen de la matriz de confusión.
    
    La matriz de confusión me ayuda a ver:
    - En la diagonal: predicciones correctas
    - Fuera de la diagonal: errores (confusiones entre personas)
    
    Args:
        conf_matrix: Matriz de confusión de numpy
        class_names: Lista con nombres de las personas
        model_name: Nombre para el archivo
    """
    plt.figure(figsize=(12, 10))
    
    # Usar seaborn para un heatmap más bonito
    sns.heatmap(
        conf_matrix,
        annot=True,           # Mostrar números en cada celda
        fmt='d',              # Formato entero
        cmap='Blues',         # Colores azules
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Cantidad'}
    )
    
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    plt.xticks(rotation=45, ha='right')  # Rotar etiquetas para que se lean mejor
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Guardar imagen
    save_path = RESULTS_DIR / "confusion_matrices" / f"{model_name}_confusion_matrix.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Matriz de confusión guardada en {save_path}")


def plot_training_history(history, model_name):
    """
    Genera gráficas de accuracy y loss durante el entrenamiento.
    
    Estas gráficas me ayudan a ver:
    - Si el modelo está aprendiendo (loss baja, accuracy sube)
    - Si hay overfitting (train mejora pero val empeora)
    
    Args:
        history: Objeto History retornado por model.fit()
        model_name: Nombre para el archivo
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfica de Accuracy
    axes[0].plot(history.history['accuracy'], label='Entrenamiento', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validación', linewidth=2)
    axes[0].set_title('Accuracy durante el entrenamiento')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfica de Loss
    axes[1].plot(history.history['loss'], label='Entrenamiento', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validación', linewidth=2)
    axes[1].set_title('Loss durante el entrenamiento')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar imagen
    save_path = RESULTS_DIR / "accuracy_loss_plots" / f"{model_name}_training_history.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Gráficas de entrenamiento guardadas en {save_path}")
