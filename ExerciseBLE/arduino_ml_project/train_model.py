"""
Arduino Movement Recognition - Training Script
===============================================
Entrena un modelo de red neuronal para clasificar 4 movimientos usando datos IMU.

Archivos requeridos en carpeta 'data/':
- ANDAR.csv
- SENTARSE.csv  
- ACOSTARSE.csv
- CAER.csv

Uso:
    python train_model.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow warnings

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

# Movimientos (en el orden de los archivos CSV)
MOVEMENTS = [
    "ANDAR",
    "SENTARSE",
    "ACOSTARSE",  # tumbarse
    "CAER"        # caerse
]

# Nombres para mostrar
MOVEMENT_NAMES = [
    "Andar",
    "Sentarse",
    "Acostarse",
    "Caer"
]

# Parámetros del dataset
SAMPLES_PER_MOVEMENT = 200
NUM_MOVEMENTS = len(MOVEMENTS)

# Parámetros del modelo
SEED = 1337
LEARNING_RATE = 0.001
EPOCHS = 300
BATCH_SIZE = 16
EARLY_STOPPING_PATIENCE = 50

# ============================================================================
# FUNCIONES
# ============================================================================

def setup_gpu():
    """Configura GPU si está disponible"""
    print("=" * 70)
    print("CONFIGURACIÓN GPU")
    print("=" * 70)
    
    print(f"TensorFlow version: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs disponibles: {len(gpus)}")
    
    if gpus:
        print("\nGPU detectada:")
        for gpu in gpus:
            print(f"  {gpu}")
        
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("\n✅ GPU configurada correctamente (memory growth habilitado)")
        except RuntimeError as e:
            print(f"\n⚠️  Error configurando GPU: {e}")
    else:
        print("\n⚠️  No se detectó GPU. Se usará CPU (más lento)")
    
    print("=" * 70 + "\n")
    return len(gpus) > 0


def load_and_prepare_data():
    """Carga y prepara los datos de los CSVs"""
    print("=" * 70)
    print("CARGANDO DATOS")
    print("=" * 70)
    
    # Verificar que exista el directorio
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"\n❌ ERROR: Carpeta '{DATA_DIR}' no encontrada!\n"
            f"Por favor, crea la carpeta '{DATA_DIR}' y coloca los CSVs allí."
        )
    
    # One-hot encoding para las salidas
    one_hot_encoded = np.eye(NUM_MOVEMENTS)
    
    inputs = []
    outputs = []
    
    # Cargar cada archivo CSV
    for movement_index, movement in enumerate(MOVEMENTS):
        print(f"\nProcesando: {movement}.csv")
        
        file_path = DATA_DIR / f"{movement}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"\n❌ ERROR: Archivo no encontrado: {file_path}\n"
                f"Asegúrate de que el archivo existe en la carpeta '{DATA_DIR}'"
            )
        
        # Leer CSV
        df = pd.read_csv(file_path)
        print(f"  Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
        
        # Manejar formato CSV (6 o 7 columnas)
        required_columns = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
        
        if df.shape[1] == 7:
            print(f"  Formato detectado: 7 columnas (ignorando primera columna)")
            # Seleccionar solo columnas de sensores
            if all(col in df.columns for col in required_columns):
                df = df[required_columns]
            else:
                # Asumir orden: evento, aX, aY, aZ, gX, gY, gZ
                df = df.iloc[:, 1:7]
                df.columns = required_columns
        elif df.shape[1] == 6:
            print(f"  Formato detectado: 6 columnas")
            if not all(col in df.columns for col in required_columns):
                df.columns = required_columns
        else:
            raise ValueError(
                f"❌ ERROR: Se esperaban 6 o 7 columnas, se encontraron {df.shape[1]}"
            )
        
        # Calcular número de grabaciones
        num_recordings = int(df.shape[0] / SAMPLES_PER_MOVEMENT)
        print(f"  Grabaciones encontradas: {num_recordings}")
        
        if num_recordings == 0:
            raise ValueError(
                f"❌ ERROR: No hay suficientes datos en {movement}.csv\n"
                f"Se necesitan al menos {SAMPLES_PER_MOVEMENT} muestras"
            )
        
        # Crear tensores de entrada
        output = one_hot_encoded[movement_index]
        
        for i in range(num_recordings):
            tensor = []
            for j in range(SAMPLES_PER_MOVEMENT):
                idx = i * SAMPLES_PER_MOVEMENT + j
                
                # Normalizar datos entre 0 y 1
                tensor += [
                    (df['aX'].iloc[idx] + 4) / 8,
                    (df['aY'].iloc[idx] + 4) / 8,
                    (df['aZ'].iloc[idx] + 4) / 8,
                    (df['gX'].iloc[idx] + 2000) / 4000,
                    (df['gY'].iloc[idx] + 2000) / 4000,
                    (df['gZ'].iloc[idx] + 2000) / 4000
                ]
            
            inputs.append(tensor)
            outputs.append(output)
    
    # Convertir a arrays numpy
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    
    print("\n" + "=" * 70)
    print("DATOS CARGADOS - VALIDACIÓN")
    print("=" * 70)
    print(f"Total de grabaciones: {len(inputs)}")
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Cada input tiene {inputs.shape[1]} valores (200 samples × 6 sensores)")
    
    # VALIDACIÓN CRÍTICA
    print("\nValidación de datos:")
    
    # 1. Verificar NaN o Infinitos
    if np.any(np.isnan(inputs)):
        print("  ⚠️  ADVERTENCIA: Hay valores NaN en los datos")
    else:
        print("  ✅ Sin valores NaN")
    
    if np.any(np.isinf(inputs)):
        print("  ⚠️  ADVERTENCIA: Hay valores infinitos en los datos")
    else:
        print("  ✅ Sin valores infinitos")
    
    # 2. Verificar rangos (después de normalización deberían estar en [0,1])
    print(f"\n  Rango de valores normalizados:")
    print(f"    Mínimo: {inputs.min():.4f}")
    print(f"    Máximo: {inputs.max():.4f}")
    print(f"    Media:  {inputs.mean():.4f}")
    
    if inputs.min() < -0.1 or inputs.max() > 1.1:
        print("  ⚠️  ADVERTENCIA: Valores fuera del rango esperado [0,1]")
    else:
        print("  ✅ Valores en rango correcto [0,1]")
    
    # 3. Verificar distribución de clases
    labels = np.argmax(outputs, axis=1)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n  Distribución de clases:")
    for cls, count in zip(unique, counts):
        print(f"    Clase {cls} ({MOVEMENT_NAMES[cls]}): {count} muestras")
    
    # Verificar balance
    if len(set(counts)) == 1:
        print("  ✅ Clases perfectamente balanceadas")
    else:
        min_count, max_count = min(counts), max(counts)
        ratio = max_count / min_count
        if ratio > 1.2:
            print(f"  ⚠️  ADVERTENCIA: Desbalance entre clases (ratio: {ratio:.2f}x)")
        else:
            print(f"  ✅ Clases razonablemente balanceadas (ratio: {ratio:.2f}x)")
    
    print("=" * 70 + "\n")
    
    return inputs, outputs


def analyze_data_samples(inputs, outputs):
    """Analiza muestras de datos para debugging"""
    print("=" * 70)
    print("ANÁLISIS DE MUESTRAS DE DATOS")
    print("=" * 70)
    
    # Tomar primera muestra de cada clase
    labels = np.argmax(outputs, axis=1)
    
    for cls in range(NUM_MOVEMENTS):
        # Encontrar primera muestra de esta clase
        idx = np.where(labels == cls)[0][0]
        sample = inputs[idx]
        
        # Los primeros 600 valores son aceleración (3 ejes × 200 samples)
        # Los siguientes 600 valores son giroscopio (3 ejes × 200 samples)
        
        # Reshapear para análisis (200 samples × 6 sensors)
        sample_reshaped = sample.reshape(200, 6)
        
        acc_x = sample_reshaped[:, 0]  # Aceleración X
        acc_y = sample_reshaped[:, 1]  # Aceleración Y
        acc_z = sample_reshaped[:, 2]  # Aceleración Z
        gyro_x = sample_reshaped[:, 3]  # Giroscopio X
        gyro_y = sample_reshaped[:, 4]  # Giroscopio Y
        gyro_z = sample_reshaped[:, 5]  # Giroscopio Z
        
        print(f"\nClase {cls} - {MOVEMENT_NAMES[cls]} (muestra #{idx}):")
        print(f"  Aceleración X: min={acc_x.min():.3f}, max={acc_x.max():.3f}, mean={acc_x.mean():.3f}")
        print(f"  Aceleración Y: min={acc_y.min():.3f}, max={acc_y.max():.3f}, mean={acc_y.mean():.3f}")
        print(f"  Aceleración Z: min={acc_z.min():.3f}, max={acc_z.max():.3f}, mean={acc_z.mean():.3f}")
        print(f"  Giroscopio  X: min={gyro_x.min():.3f}, max={gyro_x.max():.3f}, mean={gyro_x.mean():.3f}")
        print(f"  Giroscopio  Y: min={gyro_y.min():.3f}, max={gyro_y.max():.3f}, mean={gyro_y.mean():.3f}")
        print(f"  Giroscopio  Z: min={gyro_z.min():.3f}, max={gyro_z.max():.3f}, mean={gyro_z.mean():.3f}")
        
        # Verificar si los valores tienen varianza (no son constantes)
        if acc_x.std() < 0.01 and acc_y.std() < 0.01 and acc_z.std() < 0.01:
            print(f"  ⚠️  ADVERTENCIA: Aceleración casi constante (poca variación)")
        
        if gyro_x.std() < 0.01 and gyro_y.std() < 0.01 and gyro_z.std() < 0.01:
            print(f"  ⚠️  ADVERTENCIA: Giroscopio casi constante (poca variación)")
    
    print("=" * 70 + "\n")


def split_data(inputs, outputs):
    from sklearn.model_selection import train_test_split
    
    print("=" * 70)
    print("DIVIDIENDO DATOS (CON ESTRATIFICACIÓN)")
    print("=" * 70)
    
    num_inputs = len(inputs)
    
    # Convertir one-hot a labels para estratificación
    labels = np.argmax(outputs, axis=1)
    
    # Mostrar distribución original
    print("\nDistribución original por clase:")
    unique, counts = np.unique(labels, return_counts=True)
    for i, (cls, count) in enumerate(zip(unique, counts)):
        print(f"  {MOVEMENT_NAMES[cls]}: {count} muestras ({count/num_inputs*100:.1f}%)")
    
    # Primera división: 80% (train+val) / 20% (test) - ESTRATIFICADO
    inputs_temp, inputs_test, outputs_temp, outputs_test = train_test_split(
        inputs, outputs, 
        test_size=0.2, 
        stratify=labels,
        random_state=SEED
    )
    
    # Segunda división: 75% train / 25% val del 80% anterior - ESTRATIFICADO
    # Esto da 60% train / 20% val del total
    labels_temp = np.argmax(outputs_temp, axis=1)
    inputs_train, inputs_validate, outputs_train, outputs_validate = train_test_split(
        inputs_temp, outputs_temp,
        test_size=0.25,  # 25% de 80% = 20% del total
        stratify=labels_temp,
        random_state=SEED
    )
    
    print("\n" + "=" * 70)
    print("RESUMEN DE DIVISIÓN")
    print("=" * 70)
    print(f"  Entrenamiento:  {len(inputs_train):3d} muestras ({len(inputs_train)/num_inputs*100:.1f}%)")
    print(f"  Validación:     {len(inputs_validate):3d} muestras ({len(inputs_validate)/num_inputs*100:.1f}%)")
    print(f"  Test:           {len(inputs_test):3d} muestras ({len(inputs_test)/num_inputs*100:.1f}%)")
    
    # Verificar distribución por clase en cada conjunto
    print("\nDistribución por clase en cada conjunto:")
    for set_name, set_outputs in [("Train", outputs_train), 
                                    ("Val", outputs_validate), 
                                    ("Test", outputs_test)]:
        set_labels = np.argmax(set_outputs, axis=1)
        unique, counts = np.unique(set_labels, return_counts=True)
        print(f"\n  {set_name}:")
        for cls, count in zip(unique, counts):
            print(f"    {MOVEMENT_NAMES[cls]}: {count} muestras")
    
    print("=" * 70 + "\n")
    
    return (inputs_train, outputs_train, 
            inputs_validate, outputs_validate,
            inputs_test, outputs_test)


def build_model():
    """Construye el modelo de red neuronal"""
    print("=" * 70)
    print("CONSTRUYENDO MODELO")
    print("=" * 70)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(1200,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(NUM_MOVEMENTS, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    total_params = model.count_params()
    print(f"\nParámetros totales: {total_params:,}")
    print(f"Tamaño estimado (float32): {total_params * 4 / 1024:.2f} KB")
    print(f"Tamaño estimado (int8):    {total_params / 1024:.2f} KB")
    print("=" * 70 + "\n")
    
    return model


def train_model(model, inputs_train, outputs_train, inputs_validate, outputs_validate):
    """Entrena el modelo"""
    print("=" * 70)
    print("ENTRENANDO MODELO")
    print("=" * 70)
    print(f"Épocas: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print("=" * 70 + "\n")
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(OUTPUT_DIR / 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
    
    # ReduceLROnPlateau - reduce learning rate si no mejora
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        min_lr=1e-7,
        verbose=1
    )
    
    # Entrenar
    history = model.fit(
        inputs_train,
        outputs_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(inputs_validate, outputs_validate),
        callbacks=[early_stopping, checkpoint, reduce_lr],
        verbose=1  # Mostrar progreso detallado
    )
    
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    
    # Mostrar información del entrenamiento
    final_epoch = len(history.history['loss'])
    print(f"\nÉpocas completadas: {final_epoch} de {EPOCHS}")
    print(f"Mejor val_accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Mejor val_loss: {min(history.history['val_loss']):.4f}")
    print(f"Train accuracy final: {history.history['accuracy'][-1]:.4f}")
    print(f"Val accuracy final: {history.history['val_accuracy'][-1]:.4f}")
    
    # Detectar overfitting
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    gap = train_acc - val_acc
    
    if gap > 0.15:
        print(f"\n⚠️  ADVERTENCIA: Posible overfitting detectado")
        print(f"   Gap train-val accuracy: {gap:.4f}")
    
    print("=" * 70 + "\n")
    
    return history


def plot_training_history(history):
    """Grafica el historial de entrenamiento"""
    print("Generando gráficas de entrenamiento...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'training_history.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Guardado: {output_path}")
    plt.close()


def evaluate_model(model, inputs_test, outputs_test):
    """Evalúa el modelo en el conjunto de test"""
    print("\n" + "=" * 70)
    print("EVALUACIÓN EN CONJUNTO DE TEST")
    print("=" * 70)
    
    test_loss, test_accuracy = model.evaluate(inputs_test, outputs_test, verbose=0)
    
    print(f"Loss:     {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy*100:.2f}%")
    print("=" * 70 + "\n")
    
    return test_loss, test_accuracy


def plot_confusion_matrix(model, inputs_test, outputs_test):
    """Genera y guarda la matriz de confusión"""
    print("Generando matriz de confusión...")
    
    # Predicciones
    predictions = model.predict(inputs_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(outputs_test, axis=1)
    
    # Matriz de confusión
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=MOVEMENT_NAMES,
                yticklabels=MOVEMENT_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Guardado: {output_path}")
    plt.close()
    
    # Classification report
    print("\n" + "=" * 70)
    print("REPORTE DE CLASIFICACIÓN")
    print("=" * 70)
    print(classification_report(true_classes, predicted_classes,
                              target_names=MOVEMENT_NAMES, digits=4))
    print("=" * 70 + "\n")


def convert_to_tflite(model, inputs_train):
    """Convierte el modelo a TensorFlow Lite con cuantización INT8"""
    print("=" * 70)
    print("CONVERSIÓN A TENSORFLOW LITE")
    print("=" * 70)
    
    # Sin cuantización
    print("\n1. Modelo sin cuantización...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    tflite_path = OUTPUT_DIR / "movement_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    basic_size = os.path.getsize(tflite_path)
    print(f"   Tamaño: {basic_size:,} bytes ({basic_size/1024:.2f} KB)")
    print(f"   Guardado: {tflite_path}")
    
    # Con cuantización INT8
    print("\n2. Modelo con cuantización INT8...")
    converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Dataset representativo
    def representative_dataset():
        for i in range(min(100, len(inputs_train))):
            yield [inputs_train[i:i+1].astype(np.float32)]
    
    converter_quant.representative_dataset = representative_dataset
    converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_quant.inference_input_type = tf.int8
    converter_quant.inference_output_type = tf.int8
    
    tflite_model_quant = converter_quant.convert()
    
    tflite_quant_path = OUTPUT_DIR / "movement_model_quantized.tflite"
    with open(tflite_quant_path, "wb") as f:
        f.write(tflite_model_quant)
    
    quant_size = os.path.getsize(tflite_quant_path)
    print(f"   Tamaño: {quant_size:,} bytes ({quant_size/1024:.2f} KB)")
    print(f"   Guardado: {tflite_quant_path}")
    print(f"\n   Compresión: {basic_size/quant_size:.2f}x")
    
    # Estimación de memoria para Arduino
    print("\n" + "=" * 70)
    print("ESTIMACIÓN DE MEMORIA PARA ARDUINO")
    print("=" * 70)
    print(f"Modelo (Flash):           {quant_size/1024:>6.0f} KB")
    print(f"TFLite Runtime (Flash):   ~  125 KB")
    print(f"Arduino Code (Flash):     ~   35 KB")
    print("-" * 70)
    total_flash = quant_size/1024 + 125 + 35
    print(f"TOTAL Flash estimado:     {total_flash:>6.0f} KB")
    print(f"Flash disponible:          1024 KB")
    print(f"Flash restante:           {1024 - total_flash:>6.0f} KB")
    print("=" * 70)
    
    if total_flash < 1024:
        print("\n✅ El modelo cabe en la memoria Flash de Arduino!\n")
    else:
        print("\n⚠️  ADVERTENCIA: El modelo podría ser demasiado grande!\n")
    
    return tflite_quant_path


def generate_arduino_header(tflite_quant_path):
    """Genera el archivo header para Arduino"""
    print("\n" + "=" * 70)
    print("GENERANDO HEADER PARA ARDUINO")
    print("=" * 70)
    
    model_h_path = OUTPUT_DIR / "model.h"
    
    with open(model_h_path, 'w') as f:
        f.write("const unsigned char model[] = {\n")
        
        # Leer modelo cuantizado
        with open(tflite_quant_path, 'rb') as model_file:
            model_data = model_file.read()
        
        # Convertir a array hexadecimal
        hex_values = [f'0x{byte:02x}' for byte in model_data]
        
        # Escribir en líneas de 12 bytes
        bytes_per_line = 12
        for i in range(0, len(hex_values), bytes_per_line):
            line = ', '.join(hex_values[i:i+bytes_per_line])
            if i + bytes_per_line < len(hex_values):
                f.write(f"  {line},\n")
            else:
                f.write(f"  {line}\n")
        
        f.write("};\n")
        f.write(f"const unsigned int model_len = {len(model_data)};\n")
    
    header_size = os.path.getsize(model_h_path)
    
    print(f"Archivo: {model_h_path}")
    print(f"Tamaño: {header_size:,} bytes")
    print("\n" + "=" * 70)
    print("INSTRUCCIONES PARA ARDUINO")
    print("=" * 70)
    print("1. Copia 'model.h' a tu carpeta de sketch de Arduino")
    print("2. Incluye en tu código: #include \"model.h\"")
    print("3. El modelo está disponible como:")
    print("   - const unsigned char model[]")
    print("   - const unsigned int model_len")
    print("=" * 70 + "\n")


def print_summary(test_accuracy):
    """Imprime resumen final"""
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"\n📊 Movimientos detectados: {', '.join(MOVEMENT_NAMES)}")
    print(f"📁 Total de grabaciones: {len(MOVEMENTS)} × 50 = 200")
    print(f"🎯 Accuracy en test: {test_accuracy*100:.2f}%")
    print(f"\n📂 Archivos generados en '{OUTPUT_DIR}/':")
    print(f"   ✓ best_model.keras")
    print(f"   ✓ training_history.png")
    print(f"   ✓ confusion_matrix.png")
    print(f"   ✓ movement_model.tflite")
    print(f"   ✓ movement_model_quantized.tflite")
    print(f"   ✓ model.h  ← USAR ESTE EN ARDUINO")
    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Función principal"""
    print("\n")
    print("=" * 70)
    print(" ARDUINO MOVEMENT RECOGNITION - ENTRENAMIENTO DE MODELO")
    print("=" * 70)
    print("\n")
    
    # Configurar seeds
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    # Crear carpeta de salida
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. Setup GPU
    has_gpu = setup_gpu()
    
    # 2. Cargar datos
    inputs, outputs = load_and_prepare_data()
    
    # 2.5. Analizar muestras de datos (DEBUGGING)
    analyze_data_samples(inputs, outputs)
    
    # 3. Dividir datos
    data_splits = split_data(inputs, outputs)
    inputs_train, outputs_train = data_splits[0], data_splits[1]
    inputs_validate, outputs_validate = data_splits[2], data_splits[3]
    inputs_test, outputs_test = data_splits[4], data_splits[5]
    
    # 4. Construir modelo
    model = build_model()
    
    # 5. Entrenar
    history = train_model(model, inputs_train, outputs_train, 
                         inputs_validate, outputs_validate)
    
    # 6. Graficar entrenamiento
    plot_training_history(history)
    
    # 7. Evaluar
    test_loss, test_accuracy = evaluate_model(model, inputs_test, outputs_test)
    
    # 8. Matriz de confusión
    plot_confusion_matrix(model, inputs_test, outputs_test)
    
    # 9. Convertir a TFLite
    tflite_quant_path = convert_to_tflite(model, inputs_train)
    
    # 10. Generar header Arduino
    generate_arduino_header(tflite_quant_path)
    
    # 11. Resumen
    print_summary(test_accuracy)


if __name__ == "__main__":
    main()
