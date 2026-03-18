"""
Validador de CSVs para Arduino Movement Recognition
====================================================
Este script verifica que tus archivos CSV tengan el formato correcto.

Uso:
    python validate_data.py
"""

import pandas as pd
from pathlib import Path
import sys

# Configuración
DATA_DIR = Path("data")
EXPECTED_FILES = ["ANDAR.csv", "SENTARSE.csv", "ACOSTARSE.csv", "CAER.csv"]
EXPECTED_ROWS = 10000
EXPECTED_COLS = 7
REQUIRED_COLUMNS = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']

def check_file_exists():
    """Verifica que existan todos los archivos"""
    print("=" * 70)
    print("1. VERIFICANDO ARCHIVOS")
    print("=" * 70)
    
    if not DATA_DIR.exists():
        print(f"❌ ERROR: La carpeta '{DATA_DIR}' no existe!")
        print(f"\nCrea la carpeta con: mkdir {DATA_DIR}")
        return False
    
    all_exist = True
    for filename in EXPECTED_FILES:
        file_path = DATA_DIR / filename
        if file_path.exists():
            print(f"✅ {filename} - Encontrado")
        else:
            print(f"❌ {filename} - NO ENCONTRADO")
            all_exist = False
    
    print("=" * 70 + "\n")
    return all_exist


def validate_csv(filename):
    """Valida un archivo CSV individual"""
    file_path = DATA_DIR / filename
    
    print(f"\nValidando: {filename}")
    print("-" * 70)
    
    try:
        df = pd.read_csv(file_path)
        
        errors = []
        warnings = []
        
        # 1. Verificar número de columnas
        if df.shape[1] != EXPECTED_COLS:
            if df.shape[1] == 6:
                warnings.append(f"⚠️  Tiene {df.shape[1]} columnas (6). Debería tener 7 (con evento)")
            else:
                errors.append(f"❌ Número de columnas incorrecto: {df.shape[1]} (esperado: {EXPECTED_COLS})")
        else:
            print(f"✅ Columnas: {df.shape[1]}")
        
        # 2. Verificar número de filas
        if df.shape[0] != EXPECTED_ROWS:
            errors.append(f"❌ Número de filas incorrecto: {df.shape[0]} (esperado: {EXPECTED_ROWS})")
        else:
            print(f"✅ Filas: {df.shape[0]}")
        
        # 3. Verificar nombres de columnas (si tiene 7 columnas)
        if df.shape[1] == EXPECTED_COLS:
            # Verificar si las columnas de sensores existen
            sensor_cols = [col for col in REQUIRED_COLUMNS if col in df.columns]
            if len(sensor_cols) == len(REQUIRED_COLUMNS):
                print(f"✅ Nombres de columnas correctos")
            else:
                warnings.append(f"⚠️  No se encontraron nombres de columna estándar. Se usará indexación por posición.")
        
        # 4. Verificar que no haya valores nulos
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            errors.append(f"❌ Valores nulos encontrados:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"     - Columna '{col}': {count} valores nulos")
        else:
            print(f"✅ Sin valores nulos")
        
        # 5. Verificar rangos de valores (asumiendo columnas 2-7 son sensores)
        if df.shape[1] >= 6:
            # Tomar las últimas 6 columnas como sensores
            sensor_df = df.iloc[:, -6:]
            
            # Aceleración (primeras 3 columnas de sensores)
            acc_cols = sensor_df.iloc[:, 0:3]
            acc_min, acc_max = acc_cols.min().min(), acc_cols.max().max()
            
            if acc_min < -5 or acc_max > 5:
                warnings.append(f"⚠️  Aceleración fuera de rango típico: [{acc_min:.2f}, {acc_max:.2f}]")
            else:
                print(f"✅ Rango aceleración: [{acc_min:.2f}, {acc_max:.2f}]")
            
            # Giroscopio (últimas 3 columnas de sensores)
            gyro_cols = sensor_df.iloc[:, 3:6]
            gyro_min, gyro_max = gyro_cols.min().min(), gyro_cols.max().max()
            
            if gyro_min < -2500 or gyro_max > 2500:
                warnings.append(f"⚠️  Giroscopio fuera de rango típico: [{gyro_min:.2f}, {gyro_max:.2f}]")
            else:
                print(f"✅ Rango giroscopio: [{gyro_min:.2f}, {gyro_max:.2f}]")
        
        # 6. Verificar que sean valores numéricos
        if df.shape[1] >= 6:
            sensor_df = df.iloc[:, -6:]
            non_numeric = []
            for col in sensor_df.columns:
                if not pd.api.types.is_numeric_dtype(sensor_df[col]):
                    non_numeric.append(col)
            
            if non_numeric:
                errors.append(f"❌ Columnas no numéricas: {non_numeric}")
            else:
                print(f"✅ Todos los valores son numéricos")
        
        # Imprimir warnings
        if warnings:
            print("\n⚠️  ADVERTENCIAS:")
            for warning in warnings:
                print(f"  {warning}")
        
        # Imprimir errores
        if errors:
            print("\n❌ ERRORES:")
            for error in errors:
                print(f"  {error}")
            return False
        else:
            if not warnings:
                print("\n✅ ARCHIVO VÁLIDO")
            else:
                print("\n⚠️  ARCHIVO VÁLIDO CON ADVERTENCIAS")
            return True
            
    except Exception as e:
        print(f"\n❌ ERROR al leer el archivo: {e}")
        return False


def main():
    """Función principal"""
    print("\n")
    print("=" * 70)
    print(" VALIDADOR DE CSVs - ARDUINO MOVEMENT RECOGNITION")
    print("=" * 70)
    print("\n")
    
    # Verificar que existan los archivos
    if not check_file_exists():
        print("❌ Faltan archivos. Por favor, coloca todos los CSVs en la carpeta 'data/'")
        sys.exit(1)
    
    # Validar cada archivo
    print("=" * 70)
    print("2. VALIDANDO CONTENIDO")
    print("=" * 70)
    
    all_valid = True
    for filename in EXPECTED_FILES:
        if not validate_csv(filename):
            all_valid = False
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    
    if all_valid:
        print("\n✅ TODOS LOS ARCHIVOS SON VÁLIDOS")
        print("\nPuedes ejecutar el entrenamiento con:")
        print("  python train_model.py")
    else:
        print("\n❌ ALGUNOS ARCHIVOS TIENEN ERRORES")
        print("\nPor favor, corrige los errores antes de entrenar el modelo.")
    
    print("\n" + "=" * 70 + "\n")
    
    return all_valid


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
