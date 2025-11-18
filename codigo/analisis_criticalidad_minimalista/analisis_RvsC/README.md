# Análisis R vs C

Este directorio contiene herramientas para calcular el parámetro de orden R en función del parámetro de acoplamiento C.

## Archivos

- `calculo_R_vs_C_autocontenido.ipynb`: Notebook Jupyter autocontenido para Google Colab ⭐ **RECOMENDADO**
- `calcular_r_vs_c.py`: Script Python standalone (alternativa local)
- `REPRODUCIBILIDAD.md`: Guía completa de reproducibilidad 🔐

## Uso Rápido

### Opción 1: Google Colab ⭐ **RECOMENDADO**

1. Subir `calculo_R_vs_C_autocontenido.ipynb` a Google Colab
2. Configurar GPU: `Entorno de ejecución → Cambiar tipo → GPU (T4)`
3. Ejecutar todas las celdas: `Entorno → Ejecutar todo`
4. Descargar resultados automáticamente al finalizar

**Ventajas:**
- GPU gratuita (T4 o A100)
- Completamente autocontenido (no requiere instalaciones)
- Reproducibilidad garantizada con semillas fijas
- Tiempo estimado: **~3-5 horas** (60,000 imágenes)

### Opción 2: Ejecución Local

```bash
# Activar entorno virtual
cd /Users/acrcr/Documents/Unal-2025-2/IntroInvTeorica/proyecto/codigo
source .venv/bin/activate

# Ejecutar script
cd analisis_criticalidad_minimalista/analisis_RvsC
python calcular_r_vs_c.py
```

**Tiempo estimado (Apple M3):** 5-7 horas

## Estructura de la Base de Datos

Base de datos SQLite similar a `analisis_alpha_c/resultados_c_critical/mnist_c_critical.db`:

- **Tablas**: `clase_0` a `clase_9` (una por dígito MNIST)
- **Columnas**:
  - `id`: INTEGER PRIMARY KEY AUTOINCREMENT
  - `imagen_idx`: INTEGER NOT NULL UNIQUE (índice en dataset MNIST)
  - `clase`: INTEGER NOT NULL (0-9)
  - `c_values`: BLOB (array numpy de 50 valores C pickleados)
  - `r_values`: BLOB (array numpy de 50 valores R correspondientes)
  - `c_critico`: REAL (valor C donde R ≈ 0.5)
  - `seed_usado`: INTEGER NOT NULL (= 42 + imagen_idx)
  - `timestamp`: TEXT (ISO 8601 format)

- **Tabla metadata**: Almacena configuración (SEED, c_min, c_max, n_c_points, t_steps, device, etc.)

### 🔐 Reproducibilidad Garantizada

Cada imagen usa semilla **determinista**:
```python
seed_imagen = 42 + imagen_idx
```

Ver `REPRODUCIBILIDAD.md` para detalles completos.

## Configuración

- **Rango C**: [0.0, 0.4]
- **Puntos C**: 50 (espaciado: 0.0082)
- **T_STEPS**: 30
- **Checkpoints**: cada 50 imágenes

## Resultados

- Base de datos: `resultados_R_vs_C/mnist_R_vs_C.db`
- Tamaño estimado: ~500-800 MB
- Tiempo estimado (M3): 5-7 horas
