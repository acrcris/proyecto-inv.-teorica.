# Análisis de Alpha Crítico (α_c)

Esta carpeta contiene todos los scripts y datos relacionados con el análisis de criticalidad mediante el parámetro α_c en imágenes MNIST usando dinámica de Kuramoto.

## 📁 Estructura

### Scripts Principales

#### 🔬 Análisis con SQLite (Recomendado)
- **`analizar_con_sqlite.py`** - Script robusto con base de datos SQLite
  - Persistencia con ACID transactions
  - Identificación única vía `dataset_idx` (0-59,999)
  - Auto-commit cada 100 imágenes
  - Retry automático con backoff exponencial
  - Capacidad de reanudar proceso interrumpido

#### 📊 Análisis Incremental (Alternativo)
- **`analizar_incremental_clases_impares.py`** - Análisis por lotes con checkpoints PyTorch
- **`analizar_imagen_individual.py`** - Análisis de una sola imagen
- **`test_single_image_referencia.py`** - Test de referencia para validación

### Scripts de Búsqueda de α_c
- **`encontrar_alpha_critico.py`** - Encuentra α_c óptimo para conjunto de imágenes
- **`encontrar_alpha_critico_clase3.py`** - Versión especializada para clase 3

### Scripts de Visualización
- **`visualizar_alpha_critico.py`** - Visualización general de distribuciones α_c
- **`visualizar_clase_unica.py`** - Análisis visual de una clase específica
- **`visualizar_puntos_criticos.py`** - Gráficas de puntos críticos
- **`graficar_distribucion_criticalidad.py`** - Distribución de α_c con histogramas

### Scripts de Generación de Gráficas
- **`generar_graficas_checkpoints.py`** - Genera gráficas desde checkpoints
- **`generar_graficas_por_clase.py`** - Gráficas separadas por clase MNIST
- **`generar_graficas_pdf.py`** - Exporta gráficas a formato PDF
- **`graficar_distribuciones_checkpoints_impares.py`** - Distribuciones para clases impares

### Scripts de Verificación
- **`verificar_consistencia.py`** - Verifica consistencia de resultados
- **`verificar_scripts_alpha_c.py`** - Validación de scripts de análisis

## 📊 Datos

### Base de Datos SQLite
- **`resultados_criticalidad.db`** - Base de datos con resultados de α_c
  - Tabla `resultados`: (id, dataset_idx, clase, alpha_c, timestamp, img_hash)
  - Tabla `metadata`: Configuración del análisis
  - Índices en `clase` y `alpha_c` para queries rápidas

### Logs
- **`analisis_sqlite.log`** - Log del proceso actual de análisis

### Gráficas
- **`graficas_alpha_c/`** - Directorio con visualizaciones generadas
  - `distribucion_alpha_c.png`
  - `comparacion_curvas_R_alpha.png`
  - `curvas_R_alpha_por_clase.png`

## 📚 Documentación
- **`alpha_c_explicacion.md`** - Explicación detallada del concepto de α_c y su cálculo

## 🚀 Uso

### Ejecutar análisis SQLite (Recomendado)
```bash
# Lanzar en background
nohup python analizar_con_sqlite.py > analisis_sqlite.log 2>&1 &

# Monitorear progreso
tail -f analisis_sqlite.log

# Consultar resultados (modo read-only)
sqlite3 -readonly resultados_criticalidad.db "SELECT COUNT(*) FROM resultados;"
sqlite3 -readonly resultados_criticalidad.db "SELECT clase, COUNT(*) FROM resultados GROUP BY clase;"
```

### Visualizar resultados
```bash
# Generar gráficas desde la base de datos
python visualizar_alpha_critico.py

# Generar gráficas por clase
python generar_graficas_por_clase.py
```

## ⚙️ Configuración

### Parámetros del análisis
- **Rango de α**: [0.0, 0.1] con 201 puntos
- **Dataset**: MNIST train set (60,000 imágenes)
- **GPU**: Metal Performance Shaders (MPS) en macOS
- **Tiempo estimado**: ~17-18 seg/imagen (~283 horas totales)

## 📝 Notas

- ⚠️ **NO abrir la base de datos con DB Browser mientras el script está corriendo** (causa bloqueos)
- ✅ Usar `-readonly` al consultar la DB durante ejecución
- 🔄 El script puede retomar desde donde se quedó si se interrumpe
- 💾 Auto-commit cada 100 imágenes para evitar pérdida de datos

## 🔗 Referencias
Ver `alpha_c_explicacion.md` para detalles teóricos sobre el parámetro de criticalidad.
