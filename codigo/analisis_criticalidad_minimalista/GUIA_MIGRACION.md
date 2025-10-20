# 📘 Guía de Migración: Notebook → Módulo

Esta guía te ayuda a migrar código del notebook `kuramoto_pruebas_basico.ipynb` al módulo `analisis_criticalidad_minimalista`.

---

## 🔄 Conversión de Código

### 1. Importaciones

#### Antes (Notebook)
```python
# Las funciones están definidas en celdas del notebook
# No hay importaciones
```

#### Después (Módulo)
```python
from analisis_criticalidad_minimalista import (
    KBlock, KConv2d, ModReLU,
    KuramotoMetrics, DFA, PSD, Entropia,
    Visualizador, Animaciones
)
```

---

### 2. Modelo de Kuramoto

#### Antes (Notebook)
```python
# Celda con definición de clase KBlock
kblock = KBlock(n=4, ch=4, connectivity='conv', T=100, ksize=3, 
                init_omg=0.1, c_norm=None, use_omega_c=False)
```

#### Después (Módulo)
```python
from analisis_criticalidad_minimalista import KBlock

kblock = KBlock(n=4, ch=4, connectivity='conv', T=100, ksize=3, 
                init_omg=0.1, c_norm=None, use_omega_c=False)
# ¡Misma sintaxis! Solo necesitas importar
```

---

### 3. Parámetro de Orden de Kuramoto

#### Antes (Notebook)
```python
def kuramoto_order(xs, ch_index_pair=None):
    R = []
    for x in xs:
        x0 = x[0].detach().cpu().numpy()
        a = x0[0]
        b = x0[1]
        phase = np.arctan2(b, a)
        z = np.exp(1j * phase).ravel()
        R.append(np.abs(z.mean()))
    return np.array(R)

R = kuramoto_order(xs)
```

#### Después (Módulo)
```python
from analisis_criticalidad_minimalista import KuramotoMetrics

R = KuramotoMetrics.order_parameter(xs, ch_pair=(0, 1))
```

---

### 4. Series de Magnitud

#### Antes (Notebook)
```python
def magnitudes_mean_series(xs):
    T = len(xs)
    _, ch, H, W = xs[0].shape
    series = np.zeros((T, ch))
    for t, x in enumerate(xs):
        x_np = x[0].detach().cpu().numpy()
        mag = np.abs(x_np)
        series[t] = mag.mean(axis=(1,2))
    return series

series = magnitudes_mean_series(xs)
```

#### Después (Módulo)
```python
from analisis_criticalidad_minimalista import KuramotoMetrics

series = KuramotoMetrics.magnitudes_mean_series(xs)
```

---

### 5. DFA (Detrended Fluctuation Analysis)

#### Antes (Notebook)
```python
def dfa(series, scales=None):
    x = np.cumsum(series - np.mean(series))
    N = len(x)
    if scales is None:
        scales = np.floor(np.logspace(0.5, np.log10(N/4), 20)).astype(int)
    F = []
    for s in scales:
        nseg = N // s
        if nseg < 2: continue
        rms = []
        for i in range(nseg):
            seg = x[i*s:(i+1)*s]
            t = np.arange(len(seg))
            p = np.polyfit(t, seg, 1)
            trend = np.polyval(p, t)
            rms.append(np.sqrt(np.mean((seg-trend)**2)))
        F.append(np.mean(rms))
    coef = np.polyfit(np.log(scales[:len(F)]), np.log(F), 1)
    alpha = coef[0]
    return scales[:len(F)], F, alpha

scales, F, alpha = dfa(global_series)
```

#### Después (Módulo)
```python
from analisis_criticalidad_minimalista import DFA

scales, F, alpha = DFA.dfa(global_series)
```

---

### 6. PSD (Power Spectral Density)

#### Antes (Notebook)
```python
from scipy.signal import welch

def psd_slope(series, fs=1.0):
    f, Pxx = welch(series, fs=fs, nperseg=min(len(series), 256))
    mask = f>0
    s = np.polyfit(np.log(f[mask]), np.log(Pxx[mask]), 1)
    slope = s[0]
    return f, Pxx, slope

f, P, slope = psd_slope(global_series)
```

#### Después (Módulo)
```python
from analisis_criticalidad_minimalista import PSD

f, P, slope = PSD.psd_slope(global_series)
```

---

### 7. Información Mutua

#### Antes (Notebook)
```python
def mutual_info(x, y, bins=32):
    Hxy, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = Hxy / Hxy.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    eps = 1e-12
    I = np.nansum(pxy * (np.log(pxy+eps) - np.log(px[:,None]+eps) - np.log(py[None,:]+eps)))
    return I

MI = np.zeros((ch, ch))
for i in range(ch):
    for j in range(ch):
        MI[i, j] = mutual_info(series[:, i], series[:, j])
```

#### Después (Módulo)
```python
from analisis_criticalidad_minimalista import MutualInformation

MI = np.zeros((ch, ch))
for i in range(ch):
    for j in range(ch):
        MI[i, j] = MutualInformation.mutual_info(series[:, i], series[:, j])
```

---

### 8. Entropía de Shannon

#### Antes (Notebook)
```python
from scipy.stats import entropy

def Entropy(xs):
    results = {}
    if xs.dim() == 5:
        xs = xs[:, 0]
    T, C, H, W = xs.shape
    for ch in range(C):
        magnitudes = np.mean(np.abs(xs[:, ch, :, :].detach().cpu().numpy()), axis=(1, 2))
        magnitudes = np.nan_to_num(magnitudes, nan=0.0, posinf=1.0, neginf=0.0)
        hist, _ = np.histogram(magnitudes, bins=30, density=True)
        S = entropy(hist + 1e-10)
        results[f'Canal_{ch}'] = {'Entropía de Shannon': S}
    return results

if isinstance(xs, list):
    xs = torch.stack(xs)
results = Entropy(xs)
```

#### Después (Módulo)
```python
from analisis_criticalidad_minimalista import Entropia

results = Entropia.entropy_analysis(xs)  # Maneja automáticamente lista o tensor
```

---

### 9. Animaciones

#### Antes (Notebook)
```python
def animate_magnitude(xs, interval=100, filename='magnitud.gif'):
    frames = [torch.linalg.norm(x[0].detach().cpu(), dim=0).numpy() for x in xs]
    fig, ax = plt.subplots(figsize=(5,5))
    img = ax.imshow(frames[0], cmap='inferno', animated=True, origin='lower')
    ax.axis('off')
    plt.title("Evolución de la magnitud ||x||")
    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Magnitud ||x||', rotation=270, labelpad=15)
    
    def update(frame):
        img.set_array(frame)
        return [img]
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    plt.close(fig)
    ani.save(filename, writer='pillow', fps=1000//interval)
    print(f"✅ Animación guardada como {filename}")
    return HTML(ani.to_jshtml())

animate_magnitude(xs, interval=100, filename='magnitud.gif')
```

#### Después (Módulo)
```python
from analisis_criticalidad_minimalista import Animaciones

Animaciones.animate_magnitude(xs, interval=100, filename='magnitud.gif')
```

---

### 10. Visualizaciones de Series

#### Antes (Notebook)
```python
plt.figure(figsize=(8,3))
for i in range(series.shape[1]):
    plt.plot(series[:,i], label=f'canal {i}')
plt.legend()
plt.xlabel('t')
plt.ylabel('magnitud media')
plt.show()
```

#### Después (Módulo)
```python
from analisis_criticalidad_minimalista import Visualizador

Visualizador.plot_series(series, ylabel='magnitud media', 
                         title='Evolución temporal')
```

---

### 11. Matriz de Correlación

#### Antes (Notebook)
```python
corr_matrix = np.corrcoef(serie_channel_mean.T)

plt.figure(figsize=(6,5))
plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
plt.colorbar(label='Pearson corr')
plt.title('Correlación entre series de magnitud media (canales)')
plt.xticks(range(ch))
plt.yticks(range(ch))
plt.show()
```

#### Después (Módulo)
```python
from analisis_criticalidad_minimalista import Correlacion, Visualizador

corr_matrix = Correlacion.pearson_matrix(series)
Visualizador.plot_matrix(corr_matrix, 
                         title='Correlación entre canales',
                         cmap='coolwarm')
```

---

### 12. Análisis Completo de Series

#### Antes (Notebook)
```python
# Código disperso en múltiples celdas
serie_channel_mean = []
serie_channel_std = []
for x in xs:
    x_np = x[0].detach().cpu()
    abs_map = torch.abs(x_np)
    serie_channel_mean.append(abs_map.mean(dim=(1,2)).numpy())
    serie_channel_std.append(abs_map.std(dim=(1,2)).numpy())

serie_channel_mean = np.stack(serie_channel_mean)
serie_channel_std = np.stack(serie_channel_std)

# ... más código para osciladores, cambios relativos, etc.
```

#### Después (Módulo)
```python
from analisis_criticalidad_minimalista import SeriesAnalysis

stats = SeriesAnalysis.compute_channel_statistics(xs)
# Devuelve dict con:
# - serie_channel_mean
# - serie_channel_std
# - serie_osc_mean
# - rel_change
# - rel_change_osc
# - corr_matrix
```

---

## 🎯 Ejemplo de Migración Completa

### Antes: Código del Notebook (Fragmentado)

```python
# Celda 1: Imports
import torch
import numpy as np
from scipy.signal import welch
# ... más imports

# Celda 2: Definir KBlock
class KBlock(nn.Module):
    def __init__(self, ...):
        # ... código largo

# Celda 3: Definir funciones de análisis
def kuramoto_order(xs, ...):
    # ...

def dfa(series, ...):
    # ...

# Celda 4: Cargar datos
train_dataset = torchvision.datasets.MNIST(...)

# Celda 5: Ejecutar modelo
kblock = KBlock(...)
x, xs, es = kblock(...)

# Celda 6: Calcular métricas
R = kuramoto_order(xs)
scales, F, alpha = dfa(series)
# ...

# Celda 7: Visualizar
plt.plot(R)
plt.show()
# ...
```

### Después: Código con Módulo (Unificado)

```python
# Todas las importaciones en una línea
from analisis_criticalidad_minimalista import (
    KBlock, KuramotoMetrics, DFA, PSD, 
    Entropia, Visualizador, Animaciones
)
import torch
import torchvision

# Cargar datos
train_dataset = torchvision.datasets.MNIST(...)

# Ejecutar modelo
kblock = KBlock(n=4, ch=4, T=100)
x, xs, es = kblock(x, c, T=100, gamma=0.7, del_t=0.9,
                   return_xs=True, return_es=True)

# Calcular todas las métricas
R = KuramotoMetrics.order_parameter(xs)
series = KuramotoMetrics.magnitudes_mean_series(xs)
scales, F, alpha = DFA.dfa(series.mean(axis=1))
f, P, slope = PSD.psd_slope(series.mean(axis=1))
entropy_results = Entropia.entropy_analysis(xs)

# Visualizar
Visualizador.plot_series(series, title='Magnitudes')
Visualizador.plot_energy(es)
Animaciones.animate_magnitude(xs, filename='output.gif')
```

**Beneficios:**
- ✅ Código más corto (reducción del 70%)
- ✅ Más legible y mantenible
- ✅ Reutilizable en otros proyectos
- ✅ Documentado y testeado
- ✅ Fácil de compartir

---

## 📋 Checklist de Migración

Para migrar tu código del notebook al módulo:

- [ ] Reemplazar definiciones de funciones por imports
- [ ] Cambiar llamadas a funciones → métodos estáticos
- [ ] Actualizar rutas de importación
- [ ] Verificar que las animaciones usen `Animaciones.animate_*()`
- [ ] Usar `Visualizador` para gráficos estáticos
- [ ] Reemplazar análisis manual con `SeriesAnalysis`
- [ ] Ejecutar `test_instalacion.py` para verificar

---

## 🚀 Plantilla de Script

Usa esta plantilla para crear nuevos scripts:

```python
"""
Descripción de tu análisis.
"""
import torch
import numpy as np
from analisis_criticalidad_minimalista import (
    KBlock,
    KuramotoMetrics,
    DFA, PSD, Entropia,
    SeriesAnalysis,
    Visualizador, Animaciones
)

def main():
    # 1. Preparar datos
    x = torch.randn(1, 4, 64, 64)
    c = torch.randn(1, 4, 64, 64)
    
    # 2. Ejecutar modelo
    kblock = KBlock(n=4, ch=4, T=100)
    x_final, xs, es = kblock(x, c, T=100, gamma=0.7, del_t=0.9,
                             return_xs=True, return_es=True)
    
    # 3. Analizar
    R = KuramotoMetrics.order_parameter(xs)
    series = KuramotoMetrics.magnitudes_mean_series(xs)
    stats = SeriesAnalysis.compute_channel_statistics(xs)
    
    # 4. Visualizar
    Visualizador.plot_series(series)
    Visualizador.plot_energy(es)
    
    # 5. Reportar
    print(f"R final: {R[-1]:.4f}")
    print(f"Varianza: {stats['serie_channel_std'].mean():.4f}")

if __name__ == "__main__":
    main()
```

---

## 💡 Tips y Mejores Prácticas

1. **Imports al inicio**: Todas las importaciones al principio del archivo
2. **Usar métodos estáticos**: Todas las funciones de análisis son métodos estáticos
3. **Documentación**: Lee los docstrings con `help(KuramotoMetrics.order_parameter)`
4. **Verificación**: Ejecuta `test_instalacion.py` después de cambios
5. **Ejemplos**: Consulta `ejemplo_completo.py` para ver uso completo

---

## 🎓 Recursos Adicionales

- **README.md**: Documentación completa del módulo
- **ejemplo_completo.py**: Script que replica el notebook completo
- **test_instalacion.py**: Verifica que todo funcione
- **RESUMEN_IMPLEMENTACION.md**: Resumen de todas las funciones

---

¡Feliz codificación! 🎉
