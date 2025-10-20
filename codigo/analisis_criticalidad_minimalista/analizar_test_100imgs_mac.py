"""
Analiza los resultados del test de 100 imágenes (10 por clase)
Genera estadísticas y distribuciones preliminares por clase
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

RESULTS_DIR = "test_resultados_MAC_100imgs"
FINAL_FILE = os.path.join(RESULTS_DIR, "metricas_test_100imgs_MAC.pt")
OUT_CSV = os.path.join(RESULTS_DIR, "metricas_por_imagen_test.csv")
OUT_RESUMEN = os.path.join(RESULTS_DIR, "resumen_por_clase_test.csv")


def _to_json_array(x):
    if x is None:
        return json.dumps(None)
    arr = np.asarray(x).tolist()
    return json.dumps(arr, separators=(",", ":"))


def cargar_y_construir_df():
    print("📂 Cargando resultados del test...")
    data = torch.load(FINAL_FILE, weights_only=False)
    print(f"   Total imágenes: {len(data['metricas'])}")
    print(f"   Dispositivo usado: {data.get('device', 'N/A')}")
    print()
    
    filas = []
    for m in data['metricas']:
        fila = {
            'idx': m.get('idx'),
            'label': m.get('label'),
            'R_stationary': m.get('R_stationary', np.nan),
            'PSD_slope': m.get('PSD_slope', np.nan),
            'DFA_alpha': m.get('DFA_alpha', np.nan),
            'mutual_info': m.get('mutual_info', np.nan),
            'shannon_entropy_by_channel': _to_json_array(m.get('shannon_entropy_by_channel')),
            'mag_channel_mean_final': _to_json_array(m.get('mag_channel_mean_final')),
        }
        filas.append(fila)
    
    df = pd.DataFrame(filas)
    return df, data


def generar_resumen(df):
    cols_escalar = ['R_stationary', 'PSD_slope', 'DFA_alpha', 'mutual_info']
    resumen = (
        df.groupby('label')[cols_escalar]
          .agg(['count', 'median', 'mean', 'std', 'min', 'max'])
    )
    resumen.columns = [f"{c[0]}_{c[1]}" for c in resumen.columns]
    resumen = resumen.reset_index()
    return resumen


def plot_distribuciones(df):
    """Genera gráficas de distribuciones por clase"""
    cols = ['R_stationary', 'PSD_slope', 'DFA_alpha', 'mutual_info']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(cols):
        ax = axes[idx]
        
        # Boxplot por clase
        data_to_plot = [df[df['label'] == c][col].dropna().values for c in sorted(df['label'].unique())]
        bp = ax.boxplot(data_to_plot, labels=sorted(df['label'].unique()),
                        patch_artist=True, showmeans=True)
        
        # Colorear cajas
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_xlabel('Clase')
        ax.set_ylabel(col)
        ax.set_title(f'Distribución de {col} por clase (n=10 por clase)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'distribuciones_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Gráfica guardada: {output_path}")
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("="*70)
    print("ANÁLISIS DE RESULTADOS TEST - 100 IMÁGENES")
    print("="*70)
    print()
    
    # Cargar datos
    df, data = cargar_y_construir_df()
    
    # Guardar CSV
    print("💾 Guardando CSV...")
    df.to_csv(OUT_CSV, index=False)
    print(f"   ✅ {OUT_CSV}")
    print()
    
    # Generar resumen
    print("📊 Generando resumen por clase...")
    resumen = generar_resumen(df)
    resumen.to_csv(OUT_RESUMEN, index=False)
    print(f"   ✅ {OUT_RESUMEN}")
    print()
    
    # Mostrar resumen en consola
    print("RESUMEN POR CLASE (métricas escalares):")
    print(resumen.to_string(index=False))
    print()
    
    # Generar gráficas
    print("📈 Generando gráficas de distribuciones...")
    try:
        plot_distribuciones(df)
    except Exception as e:
        print(f"   ⚠️  Error al generar gráficas: {e}")
    print()
    
    # Verificar métricas vectoriales (por canal)
    print("🔍 VERIFICANDO MÉTRICAS VECTORIALES:")
    ejemplo = df.iloc[0]
    
    try:
        shannon = json.loads(ejemplo['shannon_entropy_by_channel'])
        if shannon is not None:
            print(f"   shannon_entropy_by_channel: {len(shannon)} canales")
            print(f"   Valores: {shannon}")
    except:
        print("   ⚠️  shannon_entropy_by_channel no disponible")
    
    try:
        mag = json.loads(ejemplo['mag_channel_mean_final'])
        if mag is not None:
            print(f"   mag_channel_mean_final: {len(mag)} canales")
            print(f"   Valores: {mag}")
    except:
        print("   ⚠️  mag_channel_mean_final no disponible")
    
    print()
    print("="*70)
    print("✅ ANÁLISIS COMPLETO")
    print("="*70)
    print()
    print("Archivos generados:")
    print(f"   - {OUT_CSV}")
    print(f"   - {OUT_RESUMEN}")
    print(f"   - {RESULTS_DIR}/distribuciones_test.png")
    print()
    print("📊 Si todo se ve bien, procede con el procesamiento completo:")
    print("   nohup python3 run_kuramoto_TRAIN_MAC.py > kuramoto_train_mac.log 2>&1 &")
    print("   echo $! > kuramoto_train_mac.pid")
    print()


if __name__ == "__main__":
    main()
