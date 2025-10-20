
"""
Pipeline para análisis de criticalidad minimalista sobre MNIST agrupado por clase.
Calcula la distribución de medidas de criticalidad para cada grupo de imágenes (0–9).
"""

from datasets.loader import MNISTLoader
from kuramoto.modelo import KBlock
from analisis.criticalidad import KuramotoMetrics, Entropia, DFA, PSD, MutualInformation, Correlacion
from utils.visualizacion import Visualizador
import torch
import numpy as np

def analizar_grupo(imagenes, medidas=["kuramoto", "entropia", "dfa", "psd"]):
    """
    Ejecuta el modelo Kuramoto minimalista sobre todas las imágenes de un grupo y calcula las métricas indicadas.
    Devuelve un dict con listas de resultados por métrica.
    """
    resultados = {m: [] for m in medidas}
    for img in imagenes:
        # Prepara entrada para Kuramoto (simula dinámica)
        img_resized = torch.nn.functional.interpolate(img.unsqueeze(0), size=(64,64), mode="bilinear")
        img_channels = img_resized.repeat(4, 1, 1)   # [4, 64, 64]
        c = img_channels.unsqueeze(0)
        ch, n, h, w, T = 4, 4, 64, 64, 50
        kblock = KBlock(n=n, ch=ch, connectivity='conv', T=T, ksize=3, init_omg=0.1, c_norm=None, use_omega_c=False)
        x = torch.randn(1, ch, h, w)
        x, xs, es = kblock(x, c, T=T, gamma=0.7, del_t=0.9, return_xs=True, return_es=True)

        # xs: lista de estados temporales [B, ch, H, W]
        # Medidas
        if "kuramoto" in medidas:
            R = KuramotoMetrics.order_parameter(xs)
            resultados["kuramoto"].append(R)
        if "entropia" in medidas:
            serie = np.mean([x[0].detach().cpu().numpy() for x in xs], axis=(1,2,3))
            S = Entropia.shannon(serie)
            resultados["entropia"].append(S)
        if "dfa" in medidas:
            serie = np.mean([x[0].detach().cpu().numpy() for x in xs], axis=(1,2,3))
            _, _, alpha = DFA.dfa(serie)
            resultados["dfa"].append(alpha)
        if "psd" in medidas:
            serie = np.mean([x[0].detach().cpu().numpy() for x in xs], axis=(1,2,3))
            _, _, slope = PSD.psd_slope(serie)
            resultados["psd"].append(slope)
    return resultados

def main():
    # Carga MNIST y agrupa por clase
    loader = MNISTLoader()
    train_dataset = loader.train_dataset
    imagenes_por_clase = {i: [] for i in range(10)}
    for i in range(len(train_dataset)):
        img, label = train_dataset[i]
        imagenes_por_clase[label].append(img)

    # Analiza cada grupo
    medidas = ["kuramoto", "entropia", "dfa", "psd"]
    distribuciones = {m: {} for m in medidas}
    for clase in range(10):
        print(f"Analizando clase {clase} ({len(imagenes_por_clase[clase])} imágenes)...")
        resultados = analizar_grupo(imagenes_por_clase[clase], medidas=medidas)
        for m in medidas:
            distribuciones[m][clase] = resultados[m]

    # Visualiza distribuciones
    for m in medidas:
        print(f"\nDistribución de {m} por clase:")
        for clase in range(10):
            vals = distribuciones[m][clase]
            print(f"Clase {clase}: media={np.mean(vals):.4f}, std={np.std(vals):.4f}, n={len(vals)}")
        # Ejemplo de visualización
        Visualizador.plot_series(np.array([np.mean(distribuciones[m][clase]) for clase in range(10)]).reshape(-1,1), labels=[m], title=f"Media de {m} por clase", ylabel=m)

if __name__ == "__main__":
    main()
