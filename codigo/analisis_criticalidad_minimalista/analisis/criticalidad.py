"""
Métricas y análisis de criticalidad: orden de Kuramoto, entropía, DFA, PSD, correlación, MI, etc.
"""
import numpy as np
import torch
from scipy.stats import entropy
from scipy.signal import welch

class KuramotoMetrics:
    """Cálculo del parámetro de orden de Kuramoto y variantes."""
    @staticmethod
    def order_parameter(xs, ch_pair=(0,1)):
        """
        Calcula el parámetro de orden de Kuramoto R(t).
        
        Args:
            xs: lista de tensores [B, ch, H, W] o [ch, H, W]
            ch_pair: tupla con índices de canales para calcular fase
            
        Returns:
            R: array de parámetros de orden en cada tiempo (T,)
        """
        R = []
        for x in xs:
            if x.ndim == 4:
                x0 = x[0].detach().cpu().numpy()
            else:
                x0 = x.detach().cpu().numpy()
            a = x0[ch_pair[0]]
            b = x0[ch_pair[1]]
            phase = np.arctan2(b, a)
            z = np.exp(1j * phase).ravel()
            R.append(np.abs(z.mean()))
        return np.array(R)
    
    @staticmethod
    def magnitudes_mean_series(xs):
        """
        Calcula series temporales de magnitud media por canal.
        
        Args:
            xs: lista de tensores [B, ch, H, W]
            
        Returns:
            series: array (T, ch) con magnitudes medias por canal en cada tiempo
        """
        T = len(xs)
        _, ch, H, W = xs[0].shape
        series = np.zeros((T, ch))
        for t, x in enumerate(xs):
            x_np = x[0].detach().cpu().numpy()  # [ch, H, W]
            mag = np.abs(x_np)
            series[t] = mag.mean(axis=(1, 2))
        return series
    
    @staticmethod
    def magnitude_per_oscillator(x_t):
        """
        Calcula la magnitud total por pixel (norma sobre todos los canales).
        
        Args:
            x_t: tensor [1, ch, H, W] o [ch, H, W]
            
        Returns:
            mag_map: tensor [H, W] con magnitudes por pixel
        """
        if x_t.ndim == 4:
            x = x_t[0].detach().cpu()  # [ch, H, W]
        else:
            x = x_t.detach().cpu()
        mag_map = torch.linalg.norm(x, dim=0)  # shape [H, W]
        return mag_map

class Entropia:
    """Entropía de Shannon sobre series temporales."""
    @staticmethod
    def shannon(series, bins=30):
        """
        Calcula la entropía de Shannon de una serie temporal.
        
        Args:
            series: array 1D con valores de la serie
            bins: número de bins para el histograma
            
        Returns:
            S: entropía de Shannon (nats)
        """
        hist, _ = np.histogram(series, bins=bins, density=True)
        return entropy(hist + 1e-10)
    
    @staticmethod
    def entropy_analysis(xs):
        """
        Analiza entropía para todos los canales de una evolución temporal.
        
        Args:
            xs: tensor [T, B, C, H, W] o lista de tensores
            
        Returns:
            results: dict con entropía por canal
        """
        # Convertir lista a tensor si es necesario
        if isinstance(xs, list):
            xs = torch.stack(xs)
        
        results = {}
        
        # Asegurar forma [T, C, H, W]
        if xs.dim() == 5:
            xs = xs[:, 0]
        
        T, C, H, W = xs.shape
        
        for ch in range(C):
            # Magnitud promedio del canal en cada tiempo
            magnitudes = np.mean(np.abs(xs[:, ch, :, :].detach().cpu().numpy()), axis=(1, 2))
            magnitudes = np.nan_to_num(magnitudes, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Entropía de Shannon
            S = Entropia.shannon(magnitudes)
            
            results[f'Canal_{ch}'] = {
                'Entropía de Shannon': S
            }
        
        return results

class DFA:
    """Análisis de fluctuación sin tendencia (Detrended Fluctuation Analysis)."""
    @staticmethod
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

class PSD:
    """Análisis espectral de densidad de potencia y pendiente 1/f."""
    @staticmethod
    def psd_slope(series, fs=1.0):
        f, Pxx = welch(series, fs=fs, nperseg=min(len(series), 256))
        mask = f>0
        s = np.polyfit(np.log(f[mask]), np.log(Pxx[mask]), 1)
        slope = s[0]
        return f, Pxx, slope

class MutualInformation:
    """Cálculo de información mutua entre dos series."""
    @staticmethod
    def mutual_info(x, y, bins=32):
        Hxy, _, _ = np.histogram2d(x, y, bins=bins)
        pxy = Hxy / Hxy.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        eps = 1e-12
        I = np.nansum(pxy * (np.log(pxy+eps) - np.log(px[:,None]+eps) - np.log(py[None,:]+eps)))
        return I

class Correlacion:
    """Correlación de Pearson entre canales o series."""
    @staticmethod
    def pearson_matrix(series):
        return np.corrcoef(series.T)
