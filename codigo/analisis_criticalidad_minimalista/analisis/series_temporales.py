"""
Análisis de series temporales y estadísticas avanzadas.
"""
import numpy as np
import torch


class SeriesAnalysis:
    """Análisis estadístico de series temporales de osciladores."""
    
    @staticmethod
    def compute_channel_statistics(xs):
        """
        Calcula estadísticas detalladas por canal y por tiempo.
        
        Args:
            xs: lista de tensores [B, ch, H, W]
            
        Returns:
            dict con medias, desviaciones, cambios relativos, correlaciones
        """
        B, ch, H, W = xs[0].shape
        
        # Magnitud por canal (valor absoluto medio espacial)
        serie_channel_mean = []
        serie_channel_std = []
        for x in xs:
            x_np = x[0].detach().cpu()  # shape: [ch, H, W]
            abs_map = torch.abs(x_np)
            serie_channel_mean.append(abs_map.mean(dim=(1,2)).numpy())  # shape (ch,)
            serie_channel_std.append(abs_map.std(dim=(1,2)).numpy())
        
        serie_channel_mean = np.stack(serie_channel_mean)  # shape (T, ch)
        serie_channel_std = np.stack(serie_channel_std)    # shape (T, ch)
        
        # Magnitud por oscilador (norma sobre canales)
        serie_osc_mean = []
        for x in xs:
            x_cpu = x[0].detach().cpu()  # [ch, H, W]
            mag_map = torch.linalg.norm(x_cpu, dim=0)  # [H, W]
            serie_osc_mean.append(mag_map.mean().numpy())
        
        serie_osc_mean = np.array(serie_osc_mean)  # shape (T,)
        
        # Cambios relativos respecto a t0
        eps = 1e-8
        m0 = serie_channel_mean[0]  # (ch,)
        rel_change = (serie_channel_mean - m0[None, :]) / (m0[None, :] + eps)
        
        m0_osc = serie_osc_mean[0]
        rel_change_osc = (serie_osc_mean - m0_osc) / (m0_osc + eps)
        
        # Correlación temporal entre canales
        corr_matrix = np.corrcoef(serie_channel_mean.T)
        
        return {
            'serie_channel_mean': serie_channel_mean,
            'serie_channel_std': serie_channel_std,
            'serie_osc_mean': serie_osc_mean,
            'rel_change': rel_change,
            'rel_change_osc': rel_change_osc,
            'corr_matrix': corr_matrix
        }
    
    @staticmethod
    def compute_phase_statistics(xs, ch_pair=(0, 1)):
        """
        Calcula estadísticas de fase para un par de canales.
        
        Args:
            xs: lista de tensores [B, ch, H, W]
            ch_pair: tupla (i, j) con índices de canales para calcular fase
            
        Returns:
            dict con mapas de fase, coherencia, dispersión
        """
        phases = []
        coherence = []
        
        for x in xs:
            x_np = x[0].detach().cpu().numpy()
            a = x_np[ch_pair[0]]
            b = x_np[ch_pair[1]]
            
            # Fase en cada pixel
            phase = np.arctan2(b, a)
            phase = (phase + 2 * np.pi) % (2 * np.pi)
            phases.append(phase)
            
            # Coherencia de fase (parámetro de orden local)
            z = np.exp(1j * phase)
            r = np.abs(z.mean())
            coherence.append(r)
        
        return {
            'phases': np.array(phases),  # (T, H, W)
            'coherence': np.array(coherence),  # (T,)
            'final_phase': phases[-1],
            'mean_coherence': np.mean(coherence)
        }
    
    @staticmethod
    def compute_variance_temporal(series):
        """
        Calcula varianza temporal por canal.
        
        Args:
            series: array (T, ch)
            
        Returns:
            var_per_channel: array (ch,)
        """
        return series.var(axis=0)
