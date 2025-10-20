"""
Funciones de clustering y segmentación para análisis de criticalidad.
"""
import numpy as np
from sklearn.cluster import KMeans

class SegmentadorKMeans:
    """Segmenta embeddings usando KMeans."""
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)

    def segmentar(self, features):
        """
        features: np.ndarray de shape [N, D] o [C, H, W] (se reordena)
        Devuelve: etiquetas [H, W]
        """
        if features.ndim == 3:
            C, H, W = features.shape
            feats = features.transpose(1,2,0).reshape(-1, C)
        else:
            feats = features
            H = W = int(np.sqrt(features.shape[0]))
        labels = self.model.fit_predict(feats).reshape(H, W)
        return labels
