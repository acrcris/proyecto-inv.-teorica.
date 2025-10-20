"""
Implementación del modelo Kuramoto minimalista y utilidades asociadas.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops


# ==================== FUNCIONES AUXILIARES ====================

def reshape(x, n):
    """Reorganiza los canales en grupos de n osciladores."""
    return einops.rearrange(x, 'b (c n) ... -> b c n ...', n=n)


def reshape_back(x):
    """Restaura la forma original de los canales."""
    return einops.rearrange(x, 'b c n ... -> b (c n) ...')


def nrm(x):
    """Calcula la norma euclidiana sobre la dimensión de osciladores."""
    return torch.linalg.norm(x, dim=2)


def gaussian_kernel_2d(size, sigma):
    """Crea un kernel gaussiano 2D."""
    kernel = np.fromfunction(
        lambda x, y: np.exp(
            -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma**2)
        ),
        (size, size),
    )
    return torch.Tensor(kernel)


# ==================== CLASES DE MODELO ====================

class ModReLU(nn.Module):
    """
    Activación ModReLU: Normaliza la magnitud de vectores complejos
    preservando la dirección de fase.
    """
    def __init__(self, n, ch, norm="gn"):
        super().__init__()
        self.n = n
        if norm == "bn":
            self.norm = nn.BatchNorm2d(ch // n)
        elif norm == "gn":
            self.norm = nn.GroupNorm(ch // n, ch // n)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = reshape(x, self.n)  # (B, C//n, n, H, W)
        m = torch.linalg.norm(x, dim=2)  # [B, C//n, H, W] - magnitud
        m = torch.nn.ReLU()(self.norm(m))  # Normalización + ReLU
        x = m.unsqueeze(2) * F.normalize(x, dim=2)  # Restaura dirección
        x = reshape_back(x)
        return x


class KConv2d(nn.Module):
    """
    Capa convolucional de Kuramoto que define la conectividad entre osciladores.
    Implementa las ecuaciones de acoplamiento y rotación de fase.
    """
    def __init__(self, n, ch, connectivity='conv', ksize=3, init_omg=1.0, 
                 hw=(16,16), use_omega=True, use_omega_c=True):
        super().__init__()
        assert (ch % n) == 0
        self.n = n
        self.ch = ch

        # Conectividad entre osciladores
        if connectivity == 'conv':
            self.connectivity = nn.Conv2d(ch, ch, ksize, 1, ksize//2, bias=False)
        elif connectivity == 'conv_mlp':
            self.connectivity = nn.Sequential(
                nn.Conv2d(ch, ch, ksize, 1, ksize//2, bias=False),
                ModReLU(n, ch),
                nn.Conv2d(ch, ch, ksize, 1, ksize//2, bias=False))
        else:
            raise NotImplementedError

        # Parámetros de frecuencia natural (omega)
        self.use_omega = use_omega
        self.use_omega_c = use_omega_c
        if use_omega or use_omega_c:
            if n == 2:
                self.omg_param = nn.Parameter(torch.randn(ch//2, 2))
            else:
                self.omg_param = nn.Parameter(init_omg * (1/np.sqrt(n)) * torch.randn(ch//n, n, n))

    def omg(self, p):
        """Genera la matriz antisimétrica de rotación."""
        if self.n == 2:
            p = torch.linalg.norm(p, dim=1)
            return torch.stack(
                [torch.stack([torch.zeros_like(p), p], -1),
                 torch.stack([-p, torch.zeros_like(p)], -1)],
                -1)
        else:
            return p - p.transpose(1, 2)

    def forward(self, x, c=None):
        y = self.connectivity(x)
        if c is not None:
            y = y + c
        y = reshape(y, self.n)
        x = reshape(x, self.n)

        # Término de frecuencia natural
        omg_x = torch.einsum('cnm,bcmhw->bcnhw', self.omg(self.omg_param), x) if self.use_omega else torch.zeros_like(x)
        
        # Proyección ortogonal (acoplamiento)
        proj = y - torch.sum(y*x, 2, keepdim=True) * x
        
        if c is not None:
            c = reshape(c, self.n)
            omg_c = torch.einsum('cnm,bcmhw->bcnhw', self.omg(self.omg_param), c) if self.use_omega_c else torch.zeros_like(c)
            return reshape_back(omg_x + proj), reshape_back(omg_c)
        else:
            return reshape_back(omg_x + proj)

    def compute_energy(self, x, c=None):
        """Calcula la energía del sistema (función de Lyapunov)."""
        y = self.connectivity(x)
        if c is not None:
            y = y + c
        B = x.shape[0]
        return - torch.sum(x.view(B, -1) * y.view(B, -1), -1)


class KBlock(nn.Module):
    """
    Bloque principal de dinámica de Kuramoto.
    Integra numéricamente las ecuaciones de osciladores acoplados.
    """
    def __init__(self, n, ch, connectivity='conv', T=4, ksize=7, init_omg=0.1, 
                 c_norm='gn', use_omega=True, use_omega_c=True):
        super().__init__()
        self.n = n
        self.ch = ch
        self.T = T
        self.kconv = KConv2d(n, ch, connectivity=connectivity, ksize=ksize, 
                            init_omg=init_omg, use_omega=use_omega, use_omega_c=use_omega_c)
        self.monitor_count = 0
        
        if c_norm == 'gn':
            self.c_norm = nn.GroupNorm(ch//n, ch)
        else:
            self.c_norm = lambda x: x

    def normalize(self, x, y=None):
        """Normaliza vectores de estado a norma unitaria."""
        x = reshape(x, self.n)
        x = torch.nn.functional.normalize(x, dim=2)
        if y is not None:
            x = torch.linalg.norm(reshape(y, self.n), dim=2, keepdim=True) * x
        x = reshape_back(x)
        return x

    def monitor_norms(self, dt, c, x):
        """Monitoreo de normas para debugging."""
        def print_norm(x, name):
            x = x.view(x.shape[0], -1).detach()
            x = torch.linalg.norm(x, dim=1).mean(0)
            print(f"avg norms of {name}: {x:.6f}")
        for x, name in ((dt, 'dt'), (c, 'c'), (x, 'x')):
            print_norm(x, name)

    def forward(self, x, c, T, gamma, del_t=1.0, return_xs=False, return_es=False, T_noc=None):
        """
        Integra la dinámica de Kuramoto por T pasos temporales.
        
        Args:
            x: Estados iniciales [B, ch, H, W]
            c: Campo de acoplamiento externo [B, ch, H, W]
            T: Número de pasos de integración
            gamma: Factor de acoplamiento
            del_t: Paso temporal
            return_xs: Si devolver todos los estados intermedios
            return_es: Si devolver energías en cada paso
            
        Returns:
            x: Estado final
            xs: Lista de estados (si return_xs=True)
            es: Lista de energías (si return_es=True)
        """
        x = self.normalize(x)
        c = self.c_norm(c)
        xs = [x]
        es = []
        
        if return_es:
            energy = self.kconv.compute_energy(x, c)
            es.append(energy)

        if self.monitor_count >= 50:
            do_monitoring = False
            self.monitor_count = 0
        else:
            do_monitoring = False
            self.monitor_count += 1

        # Integración numérica (Euler explícito)
        for t in range(T):
            dxdt, dcdt = self.kconv(x, c)
            _c = c + gamma * del_t * dcdt
            c = self.normalize(_c, c)
            x = x + gamma * del_t * dxdt
            x = self.normalize(x)

            if return_es:
                energy = self.kconv.compute_energy(x, c)
                es.append(energy)

            if return_xs:
                xs.append(x)

        if return_es:
            return x, xs, es
        else:
            return x, xs


class Reshape(nn.Module):
    """Capa auxiliar para cambiar forma de tensores en redes neuronales."""
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
