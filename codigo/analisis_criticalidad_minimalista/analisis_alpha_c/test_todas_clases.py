#!/usr/bin/env python3
"""
Test de reproducibilidad para todas las clases MNIST (0-9)
Calcula C_crítico para la primera imagen de cada clase
y genera visualizaciones comparativas
"""

import os
import sys
import torch
import torchvision
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as TF
import einops

# Parámetros exactos del notebook
SEED = 1
CH = 3
N = 3
H, W = 128, 128
T = 30
KSIZE = 3
INIT_OMG = 0.1
GAMMA = 0.7
DEL_T = 0.9

# ============================================================================
# Definiciones de clases (copiadas del notebook)
# ============================================================================

def reshape(x, n):
    return einops.rearrange(x, 'b (c n) ... -> b c n ...', n=n)

def reshape_back(x):
    return einops.rearrange(x, 'b c n ... -> b (c n) ...')

class ModReLU(nn.Module):
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
        x = reshape(x, self.n)
        m = torch.linalg.norm(x, dim=2)
        m = torch.nn.ReLU()(self.norm(m))
        x = m.unsqueeze(2) * torch.nn.functional.normalize(x, dim=2)
        x = reshape_back(x)
        return x

class KConv2d(nn.Module):
    def __init__(self, n, ch, connectivity='conv', ksize=3, init_omg=1.0, hw=(16,16), use_omega=True, use_omega_c=True):
        super().__init__()
        assert (ch % n) == 0
        self.n = n
        self.ch = ch

        if connectivity == 'conv':
            self.connectivity = nn.Conv2d(ch, ch, ksize, 1, ksize//2, bias=False)
        elif connectivity == 'conv_mlp':
            self.connectivity = nn.Sequential(
                nn.Conv2d(ch, ch, ksize, 1, ksize//2, bias=False),
                ModReLU(n, ch),
                nn.Conv2d(ch, ch, ksize, 1, ksize//2, bias=False))
        else:
            raise NotImplementedError

        self.use_omega = use_omega
        self.use_omega_c = use_omega_c
        if use_omega or use_omega_c:
            if n == 2:
                self.omg_param = nn.Parameter(torch.randn(ch//2, 2))
            else:
                self.omg_param = nn.Parameter(init_omg * (1/np.sqrt(n))* torch.randn(ch//n, n, n))

    def omg(self, p):
        if self.n==2:
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

        omg_x = torch.einsum('cnm,bcmhw->bcnhw', self.omg(self.omg_param), x) if self.use_omega else torch.zeros_like(x)
        proj = y - torch.sum(y*x, 2, keepdim=True) * x
        if c is not None:
            c = reshape(c, self.n)
            omg_c = torch.einsum('cnm,bcmhw->bcnhw', self.omg(self.omg_param), c) if self.use_omega_c else torch.zeros_like(c)
            return reshape_back(omg_x + proj), reshape_back(omg_c)
        else:
            return reshape_back(omg_x + proj)

    def compute_energy(self, x, c=None):
        y = self.connectivity(x)
        y = y + c
        B = x.shape[0]
        return - torch.sum(x.view(B, -1) * y.view(B, -1), -1)

class KBlock(nn.Module):
    def __init__(self, n, ch, connectivity='conv', T=4, ksize=7, init_omg=0.1, c_norm='gn', use_omega=True, use_omega_c=True):
        super().__init__()
        self.n = n
        self.ch = ch
        self.T = T
        self.kconv = KConv2d(n, ch, connectivity=connectivity, ksize=ksize, init_omg=init_omg, use_omega=use_omega, use_omega_c=use_omega_c)
        self.monitor_count = 0
        if c_norm == 'gn':
            self.c_norm = nn.GroupNorm(ch//n, ch)
        else:
            self.c_norm = lambda x: x

    def normalize(self, x, y=None):
        x = reshape(x, self.n)
        x = torch.nn.functional.normalize(x, dim=2)
        if y is not None:
            x = torch.linalg.norm(reshape(y, self.n), dim=2, keepdim=True) * x
        x = reshape_back(x)
        return x

    def forward(self, x, c, T, gamma, del_t=1.0, return_xs=False, return_es=False, T_noc=None):
        x = self.normalize(x)
        c = self.c_norm(c)
        xs = [x]
        es = []
        if return_es:
            energy = self.kconv.compute_energy(x, c)
            es.append(energy)

        for t in range(T):
            dxdt, dcdt = self.kconv(x, c)
            _c = c + gamma*del_t*dcdt
            c = self.normalize(_c, c)
            x = x + gamma*del_t*dxdt
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

def kuramoto_order_parameter(xs):
    """
    Calcula el parámetro de orden de Kuramoto r(t)
    """
    r_values = []
    for x in xs:
        x = x[0].detach().cpu()
        theta = torch.atan2(x[1], x[0])
        complex_phase = torch.exp(1j * theta)
        order_param = complex_phase.mean()
        r = torch.abs(order_param)
        r_values.append(r.item())
    return np.array(r_values)

# ============================================================================
# Función principal
# ============================================================================

def main():
    print("="*70)
    print("TEST DE REPRODUCIBILIDAD - TODAS LAS CLASES")
    print("Analizando primera imagen de cada clase (0-9)")
    print("="*70)
    print()
    
    # Configurar dispositivo
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Dispositivo: {device.type.upper()}", end="")
    if device.type == "mps":
        print(" (Apple Metal)")
    else:
        print()
    print()
    
    # 1. Cargar datos
    print("1. Cargando dataset MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='../data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Encontrar primera imagen de cada clase
    class_indices = {}
    for idx in range(len(train_dataset)):
        _, label = train_dataset[idx]
        if label not in class_indices:
            class_indices[label] = idx
        if len(class_indices) == 10:
            break
    
    print(f"   Índices encontrados: {class_indices}")
    print()
    
    # Resultados para todas las clases
    resultados = {}
    
    # 2. Procesar cada clase
    print("2. Procesando cada clase...")
    for clase in range(10):
        print(f"\n   📊 Clase {clase}:")
        idx = class_indices[clase]
        img, label = train_dataset[idx]
        
        # Configurar semilla
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        
        # Crear modelo
        kblock = KBlock(
            n=N, 
            ch=CH, 
            connectivity='conv', 
            T=T, 
            ksize=KSIZE,
            init_omg=INIT_OMG, 
            c_norm=None, 
            use_omega_c=False
        ).to(device)
        
        # Redimensionar imagen
        img_resized = TF.resize(img, [H, W])
        img_channels = img_resized.repeat(CH, 1, 1).to(device)
        
        # Calcular curva R vs C
        c_range = np.arange(0.0, 2.0, 0.01)
        R_final = []
        
        x1 = torch.randn(1, CH, H, W).to(device)
        
        for c_val in c_range:
            x = x1.clone()
            c = img_channels.unsqueeze(0) * c_val
            x, xs, es = kblock(x, c, T=T, gamma=GAMMA, del_t=DEL_T, return_xs=True, return_es=True)
            R = kuramoto_order_parameter(xs)
            R_final.append(R[-1])
        
        R_final = np.array(R_final)
        
        # Calcular C_crítico
        df = np.gradient(R_final, c_range)
        i_c = np.argmax(df)
        c_critica = c_range[i_c]
        r_critica = R_final[i_c]
        max_slope = df[i_c]
        
        print(f"      C_crítico = {c_critica:.4f}, R = {r_critica:.4f}, pendiente = {max_slope:.4f}")
        
        # Guardar resultados
        resultados[clase] = {
            'img': img,
            'c_range': c_range,
            'R_final': R_final,
            'df': df,
            'c_critica': c_critica,
            'r_critica': r_critica,
            'max_slope': max_slope,
            'idx': idx
        }
    
    print("\n3. Generando gráficas comparativas...")
    
    # Gráfica grande con todas las clases
    fig = plt.figure(figsize=(20, 16))
    
    for clase in range(10):
        r = resultados[clase]
        
        # Fila 1: Imágenes
        ax = plt.subplot(4, 10, clase + 1)
        ax.imshow(r['img'].squeeze().cpu(), cmap='gray', interpolation='none')
        ax.set_title(f'Clase {clase}', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Fila 2: R vs C
        ax = plt.subplot(4, 10, 10 + clase + 1)
        ax.plot(r['c_range'], r['R_final'], 'b-', linewidth=1.5)
        ax.axvline(x=r['c_critica'], color='r', linestyle='--', linewidth=1)
        ax.set_xlabel('C', fontsize=8)
        ax.set_ylabel('R', fontsize=8)
        ax.set_title(f'$C_c$ = {r["c_critica"]:.3f}', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        
        # Fila 3: Derivada
        ax = plt.subplot(4, 10, 20 + clase + 1)
        ax.plot(r['c_range'], r['df'], 'g-', linewidth=1.5)
        ax.axvline(x=r['c_critica'], color='r', linestyle='--', linewidth=1)
        ax.set_xlabel('C', fontsize=8)
        ax.set_ylabel('dR/dC', fontsize=8)
        ax.set_title(f'max = {r["max_slope"]:.2f}', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        
        # Fila 4: Zoom
        ax = plt.subplot(4, 10, 30 + clase + 1)
        zoom_mask = (r['c_range'] >= r['c_critica'] - 0.3) & (r['c_range'] <= r['c_critica'] + 0.3)
        ax.plot(r['c_range'][zoom_mask], r['R_final'][zoom_mask], 'b-', linewidth=1.5)
        ax.axvline(x=r['c_critica'], color='r', linestyle='--', linewidth=1)
        ax.set_xlabel('C', fontsize=8)
        ax.set_ylabel('R', fontsize=8)
        ax.set_title('Zoom', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    output_file = 'test_todas_clases_comparacion.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gráfica completa: {os.path.abspath(output_file)}")
    
    # Gráfica adicional: todas las curvas R vs C superpuestas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for clase in range(10):
        r = resultados[clase]
        ax1.plot(r['c_range'], r['R_final'], color=colors[clase], 
                linewidth=2, label=f'Clase {clase} ($C_c$={r["c_critica"]:.3f})', alpha=0.7)
        ax2.plot(r['c_range'], r['df'], color=colors[clase], 
                linewidth=2, label=f'Clase {clase}', alpha=0.7)
    
    ax1.set_xlabel('Acoplamiento (C)', fontsize=12)
    ax1.set_ylabel('Parámetro de orden R(T)', fontsize=12)
    ax1.set_title('Curvas R vs C - Todas las clases', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='best')
    
    ax2.set_xlabel('Acoplamiento (C)', fontsize=12)
    ax2.set_ylabel('dR/dC', fontsize=12)
    ax2.set_title('Derivadas - Todas las clases', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    output_file2 = 'test_todas_clases_superpuestas.png'
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Curvas superpuestas: {os.path.abspath(output_file2)}")
    
    # Tabla resumen
    print("\n" + "="*70)
    print("RESUMEN DE RESULTADOS POR CLASE")
    print("="*70)
    print(f"{'Clase':>6} {'Índice':>8} {'C_crítico':>12} {'R(C_c)':>10} {'Pendiente':>12}")
    print("-"*70)
    for clase in range(10):
        r = resultados[clase]
        print(f"{clase:>6} {r['idx']:>8} {r['c_critica']:>12.4f} {r['r_critica']:>10.4f} {r['max_slope']:>12.4f}")
    print("="*70)
    
    # Estadísticas generales
    c_criticos = [resultados[i]['c_critica'] for i in range(10)]
    print(f"\nEstadísticas de C_crítico:")
    print(f"  Media: {np.mean(c_criticos):.4f}")
    print(f"  Desv. estándar: {np.std(c_criticos):.4f}")
    print(f"  Mínimo: {np.min(c_criticos):.4f} (Clase {np.argmin(c_criticos)})")
    print(f"  Máximo: {np.max(c_criticos):.4f} (Clase {np.argmax(c_criticos)})")
    print("\n✅ Análisis completado para todas las clases")
    print("="*70)

if __name__ == "__main__":
    main()
