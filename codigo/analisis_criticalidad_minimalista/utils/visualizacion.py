"""
Funciones de visualización para análisis de criticalidad minimalista.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
try:
    from IPython.display import HTML
except ImportError:
    HTML = None


class Visualizador:
    """Clase con métodos estáticos para visualización de análisis."""
    
    @staticmethod
    def plot_series(series, labels=None, title=None, ylabel=None):
        """Grafica series temporales múltiples."""
        plt.figure(figsize=(8,3))
        for i in range(series.shape[1]):
            plt.plot(series[:,i], label=labels[i] if labels else f'canal {i}')
        plt.legend(); plt.xlabel('t')
        if ylabel: plt.ylabel(ylabel)
        if title: plt.title(title)
        plt.show()

    @staticmethod
    def plot_matrix(matrix, title=None, cmap='coolwarm', labels=None, vmin=-1, vmax=1):
        """Visualiza una matriz como mapa de calor."""
        plt.figure(figsize=(6,5))
        plt.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(label='Valor')
        if title: plt.title(title)
        if labels:
            plt.xticks(range(len(labels)), labels)
            plt.yticks(range(len(labels)), labels)
        plt.show()
    
    @staticmethod
    def plot_energy(energies, normalize=True):
        """Grafica la evolución de energía del sistema."""
        energies = np.array([e.item() if torch.is_tensor(e) else e for e in energies])
        if normalize:
            energie_norm = energies - np.min(energies)
            energy = energie_norm / np.max(energie_norm)
        else:
            energy = energies
        
        plt.figure(figsize=(8,4))
        plt.plot(energy)
        plt.title('Energía durante la evolución')
        plt.xlabel('Paso de tiempo')
        plt.ylabel('Energía' + (' (normalizada)' if normalize else ''))
        plt.grid(True)
        plt.show()


class Animaciones:
    """Clase con métodos para crear animaciones de la dinámica de Kuramoto."""
    
    @staticmethod
    def animate_dynamics(xs, channel=0, interval=100, filename='animacion.gif'):
        """
        Anima la evolución de un canal específico.
        
        Args:
            xs: lista de tensores [B, ch, H, W]
            channel: índice del canal a visualizar
            interval: milisegundos entre frames
            filename: nombre del archivo de salida
        """
        frames = [x[0, channel].detach().cpu().numpy() for x in xs]

        fig, ax = plt.subplots()
        img = ax.imshow(frames[0], animated=True)
        ax.axis('off')
        cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

        def update(frame):
            img.set_array(frame)
            return [img]

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
        plt.close(fig)
        
        ani.save(filename, writer='pillow', fps=1000//interval)
        print(f"✅ Animación guardada como {filename}")
        
        if HTML is not None:
            return HTML(ani.to_jshtml())
        return ani
    
    @staticmethod
    def animate_magnitude(xs, interval=100, filename='magnitud.gif'):
        """
        Anima la evolución de la magnitud total (norma sobre canales).
        
        Args:
            xs: lista de tensores [B, ch, H, W]
            interval: milisegundos entre frames
            filename: nombre del archivo de salida
        """
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
        
        if HTML is not None:
            return HTML(ani.to_jshtml())
        return ani
    
    @staticmethod
    def animate_vector_field(xs, step=2, interval=100, filename='campo_vectorial.gif'):
        """
        Anima el campo vectorial formado por los dos primeros canales.
        
        Args:
            xs: lista de tensores [B, ch, H, W]
            step: espaciado de la malla de vectores
            interval: milisegundos entre frames
            filename: nombre del archivo de salida
        """
        frames = []
        for x in xs:
            x_t = x[0].detach().cpu()
            u = x_t[0]
            v = x_t[1]
            frames.append((u, v))

        Y, X = torch.meshgrid(
            torch.arange(0, frames[0][0].shape[0], step),
            torch.arange(0, frames[0][0].shape[1], step),
            indexing='ij'
        )

        fig, ax = plt.subplots(figsize=(6,6))
        qv = ax.quiver(X, Y, frames[0][0][::step, ::step], frames[0][1][::step, ::step], 
                      scale=30, color='teal')
        ax.axis('equal')
        ax.axis('off')
        plt.title("Evolución del campo vectorial (x₁, x₂)")

        def update(frame_data):
            u, v = frame_data
            qv.set_UVC(u[::step, ::step], v[::step, ::step])
            return [qv]

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
        plt.close(fig)

        ani.save(filename, writer='pillow', fps=1000//interval)
        print(f"✅ Animación guardada como {filename}")
        
        if HTML is not None:
            return HTML(ani.to_jshtml())
        return ani
    
    @staticmethod
    def animate_phase_evolution(xs, interval=100, filename="sync_fases.gif"):
        """
        Anima la evolución de las fases de todos los canales en paralelo.
        
        Args:
            xs: lista de tensores [B, ch, H, W]
            interval: milisegundos entre frames
            filename: nombre del archivo de salida
        """
        # Determinar número de canales
        num_channels = xs[0].shape[1]
        
        fig, ax = plt.subplots(1, num_channels, figsize=(3*num_channels, 4))
        if num_channels == 1:
            ax = [ax]
        ims = []
        
        for t, x in enumerate(xs):
            ch_phases = []
            for i in range(num_channels):
                phase = torch.atan2(x[0, i, :, :], x[0, 0, :, :]).detach().cpu().numpy()
                im = ax[i].imshow(phase, cmap='hsv', origin='lower', 
                                vmin=-np.pi, vmax=np.pi, animated=True)
                ax[i].set_title(f"Canal {i+1}")
                ax[i].axis('off')
                ch_phases.append(im)
            ims.append(ch_phases)
        
        ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True)
        plt.close(fig)
        ani.save(filename, writer='pillow', fps=1000//interval)
        print(f"✅ Animación guardada como {filename}")
        
        if HTML is not None:
            return HTML(ani.to_jshtml())
        return ani
    
    @staticmethod
    def animate_magnitude_evolution(xs, interval=100, filename="sync_magnitudes.gif"):
        """
        Anima la evolución de las magnitudes de todos los canales en paralelo.
        
        Args:
            xs: lista de tensores [B, ch, H, W]
            interval: milisegundos entre frames
            filename: nombre del archivo de salida
        """
        num_channels = xs[0].shape[1]
        
        fig, ax = plt.subplots(1, num_channels, figsize=(3*num_channels, 4))
        if num_channels == 1:
            ax = [ax]
        ims = []

        for t, x in enumerate(xs):
            ch_mags = []
            for i in range(num_channels):
                mag = torch.abs(x[0, i, :, :]).detach().cpu().numpy()
                im = ax[i].imshow(mag, cmap='inferno', origin='lower', animated=True)
                ax[i].set_title(f"Canal {i+1}")
                ax[i].axis('off')
                ch_mags.append(im)
            ims.append(ch_mags)

        ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True)
        plt.close(fig)
        ani.save(filename, writer='pillow', fps=1000//interval)
        print(f"✅ Animación guardada como {filename}")
        
        if HTML is not None:
            return HTML(ani.to_jshtml())
        return ani
