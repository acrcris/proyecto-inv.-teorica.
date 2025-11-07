"""Genera y dibuja el grafo de conectividad local de la primera KLayer (modo conv).

Ahora permite recortar una sub‑rejilla centrada y muestrear con una
"stride" para reducir la cantidad de nodos en la visualización.

Uso rápido (menos nodos por defecto):
    python akorn_structure_graph.py

Parámetros opcionales:
    --crop  Tamaño del recorte centrado (lado, en celdas). Por defecto 12.
    --step  Muestreo cada "step" celdas (stride). Por defecto 1.
    --no-exclude-self  Conserva self-loops (por defecto se excluyen).
    --imsize, --psize, --ksize  Sobre-escriben hiper-parámetros del modelo.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

# Preparar directorio de caché de Matplotlib si se usa, pero no importarlo aún.
REPO_ROOT = Path(__file__).resolve().parents[2]
_mpldir = REPO_ROOT / "docs" / ".mplconfig"
try:
    _mpldir.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
os.environ.setdefault("MPLCONFIGDIR", str(_mpldir))

# Evitar importar paquetes pesados en sandbox a menos que se pidan explícitamente
NX_AVAILABLE = False
TORCH_AVAILABLE = False
nx = None  # type: ignore
torch = None  # type: ignore

AKORN_DIR = REPO_ROOT / "alternativa1" / "codigo" / "akorn"
if not AKORN_DIR.exists():
    AKORN_DIR = REPO_ROOT / "codigo" / "akorn"
if str(AKORN_DIR) not in sys.path:
    sys.path.insert(0, str(AKORN_DIR))


def _kernel_offsets_from_tensor(mask) -> Dict[Tuple[int, int], float]:
    k = mask.shape[-1]
    center = k // 2
    weights: Dict[Tuple[int, int], float] = {}
    for dy in range(k):
        for dx in range(k):
            # Solo offsets con algun peso distinto de cero
            if float(abs(mask[..., dy, dx]).sum()) > 0:
                offsets = (dy - center, dx - center)
                # usar promedio como peso
                try:
                    weights[offsets] = float(mask[..., dy, dx].mean())
                except Exception:
                    weights[offsets] = 1.0
    return weights


def _kernel_offsets_default(ksize: int) -> Dict[Tuple[int, int], float]:
    """Crea offsets completos k×k con peso 1.0.

    Útil cuando PyTorch o el modelo no están disponibles.
    """
    k = max(1, int(ksize))
    c = k // 2
    weights: Dict[Tuple[int, int], float] = {}
    for dy in range(k):
        for dx in range(k):
            offsets = (dy - c, dx - c)
            weights[offsets] = 1.0
    return weights


def build_grid_graph(
    h: int,
    w: int,
    offsets: Dict[Tuple[int, int], float],
    *,
    y_range: Tuple[int, int] | None = None,
    x_range: Tuple[int, int] | None = None,
    step: int = 1,
    exclude_self: bool = True,
) -> object:
    """Construye un grafo dirigido sobre una sub‑rejilla opcional.

    - y_range/x_range: límites [ini, fin) en coordenadas de parches.
    - step: solo incluye nodos donde y y x avanzan de "step" en "step".
    - exclude_self: si True, no añade aristas con offset (0, 0).
    """
    y0, y1 = (0, h) if y_range is None else y_range
    x0, x1 = (0, w) if x_range is None else x_range
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)

    ys = list(range(y0, y1, max(1, step)))
    xs = list(range(x0, x1, max(1, step)))
    selected = {(y, x) for y in ys for x in xs}

    if NX_AVAILABLE:
        graph = nx.DiGraph()
        for node in selected:
            graph.add_node(node)
        for (dy, dx), weight in offsets.items():
            if exclude_self and dy == 0 and dx == 0:
                continue
            for y in ys:
                for x in xs:
                    ny, nx_ = y + dy, x + dx
                    if (ny, nx_) in selected:
                        graph.add_edge((y, x), (ny, nx_), weight=weight)
        return graph
    else:
        # Fallback ligero: devolvemos lista de nodos y aristas
        nodes = list(selected)
        edges: list[tuple[tuple[int, int], tuple[int, int], float]] = []
        for (dy, dx), weight in offsets.items():
            if exclude_self and dy == 0 and dx == 0:
                continue
            for y in ys:
                for x in xs:
                    ny, nx_ = y + dy, x + dx
                    if (ny, nx_) in selected:
                        edges.append(((y, x), (ny, nx_), weight))
        return {"nodes": nodes, "edges": edges}


def _export_svg(graph: object, *, title: str, savepath: str) -> None:
    # Extraer nodos y aristas en formato simple
    if NX_AVAILABLE:
        nodes = list(graph.nodes)  # type: ignore[attr-defined]
        edges = [(u, v) for (u, v) in graph.edges]  # type: ignore[attr-defined]
    else:
        data = graph  # type: ignore[assignment]
        nodes = list(data["nodes"])  # type: ignore[index]
        edges = [(u, v) for (u, v, _w) in data["edges"]]  # type: ignore[index]

    # Posiciones: (y, x) -> (x, -y)
    pos = {(y, x): (float(x), float(-y)) for (y, x) in nodes}
    if not pos:
        raise ValueError("No hay nodos para exportar")

    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    # Escala y márgenes para una vista legible
    s = 36.0
    m = 24.0
    width = (maxx - minx) * s + 2 * m
    height = (maxy - miny) * s + 2 * m

    def tx(x: float) -> float:
        return (x - minx) * s + m

    def ty(y: float) -> float:
        return (y - miny) * s + m

    parts: list[str] = []
    parts.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width:.0f}' height='{height:.0f}' viewBox='0 0 {width:.0f} {height:.0f}'>")
    parts.append(f"  <title>{title}</title>")
    # Edges
    for (src, dst) in edges:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        parts.append(
            f"  <line x1='{tx(x0):.2f}' y1='{ty(y0):.2f}' x2='{tx(x1):.2f}' y2='{ty(y1):.2f}' stroke='#555' stroke-opacity='0.6' stroke-width='1' />"
        )
    # Nodes
    for (y, x) in nodes:
        xx, yy = pos[(y, x)]
        parts.append(
            f"  <circle cx='{tx(xx):.2f}' cy='{ty(yy):.2f}' r='3.5' fill='#4daf4a' />"
        )
    parts.append("</svg>")

    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    Path(savepath).write_text("\n".join(parts), encoding="utf-8")


def plot_graph(
    graph: object,
    *,
    title: str = "KLayer connectivity",
    savepath: str | None = None,
    renderer: str = "mpl",
) -> None:
    """Dibuja o exporta el grafo con Matplotlib (mpl) o SVG puro (svg)."""
    if renderer == "svg":
        out = savepath or str(REPO_ROOT / "docs" / "figs" / "akorn_graph.svg")
        if not out.endswith(".svg"):
            out = str(Path(out).with_suffix(".svg"))
        _export_svg(graph, title=title, savepath=out)
        return

    # Modo Matplotlib
    import matplotlib.pyplot as plt  # type: ignore

    plt.figure(figsize=(8, 8))
    if NX_AVAILABLE:
        pos = {(y, x): (x, -y) for y, x in graph.nodes}  # type: ignore[attr-defined]
        nx.draw_networkx_nodes(graph, pos, node_size=30, node_color="#4daf4a")  # type: ignore[name-defined]
        nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle="->", alpha=0.4)  # type: ignore[name-defined]
    else:
        data = graph  # type: ignore[assignment]
        nodes = data["nodes"]
        edges = data["edges"]
        xs = [x for (_, x) in nodes]
        ys = [-y for (y, _) in nodes]
        plt.scatter(xs, ys, s=20, c="#4daf4a")
        for (y0, x0), (y1, x1), _w in edges:
            plt.plot([x0, x1], [-y0, -y1], color="0.3", linewidth=0.6, alpha=0.6)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if savepath is None:
        plt.show()
    else:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=200)


def _setup_backend(backend: str | None, default_to_agg: bool) -> None:
    """Configura el backend si se especifica.

    Si `backend` es None y `default_to_agg` es True y no hay MPLBACKEND en el
    entorno, usa 'Agg' (headless) para guardar imágenes sin GUI.
    """
    import matplotlib  # type: ignore
    env_backend = os.environ.get("MPLBACKEND")
    if backend:
        matplotlib.use(backend)
    elif default_to_agg and not env_backend:
        matplotlib.use("Agg")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Visualiza conectividad local de AKOrN")
    parser.add_argument("--crop", type=int, default=12, help="lado del recorte centrado (en celdas)")
    parser.add_argument("--step", type=int, default=1, help="stride de muestreo en la rejilla")
    parser.add_argument("--no-exclude-self", action="store_true", help="no eliminar self-loops")
    parser.add_argument("--imsize", type=int, default=128)
    parser.add_argument("--psize", type=int, default=4)
    parser.add_argument("--ksize", type=int, default=3)
    parser.add_argument("--save", type=str, default=None, help="ruta para guardar la figura")
    parser.add_argument(
        "--no-torch",
        action="store_true",
        help="forzar modo sin PyTorch (usa offsets por defecto)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend de Matplotlib (p.ej., 'MacOSX', 'TkAgg', 'Agg')",
    )
    parser.add_argument(
        "--render",
        type=str,
        choices=["mpl", "svg"],
        default="mpl",
        help="motor de renderizado: 'mpl' (Matplotlib) o 'svg' (sin dependencias)",
    )

    args = parser.parse_args()

    # Preparar entorno de visualización según motor
    if args.render == "mpl":
        # Importar opcionalmente networkx para dibujar directamente desde el grafo
        global NX_AVAILABLE, nx
        try:  # type: ignore
            import networkx as nx  # type: ignore
            NX_AVAILABLE = True
        except Exception:
            NX_AVAILABLE = False
            nx = None  # type: ignore
        # Elegir backend: si guardamos y el usuario no fija backend, usar 'Agg'.
        _setup_backend(args.backend, default_to_agg=bool(args.save))
    else:
        # En modo SVG no importamos Matplotlib ni NetworkX.
        pass

    # Obtener offsets del kernel desde el modelo si está Torch disponible
    # Importar PyTorch solo si el usuario no desactiva su uso
    if not args.__dict__["no_torch"]:
        global TORCH_AVAILABLE, torch
        try:
            import torch  # type: ignore
            TORCH_AVAILABLE = True
        except Exception:
            TORCH_AVAILABLE = False
            torch = None  # type: ignore

    if TORCH_AVAILABLE and not args.__dict__["no_torch"]:
        from source.models.objs.knet import AKOrN  # noqa: WPS433 (import interno)

        model = AKOrN(
            n=4,
            ch=4,
            L=1,
            T=8,
            J="conv",
            imsize=args.imsize,
            psize=args.psize,
            ksize=args.ksize,
        )
        klayer = model.layers[0][0]
        conv = klayer.connectivity  # nn.Conv2d
        kernel = conv.weight.detach().cpu().numpy()
        offsets = _kernel_offsets_from_tensor(kernel)
        h = w = model.imsize // model.psize
    else:
        # Fallback sin PyTorch: usar offsets por defecto y dimensiones derivadas
        offsets = _kernel_offsets_default(args.ksize)
        h = w = args.imsize // args.psize

    # Calcular recorte centrado.
    crop = min(max(1, int(args.crop)), max(h, w)) if args.crop is not None else None
    if crop is not None and (crop < h or crop < w):
        y0 = max(0, (h - crop) // 2)
        x0 = max(0, (w - crop) // 2)
        y_range = (y0, y0 + min(crop, h))
        x_range = (x0, x0 + min(crop, w))
    else:
        y_range = x_range = None

    graph = build_grid_graph(
        h,
        w,
        offsets,
        y_range=y_range,
        x_range=x_range,
        step=args.step,
        exclude_self=not args.__dict__["no_exclude_self"],
    )
    n_nodes = graph.number_of_nodes() if NX_AVAILABLE else len(graph["nodes"])  # type: ignore[index, attr-defined]
    title = f"AKOrN KLayer (conv) connectivity | nodes: {n_nodes}"
    # Asegurar directorio del archivo de salida si se pasa --save
    if args.save is not None:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    plot_graph(graph, title=title, savepath=args.save, renderer=args.render)


if __name__ == "__main__":
    main()
