import sys, os
import argparse
import math
from typing import List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt

from ema_pytorch import EMA

from source.models.sudoku.knet import SudokuAKOrN
from source.data.datasets.sudoku.sudoku import (
    SudokuDataset,
    HardSudokuDataset,
    convert_onehot_to_int,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_int_labels(Y_onehot: torch.Tensor) -> torch.Tensor:
    # [B, 9, 9, 9] -> [B, 9, 9] in 1..9
    return Y_onehot.argmax(dim=-1) + 1


def board_accuracy(pred_logits: torch.Tensor, Y_onehot: torch.Tensor, is_input: torch.Tensor) -> Tuple[int, int, int]:
    # pred_logits: [B, 9, 9, 9]
    # Y_onehot: [B, 9, 9, 9]
    # is_input: [B, 9, 9, 1]
    B = pred_logits.shape[0]
    pred = pred_logits.reshape((B, -1, 9)).argmax(-1)
    Y = Y_onehot.argmax(dim=-1).reshape(B, -1)
    mask = 1 - is_input.reshape(B, -1)
    num_blanks = int(mask.sum(1)[0].item())
    num_correct = int((mask * (pred == Y)).sum(1)[0].item())
    board_correct = int(num_correct == num_blanks)
    return num_blanks, num_correct, board_correct


def draw_sudoku(ax, given: np.ndarray, pred: np.ndarray = None, sol: np.ndarray = None, title: str = ""):
    # given, pred, sol: [9, 9] ints. given has 0 in blanks
    ax.set_aspect("equal")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    # Grid lines
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.plot([i, i], [0, 9], color="black", linewidth=lw)
        ax.plot([0, 9], [i, i], color="black", linewidth=lw)

    # Numbers
    for r in range(9):
        for c in range(9):
            if given[r, c] > 0:
                ax.text(c + 0.5, r + 0.6, str(int(given[r, c])), ha="center", va="center",
                        fontsize=14, color="black", fontweight="bold")
            elif pred is not None and pred[r, c] > 0:
                color = "tab:blue"
                if sol is not None and pred[r, c] != sol[r, c]:
                    color = "tab:red"
                ax.text(c + 0.5, r + 0.6, str(int(pred[r, c])), ha="center", va="center",
                        fontsize=14, color=color)
    ax.set_title(title, fontsize=11)


def select_examples_by_difficulty(dataset, k_per_group=2, seed=None):
    # Difficulty via number of blanks: more blanks => harder
    rng = np.random.default_rng(seed)
    blanks = []
    for i in range(len(dataset)):
        X, Y, is_input = dataset[i]
        nb = int((1 - is_input.reshape(-1)).sum().item())
        blanks.append(nb)
    blanks = np.array(blanks)
    q1, q2 = np.quantile(blanks, [0.33, 0.66])
    easy_idx = np.where(blanks <= q1)[0]
    med_idx = np.where((blanks > q1) & (blanks <= q2))[0]
    hard_idx = np.where(blanks > q2)[0]
    def sample(arr):
        if len(arr) == 0:
            return []
        idx = rng.choice(arr, size=min(k_per_group, len(arr)), replace=False)
        return idx.tolist()
    return {
        "easy": sample(easy_idx),
        "medium": sample(med_idx),
        "hard": sample(hard_idx),
    }


def main():
    parser = argparse.ArgumentParser(description="Visualiza la dinámica de resolución de Sudoku")
    parser.add_argument("--model_path", type=str, required=True, help="Ruta a ema_*.pth o ema_model.pth")
    parser.add_argument("--data", type=str, default="id", choices=["id", "ood"], help="Dataset: id u ood")
    parser.add_argument("--data_root", type=str, default=None, help="Ruta raíz alternativa a los datos")
    parser.add_argument("--outdir", type=str, default="visualizacion/outputs", help="Directorio de salida")
    parser.add_argument("--samples_per_group", type=int, default=2, help="Muestras por grupo de dificultad (solo ID)")
    parser.add_argument("--indices", type=str, default=None, help="Lista de índices manuales (coma) en lugar de muestreo por dificultad")
    parser.add_argument("--seed", type=int, default=None, help="Semilla para muestrear ejemplos por dificultad (None=aleatorio)")
    parser.add_argument("--T", type=int, default=128, help="Timesteps para la evaluación/visualización")
    parser.add_argument("--timesteps_to_show", type=str, default=None, help="Lista de pasos a mostrar, p.ej. '1,8,16,32,64,128'")
    parser.add_argument("--snap_every", type=int, default=None, help="Guardar una instantánea cada N pasos (incluye T)")
    parser.add_argument("--num_snaps", type=int, default=None, help="Número de instantáneas equiespaciadas (por defecto más denso)")
    parser.add_argument("--grid_cols", type=int, default=6, help="Columnas máximas en la grilla de figuras")
    parser.add_argument("--save_separate", action="store_true", help="Guardar además una imagen por cada paso seleccionado")
    # Arquitectura (deben coincidir con la usada para entrenar)
    parser.add_argument("--ch", type=int, default=512)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--J", type=str, default="attn")
    parser.add_argument("--use_omega", type=lambda x: x.lower() in ("1","true","yes"), default=True)
    parser.add_argument("--global_omg", type=lambda x: x.lower() in ("1","true","yes"), default=True)
    parser.add_argument("--init_omg", type=float, default=0.1)
    parser.add_argument("--learn_omg", type=lambda x: x.lower() in ("1","true","yes"), default=False)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # Carga dataset
    if args.data == "id":
        droot = args.data_root if args.data_root is not None else "./data/sudoku"
        dataset = SudokuDataset(droot, train=False)
    else:
        droot = args.data_root if args.data_root is not None else "./data/sudoku-rrn"
        dataset = HardSudokuDataset(droot, split="test")

    # Carga modelo EMA
    net = SudokuAKOrN(
        n=args.N,
        ch=args.ch,
        L=args.L,
        T=args.T,
        gamma=args.gamma,
        J=args.J,
        use_omega=args.use_omega,
        global_omg=args.global_omg,
        init_omg=args.init_omg,
        learn_omg=args.learn_omg,
        nl=True,
        heads=args.heads,
    )
    net.to(device)

    ema = EMA(net).to(device)
    ckpt = torch.load(args.model_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ema.load_state_dict(state)
    model = ema.ema_model.to(device)
    model.eval()

    # Timesteps a visualizar
    if args.timesteps_to_show is not None:
        ts = sorted({max(1, min(args.T, int(t.strip()))) for t in args.timesteps_to_show.split(",") if t.strip()})
    elif args.snap_every is not None and args.snap_every > 0:
        ts = list(range(1, args.T + 1, args.snap_every))
        if ts[-1] != args.T:
            ts.append(args.T)
    else:
        # Por defecto, más denso que antes
        num = args.num_snaps if args.num_snaps is not None else min(12, args.T)
        ts = sorted({max(1, int(round(x))) for x in np.linspace(1, args.T, num)})

    # Selección de ejemplos
    groups = {}
    if args.indices is not None:
        idxs = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
        groups = {"custom": idxs}
    elif args.data == "id":
        groups = select_examples_by_difficulty(
            dataset,
            k_per_group=args.samples_per_group,
            seed=args.seed,
        )
    else:
        # Para OOD simplemente toma primeras N muestras
        groups = {"hard": list(range(min(args.samples_per_group, len(dataset))))}

    # Visualización
    for group_name, idx_list in groups.items():
        for idx in idx_list:
            X, Y, is_input = dataset[idx]
            X = X.unsqueeze(0).to(torch.int32).to(device)  # [1, 9, 9, 9]
            Y = Y.unsqueeze(0).to(device)
            is_input = is_input.unsqueeze(0).to(device)

            with torch.no_grad():
                ret = model(X, is_input, return_xs=True, return_es=True)
            if isinstance(ret, list) and len(ret) == 3:
                out_logits, xs_layers, es_layers = ret
            else:
                # Fallback: solo logits
                out_logits = ret
                xs_layers, es_layers = None, None

            given = convert_onehot_to_int(X).squeeze(0).cpu().numpy()  # [9,9], 0 en blancos
            sol = to_int_labels(Y).squeeze(0).cpu().numpy()

            # Predicciones por paso usando la última capa
            preds_over_time: List[np.ndarray] = []
            if xs_layers is not None and len(xs_layers) > 0:
                last_layer_xs: List[torch.Tensor] = xs_layers[-1]  # len T, cada [1, ch, 9, 9]
                for t in ts:
                    x_t = last_layer_xs[t - 1]
                    logits_t = model.out(x_t).permute(0, 2, 3, 1)  # [1,9,9,9]
                    pred_t = (logits_t.argmax(dim=-1) + 1).squeeze(0).cpu().numpy()  # [9,9]
                    preds_over_time.append(pred_t)
            else:
                # Si no hay xs, usa solo la salida final para todos
                pred_final = (out_logits.argmax(dim=-1) + 1).squeeze(0).cpu().numpy()
                preds_over_time = [pred_final for _ in ts]

            # Métrica final
            nb, nc, bc = board_accuracy(out_logits, Y, is_input)

            # Figura en grilla (varias instantáneas)
            total_panels = 1 + len(ts)
            cols = min(args.grid_cols, total_panels)
            rows = math.ceil(total_panels / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
            if isinstance(axes, np.ndarray):
                axes = axes.flatten().tolist()
            else:
                axes = [axes]
            draw_sudoku(axes[0], given, None, sol, title=f"Inicial (idx={idx})")
            for j, t in enumerate(ts, start=1):
                draw_sudoku(axes[j], given, preds_over_time[j - 1], sol, title=f"Paso t={t}")
            # Apaga ejes sobrantes si hay
            for k in range(1 + len(ts), len(axes)):
                axes[k].axis('off')

            fig.suptitle(f"Grupo: {group_name} | blanks={int((given==0).sum())} | BoardCorrect={bc}", fontsize=12)
            fig.tight_layout()

            os.makedirs(os.path.join(args.outdir, group_name), exist_ok=True)
            out_dir = os.path.join(args.outdir, group_name)
            out_path = os.path.join(out_dir, f"sudoku_idx{idx}.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Guardado (grilla): {out_path}")

            # Guardar imágenes separadas por paso si se solicita
            if args.save_separate:
                for t, pred_t in zip(ts, preds_over_time):
                    fig2, axes2 = plt.subplots(1, 2, figsize=(6.4, 3.2))
                    draw_sudoku(axes2[0], given, None, sol, title="Inicial")
                    draw_sudoku(axes2[1], given, pred_t, sol, title=f"Paso t={t}")
                    fig2.tight_layout()
                    sep_path = os.path.join(out_dir, f"sudoku_idx{idx}_t{t}.png")
                    fig2.savefig(sep_path, dpi=150)
                    plt.close(fig2)
                print(f"Guardadas {len(ts)} imágenes separadas en {out_dir}")


if __name__ == "__main__":
    main()
