"""Herramientas para analizar criticalidad en AKOrN reutilizando las métricas del módulo minimalista."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from analisis_criticalidad_minimalista.analisis.criticalidad import (
    DFA,
    KuramotoMetrics,
    PSD,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AKORN_DIR = REPO_ROOT / "alternativa1" / "codigo" / "akorn"
if not AKORN_DIR.exists():
    AKORN_DIR = REPO_ROOT / "codigo" / "akorn"
if str(AKORN_DIR) not in sys.path:
    sys.path.insert(0, str(AKORN_DIR))

from source.models.objs.knet import AKOrN  # noqa: E402


def _to_list(array: np.ndarray) -> List[float]:
    return array.astype(float).tolist()


def _format_sample_result(raw: Dict[str, np.ndarray]) -> Dict[str, object]:
    return {
        "psd_slope": float(raw["psd_slope"]),
        "dfa_alpha": float(raw["dfa_alpha"]),
        "variance": float(raw["variance"]),
        "order_parameter": _to_list(raw["order_parameter"]),
        "global_series": _to_list(raw["global_series"]),
        "magnitude_series": [_to_list(row) for row in raw["magnitude_series"]],
        "energy": _to_list(raw["energy"]),
        "order_max": float(raw["order_parameter"].max()),
        "order_min": float(raw["order_parameter"].min()),
        "global_mean_final": float(raw["global_series"][-1]),
    }


class AKOrNCriticalityAnalyzer:
    def __init__(
        self,
        model_kwargs: Dict[str, object],
        *,
        checkpoint: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = AKOrN(**model_kwargs).to(self.device)
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location=self.device)
            if isinstance(state, dict):
                for key in ("model_state_dict", "state_dict", "model", "module"):
                    if key in state:
                        state = state[key]
                        break
            self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def analyze_batch(
        self,
        batch: torch.Tensor,
        *,
        capture_layers: Optional[Iterable[int]] = None,
    ) -> Dict[str, Dict[str, object]]:
        batch = batch.to(self.device)
        with torch.no_grad():
            _, xs, es = self.model(batch, return_xs=True, return_es=True)
        layer_states = xs[1:]
        layer_energies = es[1:]
        selected = set(capture_layers) if capture_layers is not None else None
        results: Dict[str, Dict[str, object]] = {}
        for idx, (states, energy) in enumerate(zip(layer_states, layer_energies)):
            if selected is not None and idx not in selected:
                continue
            summary, samples = self._analyze_layer(states, energy)
            results[f"layer_{idx}"] = {"summary": summary, "samples": samples}
        return results

    def _analyze_layer(
        self,
        states: List[torch.Tensor],
        energy: List[torch.Tensor],
    ) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
        batch_size = states[0].shape[0]
        sample_reports: List[Dict[str, np.ndarray]] = []
        for sample_idx in range(batch_size):
            sample_states = [step[sample_idx : sample_idx + 1] for step in states]
            order_series = KuramotoMetrics.order_parameter(sample_states)
            magnitude_series = KuramotoMetrics.magnitudes_mean_series(sample_states)
            global_series = magnitude_series.mean(axis=1)
            _, _, psd_slope = PSD.psd_slope(global_series)
            _, _, dfa_alpha = DFA.dfa(global_series)
            energy_series = np.array([entry[sample_idx].item() for entry in energy[1:]], dtype=float)
            sample_reports.append(
                {
                    "psd_slope": psd_slope,
                    "dfa_alpha": dfa_alpha,
                    "variance": np.var(global_series),
                    "order_parameter": order_series,
                    "global_series": global_series,
                    "magnitude_series": magnitude_series,
                    "energy": energy_series,
                }
            )
        slopes = np.array([r["psd_slope"] for r in sample_reports], dtype=float)
        dfas = np.array([r["dfa_alpha"] for r in sample_reports], dtype=float)
        variances = np.array([r["variance"] for r in sample_reports], dtype=float)
        summary = {
            "psd_slope_mean": float(slopes.mean()),
            "psd_slope_std": float(slopes.std()),
            "dfa_alpha_mean": float(dfas.mean()),
            "dfa_alpha_std": float(dfas.std()),
            "variance_mean": float(variances.mean()),
            "variance_std": float(variances.std()),
        }
        formatted = [_format_sample_result(r) for r in sample_reports]
        return summary, formatted


def _summarize_metric(
    gamma_values: List[float],
    all_results: Dict[str, object],
    metric: str,
) -> Dict[str, object]:
    series: List[Dict[str, float]] = []
    for gamma in gamma_values:
        entry = all_results[str(gamma)]
        layer_values = []
        for layer in entry["layers"].values():
            summary = layer["summary"]
            if metric in summary:
                layer_values.append(float(summary[metric]))
        if layer_values:
            series.append(
                {
                    "gamma": float(gamma),
                    "mean": float(np.mean(layer_values)),
                    "std": float(np.std(layer_values)),
                }
            )
    if not series:
        return {"series": []}
    best = max(series, key=lambda item: item["mean"])
    return {
        "series": series,
        "best_gamma": best["gamma"],
        "best_mean": best["mean"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analiza criticalidad de AKOrN con métricas minimalistas",
    )
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[8])
    parser.add_argument("--psize", type=int, default=4)
    parser.add_argument("--ksize", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--imsize", type=int, default=128)
    parser.add_argument("--use-omega", action="store_true")
    parser.add_argument("--global-omega", action="store_true")
    parser.add_argument("--init-omega", type=float, default=1.0)
    parser.add_argument("--J", choices=["attn", "conv"], default="attn")
    parser.add_argument("--disable-gta", action="store_true")
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gamma-sweep", type=float, nargs="+")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", type=str)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-tensor", type=Path)
    parser.add_argument("--input-npy", type=Path)
    parser.add_argument("--random-input", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inspect-layers", type=int, nargs="+")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_input(args: argparse.Namespace) -> torch.Tensor:
    torch.manual_seed(args.seed)
    if args.input_tensor is not None:
        data = torch.load(args.input_tensor, map_location="cpu")
    elif args.input_npy is not None:
        data = torch.tensor(np.load(args.input_npy), dtype=torch.float32)
    else:
        if not args.random_input and args.input_tensor is None and args.input_npy is None:
            args.random_input = True
        data = torch.randn(args.batch_size, 3, args.imsize, args.imsize)
    if isinstance(data, torch.Tensor):
        tensor = data.clone().detach()
    else:
        tensor = torch.tensor(data)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    return tensor.float()


def _prepare_model_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    timesteps = args.timesteps
    if len(timesteps) == 1:
        T: object = timesteps[0]
    else:
        T = timesteps
    return {
        "n": args.n,
        "ch": args.channels,
        "L": args.layers,
        "T": T,
        "psize": args.psize,
        "gta": not args.disable_gta,
        "J": args.J,
        "ksize": args.ksize,
        "imsize": args.imsize,
        "use_omega": args.use_omega,
        "init_omg": args.init_omega,
        "global_omg": args.global_omega,
        "heads": args.heads,
    }


def main() -> None:
    args = _parse_args()
    base_kwargs = _prepare_model_kwargs(args)
    batch = _load_input(args)
    gamma_values: List[float]
    if args.gamma_sweep is not None and len(args.gamma_sweep) > 0:
        gamma_values = [float(g) for g in args.gamma_sweep]
    elif args.gamma is not None:
        gamma_values = [float(args.gamma)]
    else:
        gamma_values = [1.0]
    report: Dict[str, object] = {
        "model_kwargs": base_kwargs,
        "gamma_values": gamma_values,
        "input_shape": list(batch.shape),
    }
    all_results: Dict[str, object] = {}
    for gamma in gamma_values:
        kwargs = dict(base_kwargs)
        kwargs["gamma"] = float(gamma)
        analyzer = AKOrNCriticalityAnalyzer(
            kwargs,
            checkpoint=args.checkpoint,
            device=args.device,
        )
        layer_metrics = analyzer.analyze_batch(
            batch,
            capture_layers=args.inspect_layers,
        )
        all_results[str(gamma)] = {
            "gamma": float(gamma),
            "layers": layer_metrics,
        }
    report["results"] = all_results
    report["summary"] = {
        "dfa_alpha_mean": _summarize_metric(gamma_values, all_results, "dfa_alpha_mean"),
        "variance_mean": _summarize_metric(gamma_values, all_results, "variance_mean"),
        "psd_slope_mean": _summarize_metric(gamma_values, all_results, "psd_slope_mean"),
    }
    payload = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
