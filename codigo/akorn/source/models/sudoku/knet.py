import torch
import torch.nn as nn
from source.layers.klayer import (
    KLayer,
)
from source.layers.common_layers import (
    ReadOutConv,
    BNReLUConv2d,
    FF,
    ResBlock,
)
from source.layers.common_fns import positionalencoding2d


from source.data.datasets.sudoku.sudoku import convert_onehot_to_int


class SudokuAKOrN(nn.Module):

    def __init__(
        self,
        n,
        ch=64,
        L=1,
        T=16,
        gamma=1.0,
        J="attn",
        use_omega=True,
        global_omg=True,
        init_omg=0.1,
        learn_omg=False,
        nl=True,
        heads=8,
    ):
        """
        Parámetros del constructor:
        - n: tamaño del problema (p.ej., n=9 en Sudoku) que usan KLayer y ReadOutConv.
        - ch: número de canales de características internos (ancho del embedding y de las capas).
        - L: número de bloques apilados [KLayer -> ReadOut]; controla la profundidad del modelo.
        - T: número de iteraciones/steps de evolución interna que ejecuta cada KLayer.
        - gamma: factor/ganancia de acoplamiento de la dinámica interna (parámetro aprendible).
        - J: tipo de "interacción" en KLayer (p.ej., "attn" usa atención como acoplamiento).
        - use_omega, global_omg, init_omg, learn_omg: controlan frecuencia/ganancia de los osciladores
          internos del KLayer (si es global, su valor inicial y si se aprende o no).
        - nl: si True, agrega normalización + no linealidades en el bloque de lectura; si False, identidad.
        - heads: número de cabezas de atención internas usadas por KLayer.
        """
        super().__init__()
        self.n = n  # tamaño del problema; guía dimensiones/constraints usadas por KLayer/ReadOut
        self.L = L  # número de bloques [KLayer + ReadOut] a apilar (profundidad)
        self.ch = ch  # canales de características (ancho de la representación interna)
        self.embedding = nn.Embedding(10, ch)  # mapea cada dígito 0–9 a un vector en R^ch

        hw = [9, 9]  # tamaño espacial H×W del tablero; usado por KLayer para el layout 2D

        self.layers = nn.ModuleList()  # contendrá L pares: (KLayer, proyección/lectura)
        # Apilamos L bloques. Cada bloque contiene:
        # 1) KLayer: evoluciona x durante T pasos con acoplamiento gamma condicionado por c.
        # 2) readout (nn.Sequential): proyecta la salida de KLayer y actualiza la C del siguiente bloque.
        for l in range(self.L):
            self.layers.append(
                nn.ModuleList(
                    [
                        KLayer(
                            n,
                            ch,
                            J,
                            use_omega=use_omega,
                            c_norm=None,
                            hw=hw,
                            global_omg=global_omg,
                            init_omg=init_omg,
                            heads=heads,
                            learn_omg=learn_omg,
                            gta=True,
                        ),
                        nn.Sequential(
                            ReadOutConv(ch, ch, n, 1, 1, 0),
                            ResBlock(FF(ch, ch, ch, 1, 1, 0)) if nl else nn.Identity(),
                            BNReLUConv2d(ch, ch, 1, 1, 0) if nl else nn.Identity(),
                        ),
                    ]
                )
            )

        self.out = nn.Sequential(nn.ReLU(), nn.Conv2d(ch, 9, 1, 1, 0))  # cabezal final: 9 logits por celda (dígitos 1–9)

        self.T = T  # número de pasos de evolución interna (Kuramoto) por KLayer
        self.gamma = torch.nn.Parameter(torch.Tensor([gamma]))  # gamma aprendible (tamaño de paso de la dinámica)
        self.fixed_noise = False  # si True, fija la semilla del ruido inicial (útil para depurar)
        self.x0 = nn.Parameter(torch.randn(1, ch, 9, 9))  # estado inicial aprendible opcional para el tablero 9×9

    def feature(self, inp, is_input):
        # inp: torch.Tensor of shape [B, 9, 9, 9] the last dim repreents the digit in the one-hot representation.
        inp = convert_onehot_to_int(inp)
        c = self.embedding(inp).permute(0, 3, 1, 2)
        is_input = is_input.permute(0, 3, 1, 2)
        xs = []
        es = []

        # generate random oscillatores
        if self.fixed_noise:
            n = torch.randn(
                *(c.shape), generator=torch.Generator(device="cpu").manual_seed(42)
            ).to(c.device)
            x = is_input * c + (1 - is_input) * n
        else:
            n = torch.randn_like(c)
            x = is_input * c + (1 - is_input) * n

        for _, (klayer, readout) in enumerate(self.layers):
            # Process x and c.
            _xs, _es = klayer(
                x,
                c,
                self.T,
                self.gamma,
            )
            xs.append(_xs)
            es.append(_es)
            
            x = _xs[-1]
            c = readout(x)

        return c, xs, es

    def forward(self, c, is_input, return_xs=False, return_es=False):
        out, xs, es = self.feature(c, is_input)
        out = self.out(out).permute(0, 2, 3, 1)
        ret = [out]
        if return_xs:
            ret.append(xs)
        if return_es:
            ret.append(es)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret


