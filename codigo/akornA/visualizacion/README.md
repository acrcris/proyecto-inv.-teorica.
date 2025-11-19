Visualización de la dinámica de Sudoku (AKOrN)

Uso rápido

- ID (SATNet, test) con 2 ejemplos por dificultad y T=128:
  `python visualizacion/visualizar_sudoku.py --model_path runs/sudoku_akorn/ema_49.pth --data id --samples_per_group 2 --T 128`

- OOD (RRN hard) con 3 ejemplos:
  `python visualizacion/visualizar_sudoku.py --model_path runs/sudoku_akorn/ema_49.pth --data ood --samples_per_group 3 --T 128`

- Elegir pasos específicos a mostrar:
  `--timesteps_to_show 1,8,16,32,64,128`

- Si cambiaste arquitectura al entrenar, pásala aquí también (debe coincidir):
  `--ch 512 --L 1 --heads 8 --N 4 --gamma 1.0 --J attn --use_omega true --global_omg true --init_omg 0.5 --learn_omg false`

Salida

- Se guardan PNGs en `visualizacion/outputs/<grupo>/sudoku_idx<N>.png`.
- Cada figura muestra el tablero inicial y varias instantáneas a lo largo de los timesteps.

Notas

- Asegúrate de tener los datos descargados: `data/sudoku/` (ID) y opcional `data/sudoku-rrn/` (OOD).
- El script usa MPS en Mac si está disponible, CUDA si existe, o CPU.
