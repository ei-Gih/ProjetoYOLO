"""Repete o ajuste V2 com datasets revisados; nunca substitui pesos em uso."""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

from test.caminhos import RAIZ, TREINOS, base_treinamento


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alvo', choices=('jaleco', 'maos_ouvidos'), required=True)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--conservador', action='store_true', help='Segundo ajuste de luvas/protetores com taxa de aprendizado menor.')
    args = parser.parse_args()
    jaleco = args.alvo == 'jaleco'
    conservador = args.conservador and not jaleco
    base = args.base or base_treinamento(args.alvo)
    dataset = RAIZ/'dataset'/('v2_jalecos' if jaleco else 'v2_maos_ouvidos')/'data.yaml'
    if not base.is_file() or not dataset.is_file():
        raise SystemExit('Pesos iniciais ou dataset V2 não encontrados.')
    modelo = YOLO(str(base))
    modelo.add_callback('on_pretrain_routine_start', lambda trainer: torch.set_num_threads(8))
    modelo.train(data=str(dataset), epochs=args.epochs or (18 if jaleco else 6 if conservador else 12),
                 imgsz=416, batch=8, workers=0, device='cpu', freeze=10,
                 optimizer='AdamW', lr0=.00015 if jaleco else .00003 if conservador else .0002,
                 warmup_epochs=1, warmup_bias_lr=0, cos_lr=True, patience=6 if jaleco else 4 if conservador else 5,
                 hsv_h=.08 if jaleco else .04 if conservador else .06, hsv_s=.65 if jaleco else .5 if conservador else .7, hsv_v=.3 if conservador else .4,
                 degrees=5 if conservador else 8, scale=.3 if jaleco else .2 if conservador else .35, translate=.05 if conservador else .1,
                 mosaic=.25 if jaleco else .1 if conservador else .3, close_mosaic=5 if jaleco else 2 if conservador else 4,
                 fliplr=.5, seed=2026, deterministic=True,
                 project=str(TREINOS/'novas_variacoes'), name=args.alvo, plots=True)
    print('Candidato:', Path(modelo.trainer.save_dir)/'weights/best.pt')
    print('Compare as métricas antes de substituir o modelo usado pela câmera.')


if __name__ == '__main__':
    main()
