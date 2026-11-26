"""Gera um candidato de plugues; não altera automaticamente o detector da câmera."""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path
import argparse
import torch
from ultralytics import YOLO
from test.treinar_cabeca import validar_dataset
from test.caminhos import RAIZ as ROOT, TREINOS, base_treinamento

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--verificar',action='store_true')
    args=p.parse_args();data=ROOT/'dataset/v3_plugues';validar_dataset(data)
    if args.verificar:
        print('Integridade do dataset de plugues conferida.');return
    if args.epochs<1:raise SystemExit('--epochs deve ser positivo.')
    torch.set_num_threads(2);m=YOLO(str(base_treinamento('plugue')))
    m.add_callback('on_pretrain_routine_start',lambda trainer:torch.set_num_threads(2))
    m.train(data=str(data/'data.yaml'),epochs=args.epochs,imgsz=416,batch=8,workers=0,
        device='cpu',freeze=10,optimizer='AdamW',lr0=.0007,warmup_epochs=2,warmup_bias_lr=0,
        cos_lr=True,patience=10,hsv_h=.035,hsv_s=.5,hsv_v=.3,degrees=8,scale=.2,translate=.05,
        mosaic=.1,close_mosaic=5,fliplr=.5,seed=3719,deterministic=True,
        project=str(TREINOS/'novas_variacoes'),name='plugues',plots=True)
    print('Candidato:',Path(m.trainer.save_dir)/'weights/best.pt')

if __name__=='__main__':main()
