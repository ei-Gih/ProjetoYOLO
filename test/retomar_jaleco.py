"""Retoma um treinamento de jaleco interrompido a partir de last.pt."""
from pathlib import Path
import argparse
import shutil
import torch
from ultralytics import YOLO

RAIZ=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--saida',type=Path,default=RAIZ/'jaleco_candidato.pt')
    args=p.parse_args()
    modelo=YOLO(str(args.checkpoint.resolve()))
    modelo.add_callback('on_pretrain_routine_start',lambda trainer: torch.set_num_threads(args.threads))
    modelo.train(resume=True,device='cpu',workers=0)
    best=Path(modelo.trainer.save_dir)/'weights'/'best.pt'
    shutil.copy2(best,args.saida)
    print('Treinamento concluído:',args.saida,flush=True)

if __name__=='__main__': main()
