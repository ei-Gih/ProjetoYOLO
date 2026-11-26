"""Treinamento complementar de jaleco; preserva o detector dos outros EPIs."""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path
import argparse
import shutil
import yaml
import torch
from ultralytics import YOLO

from test.caminhos import RAIZ, TREINOS, base_treinamento

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--imgsz',type=int,default=416)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument("--dataset",type=Path,default=RAIZ/"dataset"/"jaleco")
    p.add_argument("--base",type=Path,default=RAIZ/"yolo11n.pt")
    p.add_argument("--saida",type=Path,default=RAIZ/"jaleco.pt")
    p.add_argument("--threads",type=int,default=8)
    args=p.parse_args()
    if min(args.epochs,args.imgsz,args.batch,args.threads)<=0:
        raise SystemExit("Parâmetros devem ser positivos.")
    dataset=args.dataset.resolve()
    for split in ('train','val'):
        imagens=list((dataset/'images'/split).glob('*.jpg'))
        if not imagens: raise SystemExit(f'Sem imagens em {split}')
        positivos=0
        for im in imagens:
            label=dataset/'labels'/split/(im.stem+'.txt')
            if not label.is_file(): raise SystemExit(f'Falta anotação: {label}')
            for linha in label.read_text().splitlines():
                c,x,y,w,h=map(float,linha.split())
                if c!=0 or not(0<w<=1 and 0<h<=1 and -0.000001<=x-w/2 and x+w/2<=1.000001 and -0.000001<=y-h/2 and y+h/2<=1.000001):
                    raise SystemExit(f'Caixa inválida: {label}')
                positivos+=1
        print(split, 'imagens:',len(imagens),'jalecos:',positivos,flush=True)
        if positivos==0: raise SystemExit(f'Sem jalecos em {split}')
    config={'path':str(dataset.resolve()),'train':'images/train','val':'images/val','names':{0:'jaleco'}}
    dados=dataset/'data.yaml'; dados.write_text(yaml.safe_dump(config),encoding='utf-8')
    modelo=YOLO(str(args.base.resolve()))
    modelo.add_callback("on_pretrain_routine_start",lambda trainer: torch.set_num_threads(args.threads))
    resultado=modelo.train(data=str(dados),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,
        device='cpu',workers=0,freeze=10,patience=10,seed=42,deterministic=True,
        project=str(TREINOS/'train'),name='jaleco',exist_ok=False,
        plots=True,cache=False,optimizer='AdamW',lr0=0.0005)
    best=Path(resultado.save_dir)/'weights'/'best.pt'
    if not best.is_file(): raise SystemExit('Treino não gerou best.pt')
    shutil.copy2(best,args.saida)
    print(f'Detector complementar salvo em {args.saida}',flush=True)

if __name__=='__main__': main()


