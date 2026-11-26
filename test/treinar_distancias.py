"""Treino dos cinco EPIs: cabeça/mãos e jaleco separados, com cenas próximas e distantes."""
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,hashlib
from test.caminhos import DATASETS,BASES,TREINOS

def verificar(dataset,nc):
    manifest=json.loads((dataset/'manifesto.json').read_text(encoding='utf8'))
    groups={};hashes={}
    for r in manifest:
        split=r['split'];path=dataset/'images'/split/r['name']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:
            raise ValueError(f'Imagem alterada: {path}. Revise sua anotação antes de treinar.')
        if groups.setdefault(str(r['parent']),split)!=split or hashes.setdefault(r['sha256'],split)!=split:
            raise ValueError(f'Imagem ou variação repetida entre treino e reserva: {path}')
        for line in (dataset/'labels'/split/(path.stem+'.txt')).read_text().splitlines():
            c,x,y,w,h=map(float,line.split())
            if c not in range(nc) or not all(0<=v<=1 for v in (x,y,w,h)) or min(w,h)<=0:
                raise ValueError(f'Anotação inválida: {path}')
    print(f'{len(manifest)} imagens verificadas; os grupos de origem estão separados.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alvo',choices=['epis','jaleco'],default='epis')
    p.add_argument('--verificar',action='store_true')
    p.add_argument('--retomar',type=Path,help='last.pt de um treino interrompido.')
    p.add_argument('--base',type=Path,help='Peso inicial para um novo ajuste.')
    p.add_argument('--epochs',type=int)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();dataset=DATASETS/f'v5_{a.alvo}_distancias'
    verificar(dataset,4 if a.alvo=='epis' else 1)
    if a.verificar:return
    if not 1<=a.threads<=32:raise SystemExit('--threads deve estar entre 1 e 32.')
    epochs=a.epochs if a.epochs is not None else (12 if a.alvo=='epis' else 8)
    if epochs<1:raise SystemExit('--epochs deve ser positivo.')
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(a.threads)
    model=YOLO(str(a.retomar or a.base or BASES/f'{a.alvo}_v5_base.pt'))
    model.add_callback('on_pretrain_routine_start',lambda trainer:torch.set_num_threads(a.threads))
    if a.retomar:
        if model.ckpt.get('epoch',-1)<0:raise SystemExit('Este treino terminou. Use --base para iniciar outro ajuste.')
        model.train(resume=True,device='cpu',workers=0)
    else:
        expected=['capacete','mascara','oculos','luvas'] if a.alvo=='epis' else ['jaleco']
        if list(model.names.values())!=expected:raise SystemExit(f'Classes incompatíveis. Esperadas: {expected}')
        model.train(data=str(dataset/'data.yaml'),epochs=epochs,imgsz=512 if a.alvo=='epis' else 416,
            batch=8,workers=0,device='cpu',freeze=10,optimizer='AdamW',
            lr0=.0002 if a.alvo=='epis' else .00008,warmup_epochs=1,warmup_bias_lr=0,
            cos_lr=True,patience=6,cls_remap=False,hsv_h=.025,hsv_s=.4,hsv_v=.3,
            degrees=5,scale=.35,translate=.08,mosaic=.1,close_mosaic=min(3,epochs),
            fliplr=.5,seed=5020,deterministic=True,project=str(TREINOS),name=f'v5_{a.alvo}',plots=True)
    print('Avalie o candidato nas três distâncias e nos pares máscara/óculos antes de ativar os pesos.')

if __name__=='__main__':main()
