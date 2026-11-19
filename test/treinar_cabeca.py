"""Treina um candidato para protetores, máscaras e capacetes, sem ativá-lo automaticamente."""
import argparse
from pathlib import Path
import hashlib
import json
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parent

def validar_dataset(dataset):
    vistos={}
    manifest=dataset/'manifesto.json'
    registros=json.loads(manifest.read_text(encoding='utf8')) if manifest.is_file() else []
    esperados={(m['split'],m['name']):m['sha256'] for m in registros}
    for split in ('train','val','test'):
        images=list((dataset/'images'/split).glob('*'))
        if not images: raise ValueError(f'Conjunto vazio: {split}')
        for image in images:
            sha=hashlib.sha256(image.read_bytes()).hexdigest()
            esperado=esperados.get((split,image.name))
            if esperado and esperado!=sha:
                raise ValueError(f'Imagem alterada após a anotação: {image}. Revise caixas e manifesto antes de treinar.')
            if sha in vistos and vistos[sha]!=split:
                raise ValueError(f'Imagem repetida entre {vistos[sha]} e {split}: {image.name}')
            vistos[sha]=split
            label=dataset/'labels'/split/(image.stem+'.txt')
            if not label.is_file(): raise ValueError(f'Anotação ausente: {label}')
            for line in label.read_text(encoding='utf8').splitlines():
                c,x,y,w,h=map(float,line.split())
                if c not in (0,1,2) or not all(0<=v<=1 for v in (x,y,w,h)) or w<=0 or h<=0:
                    raise ValueError(f'Anotação inválida: {label}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--base',type=Path,default=ROOT/'runs/melhoria_v3/bases/cabeca_refinada.pt')
    p.add_argument('--verificar',action='store_true',help='Verifica os dados sem iniciar treinamento.')
    args=p.parse_args()
    dataset=ROOT/'dataset/v3_cabeca'
    validar_dataset(dataset)
    if args.verificar:
        print('Dataset V3 verificado: anotações presentes, hashes conferidos e sem cópias exatas entre conjuntos.')
        return
    if not args.base.is_file(): raise SystemExit('Pesos iniciais não encontrados.')
    if args.epochs<1:raise SystemExit('--epochs deve ser positivo.')
    torch.set_num_threads(8)
    m=YOLO(str(args.base))
    if list(m.names.values())!=['protetor-auricular','mascara','capacete']:
        raise SystemExit('O modelo inicial deve ter as três classes do dataset V3.')
    m.add_callback('on_pretrain_routine_start',lambda trainer:torch.set_num_threads(8))
    m.train(data=str(dataset/'data.yaml'),epochs=args.epochs,imgsz=512,batch=8,workers=0,
        device='cpu',freeze=10,optimizer='AdamW',lr0=.0001,warmup_epochs=1,warmup_bias_lr=0,
        cos_lr=True,patience=6,hsv_h=.03,hsv_s=.5,hsv_v=.3,degrees=5,scale=.15,translate=.04,
        mosaic=.05,close_mosaic=3,fliplr=.5,seed=3719,deterministic=True,
        project=str(ROOT/'runs/novas_variacoes'),name='cabeca',plots=True)
    print('Candidato:',Path(m.trainer.save_dir)/'weights/best.pt')
    print('Avalie em validação e teste antes de mudar calibracao.json.')

if __name__=='__main__':main()
