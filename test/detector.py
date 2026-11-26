"""Mesmo pipeline de inferência para a webcam e o teste de imagens."""
import json
from pathlib import Path
from test.avaliacao import EPIS, EPIS_DESATIVADOS, PESSOAS, normalizar


def mesmo_objeto(nome, caixa, anteriores, limiar=.5):
    aliases = next((a for a in EPIS.values() if nome in a), {nome})
    for outro, b in anteriores:
        if outro not in aliases:
            continue
        inter = max(0, min(caixa[2], b[2])-max(caixa[0], b[0])) * max(0, min(caixa[3], b[3])-max(caixa[1], b[1]))
        uniao = (caixa[2]-caixa[0])*(caixa[3]-caixa[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        if inter/max(uniao, 1e-9) >= limiar:
            return True
    return False

def supressao_configurada(valores):
    resultado = {normalizar(n):float(v) for n,v in valores.items()}
    if any(not 0<v<=1 for v in resultado.values()):
        raise ValueError('A supressão por classe deve estar entre 0 e 1.')
    return resultado


def limiar_calibrado(nome, limiares, confianca=0.40):
    # 0,40 é a configuração padrão da interface. Ajustes explícitos preservam
    # a diferença entre os limiares medidos para cada classe.
    base = float(limiares.get(nome, 0.40))
    return max(0.05, min(0.95, base + confianca - 0.40))


class DetectorEPIs:
    def __init__(self, modelo_path, confianca=0.40):
        from ultralytics import YOLO
        import torch
        caminho = Path(modelo_path)
        self.confianca = confianca
        config_path = caminho.parent / 'calibracao.json'
        config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
        self.camera_resolucao = (int(config.get('camera_largura',1280)),int(config.get('camera_altura',720)))
        if any(not 32<=n<=4096 for n in self.camera_resolucao):
            raise ValueError('A resolução solicitada da câmera deve estar entre 32 e 4096 pixels.')
        self.cpu_threads = int(config.get('cpu_threads', 1))
        if not 1 <= self.cpu_threads <= 32:
            raise ValueError('cpu_threads deve estar entre 1 e 32.')
        torch.set_num_threads(self.cpu_threads)
        self.imgsz = int(config.get('imgsz', 512))
        principal = YOLO(str(caminho))
        pesos_carregados = {caminho.name: principal}
        self.classes = {normalizar(n) for n in principal.names.values()}
        relevantes = PESSOAS | set().union(*EPIS.values())
        self.modelos = []
        self.complementares = set()
        self.supressoes = {}
        if caminho.name == 'sh17_ppe.pt':
            self.supressoes[id(principal)] = supressao_configurada(config.get('principal_supressao', {}))
        # A calibração é específica dos pesos SH17 e dos complementos avaliados.
        especialista = config.get('especialista_maos_ouvidos', False) and caminho.name == 'sh17_ppe.pt'
        if especialista:
            especificacoes = config.get('especialistas', [{'arquivo':'maos_ouvidos.pt',
                'classes':config.get('classes_maos_ouvidos', ['luvas', 'protetor-auricular']),
                'limiares':config.get('maos_ouvidos', {})}])
            substituidas = set()
            aliases_ativos = set()
            for especificacao in especificacoes:
                ativas = {normalizar(n) for n in especificacao['classes']} - EPIS_DESATIVADOS
                if not ativas:
                    continue
                complemento = YOLO(str(caminho.parent / especificacao['arquivo']))
                pesos_carregados[especificacao['arquivo']] = complemento
                nomes = {normalizar(n) for n in complemento.names.values()}
                permitidas = set().union(*EPIS.values())
                # Um modelo pode também conhecer pessoas; só as classes de EPI
                # explicitamente selecionadas entram como complemento.
                if not ativas or not ativas <= permitidas:
                    raise ValueError('Selecione somente classes de EPIs reconhecidas no especialista.')
                epis_ativos = {epi for epi, aliases in EPIS.items() if aliases & ativas}
                complementar = especificacao.get('complementar', False)
                duplicada = bool(epis_ativos & substituidas) and not complementar
                if not ativas <= nomes or (ativas & aliases_ativos and not complementar) or duplicada:
                    raise ValueError('Classes de especialistas inválidas ou duplicadas.')
                substituidas.update(epis_ativos)
                aliases_ativos.update(ativas)
                if not complementar:
                    for epi in epis_ativos:
                        relevantes -= EPIS[epi]
                ids_complemento = [i for i, n in complemento.names.items() if normalizar(n) in ativas]
                tamanho = int(especificacao.get('imgsz', self.imgsz))
                if tamanho < 32 or tamanho % 32:
                    raise ValueError('A resolução do especialista deve ser um múltiplo de 32.')
                self.modelos.append((complemento, ids_complemento, especificacao.get('limiares', {}), tamanho))
                self.supressoes[id(complemento)] = supressao_configurada(especificacao.get('supressao', {}))
                if complementar:
                    self.complementares.add(id(complemento))
                self.classes.update(ativas)
        ids = [i for i, n in principal.names.items() if normalizar(n) in relevantes]
        if ids:
            limites_principal = config.get('principal', {}) if caminho.name == 'sh17_ppe.pt' else {}
            self.modelos.insert(0, (principal, ids, limites_principal, 640))
        jaleco = caminho.parent / 'jaleco.pt'
        if not self.classes & EPIS['Jaleco'] and jaleco.is_file():
            complemento = YOLO(str(jaleco))
            nomes = {normalizar(n) for n in complemento.names.values()}
            if nomes != {'jaleco'}:
                raise ValueError('jaleco.pt deve conter somente a classe jaleco.')
            self.modelos.append((complemento, None, config.get('jaleco', {}), self.imgsz))
            self.supressoes[id(complemento)] = supressao_configurada(config.get('jaleco_supressao', {}))
            self.classes.update(nomes)
        if not self.classes & PESSOAS:
            base = caminho.parent / 'yolo11n.pt'
            if not base.is_file():
                raise ValueError('Modelo sem classe pessoa e yolo11n.pt não encontrado.')
            self.modelos.append((YOLO(str(base)), [0], {}, 640))
            self.classes.add('person')
        self.zoom = None
        zoom = config.get('zoom_pessoas', {})
        if zoom.get('ativo', False) and caminho.name == 'sh17_ppe.pt':
            ativas = {normalizar(n) for n in zoom['classes']}
            permitidas = EPIS['Capacete'] | EPIS['Máscara'] | EPIS['Óculos de segurança']
            if not ativas or not ativas <= permitidas:
                raise ValueError('O recorte de cabeça aceita somente capacete, máscara e óculos.')
            modelo_zoom = pesos_carregados.get(zoom['arquivo'])
            if modelo_zoom is None:
                modelo_zoom = YOLO(str(caminho.parent / zoom['arquivo']))
            nomes = {normalizar(n) for n in modelo_zoom.names.values()}
            if not ativas <= nomes:
                raise ValueError('Classes do recorte ausentes no modelo.')
            tamanho = int(zoom.get('imgsz', 512))
            max_pessoas = int(zoom.get('max_pessoas', 2))
            if tamanho < 32 or tamanho % 32 or not 1 <= max_pessoas <= 4:
                raise ValueError('Resolução ou limite de pessoas inválido para recortes.')
            self.zoom = (modelo_zoom, [i for i,n in modelo_zoom.names.items() if normalizar(n) in ativas],
                         zoom.get('limiares', {}), tamanho, max_pessoas)
            self.classes.update(ativas)

        self.contexto = None
        contexto = config.get('contexto_proximo', {})
        if contexto.get('ativo', False) and caminho.name == 'sh17_ppe.pt':
            modelo = pesos_carregados.get(contexto['arquivo'])
            if modelo is None:
                modelo = YOLO(str(caminho.parent / contexto['arquivo']))
            ativas = {normalizar(n) for n in contexto['classes']}
            permitidas = EPIS['Capacete'] | EPIS['Máscara'] | EPIS['Óculos de segurança']
            nomes = {normalizar(n) for n in modelo.names.values()}
            escala = float(contexto.get('escala', .45))
            if not ativas or not ativas <= permitidas or not ativas <= nomes or not .1 <= escala <= 1:
                raise ValueError('Classes ou escala inválidas para contexto próximo.')
            self.contexto = (modelo, [i for i,n in modelo.names.items() if normalizar(n) in ativas],
                             contexto.get('limiares', {}), escala)
            self.classes.update(ativas)

    def prever(self, frame):
        import torch
        # Algumas inicializações do backend alteram o paralelismo global.
        # Reaplica o valor medido para evitar excesso de threads entre quadros.
        if torch.get_num_threads() != self.cpu_threads:
            torch.set_num_threads(self.cpu_threads)
        deteccoes = []
        self.detalhes = []
        imagem = frame.copy()
        for modelo, ids, limiares, imgsz in self.modelos:
            limites = {i: limiar_calibrado(normalizar(n), limiares, self.confianca)
                       for i, n in modelo.names.items() if ids is None or i in ids}
            resultado = modelo.predict(frame, classes=ids, conf=min(limites.values()),
                                       imgsz=imgsz, device='cpu', agnostic_nms=False, verbose=False)[0]
            if resultado.boxes is None:
                continue
            manter = [j for j, (c, score) in enumerate(zip(resultado.boxes.cls.tolist(), resultado.boxes.conf.tolist()))
                      if score >= limites[int(c)]]
            supressoes = self.supressoes.get(id(modelo), {})
            if supressoes:
                unicas=[];anteriores=[]
                for j in sorted(manter,key=lambda j:float(resultado.boxes.conf[j]),reverse=True):
                    nome=normalizar(resultado.names[int(resultado.boxes.cls[j])]);caixa=resultado.boxes.xyxy[j].tolist()
                    if nome not in PESSOAS and nome in supressoes and mesmo_objeto(nome,caixa,anteriores,supressoes[nome]):
                        continue
                    unicas.append(j);anteriores.append((nome,caixa))
                manter=unicas
            if id(modelo) in self.complementares:
                sem_repeticoes = []
                anteriores = list(deteccoes)
                for j in sorted(manter, key=lambda j: float(resultado.boxes.conf[j]), reverse=True):
                    nome = normalizar(resultado.names[int(resultado.boxes.cls[j])])
                    caixa = resultado.boxes.xyxy[j].tolist()
                    if not mesmo_objeto(nome, caixa, anteriores):
                        sem_repeticoes.append(j)
                        anteriores.append((nome, caixa))
                manter = sem_repeticoes
            resultado = resultado[manter]
            if not manter:
                continue
            imagem = resultado.plot(img=imagem)
            deteccoes.extend((normalizar(resultado.names[int(c)]), b)
                             for c, b in zip(resultado.boxes.cls.tolist(), resultado.boxes.xyxy.tolist()))
            self.detalhes.extend({'nome':normalizar(resultado.names[int(c)]), 'box':b, 'score':s, 'origem':'quadro'}
                for c,b,s in zip(resultado.boxes.cls.tolist(),resultado.boxes.xyxy.tolist(),resultado.boxes.conf.tolist()))
        if self.zoom is not None:
            modelo, ids, limiares, imgsz, max_pessoas = self.zoom
            altura, largura = frame.shape[:2]
            pessoas = [r for r in self.detalhes if r['nome'] in PESSOAS and r['score'] >= .6
                       and 32 <= r['box'][3]-r['box'][1] < .7*altura]
            pessoas.sort(key=lambda r:r['box'][3]-r['box'][1])
            limites = {i:limiar_calibrado(normalizar(modelo.names[i]),limiares,self.confianca) for i in ids}
            for pessoa in pessoas[:max_pessoas]:
                x1,y1,x2,y2 = pessoa['box'];w,h = x2-x1,y2-y1
                x1=max(0,int(x1-.1*w));x2=min(largura,int(x2+.1*w))
                y2=min(altura,int(y1+.5*h));y1=max(0,int(y1-.05*h))
                if min(x2-x1,y2-y1)<12:
                    continue
                resultado=modelo.predict(frame[y1:y2,x1:x2],classes=ids,conf=min(limites.values()),
                    imgsz=imgsz,device='cpu',agnostic_nms=False,verbose=False)[0]
                if resultado.boxes is None:
                    continue
                # Volta do recorte para as coordenadas da câmera, sem criar pessoas.
                dados=resultado.boxes.data
                resultado.boxes.data=dados.clone() if torch.is_tensor(dados) else dados.copy()
                resultado.boxes.data[:,[0,2]]+=x1
                resultado.boxes.data[:,[1,3]]+=y1
                manter=[]
                for j,(c,b,s) in enumerate(zip(resultado.boxes.cls.tolist(),resultado.boxes.xyxy.tolist(),resultado.boxes.conf.tolist())):
                    nome=normalizar(resultado.names[int(c)])
                    if s>=limites[int(c)] and not mesmo_objeto(nome,b,deteccoes):
                        manter.append(j);deteccoes.append((nome,b))
                        self.detalhes.append({'nome':nome,'box':b,'score':s,'origem':'recorte'})
                if manter:
                    imagem=resultado[manter].plot(img=imagem)
        if self.contexto is not None:
            import cv2
            import numpy as np
            modelo, ids, limiares, escala = self.contexto
            h,w = frame.shape[:2]
            rw=max(1,int(w/max(w,h)*640*escala));rh=max(1,int(h/max(w,h)*640*escala))
            x=(640-rw)//2;y=(640-rh)//2
            canvas=np.full((640,640,3),110,dtype=np.uint8)
            canvas[y:y+rh,x:x+rw]=cv2.resize(frame,(rw,rh))
            limites={i:limiar_calibrado(normalizar(modelo.names[i]),limiares,self.confianca) for i in ids}
            resultado=modelo.predict(canvas,classes=ids,conf=min(limites.values()),imgsz=512,
                                     device='cpu',agnostic_nms=False,verbose=False)[0]
            if resultado.boxes is not None:
                dados=resultado.boxes.data
                dados=dados.clone() if torch.is_tensor(dados) else dados.copy()
                manter=[]
                for j,(c,b,s) in enumerate(zip(resultado.boxes.cls.tolist(),resultado.boxes.xyxy.tolist(),resultado.boxes.conf.tolist())):
                    inter=max(0,min(b[2],x+rw)-max(b[0],x))*max(0,min(b[3],y+rh)-max(b[1],y))
                    if s<limites[int(c)] or inter/max(1,(b[2]-b[0])*(b[3]-b[1]))<.8:
                        continue
                    b=[max(0,(b[0]-x)*w/rw),max(0,(b[1]-y)*h/rh),min(w,(b[2]-x)*w/rw),min(h,(b[3]-y)*h/rh)]
                    nome=normalizar(resultado.names[int(c)])
                    if max(0,(b[2]-b[0])*(b[3]-b[1]))**.5*640/max(w,h)<96 or mesmo_objeto(nome,b,deteccoes):
                        continue
                    for k,v in enumerate(b):dados[j,k]=v
                    manter.append(j);deteccoes.append((nome,b))
                    self.detalhes.append({'nome':nome,'box':b,'score':s,'origem':'contexto'})
                resultado.boxes.data=dados
                if manter:imagem=resultado[manter].plot(img=imagem)
        return imagem, deteccoes
