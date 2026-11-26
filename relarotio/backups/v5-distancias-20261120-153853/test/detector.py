"""Mesmo pipeline de inferência para a webcam e o teste de imagens."""
import json
from pathlib import Path
from test.avaliacao import EPIS, EPIS_DESATIVADOS, PESSOAS, normalizar


def mesmo_objeto(nome, caixa, anteriores):
    aliases = next((a for a in EPIS.values() if nome in a), {nome})
    for outro, b in anteriores:
        if outro not in aliases:
            continue
        inter = max(0, min(caixa[2], b[2])-max(caixa[0], b[0])) * max(0, min(caixa[3], b[3])-max(caixa[1], b[1]))
        uniao = (caixa[2]-caixa[0])*(caixa[3]-caixa[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        if inter/max(uniao, 1e-9) >= .5:
            return True
    return False


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
        self.cpu_threads = int(config.get('cpu_threads', 1))
        if not 1 <= self.cpu_threads <= 32:
            raise ValueError('cpu_threads deve estar entre 1 e 32.')
        torch.set_num_threads(self.cpu_threads)
        self.imgsz = int(config.get('imgsz', 512))
        principal = YOLO(str(caminho))
        self.classes = {normalizar(n) for n in principal.names.values()}
        relevantes = PESSOAS | set().union(*EPIS.values())
        self.modelos = []
        self.complementares = set()
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
            self.classes.update(nomes)
        if not self.classes & PESSOAS:
            base = caminho.parent / 'yolo11n.pt'
            if not base.is_file():
                raise ValueError('Modelo sem classe pessoa e yolo11n.pt não encontrado.')
            self.modelos.append((YOLO(str(base)), [0], {}, 640))
            self.classes.add('person')

    def prever(self, frame):
        import torch
        # Algumas inicializações do backend alteram o paralelismo global.
        # Reaplica o valor medido para evitar excesso de threads entre quadros.
        if torch.get_num_threads() != self.cpu_threads:
            torch.set_num_threads(self.cpu_threads)
        deteccoes = []
        imagem = frame.copy()
        for modelo, ids, limiares, imgsz in self.modelos:
            limites = {i: limiar_calibrado(normalizar(n), limiares, self.confianca)
                       for i, n in modelo.names.items() if ids is None or i in ids}
            resultado = modelo.predict(frame, classes=ids, conf=min(limites.values()),
                                       imgsz=imgsz, device='cpu', verbose=False)[0]
            if resultado.boxes is None:
                continue
            manter = [j for j, (c, score) in enumerate(zip(resultado.boxes.cls.tolist(), resultado.boxes.conf.tolist()))
                      if score >= limites[int(c)]]
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
        return imagem, deteccoes
