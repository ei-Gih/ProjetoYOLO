"""Verifica imagens com exatamente o mesmo pipeline da webcam."""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import cv2
from test.detector import DetectorEPIs
from test.avaliacao import EPIS, avaliar_pessoas

from test.caminhos import RAIZ, TREINOS, base_treinamento


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagem', type=Path, required=True)
    parser.add_argument('--conf', type=float, default=0.40,
                        help='0.40 usa a calibração padrão; aumenta/diminui os limiares por classe.')
    args = parser.parse_args()
    if not args.imagem.is_file():
        raise SystemExit('Imagem não encontrada.')
    if not 0 < args.conf < 1:
        raise SystemExit('conf deve estar entre 0 e 1.')
    caminho = next((RAIZ/n for n in ('best.pt', 'sh17_ppe.pt', 'yolo11n.pt') if (RAIZ/n).is_file()), None)
    if caminho is None:
        raise SystemExit('Nenhum modelo disponível.')
    detector = DetectorEPIs(caminho, args.conf)
    frame = cv2.imread(str(args.imagem))
    if frame is None:
        raise SystemExit('Não foi possível decodificar a imagem.')
    imagem, deteccoes = detector.prever(frame)
    pessoas, presentes, suportados = avaliar_pessoas(deteccoes, detector.classes)
    confirmados = sum(map(len, presentes))
    print('Pessoas:', len(pessoas), 'EPIs confirmados:', confirmados, 'Pendências:', len(EPIS)*len(pessoas)-confirmados)
    for i, itens in enumerate(presentes, 1):
        print(f'P{i}: não confirmados:', ', '.join(e for e in EPIS if e not in itens) or 'nenhum')
    pasta = RAIZ/'deteccao'
    pasta.mkdir(exist_ok=True)
    saida = pasta/(args.imagem.stem+'_resultado.jpg')
    if not cv2.imwrite(str(saida), imagem):
        raise SystemExit('Não foi possível salvar a prévia.')
    print('Prévia:', saida)


if __name__ == '__main__':
    main()
