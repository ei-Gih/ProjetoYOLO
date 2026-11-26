"""Treina e valida o detector de EPIs da ZelarIA."""
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import shutil
import math
import hashlib
import yaml
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

from test.caminhos import RAIZ, TREINOS, base_treinamento
DATASET = RAIZ / "dataset" / "data.yaml"
MODELO_BASE = RAIZ / "yolo11n.pt"
SPLITS_OBRIGATORIOS = ("train", "val")
EXTENSOES = {".jpg", ".jpeg", ".png", ".webp"}


def validar_dataset() -> None:
    """Garante que cada imagem de treino e validação tenha um label YOLO válido."""
    erros: list[str] = []
    contagem = Counter()
    hashes = {}
    config = yaml.safe_load(DATASET.read_text(encoding="utf-8"))
    esperado = {0: "person", 1: "jaleco", 2: "oculos", 3: "mascara", 4: "luvas", 5: "protetor-auricular", 6: "capacete"}
    if config.get("names") != esperado:
        erros.append("As classes de data.yaml devem corresponder aos IDs 0 a 6 documentados.")

    for split in SPLITS_OBRIGATORIOS:
        por_split = Counter()
        pasta_imagens = RAIZ / "dataset" / "images" / split
        pasta_labels = RAIZ / "dataset" / "labels" / split
        imagens = [p for p in pasta_imagens.glob("*") if p.suffix.lower() in EXTENSOES]
        if not imagens:
            erros.append(f"Não há imagens em dataset/images/{split}.")
            continue

        for imagem in imagens:
            digest = hashlib.sha256(imagem.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                erros.append(f"Imagem repetida entre treino e validação: {imagem.name}")
            hashes[digest] = split
            label = pasta_labels / f"{imagem.stem}.txt"
            if not label.is_file():
                erros.append(f"Falta o label: {label.relative_to(RAIZ)}")
                continue
            for numero_linha, linha in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                partes = linha.split()
                if len(partes) != 5:
                    erros.append(f"{label.name}, linha {numero_linha}: esperado 5 valores.")
                    continue
                try:
                    classe = int(partes[0])
                    coordenadas = [float(valor) for valor in partes[1:]]
                    if classe not in range(7) or not all(0 <= valor <= 1 for valor in coordenadas):
                        raise ValueError
                    x, y, w, h = coordenadas
                    if not all(math.isfinite(v) for v in coordenadas) or w <= 0 or h <= 0 or x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6:
                        raise ValueError
                    contagem[classe] += 1
                    por_split[classe] += 1
                except ValueError:
                    erros.append(f"{label.name}, linha {numero_linha}: label YOLO inválido.")

        ausentes = [esperado[c] for c in esperado if not por_split[c]]
        if ausentes:
            erros.append(f"Classes sem exemplos em {split}: " + ", ".join(ausentes))

    if erros:
        raise SystemExit("Dataset incompleto:\n- " + "\n- ".join(erros))
    print("Caixas encontradas por classe:", dict(sorted(contagem.items())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--validar", action="store_true", help="Somente verifica os dados, sem treinar.")
    args = parser.parse_args()

    if not DATASET.is_file() or not MODELO_BASE.is_file():
        raise SystemExit("Não foi encontrado dataset/data.yaml ou yolo11n.pt na raiz do projeto.")
    validar_dataset()
    if args.validar:
        return
    if min(args.epochs, args.batch, args.imgsz) <= 0:
        raise SystemExit("epochs, batch e imgsz devem ser positivos.")
    config = yaml.safe_load(DATASET.read_text(encoding="utf-8"))
    config["path"] = str((RAIZ / "dataset").resolve())
    config.pop("test", None)
    resolvido = RAIZ / "dataset" / "data.resolvido.yaml"
    resolvido.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    modelo = YOLO(str(MODELO_BASE))
    resultado = modelo.train(
        data=str(resolvido), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, workers=0, patience=20, project=str(TREINOS / "train"),
        name="zelaria_epis",
    )
    melhor_modelo = Path(resultado.save_dir) / "weights" / "best.pt"
    if not melhor_modelo.is_file():
        raise SystemExit("Treinamento terminou sem gerar weights/best.pt.")
    shutil.copy2(melhor_modelo, RAIZ / "best.pt")
    print(f"Modelo copiado para: {RAIZ / 'best.pt'}")


if __name__ == "__main__":
    main()
