import unicodedata

def normalizar(nome):
    texto = unicodedata.normalize("NFKD", nome.lower())
    return "-".join("".join(c for c in texto if not unicodedata.combining(c)).replace("_", " ").replace("-", " ").split())


# Ajuste os aliases para os nomes exatos das classes do seu modelo treinado.
EPIS = {
    # Inclui os nomes do modelo SH17, usado enquanto não existe best.pt próprio.
    "Jaleco": {"jaleco", "lab-coat", "labcoat"},
    "Óculos de segurança": {"oculos", "oculos-de-seguranca", "goggles", "glasses", "safety-glasses", "safety-goggles"},
    "Máscara": {"mascara", "mask", "face-mask", "face-mask-medical", "medical-mask"},
    "Luvas": {"luva", "luvas", "glove", "gloves", "safety-gloves"},
    "Protetor auricular": {"protetor-auricular", "protetor-oricular", "ear-protection", "earmuffs", "ear-mufs", "ear-muffs", "earplug", "earplugs"},
    "Capacete": {"capacete", "helmet", "hardhat", "hard-hat", "safety-helmet"},
}
PESSOAS = {"person", "pessoa", "worker", "trabalhador"}


def avaliar_pessoas(deteccoes, classes_modelo):
    """Associa cada EPI a uma única pessoa; sobreposições ambíguas não liberam."""
    suportados = {epi: bool(aliases & classes_modelo) for epi, aliases in EPIS.items()}
    pessoas = [d for d in deteccoes if d[0] in PESSOAS]
    presentes = [set() for _ in pessoas]
    for nome, caixa in deteccoes:
        epi = next((e for e, aliases in EPIS.items() if nome in aliases), None)
        if epi is None:
            continue
        x1, y1, x2, y2 = caixa
        area = max(0, x2 - x1) * max(0, y2 - y1)
        candidatos = []
        for i, (_, (px1, py1, px2, py2)) in enumerate(pessoas):
            intersecao = max(0, min(x2, px2) - max(x1, px1)) * max(0, min(y2, py2) - max(y1, py1))
            if area > 0 and intersecao / area >= 0.70:
                candidatos.append(i)
        if len(candidatos) == 1:
            presentes[candidatos[0]].add(epi)
    return pessoas, presentes, suportados

