"""Caminhos independentes da pasta usada para iniciar o programa."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATASETS = RAIZ / 'dataset'
RELATORIOS = RAIZ / 'relarotio'
TREINOS = RELATORIOS / 'treinos'
BASES = RAIZ / 'test' / 'bases'

def base_treinamento(nome):
    """Prefere a base preservada; aceita o peso instalado compatível como alternativa."""
    candidatos = {
        'jaleco': [BASES/'jaleco.pt', RAIZ/'jaleco.pt'],
        'maos_ouvidos': [BASES/'maos_ouvidos.pt', RAIZ/'maos_ouvidos.pt'],
        'cabeca': [BASES/'cabeca_refinada.pt', RAIZ/'cabeca_v3_inicial.pt'],
        'plugue': [BASES/'plugue.pt'],
    }[nome]
    return next((p for p in candidatos if p.is_file()), candidatos[0])
