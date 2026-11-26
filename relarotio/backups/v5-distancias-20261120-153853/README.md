# ZelarIA

Abra `Iniciar_ZelarIA.cmd` ou execute, na raiz do projeto:

```powershell
.\.venv\Scripts\python.exe main.py
```

## Organização

- `main.py`: interface gráfica.
- `test/`: módulos de detecção, avaliação, captura, aviso de acesso, testes e scripts de treinamento.
- `test/caminhos.py`: definição central da raiz e das pastas de dados, bases e resultados.
- `test/bases/`: pesos iniciais necessários aos treinamentos complementares.
- `dataset/`: imagens e anotações dos conjuntos de treinamento.
- `assets/`: recursos gráficos da interface.
- `docs/`: apresentação e documentação acadêmica.
- `relarotio/`: relatórios, backups das alterações e resultados de novos treinos em `relarotio/treinos/`.
- `deteccao/`: resultados dos testes com imagens.
- Modelos `.pt` e `calibracao.json`: permanecem na raiz, onde a aplicação os procura.
- `.venv/`: ambiente Python necessário para executar o projeto.

O nome da pasta `relarotio` foi preservado conforme a organização atual. Os caminhos internos partem da localização do código, e não da pasta atual do terminal. Os scripts de `test/` também podem ser executados diretamente pelo caminho completo.

## Contagem e acesso

São avaliados cinco tipos: jaleco, óculos, máscara, luvas e capacete. Os contadores somam tipos de EPI por pessoa; duas luvas contam como um tipo. Confirmados + pendentes = pessoas × 5.

O pop-up verde **ACESSO LIBERADO** aparece quando os cinco tipos, incluindo o jaleco, estiverem confirmados na mesma pessoa em dois quadros consecutivos. O aviso indica P1, P2 etc. conforme as pessoas completas naquele quadro. EPIs distribuídos entre pessoas distintas não são somados para liberar alguém. Pessoas incompletas continuam com pendências no painel.

O aviso não bloqueia a câmera. Ele fecha quando a condição muda, quando fica sem confirmação recente por três segundos ou ao encerrar a inspeção. Fechá-lo manualmente não o faz reaparecer a cada quadro: é necessário perder e recuperar a condição. Os identificadores P1/P2 são locais ao quadro, não identificações persistentes de pessoas.

EPIs soltos podem ser reconhecidos como objetos, mas não liberam acesso sem uma pessoa com todos os cinco tipos confirmados. A associação considera a posição da caixa do EPI dentro da caixa da pessoa; não certifica ajuste correto, resistência química ou vedação. Falta de detecção significa “não confirmado”.

## Testar imagens e executar verificações

```powershell
.\.venv\Scripts\python.exe -m test.testar_modelo --imagem "C:\caminho\foto.png"
.\.venv\Scripts\python.exe -m unittest discover -s test -v
.\.venv\Scripts\python.exe -m test.treinar_cabeca --verificar
```

A webcam e o teste de imagens usam `test/detector.py` e a mesma configuração de pesos e limiares. `--conf 0.40` preserva os limiares calibrados. A pontuação exibida pelo modelo não é uma garantia de acerto.

O protetor auricular foi desativado: não aparece no painel, nas caixas de detecção nem nas pendências e não é exigido para acesso. Os arquivos históricos de treino foram preservados.

## Treinamento

```powershell
.\.venv\Scripts\python.exe -m test.treinar_cabeca --epochs 12
.\.venv\Scripts\python.exe -m test.treinar_variacoes --alvo jaleco
.\.venv\Scripts\python.exe -m test.treinar_jaleco --epochs 30 --batch 4 --saida jaleco_candidato.pt
```

As bases preservadas ficam em `test/bases/`. Para jaleco e mãos/ouvidos, os scripts podem usar os pesos instalados quando não houver uma base separada. Novas execuções são registradas em `relarotio/treinos/`, com `results.csv`, `args.yaml` e pesos. Compare os candidatos antes de substituir modelos ativos.

Para retomar um checkpoint de jaleco, use `-m test.retomar_jaleco --checkpoint CAMINHO_DO_LAST_PT`. O script genérico `-m test.treinamento --validar` continua disponível, mas exige um dataset próprio completo com as sete classes; corrigir caminhos não preenche um conjunto incompleto.

Não substitua fotos mantendo as anotações da foto anterior. O teste V2 que teve imagens trocadas não deve ser usado para medir acurácia; use os conjuntos revisados e leia os respectivos manifestos. Fotos usadas no treinamento não são uma avaliação independente de generalização.

## Encerramento

`Encerrar` e o X fecham a janela, o aviso de acesso e o processo de câmera/inferência. O processamento permanece separado da interface para manter os controles responsivos.
