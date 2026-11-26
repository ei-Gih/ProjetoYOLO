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


## Atualização V5 — distâncias e sobreposição (20/11/2026)

Continuam ativos cinco EPIs: capacete, máscara, jaleco, luvas e óculos. A liberação exige os cinco confirmados na mesma pessoa. Protetor auricular permanece desativado.

Foram concluídos 12 ciclos de treino de cabeça/mãos e 8 de jaleco, com redução de escala, closes, variações de cor e oclusões parciais. Os candidatos foram comparados com os pesos anteriores; somente as melhorias medidas foram ativadas. O novo peso V5 complementa objetos grandes numa segunda escala; a leitura anterior de máscara/óculos permanece. Luvas usam o SH17 maior previamente treinado; jaleco conserva o peso anterior com limiar recalibrado. Os escores exibidos não foram artificialmente aumentados.

A supressão de caixas repetidas ocorre dentro da mesma classe: máscara e óculos sobrepostos podem coexistir. A câmera solicita 1280 × 720 quando suportado. A interface permanece igual.

Na avaliação exploratória de imagens originais e variantes próximas/distantes, houve 20 acertos de capacete (antes 19), 82 de máscara (71), 13 de óculos (7), 61 de luvas (49) e 31 de jaleco (19). Nos closes com máscara e óculos juntos, 4/15 pares foram encontrados, antes 0/15. Ainda há falhas, especialmente em objetos pequenos e oclusões; isso não equivale a validação da webcam em metros ou garantia de uso correto de EPI. O conjunto já foi consultado durante desenvolvimento, e parte dos dados públicos pode ter participado do pré-treino. Resultados completos em `relarotio/treinos/v5_distancias`.

Para verificar os dados: `.venv\Scripts\python.exe test\treinar_distancias.py --alvo epis --verificar` (ou `--alvo jaleco`). Para novo treino, retire `--verificar`; use `--base CAMINHO` para ajustar pesos de um treino concluído. `--retomar` aceita somente checkpoints de treino interrompido. Novos pesos não são ativados automaticamente: compare falsos positivos, detecções e pares antes de substituir a configuração.

As reservas dos novos datasets excluem duplicatas próximas identificadas na auditoria; o treino histórico usou a validação original, antes dessa limpeza. As fotos individuais fornecidas pelo usuário participaram do treino e não constituem teste independente.
