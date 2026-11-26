# ZelarIA

Execute na pasta do projeto: `.venv\Scripts\python.exe main.py`.

A interface original foi preservada. Iniciar inspeção abre a câmera; Encerrar e o X fecham a janela e o processo de captura/inferência. O processamento usa um processo separado para manter os controles responsivos.

## Contagem
- Pessoas na cena: pessoas detectadas no quadro atual.
- EPIs confirmados: soma dos tipos de EPI reconhecidos por pessoa, no máximo seis por pessoa. Duas luvas não contam como dois tipos.
- Com pendências: soma dos tipos de EPI ainda não confirmados por pessoa, incluindo classes não suportadas. Confirmados + pendentes = pessoas × 6.
- O painel informa quais itens não foram confirmados para P1, P2 etc. Os números identificam somente o quadro atual.

Uma detecção ausente significa não confirmado, inclusive quando há oclusão. A associação usa sobreposição com a caixa da pessoa; objetos ambíguos entre pessoas não são atribuídos. Não verifica certificação, ajuste correto do EPI nem necessariamente o par de luvas. O especialista de protetores foi ajustado para reduzir confusão com fones, mas precisa de validação no ambiente real. Colete, macacão e roupa médica genérica não confirmam jaleco.

## Treinamento
`.venv\Scripts\python.exe treinamento.py --validar`
`.venv\Scripts\python.exe treinamento.py --epochs 100 --batch 4`

O validador exige as sete classes no treino e na validação, caixas válidas e ausência de imagens idênticas entre os dois grupos. Não mede sozinho a qualidade ou diversidade do dataset. O modelo best.pt é gerado apenas ao concluir o treinamento. Os pesos existentes sh17_ppe.pt e yolo11n.pt continuam disponíveis.

Documentação: https://docs.ultralytics.com/modes/train/

## Organização
Materiais anteriores ficam em arquivo/, incluindo backups, imagens avulsas e capturas. A documentação acadêmica fica em docs/. A pasta .venv é necessária para executar sem reinstalar dependências.

## Testes
`.venv\Scripts\python.exe -m unittest test_avaliacao test_detector -v`

## Detector complementar de jaleco
O modelo `jaleco.pt`, quando disponível, é executado junto de `sh17_ppe.pt`. O treino complementar usa `dataset/jaleco/`, com fonte e curadoria registradas em `FONTES.md` e `manifesto.json`.
Para repetir: `.venv\Scripts\python.exe treinar_jaleco.py --epochs 30 --batch 4`.
Esse foi o treinamento inicial, com poucos ambientes. A rodada V2 descrita abaixo também ajustou luvas e protetores tipo concha; capacete, máscara e óculos mantêm o pré-treinamento público.

Para testar uma imagem com os modelos usados pela interface:
`.venv\Scripts\python.exe testar_modelo.py --imagem CAMINHO_DA_FOTO`
O resultado anotado é salvo em `deteccao/` e as contagens são impressas no terminal.

Para repetir o ajuste com a base ampliada:
`.venv\Scripts\python.exe treinar_jaleco.py --dataset dataset/jaleco_ampliado --base runs/train/jaleco/weights/best.pt --epochs 12 --batch 4 --saida jaleco_candidato.pt`
Compare a validação antes de substituir `jaleco.pt`. Os arquivos `runs/train/*/results.csv` registram os resultados por época. A configuração `args.yaml` de cada execução documenta os parâmetros usados.

## Variações de EPIs — V2

Consulte `MELHORIAS_V2.md` e `relatorio_v2.json` para resultados e limitações. Os datasets revisados estão em `dataset/v2_jalecos` e `dataset/v2_maos_ouvidos`.

`detector.py` aplica os mesmos pesos e limiares na webcam e no teste de imagens. Com `sh17_ppe.pt`, os especialistas listados em `calibracao.json` substituem somente as classes aprovadas na comparação, evitando duplicação. As demais classes preservam o detector anterior. Um `best.pt` próprio mantém prioridade e não recebe essa substituição automaticamente.

`calibracao.json` contém limiares por classe, escolhidos na validação; aumentar um número não torna o modelo mais preciso por si só. O parâmetro `--conf 0.40` mantém os limiares padrão; outros valores deslocam todos os limiares pela diferença em relação a 0.40. A pontuação não certifica proteção adequada.

Para repetir a rodada com os pesos iniciais preservados:

`.venv\Scripts\python.exe treinar_variacoes.py --alvo jaleco`

`.venv\Scripts\python.exe treinar_variacoes.py --alvo maos_ouvidos`

Para o segundo ajuste, use `--alvo maos_ouvidos --conservador`.

Os candidatos são salvos em `runs/novas_variacoes`, sem substituir modelos em uso. Plugues auriculares e cobertura completa de cores/materiais ainda não estão validados.
