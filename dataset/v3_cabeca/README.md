# Dataset V3 — protetores auriculares, máscaras e capacetes

Classes YOLO:

- `0`: protetor auricular (concha e plugue; caixa no dispositivo, sem englobar o cordão).
- `1`: máscara facial.
- `2`: capacete.

Este conjunto foi criado a partir de imagens SH17 já disponíveis no projeto e de 33 imagens fornecidas pela usuária, com novas caixas revisadas visualmente. Três imagens foram excluídas por ambiguidade ou por montagens publicitárias com miniaturas. A origem externa e a licença das novas imagens fornecidas não foram informadas; elas não foram publicadas por esta revisão.

Separação original: 286 imagens de treino, 55 de validação e 53 de teste. Foram acrescentados 157 recortes derivados exclusivamente do treino, totalizando 443 arquivos de treino. Os recortes não são novas cenas independentes. Validação e teste permaneceram iguais. As imagens semelhantes identificadas por hash perceptual foram mantidas no mesmo conjunto. Duas vistas da mesma pessoa com máscara também foram agrupadas manualmente. Essa verificação não garante eliminar toda semelhança de cenário.

As caixas de máscaras e capacetes do SH17 foram recuperadas das anotações originais. Não basta a foto mostrar um capacete: ele precisa ter uma caixa anotada e a classe correta para participar do treino.

## Importante sobre a pasta anterior

Na pasta `dataset/v2_maos_ouvidos/images/test`, 36 fotos foram substituídas, mas os 36 arquivos de anotação correspondentes continuavam descrevendo as fotos anteriores. Além disso, o V2 contém somente as classes luvas e protetor auricular; ele não representa máscaras nem capacetes. Suas imagens nessa pasta foram preservadas. **Não use o teste V2 alterado para calcular métricas:** suas anotações não correspondem ao conteúdo atual. Utilize este V3 revisado.

## Novas imagens

Para ensinar uma imagem ao modelo, ela deve estar no conjunto de treino, acompanhada de anotações que correspondam ao seu conteúdo. O conjunto de validação orienta a escolha de pesos e limiares. O conjunto de teste é reservado para avaliação. Uma imagem usada no treino não demonstra reconhecimento de uma imagem nova.

Nunca substitua uma foto mantendo o arquivo `.txt` da foto anterior. Revise todas as caixas e mantenha o manifesto coerente. O comando `python treinar_cabeca.py --verificar` verifica integridade, presença de anotações e duplicatas exatas entre conjuntos.

O manifesto contém o caminho de origem, a separação utilizada, as caixas e o SHA-256 de cada imagem. Os pesos iniciais já foram treinados no SH17; portanto, o teste local que contém imagens desse dataset não é uma avaliação externa independente. Há apenas quatro imagens fornecidas pela usuária no teste, três delas com plugues. Esse número é pequeno para comprovar generalização.

Este treinamento aborda aparência visual. Não determina certificação, proteção química, vedação ou atenuação acústica. A contagem de EPIs do aplicativo depende também da detecção da pessoa e da associação espacial do objeto a ela.
