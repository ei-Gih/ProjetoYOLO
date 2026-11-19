# Fontes e curadoria do detector de jaleco

Origem das imagens: SH17, Hafiz Mughees Ahmad e Afshin Rahimi.
Fonte oficial: https://github.com/ahmadmughees/SH17dataset
Artigo: https://doi.org/10.1016/j.jnlssr.2024.09.002
Espelho utilizado para download: https://huggingface.co/datasets/fathansanum/SH-17-Dataset

O repositório oficial declara CC BY-NC-SA 4.0 e orienta observar os termos do Pexels, origem das fotografias. O espelho declara CC BY 4.0, mas esta preparação conserva a restrição mais específica da fonte oficial: uso não comercial, atribuição e compartilhamento pela mesma licença. Não foi concedida nova licença às fotografias.

A classe original medical-suit não significa jaleco. Foram revisadas visualmente 103 imagens candidatas e 142 caixas, excluídas duas imagens ambíguas, selecionadas caixas de jalecos e acrescentadas caixas ausentes. Uniformes cirúrgicos, aventais cirúrgicos, macacões e outras roupas foram usados como negativos. As anotações foram revistas pelo assistente, sem validação independente de um especialista.

A seleção resulta em 101 imagens: 85 de treino (65 caixas) e 16 de validação (12 caixas). As cenas hospitalares recorrentes foram reservadas para validação; as cenas de laboratório com fundo azul e outras sessões ficaram no treino. O manifesto registra nomes originais, divisão, SHA-256 e caixas. A diversidade de pessoas, ambientes e modelos de jaleco é pequena. Não há conjunto de teste independente; os resultados de validação não comprovam desempenho na webcam do usuário.

A versão COCO do espelho usa IDs numéricos remapeados. A seleção de medical-suit foi confirmada por inspeção visual das imagens e caixas, e o detector novo usa apenas classe 0: jaleco. Nenhuma classe safety-vest foi convertida em jaleco.

O modelo sh17_ppe.pt já existente continua responsável por pessoas, óculos, máscara, luvas, proteção auricular e capacete. Apenas o detector complementar jaleco.pt foi treinado nesta revisão.
