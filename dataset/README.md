# Dataset ZelarIA

Cada imagem deve ter um label com o mesmo nome e extensão `.txt`.

```text
images/train/foto_001.jpg
labels/train/foto_001.txt
```

O label está no formato YOLO: `classe centro_x centro_y largura altura`, com coordenadas normalizadas entre 0 e 1.

Classes:

| ID | Classe |
|---:|---|
| 0 | person |
| 1 | jaleco |
| 2 | oculos |
| 3 | mascara |
| 4 | luvas |
| 5 | protetor-auricular |
| 6 | capacete |

Separe imagens por pessoa, sessão ou vídeo: fotos quase iguais não podem aparecer em grupos diferentes. Use aproximadamente 70% para `train`, 20% para `val` e 10% para `test`.

O exemplo `capacete_exemplo.jpg` só demonstra a estrutura. Ele não torna o modelo capaz de reconhecer EPIs na webcam. Adicione fotos variadas de pessoas usando os equipamentos e revise todos os labels antes do treinamento.
