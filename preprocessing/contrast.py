"CLAHE simples: equalizacao de histograma adaptativa com limite de contraste."

import numpy as np

NUMERO_DE_NIVEIS = 256  #niveis de cinza de uma imagem de 8 bits

def calcular_tabela_do_bloco(bloco, clip_limit):
    histograma = np.bincount(bloco.ravel(), minlength=NUMERO_DE_NIVEIS)
    histograma = histograma.astype(np.float32)

    quantidade_de_pixels = bloco.size
    altura_maxima = clip_limit * quantidade_de_pixels / NUMERO_DE_NIVEIS
    altura_maxima = max(1.0, altura_maxima)

    histograma_cortado = np.minimum(histograma, altura_maxima)
    pixels_cortados = histograma.sum() - histograma_cortado.sum()

    histograma_cortado = histograma_cortado + pixels_cortados / NUMERO_DE_NIVEIS

    soma_acumulada = np.cumsum(histograma_cortado)
    tabela = soma_acumulada / soma_acumulada[-1]
    return tabela

def clahe(imagem, clip_limit=2.0, tile_grid=(8, 8)):
    #clip_limit -> quanto maior, mais contraste (e mais ruído)
    #tile_grid: blocos na vertical, blocos na horizontal
    imagem = np.asarray(imagem)
    if imagem.ndim != 2 or imagem.dtype != np.uint8:
        raise ValueError("A imagem deve ser 2D em escala de cinza (uint8).")

    altura, largura = imagem.shape
    blocos_na_vertical, blocos_na_horizontal = tile_grid

    altura_do_bloco = int(np.ceil(altura / blocos_na_vertical))
    largura_do_bloco = int(np.ceil(largura / blocos_na_horizontal))

    altura_total = blocos_na_vertical * altura_do_bloco
    largura_total = blocos_na_horizontal * largura_do_bloco
    imagem_completa = np.pad(imagem, ((0, altura_total - altura), (0, largura_total - largura)), mode="reflect")

    tabelas = np.zeros((blocos_na_vertical, blocos_na_horizontal, NUMERO_DE_NIVEIS), dtype=np.float32)
    for linha_do_bloco in range(blocos_na_vertical):
        for coluna_do_bloco in range(blocos_na_horizontal):
            y_inicio = linha_do_bloco * altura_do_bloco
            x_inicio = coluna_do_bloco * largura_do_bloco
            bloco = imagem_completa[y_inicio:y_inicio + altura_do_bloco, x_inicio:x_inicio + largura_do_bloco,]
            tabelas[linha_do_bloco, coluna_do_bloco] = calcular_tabela_do_bloco(bloco, clip_limit)
    imagem_final = np.zeros((altura, largura), dtype=np.uint8)

    for y in range(altura): #posicao vertical medida em blocos, a partir do CENTRO dos blocos
        posicao_vertical = (y + 0.5) / altura_do_bloco - 0.5
        posicao_vertical = min(max(posicao_vertical, 0), blocos_na_vertical - 1)

        bloco_de_cima = int(np.floor(posicao_vertical))
        bloco_de_baixo = min(bloco_de_cima + 1, blocos_na_vertical - 1)
        peso_de_baixo = posicao_vertical - bloco_de_cima  # 0 = só o de cima
        peso_de_cima = 1 - peso_de_baixo

        for x in range(largura):
            posicao_horizontal = (x + 0.5) / largura_do_bloco - 0.5
            posicao_horizontal = min(max(posicao_horizontal, 0), blocos_na_horizontal - 1)

            bloco_da_esquerda = int(np.floor(posicao_horizontal))
            bloco_da_direita = min(bloco_da_esquerda + 1, blocos_na_horizontal - 1)
            peso_da_direita = posicao_horizontal - bloco_da_esquerda
            peso_da_esquerda = 1 - peso_da_direita

            brilho_antigo = imagem[y, x]

            novo_cima_esquerda = tabelas[bloco_de_cima, bloco_da_esquerda, brilho_antigo]
            novo_cima_direita = tabelas[bloco_de_cima, bloco_da_direita, brilho_antigo]
            novo_baixo_esquerda = tabelas[bloco_de_baixo, bloco_da_esquerda, brilho_antigo]
            novo_baixo_direita = tabelas[bloco_de_baixo, bloco_da_direita, brilho_antigo]

            em_cima = peso_da_esquerda * novo_cima_esquerda + peso_da_direita * novo_cima_direita
            em_baixo = peso_da_esquerda * novo_baixo_esquerda + peso_da_direita * novo_baixo_direita
            brilho_novo = peso_de_cima * em_cima + peso_de_baixo * em_baixo

            imagem_final[y, x] = round(brilho_novo * 255)
    return imagem_final