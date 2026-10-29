# Binariza com Otsu e detecta crateras circulares com a Transformada de Hough.

import cv2
import numpy as np

# ------------------------------------------------------------------------------
# CONFIGURAÇÕES 
# ------------------------------------------------------------------------------

CAMINHO_ENTRADA = "processada_morfologia.jpg"

CAMINHO_SAIDA = "resultado_crateras.png"

# Parâmetros do Hough:

# MIN_DIST: distância mínima entre os centros de duas crateras
# Para evitar que a mesma cratera seja marcada duas vezes***
MIN_DIST = 30

# PARAM1: limiar alto do detector de bordas Canny, que o Hough roda por dentro
PARAM1 = 50

# PARAM2: número mínimo de "votos" para aceitar um círculo
# Quanto menor, menos preciso. Quanto mais votos, mais preciso
PARAM2 = 20

# RAIO_MIN e RAIO_MAX: tamanho mínimo e máximo das crateras, em pixels
RAIO_MIN = 10
RAIO_MAX = 50

# ------------------------------------------------------------------------------
# 1. CARREGAR A IMAGEM
# ------------------------------------------------------------------------------

imagem = cv2.imread(CAMINHO_ENTRADA, cv2.IMREAD_GRAYSCALE)

if imagem is None:
    print(f"Erro: não foi possível carregar '{CAMINHO_ENTRADA}'. Verifique o nome e a pasta.")
    exit()

# ------------------------------------------------------------------------------
# 2. LIMIAR DE OTSU (binarização)
# ------------------------------------------------------------------------------


_, imagem_binaria = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ------------------------------------------------------------------------------
# 3. TRANSFORMADA DE HOUGH PARA CÍRCULOS
# ------------------------------------------------------------------------------

circulos = cv2.HoughCircles(
    imagem_binaria,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=MIN_DIST,
    param1=PARAM1,
    param2=PARAM2,
    minRadius=RAIO_MIN,
    maxRadius=RAIO_MAX,
)

# ------------------------------------------------------------------------------
# 4. DESENHAR AS CRATERAS ENCONTRADAS
# ------------------------------------------------------------------------------

imagem_resultado = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

if circulos is not None:
   
    circulos = np.round(circulos[0]).astype(int)

    print(f"Sucesso! Foram detectadas {len(circulos)} crateras.")

    # Passa por cada cratera e desenha o círculo
    for x, y, raio in circulos:
        # Borda da cratera: verde (B=0, G=255, R=0), espessura 2
        cv2.circle(imagem_resultado, (x, y), raio, (0, 255, 0), 2)
        # Centro da cratera: ponto vermelho pequeno (B=0, G=0, R=255)
        cv2.circle(imagem_resultado, (x, y), 2, (0, 0, 255), 3)
else:
    print("Nenhuma cratera detectada. Tente ajustar PARAM2, RAIO_MIN ou RAIO_MAX.")

# ------------------------------------------------------------------------------
# 5. SALVAR A SAÍDA DO MÓDULO
# ------------------------------------------------------------------------------


cv2.imwrite(CAMINHO_SAIDA, imagem_resultado)
print(f"Imagem salva em '{CAMINHO_SAIDA}'.")

# Mostra o resultado
cv2.imshow("Crateras Detectadas", imagem_resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()