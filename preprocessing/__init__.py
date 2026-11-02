from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from contrast import clahe
from edges import laplacian, sobel, threshold_edges
from smoothing import gaussian_blur

def localizar_imagem():
    caminho = Path(__file__).resolve().parent / "img-lua-part.png"
    if not caminho.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {caminho}")
    return caminho

def preprocessar(caminho=None, salvar_visualizacao=True):
    if caminho is None:
        caminho = localizar_imagem()
    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {caminho}")

    imagem_original = np.asarray(Image.open(caminho).convert("L"), dtype=np.uint8)
    imagem_contraste = clahe(imagem_original, clip_limit=2.0, tile_grid=(8, 8))
    imagem_suavizada = gaussian_blur(imagem_contraste.astype(np.float32) / 255.0, sigma=5.0)
    imagem_sobel = sobel(imagem_suavizada)
    imagem_laplaciana = laplacian(imagem_suavizada)
    imagem_bordas = threshold_edges(imagem_sobel, low_pct=90, high_pct=97)

    resultados = {"original": imagem_original,
                  "contraste": imagem_contraste,
                  "suavizada": imagem_suavizada,
                  "sobel": imagem_sobel,
                  "laplaciana": imagem_laplaciana,
                  "bordas": imagem_bordas}

    if salvar_visualizacao:
        pasta_saida = Path(__file__).resolve().parent / "Resultados"
        pasta_saida.mkdir(exist_ok=True)

        figura, eixos = plt.subplots(2, 3, figsize=(16, 10))
        imagens = [("Original", imagem_original),
                   ("Contraste (CLAHE)", imagem_contraste),
                   ("Suavização", imagem_suavizada),
                   ("Sobel", imagem_sobel),
                   ("Laplaciano", imagem_laplaciana),
                   ("Bordas", imagem_bordas)]

        for eixo, (titulo, imagem) in zip(eixos.flat, imagens):
            eixo.imshow(imagem, cmap="gray")
            eixo.set_title(titulo)
            eixo.axis("off")

        figura.tight_layout()
        figura.savefig(pasta_saida / "pipeline_preprocess.png", dpi=200)
        plt.close(figura)
    return resultados

if __name__ == "__main__":
    preprocessar()