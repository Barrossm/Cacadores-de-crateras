"""Aquisição e normalização de imagens de entrada."""

from pathlib import Path

from PIL import Image


def load_grayscale_image(path: str | Path) -> Image.Image:
    """Carrega uma imagem e devolve uma cópia em escala de cinza de 8 bits."""
    image_path = Path(path)
    if not image_path.is_file():
        raise ValueError(f"Arquivo de imagem não encontrado: {image_path}")

    with Image.open(image_path) as source:
        if source.width < 1 or source.height < 1:
            raise ValueError("A imagem precisa ter dimensões maiores que zero.")
        return source.convert("L").copy()
