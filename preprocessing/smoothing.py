"Suavizacao Gaussiana"
import numpy as np

def convolve2d(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    padded = np.pad(image, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="edge")

    out = np.zeros((h, w), dtype=np.float32)
    for u in range(kh):
        for v in range(kw):
            out += kernel[u, v] * padded[u:u + h, v:v + w]
    return out

def gaussian_kernel(sigma=1.0):
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    g = np.exp(-x**2 / (2 * sigma**2))# curva 1D
    g /= g.sum()
    return np.outer(g, g) # 1D x 1D = 2D

def gaussian_blur(image, sigma=1.0): #sigma maior -> mais borrado
    return convolve2d(image, gaussian_kernel(sigma))