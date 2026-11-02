# Detecta bordas com Sobel e Laplaciano.
import numpy as np
from scipy import ndimage
from smoothing import convolve2d

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)   # bordas verticais

SOBEL_Y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32) # bordas horizontais

LAPLACIAN = np.array([[0,  1, 0],
                      [1, -4, 1],
                      [0,  1, 0]], dtype=np.float32)

def sobel(image):
    #magnitude do gradiente -> o quao forte eh a borda em cada pixel
    gx = convolve2d(image, SOBEL_X)
    gy = convolve2d(image, SOBEL_Y)
    return np.hypot(gx, gy)

def laplacian(image):
    return np.abs(convolve2d(image, LAPLACIAN))

def normalize(image):
    low, high = image.min(), image.max()
    if high == low:
        return np.zeros_like(image)
    return (image - low) / (high - low)

def threshold_edges(magnitude, low_pct=90, high_pct=97):
    #limiar
    #forte: está entre os 3% maiores gradientes (borda certa)
    #fraco: está entre os 10% maiores (borda possível)
    low = np.percentile(magnitude, low_pct)
    high = np.percentile(magnitude, high_pct)

    strong = magnitude >= high
    weak = magnitude >= low

    regions, _ = ndimage.label(weak, structure=np.ones((3, 3)))
    keep = np.unique(regions[strong]) #regioes que tocam um pixel forte
    keep = keep[keep > 0]
    return (np.isin(regions, keep) * 255).astype(np.uint8)