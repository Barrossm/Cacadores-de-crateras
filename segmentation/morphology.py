"""Operações morfológicas para limpeza de máscaras binárias."""

import cv2
import numpy as np


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    """Converte uma máscara 2D para o formato binário aceito pelo OpenCV."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("A máscara precisa ser uma matriz bidimensional.")
    return np.where(array, 255, 0).astype(np.uint8)


def _validate_kernel_size(kernel_size: int) -> int:
    if not isinstance(kernel_size, (int, np.integer)) or kernel_size < 1:
        raise ValueError("O tamanho do elemento estruturante deve ser positivo.")
    if kernel_size % 2 == 0:
        raise ValueError("O tamanho do elemento estruturante deve ser ímpar.")
    return int(kernel_size)


def _validate_iterations(iterations: int) -> int:
    if not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("O número de iterações deve ser um inteiro positivo.")
    return int(iterations)


def create_disk(kernel_size: int = 5) -> np.ndarray:
    """Cria um elemento estruturante circular com tamanho ímpar."""
    size = _validate_kernel_size(kernel_size)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def opening(mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """Aplica erosão seguida de dilatação para remover pequenos ruídos."""
    iterations = _validate_iterations(iterations)
    return cv2.morphologyEx(
        _validate_mask(mask),
        cv2.MORPH_OPEN,
        create_disk(kernel_size),
        iterations=iterations,
    ).astype(bool)


def closing(mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """Aplica dilatação seguida de erosão para fechar pequenas falhas."""
    iterations = _validate_iterations(iterations)
    return cv2.morphologyEx(
        _validate_mask(mask),
        cv2.MORPH_CLOSE,
        create_disk(kernel_size),
        iterations=iterations,
    ).astype(bool)


def clean(mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """Remove ruído e fecha pequenas descontinuidades da máscara."""
    return closing(opening(mask, kernel_size, iterations), kernel_size, iterations)
