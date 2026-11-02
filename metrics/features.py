"""Descrição geométrica das crateras encontradas."""

import cv2
import numpy as np


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("A máscara precisa ser uma matriz bidimensional.")
    return array.astype(bool)


def _validate_circles(circles: np.ndarray | None) -> np.ndarray:
    if circles is None:
        return np.empty((0, 3), dtype=np.float32)

    array = np.asarray(circles, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Os círculos devem ter o formato (N, 3): x, y e raio.")
    if not np.all(np.isfinite(array)) or np.any(array[:, 2] <= 0):
        raise ValueError("Os círculos devem conter coordenadas finitas e raios positivos.")
    return array


def _circle_mask(shape: tuple[int, int], circle: np.ndarray) -> np.ndarray:
    height, width = shape
    x, y, radius = np.rint(circle).astype(int)
    result = np.zeros(shape, dtype=np.uint8)
    cv2.circle(result, (x, y), radius, 1, thickness=-1)
    return result.astype(bool)


def _measure_mask(mask: np.ndarray) -> tuple[float, float, float]:
    area = float(np.count_nonzero(mask))
    if area == 0:
        return 0.0, 0.0, 0.0

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return area, 0.0, 0.0

    perimeter = float(sum(cv2.arcLength(contour, closed=True) for contour in contours))
    circularity = 0.0 if perimeter == 0 else 4 * np.pi * area / (perimeter * perimeter)
    return area, perimeter, float(circularity)


def describe_craters(
    mask: np.ndarray,
    circles: np.ndarray | None,
) -> list[dict[str, object]]:
    """Calcula as características geométricas de cada círculo detectado.

    A máscara de cada resultado é a interseção da máscara de entrada com o
    disco delimitado pelo círculo. Círculos sem pixels nessa interseção são
    descartados, pois não há região válida para medir.
    """
    binary_mask = _validate_mask(mask)
    results = []

    for circle in _validate_circles(circles):
        x, y, radius = np.rint(circle).astype(int)
        crater_mask = binary_mask & _circle_mask(binary_mask.shape, circle)
        area, perimeter, circularity = _measure_mask(crater_mask)
        if area == 0 or perimeter == 0:
            continue

        results.append({
            "center": (int(x), int(y)),
            "radius": int(radius),
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "mask": crater_mask,
        })

    return results
