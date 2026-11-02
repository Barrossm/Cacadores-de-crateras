"""Avaliação das crateras detectadas contra o gabarito (Ground Truth)."""

from __future__ import annotations
import csv
import time
from pathlib import Path
from typing import Callable, TypeVar

import cv2
import numpy as np

Circle = tuple[int, int, int]
T = TypeVar("T")

# Limiar de sobreposição para considerar acerto (0.40 compensa pequenas variações
# de raio/centro na marcação manual de 2 cliques em bordas largas de crateras).
DEFAULT_IOU_THRESHOLD = 0.40


def circles_from_craters(craters: list[dict]) -> list[Circle]:
    """Extrai uma lista de tuplas (x, y, raio) a partir dos dicionários da GUI / features.py."""
    if not craters:
        return []
    arr = np.rint([[c["center"][0], c["center"][1], c["radius"]] for c in craters]).astype(int)
    return [tuple(map(int, row)) for row in arr]


def load_ground_truth(path: str | Path) -> list[Circle]:
    """Lê um arquivo CSV de gabarito (colunas x, y, r) usando conversão vetorizada do NumPy."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise ValueError(f"Arquivo de gabarito não encontrado: {csv_path}")

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for line_num, row in enumerate(csv.reader(f), start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            first = row[0].strip().lower()
            if first.startswith("#") or first in {"x", "center_x", "cx", "n"}:
                continue
            try:
                rows.append([float(row[0]), float(row[1]), float(row[2])])
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Linha inválida no CSV ({line_num}): {row}") from exc

    if not rows:
        return []

    arr = np.rint(rows).astype(int)
    arr = arr[arr[:, 2] > 0]
    return [tuple(map(int, row)) for row in arr]


def save_ground_truth(path: str | Path, circles: list[Circle]) -> None:
    """Salva uma lista de círculos (x, y, r) em um arquivo CSV simples."""
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "r"])
        if circles:
            writer.writerows(np.rint(circles).astype(int).tolist())


def circle_iou(circle_a: Circle, circle_b: Circle) -> float:
    """Calcula a sobreposição (IoU de 0.0 a 1.0) entre dois círculos (x, y, raio)."""
    x1, y1, r1 = circle_a
    x2, y2, r2 = circle_b

    if r1 <= 0 or r2 <= 0 or np.hypot(x1 - x2, y1 - y2) >= r1 + r2:
        return 0.0

    min_x, max_x = min(x1 - r1, x2 - r2), max(x1 + r1, x2 + r2)
    min_y, max_y = min(y1 - r1, y2 - r2), max(y1 + r1, y2 + r2)
    shape = (max_y - min_y + 1, max_x - min_x + 1)

    mask_a = np.zeros(shape, dtype=np.uint8)
    mask_b = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask_a, (x1 - min_x, y1 - min_y), r1, 1, thickness=-1)
    cv2.circle(mask_b, (x2 - min_x, y2 - min_y), r2, 1, thickness=-1)

    intersecao = np.count_nonzero(mask_a & mask_b)
    uniao = np.count_nonzero(mask_a | mask_b)
    return float(intersecao / uniao) if uniao > 0 else 0.0


def match_craters(
    detected: list[Circle],
    ground_truth: list[Circle],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict:
    """Compara os círculos detectados com o gabarito (1 para 1) do maior IoU para o menor."""
    n_det, n_gt = len(detected), len(ground_truth)
    if n_det == 0 or n_gt == 0:
        return {
            "matches": [],
            "unmatched_detected": list(range(n_det)),
            "unmatched_gt": list(range(n_gt)),
        }

    iou_matrix = np.array(
        [[circle_iou(det, gt) for gt in ground_truth] for det in detected],
        dtype=np.float64,
    )

    det_idx, gt_idx = np.where(iou_matrix >= iou_threshold)
    order = np.argsort(-iou_matrix[det_idx, gt_idx])

    used_det, used_gt = set(), set()
    matches = []
    for k in order:
        i, j = int(det_idx[k]), int(gt_idx[k])
        if i not in used_det and j not in used_gt:
            used_det.add(i)
            used_gt.add(j)
            matches.append((i, j, float(iou_matrix[i, j])))

    return {
        "matches": matches,
        "unmatched_detected": [i for i in range(n_det) if i not in used_det],
        "unmatched_gt": [j for j in range(n_gt) if j not in used_gt],
    }


def diagnose_metrics(metrics: dict) -> dict[str, str]:
    """Gera um veredito simples em português explicando se a avaliação foi boa."""
    precision, recall, mean_iou = metrics["precision"], metrics["recall"], metrics["mean_iou"]
    tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]

    if tp + fn == 0:
        return {
            "level": "info",
            "title": "Gabarito vazio",
            "message": "Nenhuma cratera foi marcada no gabarito para comparar.",
        }
    if tp == 0:
        return {
            "level": "bad",
            "title": "Precisa de ajuste (0 acertos)",
            "message": (
                "O programa não acertou nenhuma cratera do gabarito. "
                "Verifique se RAIO_MIN, RAIO_MAX e o PARAM2 do Hough estão adequados."
            ),
        }

    if precision >= 0.70 and recall >= 0.70:
        level, title = "good", "Excelente resultado!"
    elif precision >= 0.50 and recall >= 0.45:
        level, title = "ok", "Resultado bom / equilibrado"
    else:
        level, title = "warn", "Resultado parcial (pode melhorar)"

    tips = []
    if precision >= 0.70:
        tips.append(f"Ótima precisão ({precision * 100:.0f}%): poucos falsos alarmes.")
    elif fp > tp:
        tips.append(
            f"Precisão baixa ({precision * 100:.0f}% com {fp} falsos alarmes): "
            "subir a Exigência (PARAM2) na Tela 2 reduz marcações extras."
        )
    else:
        tips.append(f"Precisão sólida ({precision * 100:.0f}% com {fp} falsos alarmes).")

    if recall >= 0.70:
        tips.append(f"Ótimo recall ({recall * 100:.0f}%): encontrou a maioria das crateras reais.")
    elif fn > tp:
        tips.append(
            f"Recall moderado ({recall * 100:.0f}% com {fn} esquecidas): "
            "diminuir um pouco a Exigência (PARAM2) encontra mais crateras."
        )
    else:
        tips.append(f"Recall equilibrado ({recall * 100:.0f}% com {fn} esquecidas).")

    tips.append(f"Encaixe médio dos acertos (IoU): {mean_iou * 100:.0f}%.")
    return {"level": level, "title": title, "message": " ".join(tips)}


def evaluate(
    detected: list[Circle] | list[dict],
    ground_truth: list[Circle] | list[dict],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    elapsed_ms: float | None = None,
) -> dict:
    """Calcula TP, FP, FN, Precisão, Recall, F1 e IoU médio."""
    det_circles = (
        circles_from_craters(detected)
        if detected and isinstance(detected[0], dict)
        else list(detected)
    )
    gt_circles = (
        circles_from_craters(ground_truth)
        if ground_truth and isinstance(ground_truth[0], dict)
        else list(ground_truth)
    )

    pairing = match_craters(det_circles, gt_circles, iou_threshold=iou_threshold)
    matches = pairing["matches"]

    tp = len(matches)
    fp = len(pairing["unmatched_detected"])
    fn = len(pairing["unmatched_gt"])

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean([m[2] for m in matches])) if tp > 0 else 0.0

    result = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "iou_threshold": iou_threshold,
        "elapsed_ms": elapsed_ms,
        "matches": matches,
        "unmatched_detected": pairing["unmatched_detected"],
        "unmatched_gt": pairing["unmatched_gt"],
        "detected_circles": det_circles,
        "gt_circles": gt_circles,
    }
    result["diagnosis"] = diagnose_metrics(result)
    return result


def summarize_many(evaluations: list[dict]) -> dict:
    """Calcula as médias de avaliação de várias imagens usando NumPy."""
    if not evaluations:
        return {
            "count": 0, "tp": 0, "fp": 0, "fn": 0,
            "mean_precision": 0.0, "mean_recall": 0.0, "mean_f1": 0.0,
            "mean_iou": 0.0, "mean_elapsed_ms": 0.0,
        }

    times = [e["elapsed_ms"] for e in evaluations if e.get("elapsed_ms") is not None]
    return {
        "count": len(evaluations),
        "tp": int(np.sum([e["tp"] for e in evaluations])),
        "fp": int(np.sum([e["fp"] for e in evaluations])),
        "fn": int(np.sum([e["fn"] for e in evaluations])),
        "mean_precision": float(np.mean([e["precision"] for e in evaluations])),
        "mean_recall": float(np.mean([e["recall"] for e in evaluations])),
        "mean_f1": float(np.mean([e["f1"] for e in evaluations])),
        "mean_iou": float(np.mean([e["mean_iou"] for e in evaluations])),
        "mean_elapsed_ms": float(np.mean(times)) if times else 0.0,
    }


def timed(func: Callable[..., T], *args, **kwargs) -> tuple[T, float]:
    """Executa uma função e devolve (resultado, tempo_em_milissegundos)."""
    start = time.perf_counter()
    output = func(*args, **kwargs)
    return output, (time.perf_counter() - start) * 1000.0