"""
gui.py - Interface gráfica do projeto (Tkinter + Pillow).

COMO O PROGRAMA FUNCIONA
    Tela 1 (inicial):    céu com estrelas, Lua e foguete. O botão DECOLAR
                         faz o foguete subir e leva para a Tela 2.
    Tela 2 (análise):    o usuário importa uma imagem da Lua e o pipeline real
                         roda integrando os módulos de todos os integrantes:
                           1. Aquisição (preprocessing/acquisition.py)
                           2. Contraste CLAHE (preprocessing/contrast.py)
                           3. Suavização Gaussiana (preprocessing/smoothing.py)
                           4. Bordas Sobel + Laplaciano + Limiar por Histerese (preprocessing/edges.py)
                           5. Morfologia com Disco: abertura/fechamento/clean (segmentation/morphology.py)
                           6. Segmentação Otsu + Sombra em Meia-Lua + Hough Multi-Faixa + Validação Radial Dupla Escala
                           7. Descrição Geométrica: centro, raio, área, perímetro, circularidade (metrics/features.py)
    Janela de Avaliação: compara as crateras detectadas com o Ground Truth
                         usando metrics/evaluation.py, mostrando visualmente os
                         Acertos (TP), Falsos Alarmes (FP) e Esquecidas (FN),
                         além de dizer se a Precisão, o Recall e o IoU foram bons.
"""
import random
import sys
import time
from pathlib import Path

_PREPROC_DIR = str(Path(__file__).resolve().parent / "preprocessing")
if _PREPROC_DIR not in sys.path:
    sys.path.insert(0, _PREPROC_DIR)

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk
except ImportError:
    tk = None
    filedialog = messagebox = ttk = ImageTk = None

import cv2
import numpy as np
from PIL import Image, ImageDraw

from metrics import evaluation, features
from preprocessing import acquisition, contrast, edges, smoothing
from segmentation import morphology

# ----------------------------------------------------------------------------
# Identidade visual
# ----------------------------------------------------------------------------
APP_NAME = "Caçadores de Crateras"
TAGLINE = "Encontrando crateras na Lua com imagens da NASA"

BG = "#05060f"       # céu
PANEL = "#0d1024"    # painéis
LINE = "#232850"     # bordas
TEXT = "#e8ebff"
MUTED = "#9aa1cf"
CYAN = "#36e0ff"
ORANGE = "#ff8a3d"
VIOLET = "#7a5cff"
GREEN = "#3ddc84"
RED = "#ff5c77"
FONT = "Helvetica"

USING_DEMO_PIPELINE = False

# Parâmetros padrão ajustáveis na barra da Tela 2
HOUGH_PARAMS = {
    "min_dist": 20,
    "param1": 80,
    "param2": 36,       # Exigência (maior = mais rigoroso / menos falsos alarmes)
    "min_radius": 6,
    "max_radius": 110,  # Cobre desde micro-crateras até crateras gigantes (> 80 px)
    "min_circularity": 0.35,
}

# Colunas da tabela: (chave, título, largura)
COLUMNS = [
    ("n", "#", 40),
    ("center", "Centro (x, y)", 110),
    ("radius", "Raio (px)", 80),
    ("area", "Área (px²)", 90),
    ("perimeter", "Perímetro (px)", 100),
    ("circ", "Circularidade", 100),
]

# Crateras decorativas da Lua: (dx, dy, raio) como fração do raio da Lua.
MOON_DECOR = [
    (-0.35, -0.93, 0.05),
    (0.15, -0.88, 0.07),
    (0.5, -0.82, 0.045),
    (-0.7, -0.74, 0.06),
    (0.82, -0.62, 0.05),
    (-0.15, -0.8, 0.03),
]

# ----------------------------------------------------------------------------
# Estado global e cache de pré-processamento
# ----------------------------------------------------------------------------
estado = {
    "gray": None,         # imagem PIL original em escala de cinza ("L")
    "image_path": None,   # caminho do arquivo aberto
    "craters": [],        # lista de dicionários retornada pelo pipeline
    "elapsed_ms": 0.0,    # tempo da última execução em ms
    "annotated": None,    # imagem RGB com as crateras desenhadas
    "photo": None,        # referência obrigatória para o Tkinter exibir a imagem
    "launching": False,   # True enquanto o foguete está voando
    "ground_truth": [],   # lista de tuplas (x, y, r) do gabarito atual
    "last_eval": None,    # último resultado de evaluation.evaluate(...)
}
_preproc_cache = {
    "key": None,
    "data": None,
}
ui = {}      # widgets: botões, canvas, tabela, resumo
telas = {}   # frames: "inicial" e "analise"
root = None  # inicializado em main()


# ----------------------------------------------------------------------------
# Lógica: pipeline real integrado com todos os módulos
# ----------------------------------------------------------------------------
def _run_team_preprocessing(work_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Executa os módulos de pré-processamento:
      - Etapa 2: contrast.clahe (equalização adaptativa de histograma)
      - Etapa 3: smoothing.gaussian_blur (suavização Gaussiana por convolução 2D)
      - Etapa 4: edges.sobel, edges.laplacian e edges.threshold_edges (detecção de bordas)
    Guarda o resultado em cache para que o botão 'Recalcular' responda instantaneamente.
    """
    cache_key = (work_img.shape, int(work_img.sum()), int(work_img[0, 0]), int(work_img[-1, -1]))
    if _preproc_cache["key"] == cache_key and _preproc_cache["data"] is not None:
        return _preproc_cache["data"]

    # 1. Melhoria de contraste usando preprocessing/contrast.py
    if hasattr(contrast, "clahe"):
        h_w, w_w = work_img.shape
        if max(h_w, w_w) > 320:
            s_c = 320.0 / max(h_w, w_w)
            small = cv2.resize(
                work_img,
                (max(16, int(w_w * s_c)), max(16, int(h_w * s_c))),
                interpolation=cv2.INTER_AREA,
            )
            small_clahe = contrast.clahe(small, clip_limit=2.0, tile_grid=(8, 8)).astype(np.float32)
            gain = (small_clahe + 1.0) / (small.astype(np.float32) + 1.0)
            gain_full = cv2.resize(gain, (w_w, h_w), interpolation=cv2.INTER_LINEAR)
            enhanced = np.clip(np.rint(work_img.astype(np.float32) * gain_full), 0, 255).astype(np.uint8)
        else:
            enhanced = contrast.clahe(work_img, clip_limit=2.0, tile_grid=(8, 8))
    elif hasattr(contrast, "apply_clahe"):
        enhanced = contrast.apply_clahe(work_img)
    else:
        clahe_cv = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe_cv.apply(work_img)

    # 2. Suavização Gaussiana usando preprocessing/smoothing.py
    if hasattr(smoothing, "gaussian_blur"):
        blur_f = np.asarray(smoothing.gaussian_blur(enhanced, sigma=1.0), dtype=np.float32)
        blur_u8 = np.clip(np.rint(blur_f), 0, 255).astype(np.uint8)
    else:
        blur_u8 = cv2.GaussianBlur(enhanced, (5, 5), 1.0)
        blur_f = blur_u8.astype(np.float32)

    # 3. Detecção de bordas usando preprocessing/edges.py (Sobel + Laplaciano + Limiarização)
    if hasattr(edges, "sobel") and hasattr(edges, "laplacian"):
        sobel_mag = np.asarray(edges.sobel(blur_f), dtype=np.float32)
        lap_mag = np.asarray(edges.laplacian(blur_f), dtype=np.float32)
        grad = sobel_mag + 0.25 * lap_mag
        if hasattr(edges, "threshold_edges"):
            edge_binary = edges.threshold_edges(sobel_mag, low_pct=88, high_pct=96)
        else:
            edge_binary = (sobel_mag >= np.percentile(sobel_mag, 90)).astype(np.uint8) * 255
    else:
        gx_s = cv2.Sobel(blur_f, cv2.CV_32F, 1, 0, ksize=3)
        gy_s = cv2.Sobel(blur_f, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.hypot(gx_s, gy_s)
        edge_binary = (grad >= np.percentile(grad, 90)).astype(np.uint8) * 255

    result = (blur_u8, blur_f, grad, edge_binary)
    _preproc_cache["key"] = cache_key
    _preproc_cache["data"] = result
    return result


def _build_cohesive_crater_mask(
    blurred: np.ndarray,
    edge_binary: np.ndarray,
    circles_work: list[tuple[int, int, int]],
) -> np.ndarray:
    """Gera a máscara binária combinando Otsu + Bordas (preprocessing/edges.py) +
    Morfologia com disco (segmentation/morphology.py), garantindo que cada cratera
    forme uma região conexa para medir área, perímetro e circularidade.
    """
    _, mask_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    med = float(np.median(blurred))
    std = float(np.std(blurred))
    anom = (
        (np.abs(blurred.astype(np.float32) - med) > max(14.0, 0.35 * std))
        | (mask_otsu == 0)
        | (edge_binary > 0)
    )

    cleaned = morphology.closing(
        morphology.clean(anom, kernel_size=5, iterations=1),
        kernel_size=7,
        iterations=1,
    )

    h, w = cleaned.shape
    final_mask = np.zeros((h, w), dtype=np.uint8)
    for cx, cy, r in circles_work:
        x0, x1 = max(0, cx - r - 2), min(w, cx + r + 3)
        y0, y1 = max(0, cy - r - 2), min(h, cy + r + 3)

        sub_clean = cleaned[y0:y1, x0:x1].astype(np.uint8)
        circle_patch = np.zeros_like(sub_clean)
        cv2.circle(circle_patch, (cx - x0, cy - y0), r, 1, thickness=-1)
        inter = (sub_clean & circle_patch) * 255

        contours, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= 0.16 * np.pi * (r ** 2):
                hull = cv2.convexHull(largest)
                cv2.drawContours(final_mask[y0:y1, x0:x1], [hull], -1, 1, thickness=-1)
                continue

        cv2.circle(final_mask, (cx, cy), r, 1, thickness=-1)

    return final_mask.astype(bool)


def run_pipeline(gray: Image.Image, params: dict | None = None) -> list[dict]:
    """Recebe uma imagem PIL em escala de cinza ("L") e devolve a lista de crateras.

    Avalia cada candidato em DUPLA ESCALA (sigma=20 para crateras pequenas/médias e
    bordas finas em meia-lua, e sigma=60 para crateras gigantes em bacia profunda),
    garantindo que crateras grandes (raio 60 a 110 px) nunca sejam apagadas pelo
    filtro de fundo nem bloqueadas por pequenas marcações na sua borda.
    """
    cfg = dict(HOUGH_PARAMS)
    if params:
        cfg.update(params)

    orig_img = np.array(gray, dtype=np.uint8)
    orig_h, orig_w = orig_img.shape

    max_side = max(orig_h, orig_w)
    scale = 780.0 / max_side if max_side > 780 else 1.0
    if scale < 1.0:
        work_w = max(32, int(round(orig_w * scale)))
        work_h = max(32, int(round(orig_h * scale)))
        work_img = cv2.resize(orig_img, (work_w, work_h), interpolation=cv2.INTER_AREA)
    else:
        work_img = orig_img.copy()

    h, w = work_img.shape

    # 1. Identifica apenas regiões externas "sem dados" (ex: triângulo preto de borda do QuickMap)
    zero_mask = (work_img <= 1).astype(np.uint8)
    num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(zero_mask)
    no_data_mask = np.zeros((h, w), dtype=bool)
    for i in range(1, num_lbl):
        area = stats[i, cv2.CC_STAT_AREA]
        x_b = stats[i, cv2.CC_STAT_LEFT]
        y_b = stats[i, cv2.CC_STAT_TOP]
        w_b = stats[i, cv2.CC_STAT_WIDTH]
        h_b = stats[i, cv2.CC_STAT_HEIGHT]
        if (x_b == 0 or y_b == 0 or (x_b + w_b >= w) or (y_b + h_b >= h)) and area > 0.02 * (h * w):
            no_data_mask |= (lbls == i)

    if np.any(no_data_mask):
        valid_inner = ~cv2.dilate(no_data_mask.astype(np.uint8), np.ones((23, 23), np.uint8)).astype(bool)
        med_val = int(np.median(work_img[~no_data_mask])) if np.any(~no_data_mask) else 128
        img_filled = work_img.copy()
        img_filled[no_data_mask] = med_val
    else:
        valid_inner = np.ones((h, w), dtype=bool)
        img_filled = work_img

    # 2. Roda o pré-processamento do grupo (CLAHE, Gaussiano, Sobel e Laplaciano)
    blur_u8, blur_f, grad, edge_bin = _run_team_preprocessing(img_filled)

    gx = smoothing.convolve2d(blur_f, edges.SOBEL_X)
    gy = smoothing.convolve2d(blur_f, edges.SOBEL_Y)
    grad_mag = np.hypot(gx, gy) + 1e-5
    ux_grad = gx / grad_mag
    uy_grad = gy / grad_mag

    # Contraste em dupla escala:
    # - norm_diff_s (sigma=20): realça crateras pequenas/médias e sombras finas em meia-lua
    # - norm_diff_l (sigma=60): preserva o interior escuro de crateras gigantes (raio > 50 px)
    bg_small = cv2.GaussianBlur(blur_f, (0, 0), 20.0)
    bg_large = cv2.GaussianBlur(blur_f, (0, 0), 60.0)
    diff_small = blur_f - bg_small
    diff_large = blur_f - bg_large
    diff_small[~valid_inner] = 0.0
    diff_large[~valid_inner] = 0.0
    std_s = float(np.std(diff_small[valid_inner])) + 1e-5
    std_l = float(np.std(diff_large[valid_inner])) + 1e-5
    norm_diff_s = diff_small / std_s
    norm_diff_l = diff_large / std_l

    local_grad_mean = cv2.GaussianBlur(grad_mag, (0, 0), 15.0) + 6.0
    norm_grad = np.clip(grad_mag / local_grad_mean, 0.0, 4.0)
    norm_grad[~valid_inner] = 0.0

    r_min_cfg = max(4, int(round(cfg["min_radius"] * scale)))
    r_max_cfg = max(r_min_cfg + 12, int(round(max(cfg["max_radius"], 105) * scale)))
    param2_cfg = max(10, int(cfg["param2"]))

    cand_grid: dict[tuple[int, int, int], tuple[float, float, float]] = {}

    def add_cand(cx: float, cy: float, r: float) -> None:
        for rf in (0.90, 1.0, 1.10, 1.20):
            rr = r * rf
            if r_min_cfg <= rr <= r_max_cfg:
                ix, iy = int(round(cx)), int(round(cy))
                if 0 <= ix < w and 0 <= iy < h and valid_inner[iy, ix]:
                    key = (ix // 3, iy // 3, int(round(rr)) // 2)
                    if key not in cand_grid:
                        cand_grid[key] = (float(cx), float(cy), float(rr))

    # 3. Candidatos de Bacias Escuras (Otsu + Dupla Escala) e Sombras em Meia-Lua (Morfologia)
    otsu_t, _ = cv2.threshold(blur_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    shadow_sources = [
        (blur_u8 < (otsu_t * tf)) & valid_inner for tf in (0.45, 0.62, 0.78, 0.95)
    ] + [
        (norm_diff_s < -tm) & valid_inner for tm in (0.45, 0.75, 1.10, 1.50)
    ] + [
        (norm_diff_l < -tm) & valid_inner for tm in (0.55, 0.90, 1.30)
    ]

    for sh_mask in shadow_sources:
        for ksz in (3, 5, 7):
            cl = morphology.opening(sh_mask, kernel_size=ksz, iterations=1)
            cnts, _ = cv2.findContours(cl.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < 5:
                    continue
                (cx0, cy0), rad0 = cv2.minEnclosingCircle(c)
                add_cand(cx0, cy0, rad0)
                add_cand(cx0, cy0, rad0 * 1.18)

                rect = cv2.minAreaRect(c)
                (rx, ry), (rw, rh), angle_deg = rect
                major, minor = max(rw, rh), min(rw, rh)
                if major >= 6 and minor < 0.88 * major:
                    theta = np.deg2rad(angle_deg)
                    nx, ny = (np.cos(theta), np.sin(theta)) if rw < rh else (-np.sin(theta), np.cos(theta))
                    for r_est in (0.48 * major, 0.56 * major, 0.65 * major):
                        shift = max(1.2, r_est - 0.38 * minor)
                        add_cand(rx + shift * nx, ry + shift * ny, r_est)
                        add_cand(rx - shift * nx, ry - shift * ny, r_est)

    # 4. Candidatos via Transformada de Hough em 6 faixas de raio (cobrindo até as crateras gigantes)
    norm_grad_u8 = np.clip(norm_grad * 63.0, 0, 255).astype(np.uint8)
    band_edges = np.linspace(r_min_cfg, r_max_cfg, 7).astype(int)
    for b_idx in range(6):
        b_min = int(band_edges[b_idx])
        b_max = max(b_min + 4, int(band_edges[b_idx + 1]))
        b_p2 = max(11, int(round(12 + 1.5 * b_idx)))
        for src_img in (blur_u8, norm_grad_u8):
            hc = cv2.HoughCircles(
                src_img,
                cv2.HOUGH_GRADIENT,
                dp=1.0,
                minDist=max(6, int(b_min * 0.65)),
                param1=max(30, int(cfg["param1"])),
                param2=b_p2,
                minRadius=b_min,
                maxRadius=b_max,
            )
            if hc is not None:
                for cx, cy, rad in hc[0]:
                    add_cand(cx, cy, rad)

    if not cand_grid:
        return []

    # 5. Avaliação Radial Vetorizada em Dupla Escala no NumPy
    cands_arr = np.array(list(cand_grid.values()), dtype=np.float32)
    cx_v, cy_v, r_v = cands_arr[:, 0:1], cands_arr[:, 1:2], cands_arr[:, 2:3]
    r_flat = r_v[:, 0]

    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False, dtype=np.float32)
    cos_a, sin_a = np.cos(angles)[None, :], np.sin(angles)[None, :]

    def eval_on_diff_map(ndiff: np.ndarray, is_wide_scale: bool = False):
        def smp(factor: float):
            xs = np.clip((cx_v + factor * r_v * cos_a).astype(np.int32), 0, w - 1)
            ys = np.clip((cy_v + factor * r_v * sin_a).astype(np.int32), 0, h - 1)
            rad_align = ux_grad[ys, xs] * cos_a + uy_grad[ys, xs] * sin_a
            return ndiff[ys, xs], norm_grad[ys, xs], rad_align, valid_inner[ys, xs]

        d35, ng35, _, _ = smp(0.35)
        d68, _, _, _ = smp(0.68)
        d85, ng85, ra85, val85 = smp(0.85)
        d98, ng98, ra98, val98 = smp(0.98)
        d112, ng112, ra112, val112 = smp(1.12)

        in_bounds = (
            (np.mean(val85 & val98 & val112, axis=1) >= 0.88)
            & ~((cy_v[:, 0] > h - 38) & (cx_v[:, 0] > w * 0.35))
        )
        in_wall = np.minimum(d68, d85)
        out_wall = np.maximum(d98, d112)
        outward_step = out_wall - in_wall

        step_frac = np.mean(outward_step > 0.25, axis=1)
        mean_outward_step = np.mean(outward_step, axis=1)

        ng_rim = np.maximum(ng85, np.maximum(ng98, ng112))
        ra_rim = np.maximum(ra85, np.maximum(ra98, ra112))
        rim_rad_prod = ng_rim * np.clip(ra_rim, -0.2, 1.0)
        radial_edge_coherence = np.mean(rim_rad_prod, axis=1)
        radial_align_frac = np.mean((ng_rim > 0.56) & (ra_rim > 0.25), axis=1)

        quad_active = np.zeros((len(cands_arr), 4), dtype=bool)
        for q in range(4):
            q_slice = slice(q * 6, (q + 1) * 6)
            q_step = np.mean(outward_step[:, q_slice] > 0.18, axis=1)
            q_edge = np.mean(ng_rim[:, q_slice] > 0.56, axis=1)
            quad_active[:, q] = (q_step >= 0.33) & (q_edge >= 0.33)
        quads_ok = np.sum(quad_active, axis=1)

        min_inside = np.minimum(d35, in_wall)
        shadow_strength = -np.percentile(min_inside, 20, axis=1)
        rim_bright_strength = np.percentile(out_wall, 80, axis=1)

        is_shadow_angle = min_inside < -0.42
        shadow_count = np.sum(is_shadow_angle, axis=1)
        shadow_enclosed = np.sum((out_wall - min_inside) * is_shadow_angle, axis=1) / np.maximum(1, shadow_count)

        bowl_bonus = np.where(
            r_flat <= 15.0,
            np.maximum(0.0, (-np.mean(d35, axis=1)) - 0.3) * 0.9,
            0.0,
        )
        interior_clutter_pen = np.maximum(0.0, np.mean(ng35, axis=1) - 0.85 * np.mean(ng_rim, axis=1)) * 2.5
        large_weak_pen = (
            0.0
            if is_wide_scale
            else np.maximum(0.0, (r_flat - 20.0) / 15.0) * np.maximum(0.0, 0.70 - step_frac) * 10.0
        )

        sc = (
            1.10 * mean_outward_step * (step_frac ** 1.3)
            + 0.70 * shadow_enclosed
            + 0.55 * (shadow_strength + rim_bright_strength)
            + 1.80 * radial_edge_coherence * (radial_align_frac ** 1.2)
            + 0.35 * quads_ok
            + bowl_bonus
            - interior_clutter_pen
            - large_weak_pen
        )
        req_quads = np.where(r_flat >= 18.0, 4, np.where(r_flat >= 11.0, 3, 2))
        min_step_req = 0.62 if is_wide_scale else np.clip(0.50 + 0.0045 * (r_flat - 10.0), 0.48, 0.68)
        ok = (
            in_bounds
            & (quads_ok >= req_quads)
            & (shadow_count >= 3)
            & (shadow_strength >= (0.85 if is_wide_scale else 0.50))
            & (shadow_enclosed >= (0.65 if is_wide_scale else 0.42))
            & (step_frac >= min_step_req)
            & (radial_align_frac >= 0.42)
        )
        if is_wide_scale:
            ok &= (-np.mean(min_inside, axis=1) >= 0.75) & (r_flat >= 24.0)
        return sc, ok

    sc_s, ok_s = eval_on_diff_map(norm_diff_s, is_wide_scale=False)
    sc_l, ok_l = eval_on_diff_map(norm_diff_l, is_wide_scale=True)

    scores = np.where(ok_l & (~ok_s | (sc_l > sc_s)), sc_l, sc_s)
    valid_base = ok_s | ok_l

    high_roughness = float(np.std(work_img[~no_data_mask])) > 42.0
    eff_min_r = max(r_min_cfg, 9 if (high_roughness and cfg["min_radius"] <= 6) else r_min_cfg)
    base_thresh = 3.40 if high_roughness else 3.15
    min_score_thresh = base_thresh + (param2_cfg - 36) * 0.08
    max_detections = max(15, int(round((50 if high_roughness else 70) - (param2_cfg - 36))))

    valid_mask = valid_base & (r_flat >= eff_min_r) & (scores >= min_score_thresh)
    valid_idx = np.where(valid_mask)[0]
    order = valid_idx[np.argsort(-scores[valid_idx])]

    min_dist_factor = max(0.45, min(1.15, float(cfg["min_dist"]) / 20.0))
    selected: list[tuple[int, int, int]] = []
    for idx in order:
        cx = int(round(cands_arr[idx, 0]))
        cy = int(round(cands_arr[idx, 1]))
        r = int(round(cands_arr[idx, 2]))
        conflict = False
        for ex, ey, er in selected:
            dist = np.hypot(cx - ex, cy - ey)
            # 1) Dois círculos de porte comparável não podem se sobrepor
            if min(r, er) >= 0.48 * max(r, er) and dist < 0.80 * min_dist_factor * max(r, er):
                conflict = True
                break
            # 2) Um círculo grande não pode ser bloqueado por um círculo pequeno na sua borda
            if er > 1.25 * r and dist <= 1.15 * er:
                conflict = True
                break
            if dist < 0.45 * min_dist_factor * max(r, er):
                conflict = True
                break
        if not conflict:
            selected.append((cx, cy, r))
        if len(selected) >= max_detections:
            break

    if not selected:
        return []

    # 6. Limpeza de aninhamento: se uma cratera grande entrou depois de um círculo menor que
    # estava dentro dela ou na sua borda, mantém a cratera grande e remove o círculo menor redundante.
    cleaned_selected: list[tuple[int, int, int]] = []
    for i, (cx, cy, r) in enumerate(selected):
        inside_larger = False
        for j, (ex, ey, er) in enumerate(selected):
            if i != j and er > 1.28 * r and np.hypot(cx - ex, cy - ey) <= 1.12 * er:
                inside_larger = True
                break
        if not inside_larger:
            cleaned_selected.append((cx, cy, r))

    # 7. Constrói a máscara morfológica e extrai atributos geométricos com metrics/features.py
    mask_work = _build_cohesive_crater_mask(blur_u8, edge_bin, cleaned_selected)
    work_circles_arr = np.array(cleaned_selected, dtype=np.float32)
    craters_work = features.describe_craters(mask_work, work_circles_arr)

    min_circ = float(cfg.get("min_circularity", 0.35))
    inv_scale = 1.0 / scale
    results: list[dict] = []
    for c in craters_work:
        if c["circularity"] < min_circ:
            continue
        wx, wy = c["center"]
        wr = c["radius"]
        results.append({
            "center": (int(round(wx * inv_scale)), int(round(wy * inv_scale))),
            "radius": max(1, int(round(wr * inv_scale))),
            "area": float(c["area"] * (inv_scale ** 2)),
            "perimeter": float(c["perimeter"] * inv_scale),
            "circularity": float(c["circularity"]),
            "mask": c["mask"],
        })

    return results


def draw_craters(gray: Image.Image, craters: list[dict]) -> Image.Image:
    """Desenha as crateras detectadas na imagem da Tela 2."""
    rgb = gray.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    w, h = rgb.size
    line_w = max(2, int(round(max(w, h) / 400)))
    for i, c in enumerate(craters, start=1):
        x, y = c["center"]
        r = c["radius"]
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 138, 61), width=line_w)
        draw.text((x - 3, y - 5), str(i), fill=(54, 224, 255))
    return rgb


def draw_evaluation_comparison(
    gray: Image.Image,
    eval_result: dict,
) -> Image.Image:
    """Desenha a comparação entre as crateras detectadas e o Ground Truth:
      - Verde:    Acerto (TP - detectada e existe no gabarito)
      - Vermelho: Falso Alarme (FP - detectada onde não havia cratera no gabarito)
      - Azul:     Esquecida (FN - cratera real do gabarito que não foi detectada)
    """
    rgb = gray.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    w, h = rgb.size
    line_w = max(2, int(round(max(w, h) / 400)))

    det_circles = eval_result["detected_circles"]
    gt_circles = eval_result["gt_circles"]

    for gt_idx in eval_result["unmatched_gt"]:
        x, y, r = gt_circles[gt_idx]
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(54, 224, 255), width=line_w)
        draw.text((x - 8, y - r - 12), "FN", fill=(54, 224, 255))

    for det_idx in eval_result["unmatched_detected"]:
        x, y, r = det_circles[det_idx]
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 92, 119), width=line_w)
        draw.text((x - 8, y - 6), "FP", fill=(255, 92, 119))

    for det_idx, gt_idx, iou_val in eval_result["matches"]:
        gx, gy, gr = gt_circles[gt_idx]
        draw.ellipse([gx - gr, gy - gr, gx + gr, gy + gr], outline=(180, 255, 210), width=1)

        dx, dy, dr = det_circles[det_idx]
        draw.ellipse([dx - dr, dy - dr, dx + dr, dy + dr], outline=(61, 220, 132), width=line_w + 1)
        draw.text((dx - 14, dy - 6), f"TP ({iou_val:.2f})", fill=(61, 220, 132))

    return rgb


def summarize(craters: list[dict], elapsed_ms: float, eval_result: dict | None = None) -> str:
    """Monta o texto de resumo exibido na Tela 2."""
    lines = []
    if USING_DEMO_PIPELINE:
        lines.append("MODO DEMO: dados de exemplo.\n")

    if len(craters) == 0:
        lines.append("Nenhuma cratera encontrada (tente diminuir o PARAM2 abaixo).")
    else:
        n = len(craters)
        biggest = max(craters, key=lambda c: c["radius"])
        lines.append(f"Crateras encontradas: {n}")
        lines.append(f"Raio médio: {sum(c['radius'] for c in craters) / n:.1f} px")
        lines.append(f"Maior cratera: raio {biggest['radius']} px")
        lines.append(f"Circularidade média: {sum(c['circularity'] for c in craters) / n:.2f}")

    lines.append(f"Tempo de processamento: {elapsed_ms:.0f} ms")

    if eval_result is not None:
        lines.append("")
        lines.append("--- Avaliação (Ground Truth) ---")
        lines.append(
            f"Acertos (TP): {eval_result['tp']}  |  "
            f"Falsos (FP): {eval_result['fp']}  |  "
            f"Esquecidas (FN): {eval_result['fn']}"
        )
        lines.append(
            f"Precisão: {eval_result['precision'] * 100:.0f}%  |  "
            f"Recall: {eval_result['recall'] * 100:.0f}%  |  "
            f"IoU médio: {eval_result['mean_iou'] * 100:.0f}%"
        )
        lines.append(f"Veredito: {eval_result['diagnosis']['title']}")

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Componentes visuais
# ----------------------------------------------------------------------------
def round_rect(canvas, x1, y1, x2, y2, r, **kw):
    pts = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
        x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kw)


def desenhar_botao(botao, hover):
    c = botao["canvas"]
    c.delete("all")

    if botao["ativo"]:
        edge = botao["cor"]
        if hover:
            fill = botao["cor"]
            ink = BG
        else:
            fill = PANEL
            ink = botao["cor"]
    else:
        edge = LINE
        fill = PANEL
        ink = MUTED

    round_rect(
        c, 2, 2, botao["largura"] - 2, botao["altura"] - 2, 14,
        fill=fill, outline=edge, width=2,
    )
    c.create_text(
        botao["largura"] / 2, botao["altura"] / 2, text=botao["texto"],
        fill=ink, font=(FONT, botao["fonte"], "bold"),
    )


def clicar_botao(botao):
    if botao["ativo"] and botao["comando"]:
        botao["comando"]()


def ativar_botao(botao, ativo):
    botao["ativo"] = ativo
    if ativo:
        botao["canvas"].configure(cursor="hand2")
    else:
        botao["canvas"].configure(cursor="arrow")
    desenhar_botao(botao, False)


def criar_botao(
    parent, texto, comando, largura=180, altura=42, cor=CYAN, bg=BG, fonte=10
):
    """Botão neon desenhado em Canvas. Devolve um dicionário (o 'botao')."""
    canvas = tk.Canvas(
        parent, width=largura, height=altura, bg=bg, highlightthickness=0, cursor="hand2"
    )
    botao = {
        "canvas": canvas,
        "texto": texto,
        "comando": comando,
        "cor": cor,
        "largura": largura,
        "altura": altura,
        "fonte": fonte,
        "ativo": True,
    }
    desenhar_botao(botao, False)
    canvas.bind("<Enter>", lambda e: desenhar_botao(botao, True))
    canvas.bind("<Leave>", lambda e: desenhar_botao(botao, False))
    canvas.bind("<Button-1>", lambda e: clicar_botao(botao))
    return botao


# ----------------------------------------------------------------------------
# Tela 1: boas-vindas (Lua, estrelas, foguete e botão DECOLAR)
# ----------------------------------------------------------------------------
def desenhar_inicial(_event=None):
    if estado["launching"]:
        return

    c = ui["canvas_inicial"]
    w, h = c.winfo_width(), c.winfo_height()
    if w < 50 or h < 50:
        return
    c.delete("all")

    rng = random.Random(7)
    for _ in range(int(w * h / 4500)):
        x, y, r = rng.randint(0, w), rng.randint(0, h), rng.choice([1, 1, 1, 2])
        c.create_oval(
            x - r, y - r, x + r, y + r, outline="",
            fill=rng.choice(["#ffffff", "#b8c4ff", "#ffd9b0"]),
        )

    radius = min(w, h * 1.3) * 0.6
    top = h - h * 0.2
    cx, cy = w / 2, top + radius
    c.create_oval(
        cx - radius - 6, cy - radius - 6, cx + radius + 6, cy + radius + 6,
        outline="#2a3a7a", width=6,
    )
    c.create_oval(
        cx - radius, cy - radius, cx + radius, cy + radius,
        fill="#cfd2de", outline="",
    )
    for dx, dy, rr in MOON_DECOR:
        x, y, r = cx + dx * radius, cy + dy * radius, rr * radius
        c.create_oval(x - r, y - r, x + r, y + r, fill="#aeb2c4", outline="#8f94a8", width=2)
        c.create_oval(
            x - r * 0.7, y - r * 0.5, x + r * 0.7, y + r * 0.8,
            fill="#9ca0b5", outline="",
        )

    size = max(28, min(54, w // 18))
    c.create_text(w / 2 + 2, h * 0.2 + 2, text=APP_NAME, fill=VIOLET, font=(FONT, size, "bold"))
    c.create_text(w / 2, h * 0.2, text=APP_NAME, fill=TEXT, font=(FONT, size, "bold"))
    c.create_text(w / 2, h * 0.2 + size * 1.1, text=TAGLINE, fill=MUTED, font=(FONT, 14))
    c.create_window(w / 2, h * 0.46, window=ui["botao_decolar"]["canvas"])
    c.create_text(
        w / 2, h - 14, text="UFRPE  ·  Processamento de Imagens 2026.2",
        fill="#4a4f66", font=(FONT, 10),
    )

    desenhar_foguete(w / 2, top - 4)


def desenhar_foguete(x, base):
    """Foguete feito com polígonos."""
    c = ui["canvas_inicial"]
    t = "rocket"
    c.create_polygon(x - 9, base, x, base + 26, x + 9, base, fill=ORANGE, outline="", tags=t)
    c.create_polygon(
        x - 16, base - 36, x - 32, base - 6, x - 16, base - 14,
        fill=ORANGE, outline="", tags=t,
    )
    c.create_polygon(
        x + 16, base - 36, x + 32, base - 6, x + 16, base - 14,
        fill=ORANGE, outline="", tags=t,
    )
    c.create_polygon(
        x, base - 96, x + 17, base - 54, x + 17, base - 12, x - 17, base - 12,
        x - 17, base - 54, fill=TEXT, outline="#b8c4ff", width=2, smooth=False, tags=t,
    )
    c.create_oval(x - 7, base - 66, x + 7, base - 52, fill=CYAN, outline=BG, width=2, tags=t)


def decolar():
    """Foguete voa e, ao sair da tela, abre a Tela 2."""
    if estado["launching"]:
        return
    estado["launching"] = True
    ativar_botao(ui["botao_decolar"], False)
    voar(2.0)


def voar(speed):
    c = ui["canvas_inicial"]
    c.move("rocket", 0, -speed)
    box = c.bbox("rocket")
    if box is None or box[3] < 0:
        mostrar("analise")
    else:
        root.after(16, lambda: voar(speed + 0.6))


# ----------------------------------------------------------------------------
# Tela 2: importar imagem, rodar pipeline real, ajustar parâmetros e avaliar
# ----------------------------------------------------------------------------
def ler_parametros_ui():
    """Lê os controles de PARAM2, Raio Mín e Raio Máx da Tela 2, se existirem."""
    if "var_param2" not in ui:
        return dict(HOUGH_PARAMS)
    try:
        HOUGH_PARAMS["param2"] = max(10, int(ui["var_param2"].get()))
        HOUGH_PARAMS["min_radius"] = max(4, int(ui["var_min_r"].get()))
        HOUGH_PARAMS["max_radius"] = max(HOUGH_PARAMS["min_radius"] + 6, int(ui["var_max_r"].get()))
        HOUGH_PARAMS["min_dist"] = max(6, int(ui["var_min_dist"].get()))
    except (ValueError, tk.TclError):
        pass
    return dict(HOUGH_PARAMS)


def importar_imagem():
    """Escolhe o arquivo, roda o pipeline real, mostra imagem e relatório."""
    path = filedialog.askopenfilename(
        title="Escolha um recorte da Lua",
        filetypes=[("Imagens", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")],
    )
    if not path:
        return

    try:
        gray = acquisition.load_grayscale_image(path)
    except (OSError, ValueError) as exc:
        messagebox.showerror("Imagem inválida", f"Não foi possível abrir o arquivo.\n\n{exc}")
        return

    estado["gray"] = gray
    estado["image_path"] = path
    estado["ground_truth"] = []
    estado["last_eval"] = None

    executar_deteccao()
    ativar_botao(ui["botao_salvar"], True)
    ativar_botao(ui["botao_exportar_csv"], True)
    ativar_botao(ui["botao_avaliar"], True)
    if "botao_recalcular" in ui:
        ativar_botao(ui["botao_recalcular"], True)


def executar_deteccao():
    """Roda o pipeline na imagem atual com os parâmetros configurados."""
    if estado["gray"] is None:
        return

    cfg = ler_parametros_ui()
    craters, elapsed_ms = evaluation.timed(run_pipeline, estado["gray"], cfg)
    estado["craters"] = craters
    estado["elapsed_ms"] = elapsed_ms
    estado["annotated"] = draw_craters(estado["gray"], craters)

    if estado["ground_truth"]:
        estado["last_eval"] = evaluation.evaluate(
            craters, estado["ground_truth"], elapsed_ms=elapsed_ms
        )

    renderizar()
    preencher_relatorio(craters, elapsed_ms, estado["last_eval"])


def salvar_imagem():
    """Salva a imagem com as crateras desenhadas."""
    if estado["annotated"] is None:
        return
    path = filedialog.asksaveasfilename(
        defaultextension=".png", filetypes=[("PNG", "*.png")]
    )
    if path:
        estado["annotated"].save(path)


def exportar_csv_crateras():
    """Exporta as crateras detectadas para um arquivo CSV (útil para relatórios ou base de gabarito)."""
    if not estado["craters"]:
        messagebox.showinfo("Sem crateras", "Nenhuma cratera detectada para exportar.")
        return

    path = filedialog.asksaveasfilename(
        title="Salvar tabela de crateras (CSV)",
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv")],
    )
    if not path:
        return

    circles = evaluation.circles_from_craters(estado["craters"])
    evaluation.save_ground_truth(path, circles)
    messagebox.showinfo(
        "CSV exportado",
        f"Foram salvas {len(circles)} crateras em:\n{path}\n\n"
        "Formato: x, y, r (também pode ser usado como base para Ground Truth).",
    )


def renderizar():
    c = ui["canvas_imagem"]
    c.delete("all")
    w, h = c.winfo_width(), c.winfo_height()

    if estado["annotated"] is None:
        rng = random.Random(3)
        for _ in range(60):
            x, y = rng.randint(0, max(1, w)), rng.randint(0, max(1, h))
            c.create_oval(x, y, x + 2, y + 2, fill="#8890c0", outline="")
        c.create_text(
            w // 2, h // 2, text="Importe uma imagem da Lua\npara começar",
            fill=MUTED, font=(FONT, 16), justify="center",
        )
    else:
        img = estado["annotated"].copy()
        img.thumbnail((max(w - 4, 10), max(h - 4, 10)))
        estado["photo"] = ImageTk.PhotoImage(img)
        c.create_image(w // 2, h // 2, image=estado["photo"])


def preencher_relatorio(craters, elapsed_ms, eval_result=None):
    ui["summary"].configure(text=summarize(craters, elapsed_ms, eval_result), fg=TEXT)

    tree = ui["tree"]
    tree.delete(*tree.get_children())
    for i, c in enumerate(craters, start=1):
        x, y = c["center"]
        tree.insert(
            "",
            "end",
            values=(
                i,
                f"({x}, {y})",
                c["radius"],
                f"{c['area']:.0f}",
                f"{c['perimeter']:.1f}",
                f"{c['circularity']:.2f}",
            ),
        )


# ----------------------------------------------------------------------------
# Janela de Comparação e Avaliação com Ground Truth
# ----------------------------------------------------------------------------
def abrir_janela_avaliacao():
    """Abre uma janela dedicada que compara as detecções com o Ground Truth,
    mostra visualmente TP / FP / FN e responde se a Precisão e o Recall foram bons.
    Também permite marcar crateras do gabarito clicando direto na imagem
    """
    if estado["gray"] is None:
        messagebox.showinfo("Aviso", "Importe uma imagem da Lua primeiro.")
        return

    win = tk.Toplevel(root)
    win.title("Avaliação com Ground Truth (Gabarito)")
    win.geometry("1120x720")
    win.minsize(960, 640)
    win.configure(bg=BG)

    win_state = {
        "gt_circles": list(estado["ground_truth"]),
        "eval_result": estado["last_eval"],
        "photo": None,
        "scale": 1.0,
        "offset_x": 0,
        "offset_y": 0,
        "click_center": None,
    }

    top = tk.Frame(win, bg=BG)
    top.pack(fill="x", padx=20, pady=(14, 8))
    tk.Label(
        top,
        text="Comparação com Ground Truth (Gabarito)",
        bg=BG,
        fg=TEXT,
        font=(FONT, 18, "bold"),
    ).pack(side="left")

    tk.Label(
        top,
        text="Dica: carregue um CSV ou clique na imagem (1º clique = centro, 2º clique = borda) para marcar o gabarito!",
        bg=BG,
        fg=MUTED,
        font=(FONT, 10),
    ).pack(side="right")

    body = tk.Frame(win, bg=BG)
    body.pack(fill="both", expand=True, padx=20, pady=(0, 16))
    body.columnconfigure(0, weight=3)
    body.columnconfigure(1, weight=2)
    body.rowconfigure(0, weight=1)

    left = tk.Frame(body, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

    canvas_cmp = tk.Canvas(left, bg="#000000", highlightthickness=0, cursor="crosshair")
    canvas_cmp.pack(fill="both", expand=True, padx=10, pady=(10, 6))

    legend = tk.Frame(left, bg=PANEL)
    legend.pack(fill="x", padx=12, pady=(0, 4))
    tk.Label(legend, text="● Verde: Acerto (TP)", bg=PANEL, fg=GREEN, font=(FONT, 10, "bold")).pack(side="left", padx=(0, 14))
    tk.Label(legend, text="● Vermelho: Falso Alarme (FP)", bg=PANEL, fg=RED, font=(FONT, 10, "bold")).pack(side="left", padx=(0, 14))
    tk.Label(legend, text="● Azul: Esquecida (FN)", bg=PANEL, fg=CYAN, font=(FONT, 10, "bold")).pack(side="left")

    bar_gt = tk.Frame(left, bg=PANEL)
    bar_gt.pack(fill="x", padx=10, pady=(4, 10))

    right = tk.Frame(body, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
    right.grid(row=0, column=1, sticky="nsew")

    tk.Label(
        right,
        text="Resultado da Avaliação",
        bg=PANEL,
        fg=CYAN,
        font=(FONT, 15, "bold"),
    ).pack(anchor="w", padx=16, pady=(12, 6))

    verdict_frame = tk.Frame(right, bg="#141936", highlightbackground=LINE, highlightthickness=1)
    verdict_frame.pack(fill="x", padx=16, pady=(0, 10))

    lbl_verdict_title = tk.Label(
        verdict_frame,
        text="Aguardando Gabarito (Ground Truth)",
        bg="#141936",
        fg=ORANGE,
        font=(FONT, 13, "bold"),
        anchor="w",
    )
    lbl_verdict_title.pack(fill="x", padx=12, pady=(8, 4))

    lbl_verdict_msg = tk.Label(
        verdict_frame,
        text=(
            "Para avaliar se a precisão foi boa, carregue um arquivo CSV com o gabarito "
            "ou marque as crateras reais clicando direto na imagem ao lado (centro e depois borda)."
        ),
        bg="#141936",
        fg=TEXT,
        font=(FONT, 10),
        justify="left",
        anchor="w",
        wraplength=370,
    )
    lbl_verdict_msg.pack(fill="x", padx=12, pady=(0, 8))

    lbl_metrics = tk.Label(
        right,
        text="",
        bg=PANEL,
        fg=TEXT,
        font=(FONT, 10),
        justify="left",
        anchor="w",
        wraplength=370,
    )
    lbl_metrics.pack(fill="x", padx=16, pady=(0, 8))

    help_box = tk.Frame(right, bg="#090c1c", highlightbackground=LINE, highlightthickness=1)
    help_box.pack(fill="x", padx=16, pady=(4, 12), side="bottom")
    tk.Label(
        help_box,
        text="O que significa cada métrica?",
        bg="#090c1c",
        fg=CYAN,
        font=(FONT, 10, "bold"),
        anchor="w",
    ).pack(fill="x", padx=10, pady=(6, 2))
    tk.Label(
        help_box,
        text=(
            "• Ground Truth (Gabarito): crateras reais marcadas.\n"
            "• IoU (Interseção sobre União): % de encaixe entre o círculo detectado e o real.\n"
            "• Precisão: das crateras marcadas pelo programa, quantas eram reais?\n"
            "• Recall: das crateras reais que existem, quantas o programa achou?"
        ),
        bg="#090c1c",
        fg=MUTED,
        font=(FONT, 9),
        justify="left",
        anchor="w",
        wraplength=370,
    ).pack(fill="x", padx=10, pady=(0, 6))

    def recalcular_e_desenhar():
        estado["ground_truth"] = list(win_state["gt_circles"])
        if win_state["gt_circles"]:
            res = evaluation.evaluate(
                estado["craters"],
                win_state["gt_circles"],
                elapsed_ms=estado["elapsed_ms"],
            )
            win_state["eval_result"] = res
            estado["last_eval"] = res

            diag = res["diagnosis"]
            color_map = {"good": GREEN, "ok": CYAN, "warn": ORANGE, "bad": RED, "info": MUTED}
            lbl_verdict_title.configure(text=diag["title"], fg=color_map.get(diag["level"], TEXT))
            lbl_verdict_msg.configure(text=diag["message"])

            metrics_text = (
                f"Crateras no Gabarito (Ground Truth): {len(win_state['gt_circles'])}\n"
                f"Crateras detectadas pelo Pipeline:   {len(estado['craters'])}\n\n"
                f"• Acertos (TP - Verde):        {res['tp']}\n"
                f"• Falsos Alarmes (FP - Verm.): {res['fp']}\n"
                f"• Esquecidas (FN - Azul):      {res['fn']}\n\n"
                f"• Precisão (Precision):   {res['precision'] * 100:.1f}%\n"
                f"• Recall (Sensibilidade): {res['recall'] * 100:.1f}%\n"
                f"• F1-Score (Equilíbrio):  {res['f1'] * 100:.1f}%\n"
                f"• IoU Médio dos Acertos:  {res['mean_iou'] * 100:.1f}%\n"
                f"• Tempo de execução:      {estado['elapsed_ms']:.0f} ms"
            )
            lbl_metrics.configure(text=metrics_text)
            comp_img = draw_evaluation_comparison(estado["gray"], res)
        else:
            win_state["eval_result"] = None
            estado["last_eval"] = None
            lbl_verdict_title.configure(text="Aguardando Gabarito (Ground Truth)", fg=ORANGE)
            lbl_verdict_msg.configure(
                text=(
                    "Carregue um arquivo CSV ou clique na imagem ao lado "
                    "(1º clique no centro da cratera, 2º clique na borda) para marcar o gabarito."
                )
            )
            lbl_metrics.configure(
                text=(
                    f"Crateras detectadas pelo Pipeline: {len(estado['craters'])}\n"
                    "Nenhuma cratera marcada no gabarito ainda."
                )
            )
            comp_img = draw_craters(estado["gray"], estado["craters"])

        preencher_relatorio(estado["craters"], estado["elapsed_ms"], estado["last_eval"])

        canvas_cmp.delete("all")
        cw, ch = canvas_cmp.winfo_width(), canvas_cmp.winfo_height()
        if cw < 20 or ch < 20:
            return

        orig_w, orig_h = comp_img.size
        scale = min((cw - 8) / orig_w, (ch - 8) / orig_h)
        new_w = max(10, int(orig_w * scale))
        new_h = max(10, int(orig_h * scale))

        win_state["scale"] = scale
        win_state["offset_x"] = (cw - new_w) // 2
        win_state["offset_y"] = (ch - new_h) // 2

        resized = comp_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        win_state["photo"] = ImageTk.PhotoImage(resized)
        canvas_cmp.create_image(cw // 2, ch // 2, image=win_state["photo"])

        if win_state["click_center"] is not None:
            cx_img, cy_img = win_state["click_center"]
            sx = win_state["offset_x"] + cx_img * scale
            sy = win_state["offset_y"] + cy_img * scale
            canvas_cmp.create_oval(sx - 4, sy - 4, sx + 4, sy + 4, fill="#ffe066", outline="#000")

    def ao_clicar_imagem(event):
        scale = win_state["scale"]
        if scale <= 0:
            return
        ix = (event.x - win_state["offset_x"]) / scale
        iy = (event.y - win_state["offset_y"]) / scale
        orig_w, orig_h = estado["gray"].size
        if ix < 0 or iy < 0 or ix >= orig_w or iy >= orig_h:
            return

        if win_state["click_center"] is None:
            win_state["click_center"] = (ix, iy)
            recalcular_e_desenhar()
        else:
            cx, cy = win_state["click_center"]
            r = int(round(float(np.hypot(ix - cx, iy - cy))))
            win_state["click_center"] = None
            if r >= 3:
                win_state["gt_circles"].append((int(round(cx)), int(round(cy)), r))
            recalcular_e_desenhar()

    def carregar_gt_csv():
        path = filedialog.askopenfilename(
            parent=win,
            title="Escolher arquivo CSV de Ground Truth",
            filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
        )
        if not path:
            return
        try:
            win_state["gt_circles"] = evaluation.load_ground_truth(path)
            win_state["click_center"] = None
            recalcular_e_desenhar()
        except ValueError as exc:
            messagebox.showerror("Erro ao ler CSV", str(exc), parent=win)

    def salvar_gt_csv():
        if not win_state["gt_circles"]:
            messagebox.showinfo("Gabarito vazio", "Marque ou carregue crateras antes de salvar.", parent=win)
            return
        path = filedialog.asksaveasfilename(
            parent=win,
            title="Salvar Gabarito (Ground Truth) em CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if path:
            evaluation.save_ground_truth(path, win_state["gt_circles"])
            messagebox.showinfo("Salvo", f"Gabarito salvo com {len(win_state['gt_circles'])} crateras!", parent=win)

    def desfazer_ultimo_gt():
        win_state["click_center"] = None
        if win_state["gt_circles"]:
            win_state["gt_circles"].pop()
        recalcular_e_desenhar()

    def limpar_gt():
        win_state["click_center"] = None
        win_state["gt_circles"].clear()
        recalcular_e_desenhar()

    btn_load = criar_botao(bar_gt, "Carregar CSV", carregar_gt_csv, largura=135, altura=36, cor=CYAN, bg=PANEL, fonte=10)
    btn_load["canvas"].pack(side="left", padx=(0, 8))

    btn_save = criar_botao(bar_gt, "Salvar Gabarito CSV", salvar_gt_csv, largura=160, altura=36, cor=GREEN, bg=PANEL, fonte=10)
    btn_save["canvas"].pack(side="left", padx=(0, 8))

    btn_undo = criar_botao(bar_gt, "Desfazer marcação", desfazer_ultimo_gt, largura=150, altura=36, cor=ORANGE, bg=PANEL, fonte=10)
    btn_undo["canvas"].pack(side="left", padx=(0, 8))

    btn_clear = criar_botao(bar_gt, "Limpar tudo", limpar_gt, largura=115, altura=36, cor=VIOLET, bg=PANEL, fonte=10)
    btn_clear["canvas"].pack(side="left")

    canvas_cmp.bind("<Configure>", lambda e: recalcular_e_desenhar())
    canvas_cmp.bind("<Button-1>", ao_clicar_imagem)

    win.after(80, recalcular_e_desenhar)


# ----------------------------------------------------------------------------
# Montagem das telas
# ----------------------------------------------------------------------------
def criar_tela_inicial():
    frame = tk.Frame(root, bg=BG)
    canvas = tk.Canvas(frame, bg=BG, highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    ui["canvas_inicial"] = canvas
    ui["botao_decolar"] = criar_botao(
        canvas, "DECOLAR", decolar, largura=240, altura=56, cor=ORANGE, bg=BG, fonte=16
    )

    canvas.bind("<Configure>", desenhar_inicial)
    canvas.bind("<Map>", desenhar_inicial)
    telas["inicial"] = frame


def criar_tela_analise():
    frame = tk.Frame(root, bg=BG)

    top = tk.Frame(frame, bg=BG)
    top.pack(fill="x", padx=24, pady=(16, 6))
    tk.Label(
        top, text="Análise de imagem lunar", bg=BG, fg=TEXT, font=(FONT, 20, "bold")
    ).pack(side="left")
    botao_voltar = criar_botao(
        top, "Voltar ao início", voltar_ao_inicio, largura=160, altura=36, cor=VIOLET, bg=BG
    )
    botao_voltar["canvas"].pack(side="right")

    body = tk.Frame(frame, bg=BG)
    body.pack(fill="both", expand=True, padx=24, pady=(0, 16))
    body.columnconfigure(0, weight=3)
    body.columnconfigure(1, weight=2)
    body.rowconfigure(0, weight=1)

    left = tk.Frame(body, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

    canvas = tk.Canvas(left, bg="#000000", highlightthickness=0)
    canvas.pack(fill="both", expand=True, padx=12, pady=(12, 6))
    canvas.bind("<Configure>", lambda e: renderizar())
    ui["canvas_imagem"] = canvas

    tune_bar = tk.Frame(left, bg="#121633", highlightbackground=LINE, highlightthickness=1)
    tune_bar.pack(fill="x", padx=12, pady=(0, 6))

    tk.Label(tune_bar, text="Exigência (PARAM2):", bg="#121633", fg=MUTED, font=(FONT, 9, "bold")).pack(side="left", padx=(8, 4), pady=6)
    ui["var_param2"] = tk.IntVar(value=HOUGH_PARAMS["param2"])
    tk.Spinbox(tune_bar, from_=15, to=70, textvariable=ui["var_param2"], width=4, bg=PANEL, fg=TEXT, insertbackground=TEXT, relief="flat", font=(FONT, 10)).pack(side="left", padx=(0, 10))

    tk.Label(tune_bar, text="Raio mín/máx:", bg="#121633", fg=MUTED, font=(FONT, 9, "bold")).pack(side="left", padx=(0, 4))
    ui["var_min_r"] = tk.IntVar(value=HOUGH_PARAMS["min_radius"])
    tk.Spinbox(tune_bar, from_=4, to=80, textvariable=ui["var_min_r"], width=3, bg=PANEL, fg=TEXT, insertbackground=TEXT, relief="flat", font=(FONT, 10)).pack(side="left", padx=(0, 4))
    ui["var_max_r"] = tk.IntVar(value=HOUGH_PARAMS["max_radius"])
    tk.Spinbox(tune_bar, from_=15, to=160, textvariable=ui["var_max_r"], width=4, bg=PANEL, fg=TEXT, insertbackground=TEXT, relief="flat", font=(FONT, 10)).pack(side="left", padx=(0, 10))

    tk.Label(tune_bar, text="Dist. mín:", bg="#121633", fg=MUTED, font=(FONT, 9, "bold")).pack(side="left", padx=(0, 4))
    ui["var_min_dist"] = tk.IntVar(value=HOUGH_PARAMS["min_dist"])
    tk.Spinbox(tune_bar, from_=8, to=90, textvariable=ui["var_min_dist"], width=4, bg=PANEL, fg=TEXT, insertbackground=TEXT, relief="flat", font=(FONT, 10)).pack(side="left", padx=(0, 8))

    botao_recalcular = criar_botao(
        tune_bar, "Recalcular", executar_deteccao, largura=105, altura=30, cor=CYAN, bg="#121633", fonte=9
    )
    botao_recalcular["canvas"].pack(side="right", padx=8, pady=4)
    ativar_botao(botao_recalcular, False)
    ui["botao_recalcular"] = botao_recalcular

    bar = tk.Frame(left, bg=PANEL)
    bar.pack(fill="x", padx=12, pady=(2, 10))

    botao_importar = criar_botao(
        bar, "Importar imagem", importar_imagem, largura=138, altura=38, cor=CYAN, bg=PANEL, fonte=10
    )
    botao_importar["canvas"].pack(side="left")

    botao_salvar = criar_botao(
        bar, "Salvar imagem", salvar_imagem, largura=125, altura=38, cor=ORANGE, bg=PANEL, fonte=10
    )
    botao_salvar["canvas"].pack(side="left", padx=6)
    ativar_botao(botao_salvar, False)
    ui["botao_salvar"] = botao_salvar

    botao_exportar_csv = criar_botao(
        bar, "Exportar CSV", exportar_csv_crateras, largura=120, altura=38, cor=VIOLET, bg=PANEL, fonte=10
    )
    botao_exportar_csv["canvas"].pack(side="left", padx=(0, 6))
    ativar_botao(botao_exportar_csv, False)
    ui["botao_exportar_csv"] = botao_exportar_csv

    botao_avaliar = criar_botao(
        bar, "Avaliar (Ground Truth)", abrir_janela_avaliacao, largura=172, altura=38, cor=GREEN, bg=PANEL, fonte=10
    )
    botao_avaliar["canvas"].pack(side="left")
    ativar_botao(botao_avaliar, False)
    ui["botao_avaliar"] = botao_avaliar

    right = tk.Frame(body, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
    right.grid(row=0, column=1, sticky="nsew")
    tk.Label(
        right, text="Relatório", bg=PANEL, fg=CYAN, font=(FONT, 16, "bold")
    ).pack(anchor="w", padx=16, pady=(14, 4))

    summary = tk.Label(
        right,
        text="Importe uma imagem para gerar o relatório.",
        bg=PANEL,
        fg=MUTED,
        justify="left",
        anchor="w",
        wraplength=380,
        font=(FONT, 11),
    )
    summary.pack(fill="x", padx=16, pady=(0, 10))
    ui["summary"] = summary

    table = tk.Frame(right, bg=PANEL)
    table.pack(fill="both", expand=True, padx=16, pady=(0, 16))
    tree = ttk.Treeview(
        table, columns=[c[0] for c in COLUMNS], show="headings", height=10
    )
    for key, title, width in COLUMNS:
        tree.heading(key, text=title)
        tree.column(key, width=width, anchor="center", stretch=True)
    scroll = ttk.Scrollbar(table, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    tree.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")
    ui["tree"] = tree

    telas["analise"] = frame


# ----------------------------------------------------------------------------
# Janela principal
# ----------------------------------------------------------------------------
def configurar_estilo():
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(
        "Treeview",
        background=PANEL,
        fieldbackground=PANEL,
        foreground=TEXT,
        rowheight=26,
        borderwidth=0,
    )
    style.configure(
        "Treeview.Heading",
        background=LINE,
        foreground=CYAN,
        relief="flat",
        font=(FONT, 10, "bold"),
    )
    style.map("Treeview", background=[("selected", VIOLET)])
    style.configure(
        "Vertical.TScrollbar",
        background=LINE,
        troughcolor=PANEL,
        arrowcolor=TEXT,
        bordercolor=PANEL,
    )


def mostrar(nome):
    telas["inicial"].pack_forget()
    telas["analise"].pack_forget()
    telas[nome].pack(fill="both", expand=True)


def voltar_ao_inicio():
    estado["launching"] = False
    ativar_botao(ui["botao_decolar"], True)
    mostrar("inicial")


def main():
    if tk is None:
        raise RuntimeError(
            "Tkinter não está instalado neste ambiente Python. "
            "No Ubuntu/Debian, instale com: sudo apt install python3-tk"
        )
    global root
    root = tk.Tk()
    root.title(APP_NAME)
    root.geometry("1160x700")
    root.minsize(980, 620)
    root.configure(bg=BG)

    configurar_estilo()
    criar_tela_inicial()
    criar_tela_analise()
    mostrar("inicial")
    root.mainloop()


if __name__ == "__main__":
    main()