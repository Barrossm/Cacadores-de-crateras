"""
gui.py - Interface gráfica do projeto (Tkinter + Pillow).

COMO O PROGRAMA FUNCIONA
    Tela 1 (inicial):    céu com estrelas, Lua e foguete. O botão DECOLAR
                         faz o foguete subir e leva para a Tela 2.
    Tela 2 (análise):    o usuário importa uma imagem da Lua, o pipeline real
                         roda (Aquisição -> CLAHE/Gaussiano -> Otsu + Morfologia
                         + Hough Multi-Faixa + Validação Radial -> Features)
                         e as crateras aparecem desenhadas na imagem, com o
                         relatório (resumo + tabela) ao lado.
    Janela de Avaliação: compara as crateras detectadas com o Ground Truth
                         (gabarito em CSV ou marcado na própria tela com o mouse),
                         mostrando visualmente os Acertos (TP), Falsos Alarmes (FP)
                         e Esquecidas (FN), além de dizer se a Precisão, o Recall
                         e o IoU foram bons.
"""
import random
import time

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

# Parâmetros padrão ajustáveis na barra da Tela 2 (escala de trabalho de 750 px)
HOUGH_PARAMS = {
    "min_dist": 24,
    "param1": 90,
    "param2": 36,       # Exigência (maior = mais rigoroso / menos falsos alarmes)
    "min_radius": 11,
    "max_radius": 55,
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
# Estado global
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
ui = {}      # widgets: botões, canvas, tabela, resumo
telas = {}   # frames: "inicial" e "analise"
root = None  # inicializado em main()


# ----------------------------------------------------------------------------
# Lógica (sem interface): pipeline real integrado com os módulos do grupo
# ----------------------------------------------------------------------------
def _run_optional_preprocessing(img: np.ndarray) -> np.ndarray:
    """Roda as etapas de pré-processamento (CLAHE + Filtro Gaussiano)."""
    work = img.copy()

    if hasattr(contrast, "apply_clahe"):
        work = contrast.apply_clahe(work)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        work = clahe.apply(work)

    if hasattr(smoothing, "gaussian_blur"):
        work = smoothing.gaussian_blur(work)
    else:
        work = cv2.GaussianBlur(work, (5, 5), 1.0)

    return work


def _build_cohesive_crater_mask(
    blurred: np.ndarray,
    circles_work: list[tuple[int, int, int]],
) -> np.ndarray:
    """Gera a máscara binária combinando Otsu + Morfologia com disco (módulo do grupo),
    e garante que cada cratera forme uma região conexa para medir área e circularidade.
    """
    _, mask_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    med = float(np.median(blurred))
    std = float(np.std(blurred))
    anom = (np.abs(blurred.astype(np.float32) - med) > max(15.0, 0.38 * std)) | (mask_otsu == 0)

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
            if cv2.contourArea(largest) >= 0.18 * np.pi * (r ** 2):
                hull = cv2.convexHull(largest)
                cv2.drawContours(final_mask[y0:y1, x0:x1], [hull], -1, 1, thickness=-1)
                continue

        cv2.circle(final_mask, (cx, cy), r, 1, thickness=-1)

    return final_mask.astype(bool)


def run_pipeline(gray: Image.Image, params: dict | None = None) -> list[dict]:
    """Recebe uma imagem PIL em escala de cinza ("L") e devolve a lista de crateras.

    Corrige o problema de círculos gigantes do HoughCircle:
      1. Pré-processamento com CLAHE + Filtro Gaussiano e cálculo de bordas com Sobel.
      2. Geração de candidatos por Bacias Escuras (Otsu + Limiar Adaptativo + morphology.opening)
         e por Transformada de Hough dividida em faixas estreitas de raio (para que raios
         grandes não roubem os votos das crateras pequenas e médias).
      3. Validação radial vetorizada em NumPy: verifica se o candidato tem interior escuro
         (0.30r..0.55r) e salto para a borda iluminada exatamente em 0.95r..1.10r.
      4. Pós-refinamento de halo: se um círculo grande estiver envolvendo o halo escuro
         externo de uma cratera menor bem definida no centro, trava no anel interno real.
      5. Extração geométrica via metrics/features.py (centro, raio, área, perímetro, circularidade).
    """
    cfg = dict(HOUGH_PARAMS)
    if params:
        cfg.update(params)

    orig_img = np.array(gray, dtype=np.uint8)
    orig_h, orig_w = orig_img.shape

    # Trabalha em resolução de até 760 px no maior lado para preservar crateras pequenas
    # com nitidez e manter o tempo de execução rápido (~400 ms)
    max_side = max(orig_h, orig_w)
    scale = 760.0 / max_side if max_side > 760 else 1.0
    if scale < 1.0:
        work_w = max(32, int(round(orig_w * scale)))
        work_h = max(32, int(round(orig_h * scale)))
        work_img = cv2.resize(orig_img, (work_w, work_h), interpolation=cv2.INTER_AREA)
    else:
        work_img = orig_img.copy()

    blur = _run_optional_preprocessing(work_img)
    blur_f = blur.astype(np.float32)

    # Detecção de bordas (Sobel) para medir a força da parede circular da cratera
    gx_s = cv2.Sobel(blur_f, cv2.CV_32F, 1, 0, ksize=3)
    gy_s = cv2.Sobel(blur_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx_s, gy_s)

    r_min_cfg = max(6, int(cfg["min_radius"]))
    r_max_cfg = max(r_min_cfg + 6, int(cfg["max_radius"]))
    param2_cfg = max(10, int(cfg["param2"]))

    # Grade de deduplicação rápida de candidatos (cx, cy, r)
    cand_grid: dict[tuple[int, int, int], tuple[float, float, float]] = {}

    def add_cand(cx: float, cy: float, r: float) -> None:
        for rf in (0.85, 0.92, 1.0, 1.08):
            rr = r * rf
            if r_min_cfg <= rr <= r_max_cfg:
                key = (int(round(cx)) // 4, int(round(cy)) // 4, int(round(rr)) // 2)
                if key not in cand_grid:
                    cand_grid[key] = (float(cx), float(cy), float(rr))

    # 1. Candidatos via Limiar de Otsu + Morfologia com disco (segmentation/morphology.py)
    otsu_t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for tf in (0.45, 0.60, 0.75, 0.90, 1.05):
        dark = blur < (otsu_t * tf)
        for ksz in (3, 5):
            cl = morphology.opening(dark, kernel_size=ksz, iterations=1)
            cnts, _ = cv2.findContours(cl.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < 25:
                    continue
                (cx, cy), rad = cv2.minEnclosingCircle(c)
                add_cand(cx, cy, rad)

    # 2. Candidatos via Limiar Adaptativo Local + Morfologia (acha crateras em regiões claras ou escuras)
    for bsz in (25, 45, 75, 115):
        for cval in (14, 24, 34):
            ad = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bsz, cval
            )
            cl = morphology.opening(ad > 0, kernel_size=3, iterations=1)
            cnts, _ = cv2.findContours(cl.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < 25:
                    continue
                (cx, cy), rad = cv2.minEnclosingCircle(c)
                add_cand(cx, cy, rad)

    # 3. Transformada de Hough em faixas estreitas de raio (impede que raios grandes dominem)
    band_edges = np.linspace(r_min_cfg, r_max_cfg, 5).astype(int)
    for b_idx in range(4):
        b_min = int(band_edges[b_idx])
        b_max = max(b_min + 4, int(band_edges[b_idx + 1]))
        b_p2 = max(12, int(round(param2_cfg * (0.42 + 0.06 * b_idx))))
        hc = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=max(10, int(b_min * 0.8)),
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

    # 4. Avaliação Radial Vetorizada em NumPy para todos os candidatos de uma vez só
    cands_arr = np.array(list(cand_grid.values()), dtype=np.float32)
    cx_v, cy_v, r_v = cands_arr[:, 0:1], cands_arr[:, 1:2], cands_arr[:, 2:3]

    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False, dtype=np.float32)
    cos_a, sin_a = np.cos(angles)[None, :], np.sin(angles)[None, :]
    h, w = blur_f.shape

    def sample_ring(factor: float):
        xs = np.clip((cx_v + factor * r_v * cos_a).astype(np.int32), 0, w - 1)
        ys = np.clip((cy_v + factor * r_v * sin_a).astype(np.int32), 0, h - 1)
        return blur_f[ys, xs], grad[ys, xs]

    v30, _ = sample_ring(0.30)
    v55, _ = sample_ring(0.55)
    v72, g72 = sample_ring(0.72)
    v95, g95 = sample_ring(0.95)
    v110, _ = sample_ring(1.10)

    v_floor = 0.45 * v30 + 0.55 * v55
    v_rim = 0.65 * v95 + 0.35 * v110
    g_rim = np.maximum(g95, g72)

    p15_floor = np.percentile(v_floor, 15, axis=1)
    mean_floor = np.mean(v_floor, axis=1)
    mean_72 = np.mean(v72, axis=1)
    mean_rim = np.mean(v_rim, axis=1)
    p85_rim = np.percentile(v_rim, 85, axis=1)
    rim_jump_frac = np.mean((v_rim - v_floor) > 16.0, axis=1)
    grad_score = np.mean(g_rim, axis=1)

    # Trava o raio na transição exata entre o interior escuro (0.72r) e a crista clara (0.95r)
    early_bright_pen = np.maximum(0.0, (mean_72 - mean_floor) - 0.55 * (mean_rim - mean_floor)) * 3.0
    center_bright_pen = np.maximum(0.0, np.mean(v30, axis=1) - 90.0) * 2.0

    scores = (
        1.2 * (mean_rim - mean_floor)
        + 0.6 * (p85_rim - p15_floor)
        + 0.45 * grad_score
        - 0.6 * p15_floor
        - early_bright_pen
        - center_bright_pen
    ) * (rim_jump_frac ** 1.5)

    # O controle PARAM2 (Exigência) ajusta o limiar de confiança e o limite de detecções
    min_score_thresh = 55.0 + (param2_cfg - 20) * 1.8
    max_detections = max(12, int(round(82 - param2_cfg)))

    valid_mask = (
        (p15_floor <= 72.0)
        & (mean_floor <= 112.0)
        & ((mean_rim - mean_floor) >= 22.0)
        & (rim_jump_frac >= 0.55)
        & (scores >= min_score_thresh)
        & (r_v[:, 0] >= r_min_cfg)
        & (r_v[:, 0] <= r_max_cfg)
        & ~((cy_v[:, 0] > h - 38) & (cx_v[:, 0] > w * 0.35))
    )

    valid_idx = np.where(valid_mask)[0]
    order = valid_idx[np.argsort(-scores[valid_idx])]

    min_dist_factor = max(0.45, min(1.10, float(cfg["min_dist"]) / 33.0))
    selected: list[tuple[int, int, int]] = []
    for idx in order:
        cx = int(round(cands_arr[idx, 0]))
        cy = int(round(cands_arr[idx, 1]))
        r = int(round(cands_arr[idx, 2]))
        if not any(np.hypot(cx - ex, cy - ey) < min_dist_factor * max(r, er) for ex, ey, er in selected):
            selected.append((cx, cy, r))
        if len(selected) >= max_detections:
            break

    if not selected:
        return []

    # 5. Pós-refinamento de halo: se um círculo grande estiver envolvendo um cone escuro externo
    # que já possui a crista branca da cratera menor bem definida no centro, trava no anel interno!
    refined_work_circles: list[tuple[int, int, int]] = []
    for cx, cy, r in selected:
        best_r = r
        if r >= 24:
            for sub_f in (0.55, 0.62, 0.70, 0.78):
                sr = int(round(r * sub_f))
                if sr < r_min_cfg:
                    continue
                x_in = np.clip((cx + 0.45 * sr * cos_a[0]).astype(int), 0, w - 1)
                y_in = np.clip((cy + 0.45 * sr * sin_a[0]).astype(int), 0, h - 1)
                x_rm = np.clip((cx + 0.98 * sr * cos_a[0]).astype(int), 0, w - 1)
                y_rm = np.clip((cy + 0.98 * sr * sin_a[0]).astype(int), 0, h - 1)
                if blur_f[y_rm, x_rm].mean() - blur_f[y_in, x_in].mean() > 28.0 and blur_f[y_in, x_in].mean() < 85.0:
                    best_r = sr
                    break
        refined_work_circles.append((cx, cy, best_r))

    # 6. Constrói a máscara morfológica e extrai os atributos geométricos com metrics/features.py
    mask_work = _build_cohesive_crater_mask(blur, refined_work_circles)
    work_circles_arr = np.array(refined_work_circles, dtype=np.float32)
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

    # 1. Desenha as crateras ESQUECIDAS do gabarito (FN) em Azul Ciano
    for gt_idx in eval_result["unmatched_gt"]:
        x, y, r = gt_circles[gt_idx]
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(54, 224, 255), width=line_w)
        draw.text((x - 8, y - r - 12), "FN", fill=(54, 224, 255))

    # 2. Desenha os FALSOS ALARMES (FP) em Vermelho
    for det_idx in eval_result["unmatched_detected"]:
        x, y, r = det_circles[det_idx]
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 92, 119), width=line_w)
        draw.text((x - 8, y - 6), "FP", fill=(255, 92, 119))

    # 3. Desenha os ACERTOS (TP) em Verde (e o gabarito correspondente em linha fina)
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

    # Estrelas
    rng = random.Random(7)
    for _ in range(int(w * h / 4500)):
        x, y, r = rng.randint(0, w), rng.randint(0, h), rng.choice([1, 1, 1, 2])
        c.create_oval(
            x - r, y - r, x + r, y + r, outline="",
            fill=rng.choice(["#ffffff", "#b8c4ff", "#ffd9b0"]),
        )

    # Lua na parte de baixo
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

    # Título, subtítulo e botão
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
        HOUGH_PARAMS["min_radius"] = max(6, int(ui["var_min_r"].get()))
        HOUGH_PARAMS["max_radius"] = max(HOUGH_PARAMS["min_radius"] + 5, int(ui["var_max_r"].get()))
        HOUGH_PARAMS["min_dist"] = max(8, int(ui["var_min_dist"].get()))
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
    Também permite marcar crateras do gabarito clicando direto na imagem!
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

    # Topo da janela
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

    # Painel esquerdo: imagem comparativa interativa + botões do gabarito
    left = tk.Frame(body, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

    canvas_cmp = tk.Canvas(left, bg="#000000", highlightthickness=0, cursor="crosshair")
    canvas_cmp.pack(fill="both", expand=True, padx=10, pady=(10, 6))

    # Legenda de cores
    legend = tk.Frame(left, bg=PANEL)
    legend.pack(fill="x", padx=12, pady=(0, 4))
    tk.Label(legend, text="● Verde: Acerto (TP)", bg=PANEL, fg=GREEN, font=(FONT, 10, "bold")).pack(side="left", padx=(0, 14))
    tk.Label(legend, text="● Vermelho: Falso Alarme (FP)", bg=PANEL, fg=RED, font=(FONT, 10, "bold")).pack(side="left", padx=(0, 14))
    tk.Label(legend, text="● Azul: Esquecida (FN)", bg=PANEL, fg=CYAN, font=(FONT, 10, "bold")).pack(side="left")

    bar_gt = tk.Frame(left, bg=PANEL)
    bar_gt.pack(fill="x", padx=10, pady=(4, 10))

    # Painel direito: diagnóstico ("A precisão foi boa?") + métricas detalhadas
    right = tk.Frame(body, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
    right.grid(row=0, column=1, sticky="nsew")

    tk.Label(
        right,
        text="Resultado da Avaliação",
        bg=PANEL,
        fg=CYAN,
        font=(FONT, 15, "bold"),
    ).pack(anchor="w", padx=16, pady=(12, 6))

    # Card de veredito (diz se foi bom ou ruim)
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

    # Texto das métricas numéricas
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

    # Explicação rápida na própria janela para consulta
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
            "• Ground Truth (Gabarito): crateras reais marcadas por você.\n"
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

    # Topo: título + botão voltar
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

    # Esquerda: imagem + barra de ajuste rápido + botões
    left = tk.Frame(body, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

    canvas = tk.Canvas(left, bg="#000000", highlightthickness=0)
    canvas.pack(fill="both", expand=True, padx=12, pady=(12, 6))
    canvas.bind("<Configure>", lambda e: renderizar())
    ui["canvas_imagem"] = canvas

    # Barra de ajuste fino (ajuda a adaptar para qualquer foto da Lua na hora)
    tune_bar = tk.Frame(left, bg="#121633", highlightbackground=LINE, highlightthickness=1)
    tune_bar.pack(fill="x", padx=12, pady=(0, 6))

    tk.Label(tune_bar, text="Exigência (PARAM2):", bg="#121633", fg=MUTED, font=(FONT, 9, "bold")).pack(side="left", padx=(8, 4), pady=6)
    ui["var_param2"] = tk.IntVar(value=HOUGH_PARAMS["param2"])
    tk.Spinbox(tune_bar, from_=15, to=70, textvariable=ui["var_param2"], width=4, bg=PANEL, fg=TEXT, insertbackground=TEXT, relief="flat", font=(FONT, 10)).pack(side="left", padx=(0, 10))

    tk.Label(tune_bar, text="Raio mín/máx:", bg="#121633", fg=MUTED, font=(FONT, 9, "bold")).pack(side="left", padx=(0, 4))
    ui["var_min_r"] = tk.IntVar(value=HOUGH_PARAMS["min_radius"])
    tk.Spinbox(tune_bar, from_=6, to=80, textvariable=ui["var_min_r"], width=3, bg=PANEL, fg=TEXT, insertbackground=TEXT, relief="flat", font=(FONT, 10)).pack(side="left", padx=(0, 4))
    ui["var_max_r"] = tk.IntVar(value=HOUGH_PARAMS["max_radius"])
    tk.Spinbox(tune_bar, from_=15, to=120, textvariable=ui["var_max_r"], width=4, bg=PANEL, fg=TEXT, insertbackground=TEXT, relief="flat", font=(FONT, 10)).pack(side="left", padx=(0, 10))

    tk.Label(tune_bar, text="Dist. mín:", bg="#121633", fg=MUTED, font=(FONT, 9, "bold")).pack(side="left", padx=(0, 4))
    ui["var_min_dist"] = tk.IntVar(value=HOUGH_PARAMS["min_dist"])
    tk.Spinbox(tune_bar, from_=10, to=90, textvariable=ui["var_min_dist"], width=4, bg=PANEL, fg=TEXT, insertbackground=TEXT, relief="flat", font=(FONT, 10)).pack(side="left", padx=(0, 8))

    botao_recalcular = criar_botao(
        tune_bar, "Recalcular", executar_deteccao, largura=105, altura=30, cor=CYAN, bg="#121633", fonte=9
    )
    botao_recalcular["canvas"].pack(side="right", padx=8, pady=4)
    ativar_botao(botao_recalcular, False)
    ui["botao_recalcular"] = botao_recalcular

    # Botões de ação
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

    # Direita: relatório
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