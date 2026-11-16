# Binariza com Otsu e detecta crateras circulares com a Transformada de Hough.

import cv2
import numpy as np

from . import morphology

DEFAULT_PARAMS = {
    "min_dist": 20,
    "param1": 80,
    "param2": 36,       # Exigência (quanto maior, mais rigoroso)
    "min_radius": 6,
    "max_radius": 110,  # Cobre desde micro-crateras até crateras gigantes (> 80 px)
    "min_circularity": 0.35,
}

def find_circles(
    blur_u8: np.ndarray,
    blur_f: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    valid_inner: np.ndarray,
    roughness_std: float,
    scale: float,
    params: dict | None = None,
) -> list[tuple[int, int, int]]:
    """Encontra os círculos das crateras e devolve [(x, y, raio), ...] na escala de trabalho.

    blur_u8       : imagem suavizada em uint8
    blur_f        : a mesma imagem em float32
    gx, gy        : gradientes de Sobel da imagem suavizada
    valid_inner   : máscara booleana das regiões com dados válidos (longe de bordas sem dados)
    roughness_std : desvio padrão do brilho da imagem (decide se o terreno é rugoso)
    scale         : fator de redução aplicado à imagem original (1.0 = sem redução)
    params        : sobrescreve DEFAULT_PARAMS (param2, min_radius, max_radius, min_dist...)
    """
    cfg = dict(DEFAULT_PARAMS)
    if params:
        cfg.update(params)

    h, w = blur_u8.shape

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

    high_roughness = roughness_std > 42.0
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
    return cleaned_selected


def build_crater_mask(
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