"""
gui.py - Interface gráfica do projeto (Tkinter + Pillow).

COMO O PROGRAMA FUNCIONA
    Tela 1 (inicial):  céu com estrelas, Lua e foguete. O botão DECOLAR
                       faz o foguete subir e leva para a Tela 2.
    Tela 2 (análise):  o usuário importa uma imagem da Lua, o pipeline
                       roda e as crateras aparecem desenhadas na imagem,
                       com o relatório (resumo + tabela) ao lado.

DEPENDÊNCIAS: apenas Pillow (Tkinter já vem com o Python).
"""
import math
import random
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

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
FONT = "Helvetica"

# [DEPOIS] Vira False quando run_pipeline() usar o pipeline real.
# Enquanto for True, o relatório avisa que os dados são de exemplo.
USING_DEMO_PIPELINE = True

# Colunas da tabela: (chave, título, largura)
COLUMNS = [("n", "#", 40), ("center", "Centro (x, y)", 110), ("radius", "Raio (px)", 80),
           ("area", "Área (px²)", 90), ("perimeter", "Perímetro (px)", 100),
           ("circ", "Circularidade", 100)]

# Crateras decorativas da Lua: (dx, dy, raio) como fração do raio da Lua.
MOON_DECOR = [(-0.35, -0.93, 0.05), (0.15, -0.88, 0.07), (0.5, -0.82, 0.045),
              (-0.7, -0.74, 0.06), (0.82, -0.62, 0.05), (-0.15, -0.8, 0.03)]

# ----------------------------------------------------------------------------
# Estado global
# ----------------------------------------------------------------------------
estado = {
    "annotated": None,    # imagem RGB com as crateras desenhadas
    "photo": None,        # referência obrigatória, senão o Tkinter apaga a imagem
    "launching": False,   # True enquanto o foguete está voando
}
ui = {}      # widgets: botões, canvas, tabela, resumo
telas = {}   # frames: "inicial" e "analise"
root = tk.Tk()


# ----------------------------------------------------------------------------
# Lógica (sem interface): pipeline, desenho e resumo
# ----------------------------------------------------------------------------
# CONTRATO: cada cratera é um dicionário assim. Todos os módulos do grupo
# (principalmente metrics/features.py) devem devolver neste formato:
#   {"center": (x, y), "radius": r, "area": A, "perimeter": P, "circularity": C}
# Unidades em pixels da imagem original.

def run_pipeline(gray):
    """Recebe uma imagem PIL em escala de cinza ("L") e devolve a lista de crateras.

    [DEPOIS] TODO: trocar o corpo por algo como (nomes das funções são sugestões):
        import numpy as np
        from preprocessing import contrast, smoothing, edges
        from segmentation import morphology, otsu_hough
        from metrics import features

        img = np.array(gray)                      # PIL -> NumPy uint8
        img = contrast.apply_clahe(img)
        img = smoothing.gaussian_blur(img)
        edge_map = edges.edge_map(img)
        mask, circles = otsu_hough.segment(edge_map)
        mask = morphology.clean(mask)
        return features.describe_craters(mask, circles)

    [DEPOIS] TODO: validar o tamanho da imagem (a proposta usa 256 a 512 px)
    e, se ficar lento, rodar em uma thread para a janela não travar.
    """
    # --- DADOS DE MENTIRA, só para testar a interface ---
    w, h = gray.size
    rng = random.Random(w * h)
    craters = []
    for _ in range(rng.randint(4, 8)):
        r = max(3, rng.randint(int(min(w, h) * 0.04), int(min(w, h) * 0.12)))
        x = rng.randint(r, max(r, w - r))
        y = rng.randint(r, max(r, h - r))
        area = math.pi * r * r
        perimeter = 2 * math.pi * r * rng.uniform(1.0, 1.25)
        circularity = min(1.0, 4 * math.pi * area / perimeter ** 2)
        craters.append({"center": (x, y), "radius": r, "area": area,
                        "perimeter": perimeter, "circularity": circularity})
    return craters


def draw_craters(gray, craters):
    rgb = gray.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    i = 1
    for c in craters:
        x, y = c["center"]
        r = c["radius"]
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 138, 61), width=2)
        draw.text((x - 3, y - 5), str(i), fill=(54, 224, 255))
        i = i + 1
    return rgb


def summarize(craters, elapsed_ms):
    lines = []
    if USING_DEMO_PIPELINE:
        lines.append("MODO DEMO: dados de exemplo, o pipeline real ainda não foi ligado.\n")

    if len(craters) == 0:
        lines.append("Nenhuma cratera encontrada.")
    else:
        n = len(craters)
        biggest = max(craters, key=lambda c: c["radius"])
        lines.append(f"Crateras encontradas: {n}")
        lines.append(f"Raio médio: {sum(c['radius'] for c in craters) / n:.1f} px")
        lines.append(f"Maior cratera: raio {biggest['radius']} px")
        lines.append(f"Circularidade média: {sum(c['circularity'] for c in craters) / n:.2f}")

    lines.append(f"Tempo de processamento: {elapsed_ms:.0f} ms")
    # [DEPOIS] TODO: se houver ground truth (marcação manual), mostrar aqui
    # IoU, precisão e recall vindos de metrics/evaluation.py.
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Componentes visuais
# ----------------------------------------------------------------------------
def round_rect(canvas, x1, y1, x2, y2, r, **kw):
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
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

    round_rect(c, 2, 2, botao["largura"] - 2, botao["altura"] - 2, 14,
               fill=fill, outline=edge, width=2)
    c.create_text(botao["largura"] / 2, botao["altura"] / 2, text=botao["texto"],
                  fill=ink, font=(FONT, botao["fonte"], "bold"))


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


def criar_botao(parent, texto, comando, largura=180, altura=42,
                cor=CYAN, bg=BG, fonte=12):
    """Botão neon desenhado em Canvas. Devolve um dicionário (o 'botao')."""
    canvas = tk.Canvas(parent, width=largura, height=altura, bg=bg,
                       highlightthickness=0, cursor="hand2")
    botao = {"canvas": canvas, "texto": texto, "comando": comando, "cor": cor,
             "largura": largura, "altura": altura, "fonte": fonte, "ativo": True}
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
        c.create_oval(x - r, y - r, x + r, y + r, outline="",
                      fill=rng.choice(["#ffffff", "#b8c4ff", "#ffd9b0"]))

    # Lua na parte de baixo
    radius = min(w, h * 1.3) * 0.6
    top = h - h * 0.2
    cx, cy = w / 2, top + radius
    c.create_oval(cx - radius - 6, cy - radius - 6, cx + radius + 6, cy + radius + 6,
                  outline="#2a3a7a", width=6)
    c.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                  fill="#cfd2de", outline="")
    for dx, dy, rr in MOON_DECOR:
        x, y, r = cx + dx * radius, cy + dy * radius, rr * radius
        c.create_oval(x - r, y - r, x + r, y + r, fill="#aeb2c4", outline="#8f94a8", width=2)
        c.create_oval(x - r * .7, y - r * .5, x + r * .7, y + r * .8, fill="#9ca0b5", outline="")

    # Título, subtítulo e botão
    size = max(28, min(54, w // 18))
    c.create_text(w / 2 + 2, h * 0.2 + 2, text=APP_NAME, fill=VIOLET, font=(FONT, size, "bold"))
    c.create_text(w / 2, h * 0.2, text=APP_NAME, fill=TEXT, font=(FONT, size, "bold"))
    c.create_text(w / 2, h * 0.2 + size * 1.1, text=TAGLINE, fill=MUTED, font=(FONT, 14))
    c.create_window(w / 2, h * 0.46, window=ui["botao_decolar"]["canvas"])
    c.create_text(w / 2, h - 14, text="UFRPE  ·  Processamento de Imagens 2026.2",
                  fill="#4a4f66", font=(FONT, 10))

    desenhar_foguete(w / 2, top - 4)


def desenhar_foguete(x, base):
    """Foguete feito com polígonos."""
    c = ui["canvas_inicial"]
    t = "rocket"
    c.create_polygon(x - 9, base, x, base + 26, x + 9, base, fill=ORANGE, outline="", tags=t)
    c.create_polygon(x - 16, base - 36, x - 32, base - 6, x - 16, base - 14,
                     fill=ORANGE, outline="", tags=t)
    c.create_polygon(x + 16, base - 36, x + 32, base - 6, x + 16, base - 14,
                     fill=ORANGE, outline="", tags=t)
    c.create_polygon(x, base - 96, x + 17, base - 54, x + 17, base - 12, x - 17, base - 12,
                     x - 17, base - 54, fill=TEXT, outline="#b8c4ff", width=2, smooth=False, tags=t)
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
# Tela 2: importar imagem, ver resultado e relatório
# ----------------------------------------------------------------------------
def importar_imagem():
    """[PRONTO] Escolhe o arquivo, roda o pipeline, mostra imagem e relatório."""
    path = filedialog.askopenfilename(
        title="Escolha um recorte da Lua",
        filetypes=[("Imagens", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")])
    if not path:
        return

    try:
        gray = Image.open(path).convert("L")
    except Exception as exc:
        messagebox.showerror("Imagem inválida", f"Não foi possível abrir o arquivo.\n\n{exc}")
        return

    start = time.perf_counter()
    craters = run_pipeline(gray)
    elapsed_ms = (time.perf_counter() - start) * 1000

    estado["annotated"] = draw_craters(gray, craters)
    renderizar()
    preencher_relatorio(craters, elapsed_ms)
    ativar_botao(ui["botao_salvar"], True)


def salvar_imagem():
    """Salva a imagem com as crateras desenhadas."""
    if estado["annotated"] is None:
        return
    path = filedialog.asksaveasfilename(defaultextension=".png",
                                        filetypes=[("PNG", "*.png")])
    if path:
        estado["annotated"].save(path)


def renderizar():
    c = ui["canvas_imagem"]
    c.delete("all")
    w, h = c.winfo_width(), c.winfo_height()

    if estado["annotated"] is None:
        rng = random.Random(3)
        for _ in range(60):
            x, y = rng.randint(0, w), rng.randint(0, h)
            c.create_oval(x, y, x + 2, y + 2, fill="#8890c0", outline="")
        c.create_text(w // 2, h // 2, text="Importe uma imagem da Lua\npara começar",
                      fill=MUTED, font=(FONT, 16), justify="center")
    else:
        img = estado["annotated"].copy()
        img.thumbnail((max(w - 4, 10), max(h - 4, 10)))
        estado["photo"] = ImageTk.PhotoImage(img)
        c.create_image(w // 2, h // 2, image=estado["photo"])


def preencher_relatorio(craters, elapsed_ms):
    ui["summary"].configure(text=summarize(craters, elapsed_ms), fg=TEXT)

    tree = ui["tree"]
    tree.delete(*tree.get_children())
    i = 1
    for c in craters:
        x, y = c["center"]
        tree.insert("", "end", values=(
            i, f"({x}, {y})", c["radius"], f"{c['area']:.0f}",
            f"{c['perimeter']:.1f}", f"{c['circularity']:.2f}"))
        i = i + 1


# ----------------------------------------------------------------------------
# Montagem das telas
# ----------------------------------------------------------------------------
def criar_tela_inicial():
    frame = tk.Frame(root, bg=BG)
    canvas = tk.Canvas(frame, bg=BG, highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    ui["canvas_inicial"] = canvas
    ui["botao_decolar"] = criar_botao(canvas, "DECOLAR", decolar, largura=240,
                                      altura=56, cor=ORANGE, bg=BG, fonte=16)

    canvas.bind("<Configure>", desenhar_inicial)
    canvas.bind("<Map>", desenhar_inicial)   # redesenha ao voltar para esta tela
    telas["inicial"] = frame


def criar_tela_analise():
    frame = tk.Frame(root, bg=BG)

    # Topo: título + botão voltar
    top = tk.Frame(frame, bg=BG)
    top.pack(fill="x", padx=24, pady=(18, 8))
    tk.Label(top, text="Análise de imagem lunar", bg=BG, fg=TEXT,
             font=(FONT, 20, "bold")).pack(side="left")
    botao_voltar = criar_botao(top, "Voltar ao início", voltar_ao_inicio, largura=170,
                               altura=38, cor=VIOLET, bg=BG)
    botao_voltar["canvas"].pack(side="right")

    body = tk.Frame(frame, bg=BG)
    body.pack(fill="both", expand=True, padx=24, pady=(0, 20))
    body.columnconfigure(0, weight=3)
    body.columnconfigure(1, weight=2)
    body.rowconfigure(0, weight=1)

    # Esquerda: imagem + botões
    left = tk.Frame(body, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

    canvas = tk.Canvas(left, bg="#000000", highlightthickness=0)
    canvas.pack(fill="both", expand=True, padx=12, pady=(12, 6))
    canvas.bind("<Configure>", lambda e: renderizar())
    ui["canvas_imagem"] = canvas

    bar = tk.Frame(left, bg=PANEL)
    bar.pack(fill="x", padx=12, pady=(6, 12))
    botao_importar = criar_botao(bar, "Importar imagem", importar_imagem, largura=180,
                                 altura=40, cor=CYAN, bg=PANEL)
    botao_importar["canvas"].pack(side="left")
    botao_salvar = criar_botao(bar, "Salvar resultado", salvar_imagem, largura=170,
                               altura=40, cor=ORANGE, bg=PANEL)
    botao_salvar["canvas"].pack(side="left", padx=10)
    ativar_botao(botao_salvar, False)
    ui["botao_salvar"] = botao_salvar
    # [DEPOIS] TODO: botão "Exportar CSV" com a tabela das crateras.
    # [DEPOIS] TODO: botão "Comparar com ground truth" (abre a máscara manual
    #          e mostra IoU, precisão e recall via metrics/evaluation.py).

    # Direita: relatório
    right = tk.Frame(body, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
    right.grid(row=0, column=1, sticky="nsew")
    tk.Label(right, text="Relatório", bg=PANEL, fg=CYAN,
             font=(FONT, 16, "bold")).pack(anchor="w", padx=16, pady=(14, 4))

    summary = tk.Label(right, text="Importe uma imagem para gerar o relatório.",
                       bg=PANEL, fg=MUTED, justify="left", anchor="w",
                       wraplength=380, font=(FONT, 12))
    summary.pack(fill="x", padx=16, pady=(0, 10))
    ui["summary"] = summary

    table = tk.Frame(right, bg=PANEL)
    table.pack(fill="both", expand=True, padx=16, pady=(0, 16))
    tree = ttk.Treeview(table, columns=[c[0] for c in COLUMNS],
                        show="headings", height=10)
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
    style.theme_use("clam")   # tema que aceita cores personalizadas
    style.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                    foreground=TEXT, rowheight=26, borderwidth=0)
    style.configure("Treeview.Heading", background=LINE, foreground=CYAN,
                    relief="flat", font=(FONT, 10, "bold"))
    style.map("Treeview", background=[("selected", VIOLET)])
    style.configure("Vertical.TScrollbar", background=LINE, troughcolor=PANEL,
                    arrowcolor=TEXT, bordercolor=PANEL)


def mostrar(nome):
    telas["inicial"].pack_forget()
    telas["analise"].pack_forget()
    telas[nome].pack(fill="both", expand=True)


def voltar_ao_inicio():
    estado["launching"] = False
    ativar_botao(ui["botao_decolar"], True)
    mostrar("inicial")


def main():
    root.title(APP_NAME)
    root.geometry("1100x680")
    root.minsize(900, 600)
    root.configure(bg=BG)

    configurar_estilo()
    criar_tela_inicial()
    criar_tela_analise()
    mostrar("inicial")
    root.mainloop()


if __name__ == "__main__":
    main()