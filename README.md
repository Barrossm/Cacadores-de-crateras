# 🌕 Caçadores de Crateras

Detecção e caracterização automática de crateras lunares em imagens de satélite da NASA, com interface gráfica em Python.

Projeto da 2ª VA da disciplina **Processamento de Imagens** (PI 2026.2), UFRPE, prof. Lucas Fernando da Silva Cambuim.

**Integrantes:** Gabriel Barros de Morais, Elton da Costa Oliveira e Júlia Karolyne Falcão Diniz Oliveira.

> **Status:** 🚧 em desenvolvimento. A seção [Roadmap](#roadmap) mostra o que já está pronto.

---

## O problema

Crateras de impacto são uma das feições mais importantes da superfície lunar: contá-las e medi-las permite estimar a idade relativa de uma região, e elas servem de pontos de referência para a navegação de sondas sem GPS. Marcar crateras na mão é lento e sujeito a erro.

Este projeto recebe um recorte de imagem da Lua em escala de cinza, destaca as crateras sobre a própria imagem e calcula, para cada uma: **centro (a, b), raio, área, perímetro, circularidade e máscara binária**.

O desafio: sombras muito escuras dentro das crateras, bordas estouradas pelo Sol, solo cheio de textura e ruído, e crateras de tamanhos diferentes, às vezes sobrepostas ou erodidas.

## Como funciona

Pipeline clássico de processamento de imagens. O código é dividido em três módulos, que seguem a organização da proposta:

| Módulo          | Etapa                 | Técnica                                      | Arquivo                      |
| --------------- | --------------------- | -------------------------------------------- | ---------------------------- |
| `preprocessing` | Melhoria de contraste | CLAHE                                        | `preprocessing/contrast.py`  |
| `preprocessing` | Suavização            | Filtro Gaussiano                             | `preprocessing/smoothing.py` |
| `preprocessing` | Detecção de bordas    | Sobel e Laplaciano                           | `preprocessing/edges.py`     |
| `segmentation`  | Morfologia            | Abertura e fechamento com disco              | `segmentation/morphology.py` |
| `segmentation`  | Segmentação           | Otsu + Transformada de Hough para círculos   | `segmentation/otsu_hough.py` |
| `metrics`       | Descrição             | Centro, raio, área, perímetro, circularidade | `metrics/features.py`        |
| `metrics`       | Avaliação             | IoU, precisão, recall e tempo de execução    | `metrics/evaluation.py`      |
| (raiz)          | Interface             | Tkinter + Pillow                             | `gui.py`                     |

O pipeline é independente da interface, então também pode ser rodado direto em um script ou notebook.

## Estrutura do repositório

```
Cacadores-de-crateras/
├── gui.py                  # interface gráfica (ponto de entrada)
├── preprocessing/          # contraste, suavização e bordas
│   ├── __init__.py
│   ├── contrast.py
│   ├── smoothing.py
│   └── edges.py
├── segmentation/           # morfologia, Otsu e Hough
│   ├── __init__.py
│   ├── morphology.py
│   └── otsu_hough.py
├── metrics/                # descrição das crateras e avaliação
│   ├── __init__.py
│   ├── features.py
│   └── evaluation.py
├── Imagens_entrada/        # recortes LROC de teste
├── Resultados/             # imagens e tabelas geradas (não versionadas)
├── Documentacao_projeto.pdf
└── README.md
```

## Instalação

Requer **Python 3.10+**. Tkinter já vem com o Python (no Linux: `sudo apt install python3-tk`).

```bash
git clone https://github.com/Barrossm/Cacadores-de-crateras.git
cd Cacadores-de-crateras
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Como usar

```bash
python gui.py
```

1. Clique em **Anexar imagem** e escolha um recorte em escala de cinza.
2. O pipeline roda e as crateras aparecem desenhadas sobre a imagem original.
3. As medidas de cada cratera aparecem ao lado.

## Dados

Imagens públicas da câmera **LROC**, da sonda _Lunar Reconnaissance Orbiter_ (NASA), obtidas no [Planetary Data System](https://pds-imaging.jpl.nasa.gov/). Usamos recortes de 256 a 512 px de lado, em escala de cinza de 8 bits.

Para testar, coloque imagens em `Imagens_entrada/`. De onde vieram e como foi feito o ground truth (marcação manual): _a preencher_.

## Resultados

| Cenário                      | IoU | Precisão | Recall | Tempo/recorte |
| ---------------------------- | --- | -------- | ------ | ------------- |
| Crateras isoladas            |     |          |        |               |
| Muitas crateras              |     |          |        |               |
| Terreno rugoso / luz rasante |     |          |        |               |

## Roadmap

- [x] Estrutura do repositório
- [ ] `preprocessing/contrast.py`: CLAHE
- [ ] `preprocessing/smoothing.py`: filtro Gaussiano
- [ ] `preprocessing/edges.py`: Sobel e Laplaciano
- [ ] `segmentation/morphology.py`: abertura e fechamento
- [ ] `segmentation/otsu_hough.py`: Otsu + Hough
- [ ] `metrics/features.py`: descrição geométrica das crateras
- [ ] `metrics/evaluation.py`: IoU, precisão, recall e tempo
- [ ] `gui.py`: interface gráfica
- [ ] Ground truth e experimentos
- [ ] Relatório e vídeo de apresentação

## Referências

1. Gonzalez, R. C.; Woods, R. E. _Digital Image Processing_. Pearson.
2. Duda, R. O.; Hart, P. E. Use of the Hough transformation to detect lines and curves in pictures. _Communications of the ACM_, 1972.
3. Robbins, S. J. A new global database of lunar impact craters >1–2 km. _Journal of Geophysical Research: Planets_, 2019.
4. Robinson, M. S. et al. Lunar Reconnaissance Orbiter Camera (LROC) instrument overview. _Space Science Reviews_, 2010.
5. [OpenCV](https://docs.opencv.org/) · [Tkinter](https://docs.python.org/3/library/tkinter.html)
```
