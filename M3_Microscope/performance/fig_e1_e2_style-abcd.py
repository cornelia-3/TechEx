"""Graphiques e1) nombre d'images et e2) nombre de billes, billes de 0,95 µm, dans le style des graphiques a) à d)
(axes linéaires partant de 0, exactitude E_a en bleu à gauche, précision u_a en orange à droite).
Données : fig_performance/performance_resume.csv (graphiques 5a et 5b)."""
import csv, sys
import numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt

BLEU, ORANGE = "#2a78d6", "#d9622b"
mpl.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 11,
                     "axes.linewidth": 0.8, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "xtick.direction": "out", "ytick.direction": "out",
                     "savefig.dpi": 220, "axes.spines.top": False})
src, out = sys.argv[1], sys.argv[2]
D = {}
with open(src, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["graphique"] in ("5a", "5b") and r["jeu"].startswith("0,95"):
            D.setdefault(r["graphique"], []).append([float(r[k]) for k in ("valeur", "E_a_nm", "barre_E_a_nm", "u_a_nm", "barre_u_a_nm")])
D = {k: np.array(v) for k, v in D.items()}


def plafond(v):
    """Haut d'axe « rond » au-dessus de v."""
    pas = 10 ** np.floor(np.log10(v)) / 2
    return np.ceil(v / pas) * pas


def tracer(ax, g, xlabel, lettre, log=False):
    x, E, sE, u, su = D[g].T
    ax2 = ax.twinx()
    kw = dict(lw=1, capsize=2.5, elinewidth=0.9, zorder=3)
    for axe, y, sy, c, mk, ms in ((ax, E, sE, BLEU, "o", 5), (ax2, u, su, ORANGE, "s", 4.6)):
        h = axe.errorbar(x, y, yerr=sy, color=c, marker=mk, ms=ms, **kw)
        for b in h[1] + h[2]:
            b.set_alpha(0.55)                                   # barres plus pâles, comme a) à d)
    ax.set_ylim(0, plafond(np.max(E + sE) * 1.05)); ax2.set_ylim(0, plafond(np.max(u + su) * 1.05))
    if log:
        ax.set_xscale("log"); ax.set_xticks([3, 5, 10, 20, 40, 74]); ax.set_xticklabels(["3", "5", "10", "20", "40", "74"])
        ax.minorticks_off()
    ax.set_xlabel(xlabel, style="italic")
    ax.set_ylabel(r"Exactitude $E_a$ (nm)", color=BLEU, style="italic")
    ax.tick_params(axis="y", colors=BLEU); ax.spines["left"].set_color(BLEU)
    ax2.spines["top"].set_visible(False); ax2.spines["left"].set_visible(False)
    ax2.set_ylabel(r"Précision $u_a$ (nm)", color=ORANGE, style="italic")
    ax2.tick_params(axis="y", colors=ORANGE); ax2.spines["right"].set_color(ORANGE)
    if False: ax.text(-0.02, 1.035, lettre, transform=ax.transAxes, ha="right", va="bottom", fontsize=13)


PAN = [("5a", "Nombre d'images par vidéo", r"$\mathbf{e_1)}$", False, "graphique5a_nombre_images_style-abcd"),
       ("5b", "Nombre de billes", r"$\mathbf{e_2)}$", True, "graphique5b_nombre_billes_style-abcd")]
for g, xl, le, lg, nom in PAN:
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    tracer(ax, g, xl, le, lg); fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{out}/{nom}.{e}", bbox_inches="tight", facecolor="white")
fig, axs = plt.subplots(1, 2, figsize=(11.6, 3.7))
for ax, (g, xl, le, lg, _) in zip(axs, PAN):
    tracer(ax, g, xl, le, lg)
fig.tight_layout(w_pad=3)
for e in ("png", "pdf"):
    fig.savefig(f"{out}/graphiques_e1_e2_style-abcd.{e}", bbox_inches="tight", facecolor="white")
