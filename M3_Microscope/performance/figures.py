"""Figures de performance (brouillons pour la brochure). Usage : python3 figures.py [noms...]"""
import sys, glob, pickle, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import stats_sim as S
import modeles as MOD

SURF, TXT, TXT2, GRIS, GRILLE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8986", "#e6e5e1"
C20, C40 = "#2a78d6", "#eb6834"
COUL = {"20X": C20, "40X": C40}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": "#b9b8b2",
                     "axes.labelcolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT2, "text.color": TXT,
                     "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "savefig.facecolor": SURF, "figure.facecolor": SURF, "axes.facecolor": SURF})
OUT = "fig"


def axe(ax, ylab=None, xlog=True):
    ax.grid(axis="y", color=GRILLE, lw=0.8); ax.set_axisbelow(True)
    for c in ("top", "right"):
        ax.spines[c].set_visible(False)
    if xlog:
        ax.set_xscale("log")
        ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5]); ax.set_xticklabels(["0,05", "0,1", "0,2", "0,5", "1", "2", "5"])
        ax.set_xlim(0.035, 6.5)
    if ylab:
        ax.set_ylabel(ylab)


def virg(x, n=1):
    return f"{x:.{n}f}".replace(".", ",")


# --------------------------------------------------------------------------------------------- Figure 1
def fig_principale(L=4):
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.9))
    tabs = {c: [m for m in S.tableau("base", L) if m["conf"] == c] for c in ("20X", "40X")}
    # (a) exactitude
    ax = axs[0]; axe(ax, "Biais du rayon moyen (%)")
    ax.axhspan(-10, 10, color="#2a78d6", alpha=0.06, lw=0)
    for a_ in axs:
        a_.axvspan(0.15, 0.7, color=TXT2, alpha=0.07, lw=0)
    ax.text(0.325, -50, "plage fiable\n0,15 – 0,7 µm", ha="center", fontsize=8, color=TXT2)
    ax.axhline(0, color=TXT2, lw=0.8)
    for c, t in tabs.items():
        t = [m for m in t if np.isfinite(m["biais"])]
        x = np.array([m["a"] for m in t]); y = 100 * np.array([m["biais"] for m in t]); e = 100 * np.array([m["s_biais"] for m in t])
        ax.fill_between(x, y - e, y + e, color=COUL[c], alpha=0.15, lw=0)
        ax.plot(x, y, "-o", color=COUL[c], ms=4, lw=1.6, label=f"simulation {c}")
    aa = np.logspace(np.log10(0.04), np.log10(6), 60)
    paroi = [100 * (MOD.biais_paroi(a * 1e-6)[0] - 1) for a in aa]
    ax.plot(aa, paroi, "--", color=GRIS, lw=1.3, label="paroi + sédimentation (modèle)")
    exp = [(0.475, 5.1, 4.5, "20X"), (1.0, 29, 13, "20X"), (1.0, 3, 21, "40X")]
    for a, b, s, c in exp:
        ax.errorbar(a * (1.04 if c == "40X" else 1), b, s, fmt="D", color=COUL[c], mec=TXT, mew=0.8, ms=6, capsize=3, zorder=5)
    ax.annotate("test réel\n0,475 µm", (0.475, 5.1), xytext=(0.2, 32), fontsize=8, color=TXT2,
                arrowprops=dict(arrowstyle="-", color=GRIS, lw=0.7))
    ax.annotate("50 nm : aucune\nbille suivie", (0.05, 58), xytext=(0.04, 50), fontsize=8, color=TXT2)
    ax.set_ylim(-65, 62); ax.set_xlabel("Rayon a (µm)")
    ax.set_title("a) Exactitude")
    ax.text(5.8, 8, "±10 %", ha="right", va="top", fontsize=8, color=C20)
    # (b) précision
    ax = axs[1]; axe(ax, "Écart type relatif, 1 vidéo (%)")
    for c, t in tabs.items():
        t = [m for m in t if np.isfinite(m["cv"])]
        x = np.array([m["a"] for m in t]); y = 100 * np.array([m["cv"] for m in t]); e = 100 * np.array([m["s_cv"] for m in t])
        ax.errorbar(x, y, e, fmt="-o", color=COUL[c], ms=4, lw=1.6, capsize=2)
        ax.plot(x, 100 * np.array([m["inc_rel"] for m in t]), ":", color=COUL[c], lw=1.2)
    ax.errorbar(0.475, 14.9, 4.7, fmt="D", color=C20, mec=TXT, mew=0.8, ms=6, capsize=3, zorder=5)
    ax.annotate("6 vidéos du\ntest réel", (0.475, 14.9), xytext=(0.6, 19.5), fontsize=8, color=TXT2,
                arrowprops=dict(arrowstyle="-", color=GRIS, lw=0.7))
    ax.set_ylim(0, 32); ax.set_xlabel("Rayon a (µm)")
    ax.set_title("b) Précision (≈ 7 billes, 140 images)")
    ax.legend(handles=[Line2D([], [], color=TXT2, lw=1.6, marker="o", ms=4, label="dispersion de 10 vidéos simulées"),
                       Line2D([], [], color=TXT2, lw=1.2, ls=":", label="incertitude affichée par le code")],
              frameon=False, fontsize=7.5, loc="upper right")
    # (c) suivi
    ax = axs[2]; axe(ax, "Pistes retenues par bille présente")
    ax.axhline(1, color=TXT2, lw=0.8)
    for c, t in tabs.items():
        x = np.array([m["a"] for m in t]); y = np.array([m["frac"] for m in t])
        ax.plot(x, y, "-o", color=COUL[c], ms=4, lw=1.6)
    ax.text(0.042, 1.05, "1 piste par bille", fontsize=8, color=TXT2)
    ax.annotate("pistes fragmentées :\ndéplacement par image\nproche de la limite\nde liaison (1,47 µm)", (0.1, 2.3),
                xytext=(0.17, 2.0), fontsize=8, color=TXT2, arrowprops=dict(arrowstyle="-", color=GRIS, lw=0.7))
    ax.set_ylim(0, 2.6); ax.set_xlabel("Rayon a (µm)")
    ax.set_title("c) Suivi des billes")
    handles = [Line2D([], [], color=C20, lw=1.6, marker="o", ms=4, label="simulation 20X"),
               Line2D([], [], color=C40, lw=1.6, marker="o", ms=4, label="simulation 40X"),
               Line2D([], [], color=TXT2, marker="D", ls="none", ms=6, mfc="white", label="mesure (billes connues)"),
               Line2D([], [], color=GRIS, ls="--", lw=1.3, label="modèle paroi + sédimentation")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{OUT}/fig1_performance_selon_rayon.png", dpi=200)
    fig.savefig(f"{OUT}/fig1_performance_selon_rayon.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------------------------- Figure 2 : L
def fig_L():
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
    tailles = [0.1, 0.2, 0.475, 1.5]
    teinte = {0.1: 0.35, 0.2: 0.55, 0.475: 0.8, 1.5: 1.0}
    for L in (1, 2, 3, 4, 5):
        pass
    for a in tailles:
        b, cv, Ls = [], [], []
        for L in (1, 2, 3, 4, 5):
            m = [x for x in S.tableau("base", L) if x["conf"] == "20X" and abs(x["a"] - a) < 1e-9]
            if m and np.isfinite(m[0]["biais"]):
                b.append(100 * m[0]["biais"]); cv.append(100 * m[0]["cv"]); Ls.append(L)
        axs[0].plot(Ls, b, "-o", color=C20, alpha=teinte[a], ms=4, lw=1.6, label=f"a = {virg(a, 3 if a in (0.1, 0.475) else 2 if a < 1 else 1)} µm")
        axs[1].plot(Ls, cv, "-o", color=C20, alpha=teinte[a], ms=4, lw=1.6)
    for ax, yl, t in ((axs[0], "Biais (%)", "a) Biais selon L (20X, simulation)"), (axs[1], "Écart type relatif, 1 vidéo (%)", "b) Précision selon L")):
        axe(ax, yl, xlog=False); ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlabel("Nombre de décalages L ajustés dans le MSD"); ax.set_title(t)
    axs[0].axhline(0, color=TXT2, lw=0.8)
    axs[0].annotate("L = 1 : droite forcée par l'origine,\nle flou de mouvement n'est pas retiré", (1, 47), xytext=(1.6, 35),
                    fontsize=8, color=TXT2, arrowprops=dict(arrowstyle="-", color=GRIS, lw=0.7))
    axs[0].legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_choix_L.png", dpi=200); plt.close(fig)


# --------------------------------------------------------------------------------------------- Figure 3 : dt / exposition
def fig_temps():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.7))
    # a) simulation
    ax = axs[0]
    cas = [("dt025", 0.025, 0.020), ("dt046", 0.046, 0.040), ("base", 0.092, 0.090), ("dt184", 0.184, 0.090), ("dt368", 0.368, 0.090)]
    teinte = {0.15: 0.4, 0.475: 0.7, 1.5: 1.0}
    for a in (0.15, 0.475, 1.5):
        xs, ys, es = [], [], []
        for etiq, dt, te in cas:
            f = glob.glob(f"sim/20X_{a:g}_{etiq}.pkl")
            if not f:
                continue
            m = S.metriques(f[0], 2)
            if np.isfinite(m["biais"]):
                xs.append(dt); ys.append(100 * m["biais"]); es.append(100 * m["cv"])
        if xs:
            ax.errorbar(xs, ys, es, fmt="-o", color=C20, alpha=teinte[a], ms=4, lw=1.6, capsize=2, label=f"a = {virg(a, 3 if a == 0.475 else 2)} µm")
        f = glob.glob(f"sim/20X_{a:g}_dt092e045.pkl")
        if f:
            m = S.metriques(f[0], 2)
            ax.errorbar(0.092 * 1.06, 100 * m["biais"], 100 * m["cv"], fmt="s", color=C20, alpha=teinte[a], mfc="white", ms=5, capsize=2)
    axe(ax, "Biais (%) ± écart type d'une vidéo", xlog=False); ax.set_xscale("log")
    ax.set_xticks([0.025, 0.05, 0.1, 0.2, 0.4]); ax.set_xticklabels(["0,025", "0,05", "0,1", "0,2", "0,4"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(-20, 60)
    ax.annotate("0,15 µm à 368 ms : +205 %\n(hors graphique)", (0.368, 58), xytext=(0.027, 50), fontsize=7.5, color=TXT2,
                arrowprops=dict(arrowstyle="-", color=GRIS, lw=0.7))
    ax.axhline(0, color=TXT2, lw=0.8); ax.set_xlabel("Intervalle entre images Δt (s), durée totale 12,9 s")
    ax.set_title("a) Fréquence d'acquisition (simulation 20X, L = 2)")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    ax.text(0.026, -17, "carré blanc : exposition réduite à 45 ms (Δt = 92 ms)", fontsize=7.5, color=TXT2)
    # b) décimation réelle
    ax = axs[1]
    d = pickle.load(open("decim2.pkl", "rb"))
    noms = [f"inc2_p{i}_20X_C" for i in range(1, 8)]
    for mode, sty, lab in (("brut", "o", "sans correction"), ("derive", "D", "dérive d'ensemble retirée")):
        ks = [1, 2, 3, 4, 5, 6]
        mo, se = [], []
        for k in ks:
            v = [d[(n, mode, k, 2)]["a"] * 1e6 for n in noms if (n, mode, k, 2) in d]
            v = np.array(v); mo.append(np.median(v)); se.append(1.2533 * np.std(v) / np.sqrt(len(v)))
        x = np.array(ks) * 0.092 * (1.03 if mode == "derive" else 1)
        ax.errorbar(x, mo, se, fmt="-" + sty, color=C20 if mode == "derive" else GRIS, ms=5, lw=1.6, capsize=2, label=lab)
    ax.axhspan(0.465, 0.485, color=TXT2, alpha=0.12, lw=0)
    ax.text(0.2, 0.31, "bande grise : référence 0,475 ± 0,010 µm", fontsize=8, color=TXT2, ha="left", va="bottom")
    axe(ax, "Rayon médian des 7 vidéos (µm)", xlog=False)
    ax.set_xlabel("Δt (s) : une image gardée sur k (k = 1 à 6)")
    ax.set_title("b) Vidéos du test réel décimées (code de base, L = 2)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_frequence_acquisition.png", dpi=200); plt.close(fig)


# --------------------------------------------------------------------------------------------- Figure 4 : bits et pixels
def fig_degrade():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    for nom, a0, lab, alpha in (("inc2_p1_20X_C", 0.475, "vidéo 2, au foyer (σ tache 0,66 µm)", 1.0), ("inc2_p5_20X_C", 0.475, "vidéo 5, défocalisée (σ tache 1,8 µm)", 0.45)):
        try:
            d = pickle.load(open(f"deg_{nom}.pkl", "rb"))
        except FileNotFoundError:
            continue
        ref = d[("bits", 14)]["r"][2]["a"]
        bits = [b for b in (14, 12, 11, 10, 9, 8, 7, 6) if ("bits", b) in d and "r" in d[("bits", b)]]
        y = [100 * (d[("bits", b)]["r"][2]["a"] / ref - 1) for b in bits]
        e = [100 * d[("bits", b)]["r"][2]["inc"] / ref for b in bits]
        axs[0].errorbar(bits, y, e, fmt="-o", color=C20, alpha=alpha, ms=4, lw=1.6, capsize=2, label=lab)
        ks = [1] + [k for k in (2, 3, 4) if ("bin", k) in d and "r" in d[("bin", k)]]
        pe = [d[("bits", 14)]["pe"] * 1e6] + [d[("bin", k)]["pe"] * 1e6 for k in ks[1:]]
        y = [0] + [100 * (d[("bin", k)]["r"][2]["a"] / ref - 1) for k in ks[1:]]
        e = [100 * d[("bits", 14)]["r"][2]["inc"] / ref] + [100 * d[("bin", k)]["r"][2]["inc"] / ref for k in ks[1:]]
        axs[1].errorbar(pe, y, e, fmt="-o", color=C20, alpha=alpha, ms=4, lw=1.6, capsize=2, label=lab)
    axe(axs[0], "Écart au rayon à pleine profondeur (%)", xlog=False); axs[0].invert_xaxis()
    axs[0].set_xlabel("Profondeur de bits conservée"); axs[0].set_title("a) Profondeur de bits (vidéos réelles, L = 2)")
    axe(axs[1], "Écart au rayon d'origine (%)", xlog=False)
    axs[1].set_xlabel("Taille de pixel dans le plan objet (µm) ; 20X équivaut à 20/k X")
    axs[1].set_title("b) Taille de pixel (pixels regroupés k × k)")
    for ax in axs:
        ax.axhline(0, color=TXT2, lw=0.8); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_bits_pixels.png", dpi=200); plt.close(fig)


# --------------------------------------------------------------------------------------------- Figure 5 : caméra
def fig_camera():
    import os
    U = "/mnt/user-data/uploads/GIT/TechEx/M3_Microscope/techex_M3_videos/"
    fichiers = sorted(glob.glob(U + "inconnu*_C_*.json")) + sorted(glob.glob(U + "100nano1*.json"))
    fichiers = [f for f in fichiers if "part" not in f]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for f in fichiers:
        dd = json.load(open(f)); t = (np.array(dd["timestamps_ns"]) - dd["timestamps_ns"][0]) / 1e9
        dt = np.median(np.diff(t)); slot = np.round(t / dt).astype(int)
        perdues = slot - np.arange(len(slot))
        ax.step(t, perdues, where="post", color=C20 if "_20X" in f else C40, lw=1.2, alpha=0.8)
    axe(ax, "Images perdues (cumul)", xlog=False)
    ax.set_xlabel("Temps depuis le début de l'enregistrement (s)")
    ax.set_title("Pertes d'images en option C (1014 × 760, 10,87 images/s)")
    ax.legend(handles=[Line2D([], [], color=C20, lw=1.2, label="20X (7 enregistrements)"),
                       Line2D([], [], color=C40, lw=1.2, label="40X (2 enregistrements : inconnu1, 100 nm)")], frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_pertes_images.png", dpi=200); plt.close(fig)


# --------------------------------------------------------------------------------------------- Figure 6 : précision selon N
def fig_N():
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    # CV par bille (simulation 20X, 0,475 µm) selon le nombre d'images
    pts = []
    for etiq, N in (("N50", 50), ("base", 140), ("N280", 280)):
        f = glob.glob(f"sim/20X_0.475_{etiq}.pkl")
        if f:
            m = S.metriques(f[0], 2); pts.append((N, m["cv_bille"]))
    nb = np.arange(1, 121)
    teinte = {50: 0.4, 140: 0.7, 280: 1.0}
    for N, cvb in pts:
        ax.plot(nb, 100 * cvb / np.sqrt(nb), color=C20, alpha=teinte[N], lw=1.6, label=f"{N} images ({virg(N * 0.092, 1)} s), σ bille = {virg(100 * cvb, 0)} %")
    s_reel = 0.029 / 0.475 * np.sqrt(74)
    ax.plot(nb, 100 * s_reel / np.sqrt(nb), color=GRIS, lw=1.6, label=f"réel : σ effectif par bille = {virg(100 * s_reel, 0)} % (dispersion entre vidéos)")
    ax.axhline(5, color=GRIS, ls="--", lw=1); ax.text(118, 5.3, "5 %", ha="right", fontsize=8, color=TXT2)
    ax.plot(74, 100 * 0.029 / 0.475, "D", color=C20, mec=TXT, ms=6)
    ax.annotate("test réel : 74 billes,\ndispersion entre vidéos", (74, 6.1), xytext=(80, 12), fontsize=8, color=TXT2,
                arrowprops=dict(arrowstyle="-", color=GRIS, lw=0.7))
    axe(ax, "Incertitude relative sur le rayon moyen (%)", xlog=False)
    ax.set_xlabel("Nombre de billes moyennées"); ax.set_ylim(0, 30)
    ax.set_title("Précision attendue selon le nombre de billes (20X, a = 0,475 µm)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig6_nombre_billes.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    quoi = sys.argv[1:] or ["principale", "L", "temps", "degrade", "camera", "N"]
    for q in quoi:
        {"principale": fig_principale, "L": fig_L, "temps": fig_temps, "degrade": fig_degrade,
         "camera": fig_camera, "N": fig_N}[q]()
        print("ok", q)
