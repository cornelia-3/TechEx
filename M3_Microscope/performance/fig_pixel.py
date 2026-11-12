import pickle, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import traitement_video_tiff as M
from figures import SURF, TXT, TXT2, GRIS, C20, axe, virg
def met(a, px, dm):
    d = pickle.load(open(f"sim_px/{a:g}_{px:g}.pkl", "rb"))
    moy, nb = [], []
    for rr in d["out"][dm]:
        if not rr: nb.append(0); continue
        g = [q for q in rr if q["raison"] == "non" and np.isfinite(q["D"])]; nb.append(len(g))
        if g:
            Dm = np.mean([q["D"] for q in g])
            if Dm > 0: moy.append(M.kB*M.T/(6*np.pi*M.ETA*Dm)*1e6)
    moy = np.array(moy)
    if len(moy) < 2: return np.nan, np.nan, np.mean(nb)/7
    return 100*(moy.mean()/a-1), 100*moy.std(ddof=1)/a/np.sqrt(len(moy)), np.mean(nb)/7
A = [0.075, 0.1, 0.15]; PX = [0.187, 0.374, 0.748]; DM = [1.47, 2.5, 3.5]
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
alpha = {1.47: 0.35, 2.5: 0.7, 3.5: 1.0}
for dm in DM:
    b = [met(a, 0.374, dm) for a in A]
    axs[0].errorbar(A, [x[0] for x in b], [x[1] for x in b], fmt="-o", color=C20, alpha=alpha[dm], ms=5, lw=1.6, capsize=2,
                    label=f"limite {virg(dm, 2 if dm == 1.47 else 1)} µm" + (" (actuelle)" if dm == 1.47 else ""))
    axs[2].plot(A, [x[2] for x in b], "-o", color=C20, alpha=alpha[dm], ms=5, lw=1.6)
axs[0].axhspan(-5, 5, color=C20, alpha=0.07, lw=0); axs[0].axhline(0, color=TXT2, lw=0.8)
axs[0].set_title("a) Biais selon la limite de liaison (pixel 0,374 µm)", fontsize=10)
axs[0].legend(frameon=False, fontsize=8)
teinte = {0.075: 0.35, 0.1: 0.7, 0.15: 1.0}
for a in A:
    b = [met(a, px, 2.5) for px in PX]
    axs[1].errorbar(PX, [x[0] for x in b], [x[1] for x in b], fmt="-s", color=GRIS if False else C20, alpha=teinte[a], ms=5, lw=1.6, capsize=2,
                    label=f"a = {virg(a, 3)} µm")
axs[1].axhspan(-5, 5, color=C20, alpha=0.07, lw=0); axs[1].axhline(0, color=TXT2, lw=0.8)
axs[1].set_title("b) Biais selon la taille de pixel (limite 2,5 µm)", fontsize=10)
axs[1].legend(frameon=False, fontsize=8)
axs[2].axhline(1, color=TXT2, lw=0.8)
axs[2].set_title("c) Pistes retenues par bille simulée", fontsize=10)
for ax, xl in ((axs[0], "Rayon a (µm)"), (axs[1], "Pixel dans le plan objet (µm)"), (axs[2], "Rayon a (µm)")):
    axe(ax, xlog=False); ax.set_xlabel(xl)
axs[0].set_ylabel("Biais du rayon moyen (%)"); axs[2].set_ylabel("Pistes / bille")
axs[1].set_xticks(PX); axs[1].set_xticklabels(["0,187\n(option A)", "0,374\n(actuel)", "0,748\n(2 × 2)"])
for ax in (axs[0], axs[2]):
    ax.set_xticks(A); ax.set_xticklabels(["0,075", "0,1", "0,15"])
axs[0].text(0.095, 22, "50 nm : aucune bille détectée,\nquels que soient le pixel et la limite", fontsize=7.5, color=TXT2)
fig.suptitle("Borne basse : la limite de liaison décide, la taille de pixel presque pas (simulation 20X, 8 vidéos par point)",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig("fig/fig15_liaison_pixel.png", dpi=200)
