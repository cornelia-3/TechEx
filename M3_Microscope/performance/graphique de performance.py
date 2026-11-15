# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Graphiques de performance — mandat 3 (suivi de particules)
#
# Ce notebook produit les **graphiques de performance 1 à 5** de l'annexe 2 (une figure par graphique) et la
# **figure d'annexe sur les pertes d'images**, uniquement à partir de **vidéos réelles**. **Aucune simulation.**
# Deux jeux de billes de référence sont analysés et tracés sur les mêmes graphiques :
#
# * **billes de 0,95 µm** (a = 0,475 ± 0,010 µm) : les 6 vidéos du test (vidéos 2 à 7), 20X, 74 billes gardées après
#   la revue manuelle du test;
# * **billes de 2 µm** (a = 1,000 ± 0,021 µm, même tolérance relative que les billes de 0,95 µm) : les 2 vidéos du
#   2 novembre au 40X, en mode 1014 × 760. Pas de revue manuelle : les billes gardées sont celles que le test
#   d'immobilité juge mobiles (3 par vidéo). Pas de fichier .json : Δt vient de la fréquence nominale (10,8696 img/s).
#
# Pour chaque paramètre, deux métriques (axe y gauche et axe y droit, comme au mandat 2). Le poids de chaque vidéo
# est son nombre de billes, $w_v = N_v / \sum N_v$.
#
# * **Exactitude** : $E_a = \sum_v w_v\,|a_v - a_\mathrm{vrai}|$ (le rayon $a_v$ d'une vidéo vient de la moyenne des
#   $D_i$ de ses billes, comme dans `rayon_final`). Toujours positive.
# * **Précision absolue** : $u_a = \sqrt{u_\mathrm{stat}^2 + u_\mathrm{loc}^2}$, avec
#   $u_\mathrm{stat} = \sqrt{\sum_v w_v^2 u_{\mathrm{stat},v}^2}$, $u_{\mathrm{stat},v} = \mathrm{std}(a_i)/\sqrt{N_v}$
#   (évaluée par propagation à partir des $D_i$), et $u_\mathrm{loc}$ l'incertitude de localisation propagée jusqu'au
#   rayon, comme dans le code du prototype.
# * **Barres d'erreur.** Exactitude : $\sqrt{u_a^2 + u_\mathrm{inst}^2 + u_\mathrm{réf}^2}$ ($u_\mathrm{inst}$ :
#   étalonnage du grossissement et horodatages, < 1 nm). Précision : demi-largeur 16–84 % d'un bootstrap des billes.
#
# **Pour relancer** : `Run All`. Les graphiques 3 et 4 relisent les TIFF et relocalisent chaque bille dans chaque image
# dégradée (≈ 15 min la première fois); le résultat est mis en cache dans `donnees/cache_relocalisation_v2/`.

# %%
import sys, json, glob, pickle, time, csv
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ICI = Path.cwd()                                   # dossier performance/
RACINE = ICI.parent                                # dossier M3_Microscope/
sys.path.insert(0, str(RACINE))
import traitement_video_tiff_extra as M            # seulement des fonctions de calcul : rien n'est modifié

DOSSIER_JSON = RACINE / "techex_M3_videos"
CACHE = ICI / "donnees" / "cache_relocalisation_v2"
FIG = ICI / "fig_performance"
FIG.mkdir(exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)

L_PROTO = 4                               # nombre de décalages du protocole du test
N_MIN_PTS = 10                            # positions minimales pour ajuster un MSD après dégradation
N_BOOT = 1000
GRAINE = 0

# Les deux jeux de billes : fichier de données, dossier des TIFF, rayon de référence, nombre minimal de vidéos
JEUX = {
    "0,95 µm": dict(fichier="test_6videos_final.pkl", dossier="techex_M3_videos/_avant_premier_trou",
                    a_vrai=0.475, u_ref=0.010, n_vid_min=2),
    "2 µm": dict(fichier="billes_2um_40X.pkl", dossier="techex_M3_videos",
                 a_vrai=1.000, u_ref=0.021, n_vid_min=1),
}
donnees = {}
for jeu, J in JEUX.items():
    dd = pickle.load(open(ICI / "donnees" / J["fichier"], "rb"))
    J["videos"] = list(dd)
    for v, d in dd.items():
        d.update(jeu=jeu, a_vrai=J["a_vrai"], u_ref=J["u_ref"], chemin=RACINE / J["dossier"] / d["fichier"])
        donnees[v] = d

# Incertitudes d'instrument relatives sur le rayon de chaque vidéo (a ∝ Δt / pixel², pixel ∝ 1/M)
for v, d in donnees.items():
    js = d["chemin"].with_suffix(".json")
    if js.exists():
        ecarts = np.diff(np.array(json.load(open(js))["timestamps_ns"], float))[: d["n_creneaux"] - 1] / 1e9
        d["rel_dt"] = float(ecarts.std(ddof=1) / np.sqrt(len(ecarts)) / ecarts.mean())
    else:
        d["rel_dt"] = 0.0                                          # pas d'horodatages : Δt nominal
    d["rel_M"] = 2 * d["inc_M"] / d["M"]
    d["rel_inst"] = float(np.hypot(d["rel_dt"], d["rel_M"]))
    print(f"[{d['jeu']}] {v} : {len(d['billes'])} billes, {len(d['refs'])} réf., {d['n_creneaux']} images, "
          f"pixel {d['pixel_eff']*1e6:.3f} µm, Δt = {d['dt']*1e3:.2f} ms, M = {d['M']:.3f} ± {d['inc_M']:.3f}, "
          f"a officiel = {d['a_officiel']*1e6:.3f} ± {d['inc_officiel']*1e6:.3f} µm")

# %% [markdown]
# ## Métriques communes

# %%
def rayon_um(D):
    """Stokes-Einstein, a = kT/(6πηD), en µm (T = 295,15 K, η = 0,95 mPa·s, comme le code du prototype)."""
    return M.kB * M.T / (6 * np.pi * M.ETA * np.asarray(D, float)) * 1e6


def D_piste(pos_px, inc_px, dt, pe, L=L_PROTO, n_min=N_MIN_PTS):
    """(D, δD_loc) en m²/s : régression du MSD sur L décalages (fonction du prototype) et propagation des
    incertitudes de position de chaque image jusqu'à D (incertitude_D_psf du prototype)."""
    if np.isfinite(pos_px[:, 0]).sum() < max(n_min, L + 1):
        return np.nan, np.nan
    D = M.ajuste_msd(pos_px * pe, dt, L)[0]
    sD = M.incertitude_D_psf(pos_px * pe, inc_px * pe, dt, L) if np.isfinite(D) else np.nan
    return D, sD


def stats_video(D, sD, v):
    """Rayon et incertitudes d'une vidéo à partir des D_i de ses billes (au moins 2, moyenne positive)."""
    d = donnees[v]
    ok = np.isfinite(D)
    D, sD = np.asarray(D)[ok], np.asarray(sD)[ok]
    N = len(D)
    if N < 2 or D.mean() <= 0:
        return None
    a = float(rayon_um(D.mean()))
    return dict(N=N, D_moy=D.mean(), a=a, ecart=abs(a - d["a_vrai"]),
                u_stat=a * D.std(ddof=1) / np.sqrt(N) / D.mean(),
                u_loc=a * np.sqrt(np.nansum(sD**2)) / N / D.mean(),
                u_inst=a * d["rel_inst"])


def agreger(stats, jeu):
    """Combine les vidéos d'un jeu (pondération par le nombre de billes). stats : liste de dicts (None ignorés)."""
    J = JEUX[jeu]
    s = [x for x in stats if x is not None]
    if len(s) < J["n_vid_min"]:
        return None
    w = np.array([x["N"] for x in s], float); w /= w.sum()
    E = float(np.sum(w * [x["ecart"] for x in s]))
    u_stat = float(np.sqrt(np.sum(w**2 * np.square([x["u_stat"] for x in s]))))
    u_loc = float(np.sqrt(np.sum(w**2 * np.square([x["u_loc"] for x in s]))))
    u_inst = float(np.sum(w * [x["u_inst"] for x in s]))           # systématique commun : somme linéaire
    u = float(np.hypot(u_stat, u_loc))
    return dict(E=E, sE=float(np.sqrt(u**2 + u_inst**2 + J["u_ref"]**2)), u=u, u_stat=u_stat, u_loc=u_loc,
                u_inst=u_inst, a_moy=float(np.sum(w * [x["a"] for x in s])), N=int(sum(x["N"] for x in s)),
                n_vid=len(s), a_v=[x["a"] for x in s])


VIDE = dict(E=np.nan, sE=np.nan, u=np.nan, su=np.nan, u_stat=np.nan, u_loc=np.nan, u_inst=np.nan, N=0, n_vid=0)


def metriques(D_v, sD_v, videos, n_boot=N_BOOT, graine=GRAINE):
    """D_v, sD_v : listes (une par vidéo d'un même jeu) des D_i et δD_i des billes. Métriques + barre de u_a."""
    jeu = donnees[videos[0]]["jeu"]
    stats = [stats_video(np.asarray(D, float), np.asarray(s, float), v) for D, s, v in zip(D_v, sD_v, videos)]
    r = agreger(stats, jeu)
    if r is None:
        return dict(VIDE, stats=stats)
    rng = np.random.default_rng(graine)
    paires = [(np.asarray(D, float), np.asarray(s, float), v) for D, s, v in zip(D_v, sD_v, videos)]
    paires = [(D[np.isfinite(D)], s[np.isfinite(D)], v) for D, s, v in paires]
    boots = []
    for _ in range(n_boot):
        st = [stats_video(D[i], s[i], v) for D, s, v in paires if len(D) >= 2
              for i in [rng.integers(0, len(D), len(D))]]
        rb = agreger(st, jeu)
        boots.append(rb["u"] if rb else np.nan)
    r["su"] = float(np.diff(np.nanpercentile(boots, [16, 84]))[0] / 2)
    r["stats"] = stats
    return r


def combiner(liste, videos):
    """Plusieurs analyses d'un même réglage (phases de décimation, fenêtres d'images) : exactitude à partir du D
    moyen de chaque vidéo moyenné sur les analyses (l'erreur systématique); précision = médiane des analyses
    (chacune représente UNE acquisition réelle à ce réglage)."""
    jeu = donnees[videos[0]]["jeu"]
    liste = [r for r in liste if np.isfinite(r["u"])]
    if not liste:
        return dict(VIDE, stats=[None] * len(videos))
    st = []
    for j, v in enumerate(videos):
        sv = [r["stats"][j] for r in liste if r["stats"][j] is not None]
        if not sv:
            st.append(None); continue
        Dm = np.mean([x["D_moy"] for x in sv])
        a = float(rayon_um(Dm))
        st.append(dict(N=int(round(np.mean([x["N"] for x in sv]))), D_moy=Dm, a=a, ecart=abs(a - donnees[v]["a_vrai"]),
                       u_stat=np.median([x["u_stat"] for x in sv]), u_loc=np.median([x["u_loc"] for x in sv]),
                       u_inst=a * donnees[v]["rel_inst"]))
    r = agreger(st, jeu)
    if r is None:
        return dict(VIDE, stats=st)
    med = lambda c: float(np.median([x[c] for x in liste]))
    r.update(u=med("u"), su=med("su"), u_stat=med("u_stat"), u_loc=med("u_loc"),
             N=float(np.mean([x["N"] for x in liste])), n_analyses=len(liste), stats=st)
    r["sE"] = float(np.sqrt(r["u"]**2 + r["u_inst"]**2 + JEUX[jeu]["u_ref"]**2))
    return r


def resume(r, txt=""):
    if not np.isfinite(r["E"]):
        return f"{txt} (pas assez de billes)"
    return (f"{txt} E_a = {r['E']*1e3:5.0f} ± {r['sE']*1e3:3.0f} nm   u_a = {r['u']*1e3:4.0f} ± {r['su']*1e3:3.0f} nm "
            f"(stat {r['u_stat']*1e3:.0f}, loc {r['u_loc']*1e3:.1f}, inst {r['u_inst']*1e3:.2f} nm)   billes {r['N']:.0f}")


# Vérification : les positions exportées redonnent exactement les rayons de départ (L = 4)
D0 = {v: [D_piste(b["pos_msd"], b["inc_px"], d["dt"], d["pixel_eff"]) for b in d["billes"]] for v, d in donnees.items()}
for jeu, J in JEUX.items():
    V = J["videos"]
    base = metriques([[x[0] for x in D0[v]] for v in V], [[x[1] for x in D0[v]] for v in V], V)
    for v, s in zip(V, base["stats"]):
        print(f"[{jeu}] {v} : a = {s['a']:.3f} µm (départ {donnees[v]['a_officiel']*1e6:.3f}), u_loc = {s['u_loc']*1e3:.1f} nm "
              f"(départ {donnees[v]['inc_officiel_psf']*1e9:.1f})")
    print(resume(base, f"[{jeu}] réglage de départ :"), f"  rayon moyen pondéré {base['a_moy']:.3f} µm\n")

# %% [markdown]
# ## Style des figures (celui du mandat 2 : deux axes y, bleu = exactitude, orange = précision)
# Les billes de 0,95 µm sont en marqueurs pleins et traits continus, celles de 2 µm en marqueurs vides et tirets.
# Les axes y sont logarithmiques : les valeurs des billes de 2 µm sont environ 20 fois plus grandes.

# %%
BLEU, ORANGE, GRIS = "#2a78d6", "#d9622b", "#8a8986"
STYLE = {"0,95 µm": dict(ls="-", plein=True), "2 µm": dict(ls="--", plein=False)}
mpl.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 11,
    "axes.linewidth": 0.8, "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.direction": "out", "ytick.direction": "out", "savefig.dpi": 220, "figure.dpi": 110,
    "axes.spines.top": False,
})


def figure(series, xlabel, nom, log=False, xticks=None, xticklabels=None, inverser=False):
    """Une figure par graphique. series : {jeu: (x, res)}. Exactitude E_a à gauche (bleu, cercles), précision u_a à
    droite (orange, carrés); 0,95 µm en plein/continu, 2 µm en vide/tirets."""
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    ax2 = ax.twinx()
    hautE, hautu, poignees = [], [], []
    for jeu, (x, res) in series.items():
        st = STYLE[jeu]
        x = np.asarray(x, float)
        E = np.array([r["E"] for r in res]) * 1e3; sE = np.array([r["sE"] for r in res]) * 1e3
        u = np.array([r["u"] for r in res]) * 1e3; su = np.array([r["su"] for r in res]) * 1e3
        for axe, y, sy, c, mk, ms in ((ax, E, sE, BLEU, "o", 5), (ax2, u, su, ORANGE, "s", 4.6)):
            h = axe.errorbar(x, y, yerr=sy, color=c, marker=mk, ms=ms, lw=1, ls=st["ls"], capsize=2.5, elinewidth=0.9,
                             mfc=c if st["plein"] else "white", mec=c, zorder=3)
            poignees.append((h, f"{'$E_a$' if axe is ax else '$u_a$'}, billes de {jeu}"))
        hautE.append(np.nanmax(E + np.minimum(sE, E + 20))); hautu.append(np.nanmax(u + np.minimum(su, u)))
    ax.set_xlabel(xlabel, style="italic")
    ax.set_ylabel(r"Exactitude $E_a$ (nm)", color=BLEU, style="italic")
    ax.tick_params(axis="y", colors=BLEU); ax.spines["left"].set_color(BLEU)
    ax2.spines["top"].set_visible(False); ax2.spines["left"].set_visible(False)
    ax2.set_ylabel(r"Précision $u_a$ (nm)", color=ORANGE, style="italic")
    ax2.tick_params(axis="y", colors=ORANGE); ax2.spines["right"].set_color(ORANGE)
    if log:
        ax.set_xscale("log")
    if xticks is not None:
        ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
        if log:
            ax.minorticks_off()
    # échelle logarithmique en y : les billes de 2 µm ont des valeurs environ 20 fois plus grandes
    tousE = np.concatenate([np.array([r["E"] for r in res]) for _, res in series.values()]) * 1e3
    tousu = np.concatenate([np.array([r["u"] for r in res]) for _, res in series.values()]) * 1e3
    for axe in (ax, ax2):
        axe.set_yscale("log", nonpositive="clip")
        axe.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axe.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_ylim(np.nanmin(tousE) / 2.5, np.nanmax(hautE) * 1.6)   # une barre trop grande dépasse l'axe
    ax2.set_ylim(np.nanmin(tousu) / 2.5, np.nanmax(hautu) * 1.6)
    if xticks is None and not log:
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    if inverser:
        ax.invert_xaxis()
    ordre = [p for p in poignees if "E_a" in p[1]] + [p for p in poignees if "u_a" in p[1]]
    fig.legend([p[0] for p in ordre], [p[1] for p in ordre], loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / f"{nom}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{nom}.pdf", bbox_inches="tight", facecolor="white")
    plt.show()


RESULTATS = {}           # {(graphique, jeu): dict(parametre, x, res, videos)}, exporté en CSV à la fin
PAR_BILLE = []           # chaque D_i utilisé, exporté en CSV à la fin


def noter_billes(graphique, parametre, x, analyse, videos, D_v, sD_v, extra=None):
    for v, D, sD in zip(videos, D_v, sD_v):
        d = donnees[v]
        for b, Di, si in zip(d["billes"], D, sD):
            ok = np.isfinite(Di)
            PAR_BILLE.append(dict(graphique=graphique, jeu=d["jeu"], parametre=parametre, valeur=x, analyse=analyse,
                                  video=v, fichier=d["fichier"], piste=b["piste"], D_um2_s=Di * 1e12 if ok else None,
                                  incertitude_D_loc_um2_s=si * 1e12 if ok and np.isfinite(si) else None,
                                  a_um=float(rayon_um(Di)) if ok and Di > 0 else None, retenue=bool(ok),
                                  **(extra or {})))


def paires(DD):
    return [[x[0] for x in l] for l in DD], [[x[1] for x in l] for l in DD]

# %% [markdown]
# ## Graphique 1 — Nombre de décalages L
#
# On refait la régression du MSD de chaque bille gardée avec L = 1 à 12 décalages, sur les mêmes positions
# (corrigées de la vibration). Avec L = 1, la droite est forcée par l'origine (D = MSD(Δt)/(4Δt)).

# %%
LS = list(range(1, 13))
for jeu, J in JEUX.items():
    V, res = J["videos"], []
    for L in LS:
        D_v, sD_v = paires([[D_piste(b["pos_msd"], b["inc_px"], donnees[v]["dt"], donnees[v]["pixel_eff"], L=L)
                             for b in donnees[v]["billes"]] for v in V])
        res.append(metriques(D_v, sD_v, V))
        noter_billes("1", "L", L, 0, V, D_v, sD_v)
        print(resume(res[-1], f"[{jeu}] L = {L:2d} :"))
    RESULTATS[("1", jeu)] = dict(parametre="L", x=LS, res=res, videos=V)
figure({j: (LS, RESULTATS[("1", j)]["res"]) for j in JEUX}, "Nombre de décalages $L$ dans l'ajustement du MSD",
       "graphique1_lags", xticks=LS)

# %% [markdown]
# ## Graphique 2 — Fréquence d'image (exposition fixée à 90 ms)
#
# On garde une image sur k (k = 1 à 8) dans chaque piste corrigée : Δt devient k × 92 ms et l'exposition reste
# 90 ms. Chaque phase de décimation est analysée comme une acquisition séparée, puis les analyses sont combinées
# (`combiner`). L reste à 4. Une bille qui garde moins de 10 positions est retirée. La vidéo 2 (0,95 µm) est exclue :
# ses pistes sont trop courtes (26 à 43 images pour 8 de ses 9 billes). Pour les billes de 2 µm, les pistes de la
# vidéo B (25 à 51 images) tombent sous 10 positions dès k = 3 : il ne reste alors que la vidéo A.

# %%
KS = list(range(1, 9))


def D_decime(b, k, phase, dt, pe, L=L_PROTO):
    j = np.arange(len(b["pos_msd"])) + b["f0"]          # numéro d'image (créneau) de chaque position
    m = (j % k) == phase
    return D_piste(b["pos_msd"][m], b["inc_px"][m], k * dt, pe, L=L)


for jeu, J in JEUX.items():
    V = [v for v in J["videos"] if v != "Vidéo 2"]
    res, fps = [], []
    for k in KS:
        analyses = []
        for ph in range(k):
            D_v, sD_v = paires([[D_decime(b, k, ph, donnees[v]["dt"], donnees[v]["pixel_eff"]) for b in donnees[v]["billes"]]
                                for v in V])
            analyses.append(metriques(D_v, sD_v, V, n_boot=300))
            noter_billes("2", "frequence_img_s", round(1 / (k * donnees[V[0]]["dt"]), 4), ph, V, D_v, sD_v,
                         extra=dict(k_decimation=k))
        res.append(combiner(analyses, V))
        fps.append(1 / (k * np.mean([donnees[v]["dt"] for v in V])))
        print(resume(res[-1], f"[{jeu}] k = {k} ({fps[-1]:5.2f} img/s) :"))
    RESULTATS[("2", jeu)] = dict(parametre="frequence_img_s", x=fps, res=res, videos=V)
figure({j: (RESULTATS[("2", j)]["x"], RESULTATS[("2", j)]["res"]) for j in JEUX},
       "Fréquence d'image (images/s) (exposition 90 ms)", "graphique2_frequence", log=True,
       xticks=[1.36, 2, 3, 5, 10.87], xticklabels=["1,36", "2", "3", "5", "10,87"])

# %% [markdown]
# ## Relocalisation dans les images dégradées (graphiques 3 et 4)
#
# Pour chaque réglage, on dégrade chaque image de la vidéo, puis on **relocalise chaque bille gardée et chaque bille
# de référence** par un ajustement gaussien 2D (`fit_gauss_local` du code du prototype) autour de sa position connue.
# L'ajustement donne aussi l'incertitude de chaque position, propagée jusqu'à D. On retire ensuite la vibration avec
# les mêmes billes de référence (`mouvement_commun`), et on calcule D avec L = 4.
#
# * **Bits** : le capteur (IMX477) numérise sur 12 bits; la somme 2 × 2 du code d'acquisition donne des images de
#   0 à 16 380 (14 bits). Quantification `image // q * q` avec q = 2^(14 − b), de 14 à 5 bits.
# * **Pixel** : somme de k × k pixels (k = 1 à 5). Au 20X (0,95 µm) : 0,374 à 1,87 µm; au 40X (2 µm) : 0,185 à
#   0,93 µm dans le plan objet. Les positions sont ramenées en pixels d'origine.
#
# La détection et la liaison ne sont pas refaites : les graphiques isolent l'effet du paramètre sur la localisation.
# Le point 14 bits / k = 1 de cette relocalisation sert de référence aux deux graphiques.

# %%
import tifffile


def degrader(img, mode, val):
    if mode == "bits":
        q = 2 ** (14 - val)
        return (img // q * q).astype(float)
    k = val                                                        # mode == "pixel" : regroupement k × k
    H, W = img.shape[0] // k * k, img.shape[1] // k * k
    return img[:H, :W].astype(float).reshape(H // k, k, W // k, k).sum(axis=(1, 3))


def relocaliser_video(v, mode, val, frames=None):
    """Positions et incertitudes relocalisées (pixels d'origine), pour toutes les pistes (billes, puis références).
    Retourne une liste de tableaux (n, 4) : x, y, σ_x, σ_y."""
    fichier = CACHE / f"{v.replace(' ', '_').replace('–', '-')}_{mode}_{val}.pkl"
    if fichier.exists():
        return pickle.load(open(fichier, "rb"))
    d = donnees[v]
    if frames is None:
        frames = tifffile.imread(d["chemin"])
    k = val if mode == "pixel" else 1
    pistes = d["billes"] + d["refs"]
    sortie = [np.full((len(p["pos_px"]), 4), np.nan) for p in pistes]
    image_de = {int(c): i for i, c in enumerate(d["indices"])}
    for c in range(d["n_creneaux"]):
        actives = [(n, p) for n, p in enumerate(pistes)
                   if p["f0"] <= c < p["f0"] + len(p["pos_px"]) and np.isfinite(p["pos_px"][c - p["f0"], 0])]
        if not actives or c not in image_de:
            continue
        img = degrader(frames[image_de[c]], mode, val)
        fond = float(np.median(img))
        for n, p in actives:
            x0, y0 = p["pos_px"][c - p["f0"]]
            r = M.fit_gauss_local(img, (x0 + 0.5) / k - 0.5, (y0 + 0.5) / k - 0.5, fond, d["sigma_px"] / k)
            if r is not None:
                sortie[n][c - p["f0"]] = [(r[0] + 0.5) * k - 0.5, (r[1] + 0.5) * k - 0.5, r[2] * k, r[3] * k]
    pickle.dump(sortie, open(fichier, "wb"))
    return sortie


def D_relocalises(v, positions, L=L_PROTO):
    """Retire la vibration (médiane des références relocalisées) puis calcule (D, δD_loc) de chaque bille gardée."""
    d = donnees[v]
    nb = len(d["billes"])
    pistes = [dict(f0=p["f0"], pos_px=pos[:, :2]) for p, pos in zip(d["billes"] + d["refs"], positions)]
    refs_ok = [i for i in range(nb, len(pistes)) if np.isfinite(pistes[i]["pos_px"][:, 0]).sum() >= N_MIN_PTS]
    corr = M.mouvement_commun(pistes, refs_ok, d["n_creneaux"]) if len(refs_ok) >= 2 else np.zeros((d["n_creneaux"], 2))
    D, sD = [], []
    for p, pos in zip(pistes[:nb], positions[:nb]):
        x = p["pos_px"] - corr[p["f0"]:p["f0"] + len(p["pos_px"])]
        Di, si = D_piste(x, pos[:, 2:], d["dt"], d["pixel_eff"], L=L)
        D.append(Di); sD.append(si)
    frac = np.mean([np.isfinite(pos[:, 0]).sum() / max(np.isfinite(q["pos_px"][:, 0]).sum(), 1)
                    for pos, q in zip(positions[:nb], d["billes"])])
    return D, sD, frac


BITS = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5]
KPIX = [1, 2, 3, 4, 5]
t0 = time.time()
for v in donnees:
    a_faire = [("bits", b) for b in BITS] + [("pixel", k) for k in KPIX if k > 1]
    if all((CACHE / f"{v.replace(' ', '_').replace('–', '-')}_{m}_{x}.pkl").exists() for m, x in a_faire):
        continue
    frames = tifffile.imread(donnees[v]["chemin"])
    for m, x in a_faire:
        relocaliser_video(v, m, x, frames)
    print(f"{v} relocalisée : {time.time() - t0:.0f} s", flush=True)
    del frames
print("relocalisation prête")

# %% [markdown]
# ## Graphique 3 — Profondeur de bits

# %%
for jeu, J in JEUX.items():
    V, res = J["videos"], []
    for b in BITS:
        out = [D_relocalises(v, relocaliser_video(v, "bits", b)) for v in V]
        D_v, sD_v = [o[0] for o in out], [o[1] for o in out]
        r = metriques(D_v, sD_v, V); r["frac"] = float(np.mean([o[2] for o in out]))
        res.append(r)
        noter_billes("3", "bits", b, 0, V, D_v, sD_v)
        print(resume(r, f"[{jeu}] {b:2d} bits :"), f"  positions relocalisées {100*r['frac']:.0f} %")
    RESULTATS[("3", jeu)] = dict(parametre="bits", x=BITS, res=res, videos=V)
figure({j: (BITS, RESULTATS[("3", j)]["res"]) for j in JEUX}, "Profondeur de bits", "graphique3_bits", xticks=BITS,
       inverser=True)

# %% [markdown]
# ## Graphique 4 — Taille de pixel (dans le plan objet)

# %%
for jeu, J in JEUX.items():
    V, res = J["videos"], []
    pix = [k * donnees[V[0]]["pixel_eff"] * 1e6 for k in KPIX]
    for k, px in zip(KPIX, pix):
        out = [D_relocalises(v, relocaliser_video(v, "bits", 14) if k == 1 else relocaliser_video(v, "pixel", k))
               for v in V]
        D_v, sD_v = [o[0] for o in out], [o[1] for o in out]
        r = metriques(D_v, sD_v, V); r["frac"] = float(np.mean([o[2] for o in out]))
        res.append(r)
        noter_billes("4", "pixel_um", round(px, 4), 0, V, D_v, sD_v, extra=dict(k_binning=k))
        print(resume(r, f"[{jeu}] k = {k} ({px:.3f} µm) :"), f"  positions relocalisées {100*r['frac']:.0f} %")
    RESULTATS[("4", jeu)] = dict(parametre="pixel_um", x=pix, res=res, videos=V)
figure({j: (RESULTATS[("4", j)]["x"], RESULTATS[("4", j)]["res"]) for j in JEUX}, "Taille de pixel (µm)",
       "graphique4_pixel")

# %% [markdown]
# ## Graphique 5a — Nombre d'images
#
# On ne garde que n images consécutives de chaque vidéo (jusqu'à la longueur de la plus courte vidéo du jeu : 111
# images pour 0,95 µm, 51 pour 2 µm), fenêtre placée à 5 positions régulièrement espacées, analyses combinées.

# %%
N_FEN = 5


def D_fenetre(b, debut, n, dt, pe):
    c = np.arange(len(b["pos_msd"])) + b["f0"]
    m = (c >= debut) & (c < debut + n)
    return D_piste(b["pos_msd"][m], b["inc_px"][m], dt, pe)


for jeu, J in JEUX.items():
    V, res = J["videos"], []
    n_max = min(donnees[v]["n_creneaux"] for v in V)
    N_IMG = [n for n in [20, 25, 30, 40, 50, 60, 70, 80, 90, 100] if n < n_max] + [n_max]
    for n in N_IMG:
        analyses = []
        for r_ in range(N_FEN):
            DD = []
            for v in V:
                d = donnees[v]
                debut = int(round(r_ / (N_FEN - 1) * (d["n_creneaux"] - n)))
                DD.append([D_fenetre(b, debut, n, d["dt"], d["pixel_eff"]) for b in d["billes"]])
            D_v, sD_v = paires(DD)
            analyses.append(metriques(D_v, sD_v, V, n_boot=300))
            noter_billes("5a", "n_images", n, r_, V, D_v, sD_v)
        res.append(combiner(analyses, V))
        print(resume(res[-1], f"[{jeu}] n = {n:3d} images :"))
    RESULTATS[("5a", jeu)] = dict(parametre="n_images", x=N_IMG, res=res, videos=V)
figure({j: (RESULTATS[("5a", j)]["x"], RESULTATS[("5a", j)]["res"]) for j in JEUX}, "Nombre d'images par vidéo",
       "graphique5a_nombre_images")

# %% [markdown]
# ## Graphique 5b — Nombre de billes
#
# On tire au hasard N billes (avec remise) parmi les billes gardées d'un jeu, 2000 fois (74 billes pour 0,95 µm,
# 6 pour 2 µm). Pour chaque tirage : rayon a = kT/(6πη·moyenne des D), écart |a − a_vrai|, u_stat et u_loc propagés
# comme ailleurs. Exactitude : moyenne des écarts; précision : médiane des u_a des tirages.

# %%
rng = np.random.default_rng(GRAINE)
for jeu, J in JEUX.items():
    V = J["videos"]
    Dt = np.concatenate([[x[0] for x in D0[v]] for v in V]); sDt = np.concatenate([[x[1] for x in D0[v]] for v in V])
    relt = np.concatenate([[donnees[v]["rel_inst"]] * len(D0[v]) for v in V])
    ok = np.isfinite(Dt); Dt, sDt, relt = Dt[ok], sDt[ok], relt[ok]
    N_BILLES = [n for n in [3, 5, 8, 10, 15, 20, 30, 40, 50, 60] if n < len(Dt)] + [len(Dt)]
    res = []
    for N in N_BILLES:
        idx = rng.integers(0, len(Dt), size=(2000, N))
        Dm = Dt[idx].mean(axis=1)
        okm = Dm > 0
        a = rayon_um(Dm[okm])
        u_stat = a * Dt[idx][okm].std(axis=1, ddof=1) / np.sqrt(N) / Dm[okm]
        u_loc = a * np.sqrt(np.sum(sDt[idx][okm]**2, axis=1)) / N / Dm[okm]
        u = np.hypot(u_stat, u_loc)
        u_inst = float(np.median(a * relt[idx][okm].mean(axis=1)))
        r = dict(E=float(np.mean(np.abs(a - J["a_vrai"]))), u=float(np.median(u)),
                 su=float(np.diff(np.percentile(u, [16, 84]))[0] / 2), u_stat=float(np.median(u_stat)),
                 u_loc=float(np.median(u_loc)), u_inst=u_inst, N=N, n_vid=None, a_moy=float(np.mean(a)),
                 dispersion_tirages=float(a.std()))
        r["sE"] = float(np.sqrt(r["u"]**2 + u_inst**2 + J["u_ref"]**2))
        res.append(r)
        print(resume(r, f"[{jeu}] N = {N:2d} billes :"), f"  dispersion réelle des tirages {a.std()*1e3:.0f} nm")
    RESULTATS[("5b", jeu)] = dict(parametre="n_billes", x=N_BILLES, res=res, videos=None)
figure({j: (RESULTATS[("5b", j)]["x"], RESULTATS[("5b", j)]["res"]) for j in JEUX}, "Nombre de billes",
       "graphique5b_nombre_billes", log=True, xticks=[3, 5, 10, 20, 40, 74], xticklabels=["3", "5", "10", "20", "40", "74"])

# %% [markdown]
# ## Export des données en CSV
#
# Trois fichiers dans `fig_performance/` (séparateur virgule, point décimal, UTF-8), avec une colonne `jeu`
# (0,95 µm ou 2 µm) : `performance_resume.csv` (une ligne par point tracé), `performance_par_video.csv` (une ligne
# par vidéo et par point) et `performance_par_bille.csv` (une ligne par bille et par analyse).

# %%
def f6(v):
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}" if np.isfinite(v) else ""
    return v


def ecrire(nom, lignes):
    cles = list(dict.fromkeys(k for l in lignes for k in l))
    with open(FIG / nom, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cles)
        w.writeheader()
        for l in lignes:
            w.writerow({k: f6(v) for k, v in l.items()})
    print(f"{nom} : {len(lignes)} lignes")


lignes_res, lignes_vid = [], []
for (g, jeu), R in RESULTATS.items():
    for x, r in zip(R["x"], R["res"]):
        lignes_res.append(dict(graphique=g, jeu=jeu, parametre=R["parametre"], valeur=float(x),
                               E_a_nm=r["E"] * 1e3, barre_E_a_nm=r["sE"] * 1e3, u_a_nm=r["u"] * 1e3,
                               barre_u_a_nm=r["su"] * 1e3, u_stat_nm=r["u_stat"] * 1e3, u_loc_nm=r["u_loc"] * 1e3,
                               u_inst_nm=r["u_inst"] * 1e3, u_ref_nm=JEUX[jeu]["u_ref"] * 1e3,
                               rayon_moyen_pondere_um=r.get("a_moy"), N_billes=r["N"], n_videos=r.get("n_vid"),
                               fraction_positions_relocalisees=r.get("frac"),
                               dispersion_tirages_nm=r["dispersion_tirages"] * 1e3 if "dispersion_tirages" in r else None))
        if R["videos"] is None:
            continue
        stats = r["stats"]
        Ntot = sum(s["N"] for s in stats if s is not None) or 1
        for v, s in zip(R["videos"], stats):
            base = dict(graphique=g, jeu=jeu, parametre=R["parametre"], valeur=float(x), video=v)
            if s is None:
                lignes_vid.append(dict(base, N_billes=0)); continue
            lignes_vid.append(dict(base, N_billes=s["N"], poids=s["N"] / Ntot, D_moy_um2_s=s["D_moy"] * 1e12, a_v_um=s["a"],
                                   ecart_abs_nm=s["ecart"] * 1e3, u_stat_v_nm=s["u_stat"] * 1e3,
                                   u_loc_v_nm=s["u_loc"] * 1e3, u_inst_v_nm=s["u_inst"] * 1e3,
                                   a_depart_um=donnees[v]["a_officiel"] * 1e6))

ecrire("performance_resume.csv", lignes_res)
ecrire("performance_par_video.csv", lignes_vid)
ecrire("performance_par_bille.csv", PAR_BILLE)

# %% [markdown]
# ## Annexe — Pertes d'images pendant l'acquisition
#
# Chaque acquisition enregistre l'horodatage de chaque image (`timestamps_ns` du .json). Un écart de plus de 1,5 fois
# la durée d'image prévue signale des images perdues; leur nombre est l'écart arrondi divisé par la durée d'image, moins 1.

# %%
def enregistrer(fig, nom):
    fig.savefig(FIG / f"{nom}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{nom}.pdf", bbox_inches="tight", facecolor="white")


acq = []
for f in sorted(DOSSIER_JSON.glob("*.json")):
    j = json.load(open(f))
    if "timestamps_ns" not in j or "_part_" in f.name or len(j["timestamps_ns"]) < 5:
        continue
    t = (np.array(j["timestamps_ns"], float) - j["timestamps_ns"][0]) / 1e9
    T0 = j.get("duree_image_consigne_us", 92000) / 1e6
    ecarts = np.diff(t)
    perdues = np.where(ecarts > 1.5 * T0, np.round(ecarts / T0) - 1, 0)
    premier = int(np.argmax(perdues > 0)) + 1 if perdues.any() else len(t)
    nom = f.stem.split("_2026")[0]
    acq.append(dict(nom=nom, fichier=f.name, option=j.get("option", "?"), g=j.get("grossissement"), t=t, T0=T0,
                    perdues=perdues, n=len(t), n_perdues=int(perdues.sum()), duree=t[-1],
                    frac=perdues.sum() / (len(t) + perdues.sum()), premier=premier,
                    t_premier=t[premier - 1] if perdues.any() else np.nan,
                    largeur=j.get("largeur_px"), hauteur=j.get("hauteur_px")))
for a in acq:
    print(f"{a['fichier']:48s} option {a['option']} {a['largeur']}×{a['hauteur']} : {a['n']:4d} images reçues, "
          f"{a['n_perdues']:4d} perdues ({100*a['frac']:4.1f} %), durée {a['duree']:5.1f} s, "
          f"1re perte après {a['premier']} images ({a['t_premier']:.1f} s)")
PERTES = [{k: (v if not isinstance(v, np.ndarray) else None) for k, v in a.items() if k not in ("t", "perdues")}
                       for a in acq]

# %%
C_OPT = {"A": "#7d5ba6", "B": "#2a9d8f", "C": BLEU, "D": "#b5838d"}


def etiquette(a):
    if a["nom"].startswith("acq"):
        return f"2 nov., option {a['option']}"
    return {"inconnu2_20x": "vidéo 2 (test)", "inconnu2_p2": "p2 (test, écartée)"}.get(
        a["nom"], a["nom"].replace("inconnu2_p", "vidéo ").replace("_20x", "") + (" (test)" if a["nom"].startswith("inconnu2_p") else ""))


def lettre(ax, t):
    ax.text(-0.02, 1.035, t, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="right")


fig, axs = plt.subplots(1, 3, figsize=(13.2, 3.9), gridspec_kw=dict(width_ratios=[1.25, 1, 1.15]))
ax = axs[0]
for a in sorted(acq, key=lambda a: a["option"] != "C"):
    cumul = np.concatenate([[0], np.cumsum(a["perdues"])])
    test = a["nom"].startswith("inconnu2") and a["nom"] != "inconnu2_p2"
    ax.step(a["t"], cumul, where="post", color=C_OPT.get(a["option"], GRIS), lw=1.5 if test else 1.0,
            alpha=0.95 if test else 0.6, ls="-" if a["g"] == 20 else "--")
ax.set_xlabel("Temps depuis le début de l'acquisition (s)", style="italic")
ax.set_ylabel("Images perdues, cumul", style="italic")
ax.set_ylim(0, None)
lettre(ax, "a)")
for o, c in C_OPT.items():
    if any(a["option"] == o for a in acq):
        ax.plot([], [], color=c, lw=1.6, label=f"option {o}")
ax.plot([], [], color=GRIS, ls="--", lw=1, label="40X")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")

ax = axs[1]
for a in acq:
    ax.scatter(a["duree"], 100 * a["frac"], s=38, color=C_OPT.get(a["option"], GRIS), edgecolor="white", lw=0.6,
               marker="o" if a["g"] == 20 else "s", zorder=3)
ax.set_xlabel("Durée de l'acquisition (s)", style="italic")
ax.set_ylabel("Images perdues (%)", style="italic")
ax.scatter([], [], color=GRIS, marker="o", label="20X"); ax.scatter([], [], color=GRIS, marker="s", label="40X")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_ylim(-2, None)
lettre(ax, "b)")

ax = axs[2]
ordre = sorted(acq, key=lambda a: a["premier"])
y = np.arange(len(ordre))
ax.barh(y, [a["premier"] for a in ordre], color=[C_OPT.get(a["option"], GRIS) for a in ordre], height=0.62,
        hatch=[None if np.isfinite(a["t_premier"]) else "////" for a in ordre], edgecolor="white", lw=0)
ax.set_yticks(y); ax.set_yticklabels([etiquette(a) for a in ordre], fontsize=8.5)
ax.set_xlabel("Images reçues avant la 1re perte", style="italic")
for yy, a in zip(y, ordre):
    txt = f"{a['t_premier']:.0f} s" if np.isfinite(a["t_premier"]) else f"aucune perte ({a['duree']:.0f} s)"
    ax.text(a["premier"] + 2, yy, txt, va="center", fontsize=7.8, color="#52514e")
ax.set_xlim(0, max(a["premier"] for a in acq) * 1.18)
ax.spines["right"].set_visible(False)
lettre(ax, "c)")
fig.tight_layout(w_pad=2.2)
enregistrer(fig, "annexe_pertes_images"); plt.show()

