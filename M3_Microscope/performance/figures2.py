"""Figures complémentaires du brainstorm (images de vidéos, simulations, graphiques « envisagés »).
Usage : python3 figures2.py [noms...]"""
import sys, glob, pickle, json, time, contextlib, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
import tifffile
import traitement_video_tiff as M
import sim_perf as SP
import stats_sim as S
import modeles as MOD
from figures import SURF, TXT, TXT2, GRIS, GRILLE, C20, C40, axe, virg

MAN = "/tmp/claude-0/-home-claude/b34181bf-7863-54fc-b462-8a44767b4356/scratchpad/manuel"
OUT = "fig"
VERT, ORANGE, ROUGE = "#1baf7a", "#eb6834", "#e34948"
TCR = {"p1": 2, "p3": 3, "p4": 4, "p5": 5, "p6": 6, "p7": 7}   # fichier -> numéro de vidéo du rapport du TCR


def image(nom, k=0):
    with tifffile.TiffFile(f"videos/{nom}.tif") as tf:
        return tf.pages[k].asarray().astype(float)


def etirer(im, bas=1, haut=99.8):
    lo, hi = np.percentile(im, (bas, haut))
    return np.clip((im - lo) / max(hi - lo, 1), 0, 1)


def barre(ax, px_um, long_um, x0, y0, txt=None, coul="white"):
    L = long_um / px_um
    ax.add_patch(Rectangle((x0, y0), L, max(2, L * 0.06), color=coul, lw=0))
    ax.text(x0 + L / 2, y0 - 3, txt or f"{virg(long_um, 0)} µm", color=coul, ha="center", va="bottom", fontsize=8)


def sans_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ------------------------------------------------------------------------------------------------ A. vraies images
def fig_images_reelles():
    cas = [("inc2_p1_20X_C", "Test réel, a = 0,475 µm · 20X", 0.374, 0),
           ("inc1_40X_C", "Même échantillon · 40X (défocalisé)", 0.185, 0),
           ("m1um_20X_C", "a = 1 µm · 20X", 0.374, 0),
           ("n50_40X_C", "a = 50 nm · 40X", 0.185, 0)]
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.6), gridspec_kw=dict(height_ratios=[1, 1]))
    for j, (nom, titre, px, k) in enumerate(cas):
        im = image(nom, k)
        axs[0, j].imshow(etirer(im), cmap="gray"); sans_axes(axs[0, j])
        axs[0, j].set_title(titre, fontsize=9)
        h, w = im.shape
        barre(axs[0, j], px, 50, w * 0.06, h * 0.92)
        # zoom 30 µm autour de la détection la plus brillante (hors bords)
        lis = M.ndi.gaussian_filter(im, 2)
        m = 40; lis[:m] = lis[-m:] = 0; lis[:, :m] = lis[:, -m:] = 0
        y, x = np.unravel_index(np.argmax(lis), lis.shape)
        r = int(15 / px)
        sub = im[max(0, y - r):y + r, max(0, x - r):x + r]
        axs[0, j].add_patch(Rectangle((x - r, y - r), 2 * r, 2 * r, fill=False, ec="#eda100", lw=1))
        axs[1, j].imshow(etirer(sub, 1, 99.9), cmap="gray", interpolation="nearest"); sans_axes(axs[1, j])
        barre(axs[1, j], px, 5, sub.shape[1] * 0.08, sub.shape[0] * 0.9)
        axs[1, j].set_title(f"zoom 30 µm (pixel {virg(px, 3)} µm)", fontsize=8.5, color=TXT2)
    fig.suptitle("Images brutes (première image de chaque vidéo, contraste étiré)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUT}/img1_images_reelles.png", dpi=170); plt.close(fig)


# ------------------------------------------------------------------------------------------------ B. pixels et bits
def fig_pixels_bits():
    im = image("inc2_p1_20X_C", 0)
    lis = M.ndi.gaussian_filter(im, 2); m = 60
    lis[:m] = lis[-m:] = 0; lis[:, :m] = lis[:, -m:] = 0
    y, x = np.unravel_index(np.argmax(lis), lis.shape)
    r = 24
    sub = im[y - r:y + r, x - r:x + r]
    fig, axs = plt.subplots(2, 4, figsize=(12, 6.6))
    for j, k in enumerate((1, 2, 3, 4)):
        s = sub[:(2 * r // k) * k, :(2 * r // k) * k]
        b = s.reshape(s.shape[0] // k, k, s.shape[1] // k, k).mean(axis=(1, 3))
        axs[0, j].imshow(etirer(b, 0.5, 99.9), cmap="gray", interpolation="nearest"); sans_axes(axs[0, j])
        res = {1: "0 %", 2: "−2 %", 3: "−49 %", 4: "−36 %"}[k]
        axs[0, j].set_title(f"pixel {virg(0.374 * k, 2)} µm (équiv. {20 // k if k < 3 else virg(20 / k, 1)}X)\nécart sur a : {res}", fontsize=9)
    for j, b in enumerate((14, 10, 8, 6)):
        q = 2 ** (14 - b)
        s = (sub // q) * q
        f0 = np.median(sub)
        axs[1, j].imshow(s, cmap="gray", interpolation="nearest", vmin=f0 - 100, vmax=f0 + 400)
        sans_axes(axs[1, j])
        axs[1, j].set_title(f"{b} bits : pas de {q} compte{'s' if q > 1 else ''}\n(bruit : 31 comptes)", fontsize=9)
    for ax in axs[:, 0]:
        barre(ax, 0.374 if ax is axs[1, 0] else 0.374, 5, 3, ax.get_images()[0].get_array().shape[0] * 0.9)
    fig.suptitle("Une bille du test réel (vidéo 2, 20X) : regroupement de pixels (haut, contraste plein)\n"
                 "et profondeur de bits (bas, contraste serré sur le fond pour voir la quantification)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.5); fig.savefig(f"{OUT}/img2_pixels_bits.png", dpi=170); plt.close(fig)


# ------------------------------------------------------------------------------------------------ C. détections et revue manuelle
def fig_revue_manuelle():
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, p in zip(axs, ("p5", "p7")):
        d = pickle.load(open(f"{MAN}/{p}_final.pkl", "rb"))
        with tifffile.TiffFile(d["video"]) as tf:
            im = tf.pages[0].asarray().astype(float)
        ax.imshow(etirer(im, 1, 99.7), cmap="gray"); sans_axes(ax)
        excl = set(d["exclues_main"]); grp = set(d["retires_groupe"]); imm = set(d["imm_auto"]); garde = set(d["garde"])
        n = dict(garde=0, imm=0, grp=0, excl=0)
        for r in d["res"]:
            q = r["pos_px"]; ok = np.isfinite(q[:, 0])
            if not ok.any():
                continue
            xm, ym = np.nanmedian(q[:, 0]), np.nanmedian(q[:, 1])
            pid = r["piste"]
            if pid in garde:
                c, ls, key = VERT, "-", "garde"
            elif pid in grp:
                c, ls, key = ORANGE, "-", "grp"
            elif pid in excl:
                c, ls, key = ROUGE, "-", "excl"
            else:
                c, ls, key = "#8a8986", "--", "imm"
            n[key] += 1
            ax.plot(q[ok, 0], q[ok, 1], color=c, lw=0.8, alpha=0.9)
            ax.add_patch(Circle((xm, ym), 14, fill=False, ec=c, lw=1.2, ls=ls))
            if pid in garde:
                ax.text(xm + 16, ym - 10, f"{virg(r['a'] * 1e6, 2)}", color=c, fontsize=6.5)
        barre(ax, 0.374, 50, 30, im.shape[0] - 40)
        a_imm = M.rayon_final([r for r in d["res"] if not r["immobile"]])
        a_tout = M.rayon_final(d["res"])
        ax.set_title(f"Vidéo {TCR[p]} : {len(d['res'])} pistes → {n['garde']} gardées · rayon (réf. 0,475 µm) :\n"
                     f"toutes les pistes {virg(a_tout['a'] * 1e6, 3)} · sans les immobiles {virg(a_imm['a'] * 1e6, 3)} · "
                     f"après revue {virg(d['final']['a'] * 1e6, 3)} µm", fontsize=9.5)
    h = [Line2D([], [], color=VERT, marker="o", mfc="none", ls="-", label="gardée (rayon en µm)"),
         Line2D([], [], color="#8a8986", marker="o", mfc="none", ls="--", label="immobile (test D < 2σ₀)"),
         Line2D([], [], color=ORANGE, marker="o", mfc="none", label="membre d'agrégat, retiré"),
         Line2D([], [], color=ROUGE, marker="o", mfc="none", label="exclue à la main")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92)); fig.savefig(f"{OUT}/img3_revue_manuelle.png", dpi=170); plt.close(fig)


# ------------------------------------------------------------------------------------------------ D. à quoi ressemblent les simulations
def fig_simulations():
    etal = json.load(open("etalon.json"))
    tailles = [0.05, 0.15, 0.475, 1.5, 5.0]
    fig, axs = plt.subplots(2, 5, figsize=(14, 6.2))
    for j, a in enumerate(tailles):
        rng = np.random.default_rng(3)
        fr, vrai, D = SP.simuler_video(a, SP.CONF["20X"], rng, N=60, taille=200, n_part=1, etal=etal)
        x0, y0 = vrai[0, 0, :2]
        r = 40
        cx, cy = int(np.clip(x0, r, 200 - r)), int(np.clip(y0, r, 200 - r))
        sub = fr[0, cy - r:cy + r, cx - r:cx + r].astype(float)
        axs[0, j].imshow(sub, cmap="gray", interpolation="nearest", vmin=np.percentile(fr[0], 1), vmax=max(np.percentile(fr[0], 99.99), np.percentile(fr[0], 1) + 150))
        sans_axes(axs[0, j]); barre(axs[0, j], 0.374, 5, 4, 2 * r * 0.9)
        axs[0, j].set_title(f"a = {virg(a, 3 if a < 0.2 or a == 0.475 else 1)} µm\nD = {virg(D, 2)} µm²/s", fontsize=9.5)
        # trajectoire vraie sur 60 images
        t = vrai[:, 0, :2] * 0.374
        axs[1, j].plot(t[:, 0] - t[0, 0], t[:, 1] - t[0, 1], "-", color=C20, lw=1)
        axs[1, j].plot(0, 0, "o", color=TXT, ms=3)
        axs[1, j].set_aspect("equal"); axe(axs[1, j], xlog=False)
        lim = max(1.0, np.abs(t - t[0]).max() * 1.1)
        axs[1, j].set_xlim(-lim, lim); axs[1, j].set_ylim(-lim, lim)
        axs[1, j].set_xlabel("x (µm)")
        pas = np.sqrt(4 * D * 0.092)
        axs[1, j].set_title(f"pas moyen {virg(pas, 2)} µm / image", fontsize=8.5, color=TXT2)
    axs[1, 0].set_ylabel("y (µm)")
    fig.suptitle("Vidéos simulées au 20X (pixel 0,374 µm, 90 ms) : image 1 (haut) et trajectoire vraie sur 60 images (bas)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{OUT}/img4_simulations.png", dpi=170); plt.close(fig)


def fig_sim_vs_reel():
    etal = json.load(open("etalon.json"))
    rng = np.random.default_rng(11)
    fr, vrai, D = SP.simuler_video(0.475, SP.CONF["20X"], rng, N=3, taille=300, n_part=10, etal=etal)
    im = image("inc2_p1_20X_C", 0)[200:500, 300:600]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax, I, t in ((axs[0], im, "Vraie vidéo (test, vidéo 2)"), (axs[1], fr[0].astype(float), "Vidéo simulée (même bruit étalonné)")):
        ax.imshow(etirer(I, 1, 99.8), cmap="gray"); sans_axes(ax); ax.set_title(t, fontsize=10)
        barre(ax, 0.374, 20, 15, 280)
    fig.suptitle("a = 0,475 µm au 20X : champ de 112 × 112 µm", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{OUT}/img5_simulation_vs_reel.png", dpi=170); plt.close(fig)


# ------------------------------------------------------------------------------------------------ E. histogramme par bille
def fig_histogramme():
    a_all, par = [], {}
    for p in TCR:
        d = pickle.load(open(f"{MAN}/{p}_final.pkl", "rb"))
        a = [r["a"] * 1e6 for r in d["res"] if r["piste"] in d["garde"]]
        a_all += a; par[p] = a
    a_all = np.array(a_all)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bins = np.logspace(np.log10(0.15), np.log10(5), 28)
    ax.hist(a_all, bins=bins, color=C20, alpha=0.85, edgecolor=SURF, lw=1)
    ax.axvline(0.475, color=TXT, lw=1.2); ax.axvspan(0.465, 0.485, color=TXT2, alpha=0.15)
    ax.axvline(0.499, color=C20, lw=1.2, ls="--")
    med = np.median(a_all); q1, q3 = np.percentile(a_all, (25, 75))
    ax.text(0.49, ax.get_ylim()[1] * 0.95, "référence 0,475", ha="right", va="top", fontsize=8, color=TXT, rotation=90)
    ax.text(0.515, ax.get_ylim()[1] * 0.95, "moyenne (par D moyen) 0,499", ha="left", va="top", fontsize=8, color=C20, rotation=90)
    axe(ax, "Nombre de billes"); ax.set_xlabel("Rayon de chaque bille (µm), échelle log")
    ax.set_title(f"74 billes gardées du test réel : médiane {virg(med, 2)} µm, quartiles {virg(q1, 2)}–{virg(q3, 2)} µm", fontsize=10)
    ax.annotate("queue : agrégats\nnon repérés ou billes\nfreinées (D petit)", (1.6, 1), xytext=(2.2, 4.5), fontsize=8, color=TXT2,
                arrowprops=dict(arrowstyle="-", color=GRIS, lw=0.7))
    fig.tight_layout(); fig.savefig(f"{OUT}/fig7_histogramme_billes.png", dpi=200); plt.close(fig)
    return a_all


# ------------------------------------------------------------------------------------------------ F. exemple de MSD
def fig_msd():
    d = pickle.load(open(f"{MAN}/p5_final.pkl", "rb"))
    dt, pe, te = 0.092, 6.2 / 16.566 * 1e-6, 0.090
    g = [r for r in d["res"] if r["piste"] in d["garde"]]
    r = sorted(g, key=lambda r: -r["n_pts"])[0]
    pos = r["pos_px"] * pe
    lags, msd = M.msd_temporel(pos, 10)
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    t = lags * dt
    ax.plot(t, msd * 1e12, "o", color=C20, ms=6, label="MSD mesuré (une bille de la vidéo 5)")
    for L, c, ls in ((4, C20, "-"), (10, GRIS, ":")):
        D, off = M.ajuste_msd(pos, dt, L)
        tt = np.linspace(0, t[-1], 50)
        ax.plot(tt, (4 * D * tt + off) * 1e12, ls, color=c, lw=1.5, label=f"ajustement L = {L} : D = {virg(D * 1e12, 3)} µm²/s")
    D4, off4 = M.ajuste_msd(pos, dt, 4)
    ax.axvspan(0, 4 * dt + 0.02, color=C20, alpha=0.06)
    ax.annotate(f"ordonnée = 4σ² − 4D·t_exp/3\n= {virg(off4 * 1e12, 3)} µm²", (0, off4 * 1e12), xytext=(0.25, 0.04 + off4 * 1e12),
                fontsize=8, color=TXT2, arrowprops=dict(arrowstyle="-", color=GRIS, lw=0.7))
    axe(ax, "MSD (µm²)", xlog=False); ax.set_xlabel("Délai τ (s)")
    a4 = M.kB * M.T / (6 * np.pi * M.ETA * D4) * 1e6
    ax.set_title(f"MSD = 4Dτ + 4σ² − 4D·t_exp/3  →  a = kT/(6πηD) = {virg(a4, 3)} µm", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig8_exemple_msd.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------------------------------------ G. erreur de localisation vs SNR
def fig_loc_snr():
    etal = json.load(open("etalon.json"))
    rng = np.random.default_rng(5)
    g, noir, lect = etal["gain"], etal["noir"], etal["lecture"]
    B = (etal["fond"] - noir) / g
    s = 1.8                                    # σ de la tache (px), test réel au foyer
    pas = 0.374
    Y, X = np.mgrid[0:31, 0:31]
    snrs, err, err_th = [], [], []
    for Nph in np.logspace(2.3, 6, 12):
        e = []
        A_c = g * Nph / (2 * np.pi * s**2)
        for _ in range(150):
            x0, y0 = 15 + rng.uniform(-.5, .5), 15 + rng.uniform(-.5, .5)
            img = B + Nph / (2 * np.pi * s**2) * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * s**2))
            im = noir + g * rng.poisson(img) + rng.normal(0, lect, img.shape)
            r = M.fit_gauss_local(im, 15, 15, np.median(im), s)
            if r is not None:
                e.append(np.hypot(r[0] - x0, r[1] - y0))
        bruit = np.sqrt(lect**2 + g**2 * B)
        snrs.append(A_c / bruit)
        err.append(np.sqrt(np.mean(np.square(e)) / 2) * pas * 1e3 if len(e) > 20 else np.nan)
        # Thompson et coll. (2002) : σ² = (s² + a²/12)/N + 8π s⁴ b² / (a² N²), en pixels (a = 1 px)
        b2 = (lect / g)**2 + B
        th = np.sqrt((s**2 + 1 / 12) / Nph + 8 * np.pi * s**4 * b2 / Nph**2)
        err_th.append(th * pas * 1e3)
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.loglog(snrs, err, "o-", color=C20, ms=5, lw=1.6, label="simulation : ajustement gaussien du code")
    ax.loglog(snrs, err_th, "--", color=GRIS, lw=1.3, label="formule de Thompson et coll. (2002)")
    # points réels : σ_exp médian et SNR médian par vidéo
    for f in sorted(glob.glob("res/inc2_p*_20X_C.pkl")):
        d = pickle.load(open(f, "rb"))
        rr = [r for r in d["resL"][3] if r["raison"] == "non" and np.isfinite(r["sigma_exp"])]
        if rr:
            ax.plot(np.median([r["snr"] for r in rr]), np.median([r["sigma_exp"] for r in rr]) * 1e9, "D", color=C20, mec=TXT, ms=6)
    ax.axvline(4, color=ROUGE, lw=1, ls=":")
    ax.text(4.3, 600, "SNR = 4 :\nseuil du travail\npréparatoire", fontsize=8, color=ROUGE, va="top")
    ax.plot([], [], "D", color=C20, mec=TXT, label="vraies vidéos du test (σ_exp, inclut le flou)")
    axe(ax, "Erreur de localisation par axe (nm)", xlog=False); ax.set_xscale("log"); ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
    ax.text(25, 330, "vraies vidéos au-dessus de la courbe :\nflou résiduel, défocalisation, aberrations", fontsize=7.5, color=TXT2)
    ax.set_xlabel("Rapport signal sur bruit au pic de la tache")
    ax.set_title("Erreur de localisation selon le SNR (20X, σ de tache 1,8 pixel)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig9_localisation_snr.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------------------------------------ H. longueur des pistes
def fig_longueurs():
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for conf, c in (("20X", C20), ("40X", C40)):
        xs, med, q1, q3 = [], [], [], []
        for f in sorted(glob.glob(f"sim/{conf}_*_base.pkl")):
            d = pickle.load(open(f, "rb"))
            L = [l for r in d["reps"] if r["info"] for l in r["info"]["longueurs"] if l >= 5]
            if len(L) < 5:
                continue
            xs.append(d["a"]); med.append(np.median(L)); q1.append(np.percentile(L, 25)); q3.append(np.percentile(L, 75))
        o = np.argsort(xs); xs, med, q1, q3 = (np.array(v)[o] for v in (xs, med, q1, q3))
        ax.fill_between(xs, q1, q3, color=c, alpha=0.15, lw=0)
        ax.plot(xs, med, "-o", color=c, ms=4, lw=1.6, label=f"simulation {conf} (médiane, quartiles)")
    ax.axhline(140, color=TXT2, lw=0.8, ls=":"); ax.text(0.045, 143, "durée de la vidéo (140 images)", fontsize=8, color=TXT2)
    ax.axhline(20, color=ROUGE, lw=0.8, ls=":"); ax.text(0.045, 22, "minimum retenu (20 positions)", fontsize=8, color=ROUGE)
    axe(ax, "Longueur des pistes (positions)"); ax.set_xlabel("Rayon a (µm)")
    ax.set_title("Les petites billes donnent des pistes courtes (liaison cassée, sortie du foyer)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig10_longueur_pistes.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------------------------------------ I. temps de calcul
def fig_temps_calcul():
    acq = M.charger_acquisition("videos/inc2_p5_20X_C.tif")
    fr = acq["frames"][:20]
    kw, dmax, demi = M.parametres_pixels(acq["pixel_eff"])
    s, _, _ = M.estimer_sigma_psf(fr, demi=demi, **kw)
    kw["distance_min"] = max(kw["distance_min"], int(np.ceil(2 * s)))
    res = {}
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        M.localiser_toutes_frames(fr, s, **kw)
    res["base"] = (time.time() - t0) / len(fr)
    import traitement_video_tiff_extra as MX
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        MX.localiser_video_multi(fr, s, progres=False)
    res["extra"] = (time.time() - t0) / len(fr)
    json.dump(res, open("temps_calcul.json", "w"))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    noms = ["code de base\n(gaussienne par bille)", "code extra\n(ajustement multi-billes)"]
    v = [res["base"], res["extra"]]
    ax.barh(noms, v, color=[C20, GRIS], height=0.5)
    for i, x in enumerate(v):
        ax.text(x * 1.05, i, f"{virg(x, 2)} s/image  →  {virg(x / 0.092, 1)} × le temps réel", va="center", fontsize=8.5)
    ax.axvline(0.092, color=ROUGE, lw=1, ls=":"); ax.text(0.095, 1.35, "92 ms : temps réel", fontsize=8, color=ROUGE)
    ax.set_xscale("log"); axe(ax, xlog=False); ax.set_xscale("log"); ax.set_xlim(0.03, max(v) * 30)
    ax.set_xlabel("Temps de localisation par image 1014 × 760 (s), ordinateur de développement (log)")
    ax.set_title("Temps de calcul (à remesurer sur la Raspberry Pi pour le bonus)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig11_temps_calcul.png", dpi=200); plt.close(fig)
    return res


# ------------------------------------------------------------------------------------------------ J. 20X contre 40X
def fig_objectifs():
    lignes = {}
    for nom, obj in (("inc2_p1_20X_C", "20X"), ("inc2_p3_20X_C", "20X"), ("inc2_p4_20X_C", "20X"), ("inc2_p5_20X_C", "20X"),
                     ("inc2_p6_20X_C", "20X"), ("inc2_p7_20X_C", "20X"), ("m1um_20X_C", "20X"), ("m1um_20X_B", "20X"),
                     ("inc1_40X_C", "40X"), ("n50_40X_C", "40X"), ("m1umA_C_40X", "40X"), ("m1umB_C_40X", "40X")):
        d = pickle.load(open(f"res/{nom}.pkl", "rb")); i = d["info"]
        rr = [r for r in d["resL"][3] if r["raison"] == "non" and np.isfinite(r["sigma_exp"])]
        lignes.setdefault(obj, []).append(dict(psf=i["sigma_psf"] * i["pixel_eff"] * 1e6,
                                               sexp=np.median([r["sigma_exp"] for r in rr]) * 1e9 if rr else np.nan,
                                               n=len(rr), ntot=len(d["resL"][3])))
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    th = {"20X": 0.21 * 0.58 / 0.40, "40X": 0.21 * 0.58 / 0.65}
    for j, (cle, lab, unite) in enumerate((("psf", "σ de la tache mesurée", "µm"), ("sexp", "Erreur de localisation σ_exp", "nm"),
                                            ("n", "Billes mobiles suivies par vidéo", ""))):
        ax = axs[j]
        for k, obj in enumerate(("20X", "40X")):
            v = [l[cle] for l in lignes[obj] if np.isfinite(l[cle])]
            ax.scatter(np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v)), v, color=C20 if obj == "20X" else C40, s=28, zorder=3)
            ax.plot([k - 0.22, k + 0.22], [np.median(v)] * 2, color=TXT, lw=1.5)
            if cle == "psf":
                ax.plot([k - 0.22, k + 0.22], [th[obj]] * 2, color=GRIS, lw=1.2, ls="--")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["20X (NA 0,40)", "40X (NA 0,65)"])
        axe(ax, f"{lab} ({unite})" if unite else lab, xlog=False); ax.set_xlim(-0.5, 1.5)
        if cle == "psf":
            ax.set_yscale("log"); ax.text(1.25, th["40X"] * 1.1, "diffraction\nseule", fontsize=7.5, color=GRIS)
            from matplotlib.ticker import FixedLocator, NullFormatter
            ax.yaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.5, 1, 2])); ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_yticklabels(["0,2", "0,3", "0,5", "1", "2"])
    axs[0].set_title("a) Résolution", fontsize=10); axs[1].set_title("b) Localisation", fontsize=10); axs[2].set_title("c) Statistique", fontsize=10)
    fig.suptitle("20X contre 40X sur les vraies vidéos (trait noir : médiane) : le 40X ne gagne rien en pratique",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(f"{OUT}/fig12_20X_40X.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------------------------------------ K. paroi et température
def fig_paroi():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axs[0]
    aa = np.logspace(np.log10(0.05), np.log10(5), 80)
    for zf, ls, lab in ((None, "-", "plan focal là où sont les billes (a + l_g)"), (10e-6, "--", "plan focal à 10 µm du fond")):
        y = [100 * (MOD.biais_paroi(a * 1e-6, zf=zf)[0] - 1) for a in aa]
        ax.plot(aa, y, ls, color=GRIS if zf else TXT2, lw=1.6, label=lab)
    ax.errorbar([0.475], [5.1], [4.9], fmt="D", color=C20, mec=TXT, ms=6, capsize=3, label="mesures (20X)")
    ax.errorbar([1.0], [29], [13], fmt="D", color=C20, mec=TXT, ms=6, capsize=3)
    ax.errorbar([1.03], [3], [21], fmt="D", color=C40, mec=TXT, ms=6, capsize=3, label="mesure (40X, 3 billes)")
    ax.axhline(10, color=ROUGE, lw=0.8, ls=":"); ax.text(0.055, 12, "seuil 10 %", fontsize=8, color=ROUGE)
    ax.axvline(0.71, color=ROUGE, lw=0.8, ls=":"); ax.text(0.73, 120, "a_max ≈ 0,7 µm", fontsize=8, color=ROUGE)
    axe(ax, "Biais attendu sur a (%)"); ax.set_xlabel("Rayon a (µm)"); ax.set_ylim(-10, 200)
    ax2 = ax.twiny(); ax2.set_xscale("log"); ax2.set_xlim(ax.get_xlim())
    tk = [0.1, 0.2, 0.5, 1, 2, 5]; ax2.set_xticks(tk)
    ax2.set_xticklabels([virg(MOD.lg(a * 1e-6) * 1e6, 0) if MOD.lg(a * 1e-6) * 1e6 >= 10 else virg(MOD.lg(a * 1e-6) * 1e6, 2) for a in tk], fontsize=7.5, color=TXT2)
    ax2.set_xlabel("longueur de sédimentation l_g (µm)", fontsize=8, color=TXT2)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title("a) Paroi (Faxén) et sédimentation", fontsize=10, pad=28)
    ax = axs[1]
    dT = np.linspace(-3, 3, 61)
    A_, B_, C_ = 2.414e-5, 247.8, 140.0
    eta = lambda TT: A_ * 10**(B_ / (TT - C_))
    T0 = 295.15
    r = [100 * ((T0 / eta(T0)) / ((T0 + d) / eta(T0 + d)) - 1) for d in dT]
    ax.plot(dT, r, color=C20, lw=1.8); ax.axhline(0, color=TXT2, lw=0.8); ax.axvline(0, color=TXT2, lw=0.8)
    ax.axvspan(-0.5, 0.5, color=C20, alpha=0.08); ax.text(0.6, 5, "± 0,5 °C → ± 1,4 %", fontsize=8, color=C20)
    axe(ax, "Erreur sur a (%)", xlog=False); ax.set_xlabel("Température réelle − 22 °C supposés (°C)")
    ax.set_title("b) Température (viscosité de l'eau)", fontsize=10, pad=28)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig13_paroi_temperature.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------------------------------------ L. métriques du TP
def fig_metriques_tp(a_reel):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    for conf, c in (("20X", C20), ("40X", C40)):
        xs, ea, sa = [], [], []
        for f in sorted(glob.glob(f"sim/{conf}_*_base.pkl")):
            d = pickle.load(open(f, "rb")); a = d["a"]
            ab = [p["a"] * 1e6 for r in d["reps"] if r["out"] for p in r["out"][3] if p["raison"] == "non" and np.isfinite(p["a"]) and p["a"] > 0]
            if len(ab) < 10:
                continue
            ab = np.array(ab)
            xs.append(a); ea.append(100 * np.mean(np.abs(ab - a)) / a); sa.append(100 * np.std(ab) / a)
        o = np.argsort(xs)
        axs[0].plot(np.array(xs)[o], np.array(ea)[o], "-o", color=c, ms=4, lw=1.6, label=f"simulation {conf}, 140 images")
        axs[1].plot(np.array(xs)[o], np.array(sa)[o], "-o", color=c, ms=4, lw=1.6, label=f"simulation {conf}, 140 images")
    a = 0.475
    axs[0].plot(a, 100 * np.mean(np.abs(a_reel - a)) / a, "D", color=C20, mec=TXT, ms=7, label="test réel (74 billes)")
    axs[1].plot(a, 100 * np.std(a_reel) / a, "D", color=C20, mec=TXT, ms=7, label="test réel (74 billes)")
    axs[1].plot(a, 100 * 0.7413 * np.subtract(*np.percentile(a_reel, (75, 25))) / a, "D", color="white", mec=C20, ms=7,
                label="test réel, écart type robuste (quartiles)")
    axs[0].plot(1.0, 3.6, "*", color=TXT, ms=11, label="prédiction du TP (1 µm, 1500 images)")
    axs[1].plot(1.0, 4.5, "*", color=TXT, ms=11, label="prédiction du TP (1 µm, 1500 images)")
    ea_r = 100 * np.mean(np.abs(a_reel - a)) / a; sa_r = 100 * np.std(a_reel) / a
    print("réel E_a/a", ea_r, "σ_a/a", sa_r)
    for ax, v in ((axs[0], ea_r), (axs[1], sa_r)):
        if v > 70:
            ax.annotate(f"test réel : {virg(v, 0)} %\n(hors graphique)", (a, 69), xytext=(0.045, 62), fontsize=7.5,
                        color=TXT2, arrowprops=dict(arrowstyle="-", color=GRIS, lw=0.7))
    for ax, t, yl in ((axs[0], "a) E_a / a : erreur moyenne par bille", "E_a / a (%)"), (axs[1], "b) σ_a / a : dispersion par bille", "σ_a / a (%)")):
        axe(ax, yl); ax.set_xlabel("Rayon a (µm)"); ax.set_title(t, fontsize=10); ax.set_ylim(0, 70)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left" if ax is axs[0] else "upper right")
    fig.suptitle("Métriques du travail préparatoire, par bille : simulation, prédiction du TP et test réel",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(f"{OUT}/fig14_metriques_TP.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    quoi = sys.argv[1:] or ["reelles", "pixels", "revue", "sims", "simreel", "hist", "msd", "loc", "long", "calcul", "obj", "paroi", "tp"]
    a_reel = None
    for q in quoi:
        if q == "hist" or (q == "tp" and a_reel is None):
            a_reel = fig_histogramme()
        f = {"reelles": fig_images_reelles, "pixels": fig_pixels_bits, "revue": fig_revue_manuelle, "sims": fig_simulations,
             "simreel": fig_sim_vs_reel, "msd": fig_msd, "loc": fig_loc_snr, "long": fig_longueurs, "calcul": fig_temps_calcul,
             "obj": fig_objectifs, "paroi": fig_paroi}.get(q)
        if q == "tp":
            fig_metriques_tp(a_reel)
        elif f:
            print(q, f())
        print("ok", q, flush=True)
