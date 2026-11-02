import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile                      # pip install tifffile
from pathlib import Path
from scipy import ndimage as ndi
from scipy.optimize import curve_fit, linear_sum_assignment

DOSSIER_VIDEOS = Path(r"M3_Microscope\videos_20261026")
NOM_VIDEO = "20X.tif"        # ou "20X.tiff", "20X.mp4", ou un nom de dossier contenant un TIFF par frame
# pour dense_up.mp4, on ne détecte aucune particule
# pour gab.mp4, on ne détecte aucune particule
# pour 20X on détecte bien les 6 particules : a = 1.257 µm  ± 0.088 µm (σ_PSF = 0.008 ⊕ σ_stat = 0.088), 6 particules, χ²_red = 1.76
# dense_up_contraste.mp4, grossissement 4X, on détecte 15 particules : a = 0.400 µm  ± 0.022 µm (σ_PSF = 0.008 ⊕ σ_stat = 0.021), 17 particules, χ²_red = 349.89
# dense_up_contraste.mp4, grossissement 10X, on détecte 15 particules : 2.499 µm  ± 0.140 µm (σ_PSF = 0.052 ⊕ σ_stat = 0.130), 17 particules, χ²_red = 349.89
# constantes
kB = 1.380649e-23          # J/K
T = 22 + 273               # K
ETA = 0.95e-3              # Pa·s (eau à 22 °C)
GROSSISSEMENT = 20         # X
TAILLE_PIXEL = 1.55e-6     # m
REDUCTION = 4056 / 640     # nbr pixels image / nbr pixels capteur (car on n'est pas à la résolution maximale)
                           # -> mettre 1 si les TIFF sont à pleine résolution
PIXEL_EFF = TAILLE_PIXEL * REDUCTION / GROSSISSEMENT        # m/pixel dans le plan objet

EXTENSIONS_TIFF = {".tif", ".tiff"}
FPS_TIFF = 30.0            # un TIFF ne contient généralement pas de fps : à adapter à ton acquisition

#############################################################################################################
#                                       CHARGEMENT DES IMAGES
#############################################################################################################

def charger_video(chemin):
    """Lit toute la vidéo et retourne (frames, fps).
    frames : tableau uint8 de forme (n_frames, hauteur, largeur), en niveaux de gris.
    """
    cap = cv2.VideoCapture(str(chemin))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {chemin}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if not frames:
        raise RuntimeError("Aucune frame lue dans la vidéo.")
    return np.stack(frames), fps


def _vers_gris_uint8(stack):
    """Convertit un tableau TIFF quelconque (h,w), (n,h,w) ou (n,h,w,c), de type uint8/uint16/float,
    en niveaux de gris uint8 (n,h,w). Les images 16 bits sont mises à l'échelle avec un
    facteur GLOBAL (le même pour toutes les frames) pour ne pas fausser l'intensité d'une frame à l'autre."""
    stack = np.asarray(stack)
    if stack.ndim == 2:                                   # une seule image en niveaux de gris
        stack = stack[None]
    elif stack.ndim == 3 and stack.shape[-1] in (3, 4) and stack.shape[1] > 4:
        stack = stack[None]                               # une seule image couleur (h,w,c)
    if stack.ndim == 4:                                   # couleur -> gris
        if stack.shape[-1] >= 3:
            stack = stack[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])  # RGB (tifffile)
        else:
            stack = stack[..., 0]
    if stack.ndim != 3:
        raise ValueError(f"Forme TIFF non gérée : {stack.shape}")

    if stack.dtype == np.uint8:
        return stack
    s = stack.astype(float)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return np.zeros(stack.shape, dtype=np.uint8)
    return np.round(255 * (s - lo) / (hi - lo)).astype(np.uint8)


def charger_tiff(chemin, fps=FPS_TIFF):
    """Charge un TIFF multi-pages (pile d'images) OU un dossier contenant un TIFF par frame.
    Retourne (frames, fps) comme charger_video."""
    chemin = Path(chemin)
    if chemin.is_dir():
        fichiers = sorted(p for p in chemin.iterdir() if p.suffix.lower() in EXTENSIONS_TIFF)
        if not fichiers:
            raise FileNotFoundError(f"Aucun fichier TIFF dans : {chemin}")
        stack = np.stack([tifffile.imread(str(f)) for f in fichiers])
    else:
        if not chemin.exists():
            raise FileNotFoundError(f"Impossible d'ouvrir le fichier : {chemin}")
        with tifffile.TiffFile(str(chemin)) as tif:
            stack = tif.asarray()
            # fps éventuellement stocké dans les métadonnées ImageJ
            meta = tif.imagej_metadata or {}
            if meta.get("finterval"):
                fps = 1.0 / float(meta["finterval"])
            elif meta.get("fps"):
                fps = float(meta["fps"])
    frames = _vers_gris_uint8(stack)
    if len(frames) == 0:
        raise RuntimeError("Aucune frame lue dans le TIFF.")
    return frames, fps


def charger_media(chemin, fps_tiff=FPS_TIFF):
    """Choisit automatiquement le lecteur : TIFF (fichier ou dossier) ou vidéo (mp4, avi...)."""
    chemin = Path(chemin)
    if chemin.is_dir() or chemin.suffix.lower() in EXTENSIONS_TIFF:
        return charger_tiff(chemin, fps_tiff)
    return charger_video(chemin)


def stats_video(frames, fps):
    n, h, w = frames.shape
    return {
        "n_frames": n,
        "largeur": w,
        "hauteur": h,
        "fps": fps,
        "duree_s": n / fps if fps > 0 else float("nan"),
    }


def afficher_frame(frames, indice, fps=None):
    """Affiche la frame `indice` (0 = première) avec une barre d'intensité."""
    n = len(frames)
    if not 0 <= indice < n:
        raise IndexError(f"Indice {indice} hors de [0, {n - 1}]")

    gris = frames[indice]
    titre = f"Frame {indice}/{n - 1}"
    if fps:
        titre += f"  (t = {indice / fps:.3f} s)"

    fig, ax = plt.subplots()
    im = ax.imshow(gris, cmap="gray", vmin=0, vmax=255)
    ax.set_title(titre)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    fig.colorbar(im, ax=ax, label="Intensité (8 bits)")
    plt.show()
    return gris


#############################################################################################################
#                                       DÉTECTION DES PARTICULES
#############################################################################################################


def detecter_particules(image, sigma=1.5, k=5.0, distance_min=5):
    """Détecte les PSF dans une image en niveaux de gris.

    sigma        : lissage gaussien (pixels) pour réduire le bruit avant la détection
    k            : seuil = fond + k * bruit (bruit estimé par le MAD, robuste aux particules)
    distance_min : distance minimale entre deux particules (pixels)

    Retourne un tableau (N, 3) de colonnes [y, x, intensité lissée].
    """
    img = image.astype(float)
    lisse = ndi.gaussian_filter(img, sigma)

    # Estimation robuste du fond et du bruit (la médiane ignore les rares pixels brillants)
    fond = np.median(lisse)
    bruit = 1.4826 * np.median(np.abs(lisse - fond))
    seuil = fond + k * bruit

    # Maxima locaux au-dessus du seuil
    max_local = lisse == ndi.maximum_filter(lisse, size=2 * distance_min + 1)
    masque = max_local & (lisse > seuil)

    # Un plateau (pixels saturés) donne plusieurs pixels : on les regroupe en un seul
    etiquettes, n = ndi.label(masque, structure=np.ones((3, 3)))
    if n == 0:
        return np.empty((0, 3))
    centres = ndi.center_of_mass(lisse, etiquettes, range(1, n + 1))
    intensites = ndi.maximum(lisse, etiquettes, range(1, n + 1))
    return np.array([[y, x, I] for (y, x), I in zip(centres, intensites)])


def compter_particules(image, indice=None, **kwargs):
    """Détecte les particules, affiche leur nombre et retourne les positions."""
    particules = detecter_particules(image, **kwargs)
    etiquette = f"Frame {indice} : " if indice is not None else ""
    print(f"{etiquette}{len(particules)} particule(s) détectée(s)")
    return particules


def afficher_detection(image, particules, indice=None):
    """Affiche l'image avec un cercle autour de chaque particule détectée."""
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    if len(particules):
        ax.scatter(particules[:, 1], particules[:, 0], s=120,
                   facecolors="none", edgecolors="r", linewidths=1)
    titre = f"{len(particules)} particule(s)"
    if indice is not None:
        titre = f"Frame {indice} : " + titre
    ax.set_title(titre)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    plt.show()


def compter_sur_toutes_les_frames(frames, **kwargs):
    """Retourne le nombre de particules détectées dans chaque frame."""
    comptes = np.array([len(detecter_particules(f, **kwargs)) for f in frames])
    print(f"Particules par frame : min={comptes.min()}, max={comptes.max()}, "
          f"moyenne={comptes.mean():.1f}")
    return comptes

#############################################################################################################
#                                       TRACKING ET CALCUL DU RAYON
#############################################################################################################

# ---------- σ de la PSF mesuré sur la vidéo ----------
def sigma_demi_hauteur(img, x, y, fond, demi=8):
    """Première estimation de σ: aire des pixels au-dessus de la mi-hauteur du pic (aire = π (LMH/2)²)."""
    sub, X, Y = region_interet(img, x, y, demi)
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    pic = sub[iy, ix] - fond
    if pic <= 0:
        return 1.0
    etiq, _ = ndi.label((sub - fond) > 0.5 * pic)
    aire = np.sum(etiq == etiq[iy, ix])
    fwhm = 2 * np.sqrt(aire / np.pi)
    return max(fwhm / (2 * np.sqrt(2 * np.log(2))), 0.5)


def region_interet(img, cx, cy, demi_region):
    x0, x1 = max(int(round(cx)) - demi_region, 0), min(int(round(cx)) + demi_region + 1, img.shape[1])
    y0, y1 = max(int(round(cy)) - demi_region, 0), min(int(round(cy)) + demi_region + 1, img.shape[0])
    Y, X = np.mgrid[y0:y1, x0:x1]
    return img[y0:y1, x0:x1].astype(float), X, Y

def _gauss2d(coords, A, x0, y0, sigma, b):
    X, Y = coords
    return (A * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)) + b).ravel()

def msd_temporel(pos, lag_max):
    """MSD moyenné dans le temps, lags 1..lag_max (ignore les NaN)."""
    lags = np.arange(1, lag_max + 1)
    msd = np.full(lag_max, np.nan)
    for i, k in enumerate(lags):
        d2 = np.sum((pos[k:] - pos[:-k])**2, axis=1)
        d2 = d2[np.isfinite(d2)]
        if d2.size:
            msd[i] = d2.mean()
    return lags, msd


def ajuste_msd(pos, dt, lag_max):
    """MSD = 2N D t + 2N σ_exp² (Éq. 7). Retourne D (m²/s) et l'offset (m²)."""
    lags, msd = msd_temporel(pos, lag_max)
    ok = np.isfinite(msd)
    if ok.sum() < 2:
        return np.nan, np.nan
    pente, offset = np.polyfit(lags[ok] * dt, msd[ok], 1)
    return pente / (2 * pos.shape[1]), offset

def fit_gauss_local(img, x0, y0, fond, sigma_init):
    """Retourne [x, y, σ_x, σ_y, σ_PSF] en pixels, ou None si l'ajustement échoue."""
    demi = max(4, int(np.ceil(3 * sigma_init)))
    sub, X, Y = region_interet(img, x0, y0, demi)
    if sub.shape != (2 * demi + 1, 2 * demi + 1):      # trop près du bord
        return None
    p0 = [max(sub.max() - fond, 1.0), x0, y0, sigma_init, fond]
    bornes = ([0, X.min(), Y.min(), 0.3, 0], [np.inf, X.max(), Y.max(), 4 * demi, np.inf])
    try:
        p, cov = curve_fit(_gauss2d, (X, Y), sub.ravel(), p0=p0, bounds=bornes)
    except (RuntimeError, ValueError):
        return None
    err = np.sqrt(np.diag(cov))
    if not np.all(np.isfinite(err)) or err[1] > 1 or err[2] > 1 or p[3] < 0.31:
        return None
    return np.array([p[1], p[2], err[1], err[2], p[3]])


def estimer_sigma_psf(frames, n_frames_max=30, **kw_detection):
    """σ de la PSF (pixels) de CETTE vidéo: médiane des σ ajustés librement sur les pics détectés.
    Retourne (σ médian, dispersion MAD, nombre de PSF ajustées)."""
    indices = np.linspace(0, len(frames) - 1, min(n_frames_max, len(frames))).astype(int)
    sigmas = []
    for i in indices:
        img = frames[i].astype(float)
        fond = np.median(img)
        for y, x, _ in detecter_particules(frames[i], **kw_detection):
            r = fit_gauss_local(img, x, y, fond, sigma_demi_hauteur(img, x, y, fond))
            if r is not None:
                sigmas.append(r[4])
    if not sigmas:
        raise RuntimeError("Aucune PSF ajustée: revoir les paramètres de détection.")
    sigmas = np.array(sigmas)
    med = np.median(sigmas)
    return med, 1.4826 * np.median(np.abs(sigmas - med)), len(sigmas)


def localiser_toutes_frames(frames, sigma_psf, **kw_detection):
    detections = []
    for im in frames:
        img = im.astype(float)
        fond = np.median(img)
        liste = [r for y, x, _ in detecter_particules(im, **kw_detection)
                 if (r := fit_gauss_local(img, x, y, fond, sigma_psf)) is not None]
        detections.append(np.array(liste).reshape(-1, 5))
    return detections


# ---------- Incertitudes analytiques (aucun tirage aléatoire) ----------
def _paires_valides(pos, lag_max):
    """Pour chaque lag k: indices i où les positions i et i+k existent (frames manquantes = NaN)."""
    ok = np.isfinite(pos[:, 0])
    idx = {}
    for k in range(1, lag_max + 1):
        i = np.flatnonzero(ok[:-k] & ok[k:])
        if i.size:
            idx[k] = i
    return idx


def _coeffs_regression(ks, dt):
    """Pente de la régression linéaire = Σ c_k · MSD_k."""
    t = ks * dt
    return (t - t.mean()) / np.sum((t - t.mean())**2)


def incertitude_D_psf(pos, inc, dt, lag_max):
    """Propagation linéaire des incertitudes de localisation (σ_x, σ_y de l'ajustement de chaque image)
    jusqu'à D, à travers le MSD et la régression: σ_D² = Σ_j (∂D/∂x_j)² σ_xj² + (∂D/∂y_j)² σ_yj²."""
    idx = _paires_valides(pos, lag_max)
    ks = np.array(sorted(idx))
    c = _coeffs_regression(ks, dt)
    n_dim = pos.shape[1]
    g = np.zeros_like(pos)                                    # ∂D/∂position de chaque image
    for ck, k in zip(c, ks):
        i = idx[k]
        delta = pos[i + k] - pos[i]
        w = ck / (n_dim * len(i))
        g[i + k] += w * delta
        g[i] -= w * delta
    return np.sqrt(np.nansum(g**2 * inc**2))


def incertitude_D_stat(pos, D, dt, lag_max):
    """Dispersion statistique de D pour un mouvement brownien de coefficient D observé sur ces images:
    variance exacte de la régression du MSD. Les MSD de lags différents sont corrélés (pas communs):
    Cov(X², Y²) = 2 Cov(X, Y)² pour des déplacements gaussiens, avec Cov(X, Y) = 2 D dt × (pas partagés)."""
    idx = _paires_valides(pos, lag_max)
    ks = np.array(sorted(idx))
    c = _coeffs_regression(ks, dt)
    n_dim = pos.shape[1]
    s2 = 2 * D * dt                                           # variance d'un pas, par axe
    C = np.zeros((len(ks), len(ks)))                          # covariance entre MSD_k et MSD_l
    for a, k in enumerate(ks):
        for b, l in enumerate(ks):
            I, J = idx[k][:, None], idx[l][None, :]
            partages = np.clip(np.minimum(I + k, J + l) - np.maximum(I, J), 0, None)
            C[a, b] = 2 * n_dim * s2**2 * np.sum(partages**2) / (len(idx[k]) * len(idx[l]))
    return np.sqrt(c @ C @ c) / (2 * n_dim)


def rayon_piste(pos_px, inc_px, dt, pixel_eff=PIXEL_EFF, lag_max=3):
    pos, inc = pos_px * pixel_eff, inc_px * pixel_eff
    n_dim = pos.shape[1]
    D, offset = ajuste_msd(pos, dt, lag_max)
    if not np.isfinite(D) or D <= 0:
        return None
    a = kB * T / (6 * np.pi * ETA * D)                        # Stokes-Einstein (Éq. 5)
    sigma_exp = np.sqrt(max(offset, 0) / (2 * n_dim))         # erreur de localisation effective (Éq. 7)
    inc_psf = a * incertitude_D_psf(pos, inc, dt, lag_max) / D          # δa/a = δD/D
    inc_stat = a * incertitude_D_stat(pos, D, dt, lag_max) / D
    return dict(a=a, D=D, sigma_exp=sigma_exp, inc_psf=inc_psf, inc_stat=inc_stat,
                inc=np.hypot(inc_psf, inc_stat))              # quadrature


def lier_trajectoires(detections, dist_max_px=3.0, trou_max=2):
    """dist_max_px : déplacement max entre deux frames (≈ 4·√(2 D dt)/pixel_eff avec le plus grand D attendu)
    trou_max     : nb de frames consécutives où une particule peut disparaître (portée × √écart)."""
    pistes, actives = [], []
    for f, det in enumerate(detections):
        actives = [k for k in actives if f - pistes[k]["frames"][-1] <= trou_max + 1]
        liees = set()
        if actives and len(det):
            derniere = np.array([pistes[k]["data"][-1][:2] for k in actives])
            ecart = np.array([f - pistes[k]["frames"][-1] for k in actives])
            d = np.linalg.norm(derniere[:, None, :] - det[None, :, :2], axis=2)
            cout = np.where(d <= dist_max_px * np.sqrt(ecart)[:, None], d, 1e6)
            for i, j in zip(*linear_sum_assignment(cout)):
                if cout[i, j] < 1e6:
                    k = actives[i]
                    pistes[k]["frames"].append(f)
                    pistes[k]["data"].append(det[j])
                    liees.add(j)
        for j in range(len(det)):
            if j not in liees:
                pistes.append({"frames": [f], "data": [det[j]]})
                actives.append(len(pistes) - 1)
    return pistes

def pistes_en_tableaux(pistes, n_min=20):
    """Garde les pistes d'au moins n_min positions. pos_px: (n, 2) avec NaN aux frames manquantes."""
    sorties = []
    for p in pistes:
        if len(p["frames"]) < n_min:
            continue
        f0, f1 = p["frames"][0], p["frames"][-1]
        pos = np.full((f1 - f0 + 1, 2), np.nan)
        inc = pos.copy()
        for f, d in zip(p["frames"], p["data"]):
            pos[f - f0] = d[:2]
            inc[f - f0] = d[2:4]
        sorties.append({"f0": f0, "pos_px": pos, "inc_px": inc})
    return sorties


def analyser_pistes(pistes_tab, dt, **kw):
    resultats = []
    for num, p in enumerate(pistes_tab):
        r = rayon_piste(p["pos_px"], p["inc_px"], dt, **kw)
        if r is None:
            print(f"Piste {num} ignorée (D <= 0)")
            continue
        r.update(piste=num, f0=p["f0"], pos_px=p["pos_px"], n_pts=int(np.isfinite(p["pos_px"][:, 0]).sum()))
        resultats.append(r)
    return resultats


def rayon_final(resultats):
    """Moyenne pondérée, poids = 1/(incertitude RELATIVE)² (pondérer par 1/σ² biaiserait vers les petits rayons,
    car σ est proportionnelle à a). σ_PSF et σ_stat sont propagées séparément puis sommées en quadrature."""
    a = np.array([r["a"] for r in resultats])
    s_psf = np.array([r["inc_psf"] for r in resultats])
    s_stat = np.array([r["inc_stat"] for r in resultats])
    s = np.hypot(s_psf, s_stat)
    w = (a / s)**2
    W = w.sum()
    a_fin = np.sum(w * a) / W
    inc_psf = np.sqrt(np.sum(w**2 * s_psf**2)) / W
    inc_stat = np.sqrt(np.sum(w**2 * s_stat**2)) / W
    n = len(a)
    chi2_red = np.sum((a - a_fin)**2 / s**2) / (n - 1) if n > 1 else np.nan
    return dict(a=a_fin, inc_psf=inc_psf, inc_stat=inc_stat, inc=np.hypot(inc_psf, inc_stat),
                chi2_red=chi2_red, n=n)


def imprimer_resultats(resultats, final):
    print(f"{'piste':>5s} {'n_pts':>5s} {'D (µm²/s)':>10s} {'σ_exp (nm)':>10s} {'a (µm)':>8s} "
          f"{'σ_PSF':>7s} {'σ_stat':>7s} {'σ_tot':>7s}")
    for r in resultats:
        print(f"{r['piste']:5d} {r['n_pts']:5d} {r['D']*1e12:10.3f} {r['sigma_exp']*1e9:10.0f} {r['a']*1e6:8.3f} "
              f"{r['inc_psf']*1e6:7.3f} {r['inc_stat']*1e6:7.3f} {r['inc']*1e6:7.3f}")
    print(f"\nRayon final : {final['a']*1e6:.3f} µm  ± {final['inc']*1e6:.3f} µm "
          f"(σ_PSF = {final['inc_psf']*1e6:.3f} ⊕ σ_stat = {final['inc_stat']*1e6:.3f}), "
          f"{final['n']} particules, χ²_red = {final['chi2_red']:.2f}")


def afficher_resultats(frames, resultats, final):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].imshow(frames.max(axis=0), cmap="gray")
    for r in resultats:
        ax[0].plot(r["pos_px"][:, 0], r["pos_px"][:, 1], "-", lw=1.2)
        i0 = np.flatnonzero(np.isfinite(r["pos_px"][:, 0]))[0]
        ax[0].text(*r["pos_px"][i0], str(r["piste"]), color="yellow", fontsize=9)
    ax[0].set(xlabel="x (pixels)", ylabel="y (pixels)", title="Trajectoires (max. sur toutes les frames)")

    num = [r["piste"] for r in resultats]
    a = np.array([r["a"] for r in resultats]) * 1e6
    ax[1].errorbar(num, a, yerr=[r["inc"] * 1e6 for r in resultats], fmt="o", capsize=4, label="a ± σ_tot (par piste)")
    ax[1].axhline(final["a"] * 1e6, color="tab:green")
    ax[1].axhspan((final["a"] - final["inc"]) * 1e6, (final["a"] + final["inc"]) * 1e6,
                  color="tab:green", alpha=0.2, label=f"final = {final['a']*1e6:.2f} ± {final['inc']*1e6:.2f} µm")
    ax[1].set(xlabel="piste", ylabel="rayon (µm)", title="Rayon par particule")
    ax[1].legend()
    plt.tight_layout()
    plt.show()

#############################################################################################################
#                                                 MAIN
#############################################################################################################

if __name__ == "__main__":
    chemin = DOSSIER_VIDEOS / NOM_VIDEO
    frames, fps = charger_media(chemin)          # vidéo (mp4, avi...) ou TIFF (fichier / dossier)
    s = stats_video(frames, fps)

    print(f"Fichier          : {chemin}")
    print(f"Nombre de frames : {s['n_frames']}")
    print(f"Résolution       : {s['largeur']} x {s['hauteur']} pixels")
    print(f"Frame rate       : {s['fps']:.2f} fps")
    print(f"Durée            : {s['duree_s']:.2f} s")

    INDICE_FRAME = 120   # à modifier
    gris = afficher_frame(frames, INDICE_FRAME, s["fps"])

    particules = compter_particules(gris, indice=INDICE_FRAME)
    afficher_detection(gris, particules, indice=INDICE_FRAME)

    #comptes = compter_sur_toutes_les_frames(frames)

    #print(f"Intensité : min={gris.min()}, max={gris.max()}, moyenne={gris.mean():.1f}")

    dt = 1 / fps
    sigma_psf, disp_psf, n_fit = estimer_sigma_psf(frames)
    print(f"σ_PSF mesuré : {sigma_psf:.2f} ± {disp_psf:.2f} px ({sigma_psf*PIXEL_EFF*1e6:.2f} µm), {n_fit} PSF")

    detections = localiser_toutes_frames(frames, sigma_psf)
    pistes = lier_trajectoires(detections, dist_max_px=3.0, trou_max=2)
    pistes_tab = pistes_en_tableaux(pistes, n_min=20)
    resultats = analyser_pistes(pistes_tab, dt, lag_max=3)
    final = rayon_final(resultats)
    imprimer_resultats(resultats, final)
    afficher_resultats(frames, resultats, final)