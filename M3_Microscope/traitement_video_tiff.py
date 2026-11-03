"""Traitement des acquisitions du microscope : rayon des billes par suivi de particules (MSD + Stokes-Einstein).

Au lancement, deux choix dans le terminal :
  1. le FICHIER, parmi ceux du dossier TechEx/gab (piles .tif de acquisition_raspberry.py,
     dossiers de PNG de l'option E, ou anciennes vidéos .mp4);
  2. l'ANALYSE :
     1 = analyse « classique » : une image avec ses détections, puis trajectoires et rayon par particule;
     2 = analyse dynamique, représentation reprise de interface_suivi.py : la vidéo joue en boucle,
         chaque image annotée (temps, échelle µm/pixel, grossissement, exposition, fréquence, profondeur
         de bit, barre d'échelle); chaque bille suivie a sa trajectoire en couleur, un cercle de 2σ (taille
         apparente) et son rayon estimé; cercles rouges = autres détections. À droite : carte « rayon moyen »,
         MSD avec ajustements et rayon par piste (pistes visibles mises en évidence); dessous : tableau.
         Espace = pause · e = exporter <nom>_annote.mp4 + <nom>_analyse.csv à côté du fichier.

Exclusion À LA MAIN, une fois l'analyse faite (rien n'est exclu automatiquement) :
  - analyse 1 : après la figure, taper dans le terminal les numéros des pistes à exclure / réintégrer;
  - analyse 2 : cliquer sur une bille dans la vidéo, ou sur son point dans le graphique des rayons
    (r = tout réintégrer). Le rayon moyen, la carte, le tableau et l'export se mettent à jour.
Les billes immobiles (D <= 0) sont suivies, affichées (« immobile ») et comptées dans la moyenne tant
qu'elles ne sont pas exclues. Leur rayon (a = ∞) ne se moyenne pas : dès qu'une immobile est sélectionnée,
le rayon moyen est calculé à partir de la moyenne des D (a = kT / 6πη·D_moyen). Sans immobile sélectionnée,
c'est la moyenne pondérée des rayons, comme avant.

Les paramètres d'acquisition sont lus dans le .json écrit à côté de chaque pile par acquisition_raspberry.py :
  - le temps de chaque image (horodatages du capteur) -> Δt réel, et les images perdues deviennent des trous
    dans les trajectoires au lieu de fausser le MSD;
  - la taille de pixel (pixel_effectif_capteur_um / grossissement);
  - le contenu : une mosaïque de Bayer brute (options D, E) est d'abord sommée 2x2 pour donner une image grise.
Les données restent en 16 bits (aucune conversion en 8 bits, sauf pour l'affichage).
Sans .json (mp4, TIFF externe), on retombe sur les constantes ci-dessous, avec un avertissement.
"""
import csv
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile                      # pip install tifffile
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.optimize import curve_fit, linear_sum_assignment

DOSSIER_DONNEES = Path(__file__).resolve().parent.parent / "gab"   # TechEx/gab
# pour dense_up.mp4, on ne détecte aucune particule
# pour gab.mp4, on ne détecte aucune particule
# pour 20X on détecte bien les 6 particules : a = 1.257 µm  ± 0.088 µm (σ_PSF = 0.008 ⊕ σ_stat = 0.088), 6 particules, χ²_red = 1.76
# dense_up_contraste.mp4, grossissement 4X, on détecte 15 particules : a = 0.400 µm  ± 0.022 µm (σ_PSF = 0.008 ⊕ σ_stat = 0.021), 17 particules, χ²_red = 349.89
# dense_up_contraste.mp4, grossissement 10X, on détecte 15 particules : 2.499 µm  ± 0.140 µm (σ_PSF = 0.052 ⊕ σ_stat = 0.130), 17 particules, χ²_red = 349.89
# constantes
kB = 1.380649e-23          # J/K
T = 22 + 273               # K
ETA = 0.95e-3              # Pa·s (eau à 22 °C)

# Valeurs par défaut, utilisées SEULEMENT quand il n'y a pas de .json (anciens mp4, TIFF externes)
GROSSISSEMENT = 20         # X
TAILLE_PIXEL = 1.55e-6     # m
REDUCTION = 4056 / 640     # nbr pixels capteur / nbr pixels image (mp4 de 640 px de large)
PIXEL_EFF = TAILLE_PIXEL * REDUCTION / GROSSISSEMENT        # m/pixel dans le plan objet
FPS_DEFAUT = 30.0

# Paramètres de détection et de suivi, en µm dans le plan objet (convertis en pixels selon le fichier).
# Avec les anciens mp4 (0,49 µm/px), ils redonnent exactement les valeurs en pixels d'avant (1,5 / 5 / 3 / 8 px).
LISSAGE_UM = 0.74          # lissage gaussien avant la détection
DISTANCE_MIN_UM = 2.45     # distance minimale entre deux particules
DEPLACEMENT_MAX_UM = 1.47  # déplacement maximal entre deux images consécutives
DEMI_REGION_UM = 3.9       # demi-largeur de la région utilisée pour la 1re estimation de σ
SEUIL_K = 5.0              # seuil de détection = fond + k * bruit
TROU_MAX = 5               # nb d'images manquantes tolérées dans une trajectoire (images perdues comprises)
N_MIN = 20                 # nb minimal de positions pour garder une trajectoire
LAG_MAX = 3                # nb de décalages utilisés dans l'ajustement du MSD

EXTENSIONS_TIFF = {".tif", ".tiff"}
EXTENSIONS_VIDEO = {".mp4", ".avi", ".h264", ".mov"}


#############################################################################################################
#                                       CHARGEMENT DES DONNÉES
#############################################################################################################

def somme_bayer(raw):
    """Somme des 4 pixels B, G, G, R de chaque carré 2x2 -> image grise de demi-taille."""
    return raw[0::2, 0::2] + raw[0::2, 1::2] + raw[1::2, 0::2] + raw[1::2, 1::2]


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


def _en_gris(stack):
    """Ramène un tableau (h,w), (n,h,w) ou (n,h,w,c) à (n,h,w) en GARDANT la profondeur d'origine
    (pas de conversion en 8 bits). Une pile couleur est convertie en gris (float32)."""
    stack = np.asarray(stack)
    if stack.ndim == 2:
        stack = stack[None]
    elif stack.ndim == 3 and stack.shape[-1] in (3, 4) and stack.shape[1] > 4:
        stack = stack[None]                               # une seule image couleur (h,w,c)
    if stack.ndim == 4:
        stack = (stack[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
                 if stack.shape[-1] >= 3 else stack[..., 0])
    if stack.ndim != 3:
        raise ValueError(f"Forme non gérée : {stack.shape}")
    return stack


def _lire_dossier(chemin):
    fichiers = sorted(p for p in chemin.iterdir() if p.suffix.lower() in EXTENSIONS_TIFF | {".png"})
    if not fichiers:
        raise FileNotFoundError(f"Aucune image .tif ou .png dans : {chemin}")
    lire = lambda p: (tifffile.imread(str(p)) if p.suffix.lower() in EXTENSIONS_TIFF
                      else cv2.imread(str(p), cv2.IMREAD_UNCHANGED))
    return np.stack([lire(p) for p in fichiers])


def charger_acquisition(chemin):
    """Charge une acquisition et ses paramètres. Retourne un dict :
    frames (n,h,w) · temps_s (n,) · indices (n,) numéro de créneau de chaque image (trous = images perdues)
    · dt (s) · pixel_eff (m/px) · meta (contenu du .json ou {}) · source (texte)."""
    chemin = Path(chemin)
    json_path = chemin.with_suffix(".json") if chemin.is_file() else chemin.parent / (chemin.name + ".json")
    meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}

    if chemin.is_dir():
        frames = _en_gris(_lire_dossier(chemin))
        fps_fichier = None
    elif chemin.suffix.lower() in EXTENSIONS_TIFF:
        frames = _en_gris(tifffile.imread(str(chemin)))
        fps_fichier = None
    else:
        frames, fps_fichier = charger_video(chemin)

    # Mosaïque de Bayer brute (options D et E) -> somme 2x2 en gris, pixel 2x plus grand
    bayer = "bayer" in meta.get("contenu", "").lower() and "somme" not in meta.get("contenu", "").lower()
    if bayer:
        frames = np.stack([somme_bayer(f) for f in frames])

    # Taille de pixel dans le plan objet
    if meta.get("pixel_effectif_capteur_um") and meta.get("grossissement"):
        pixel_eff = meta["pixel_effectif_capteur_um"] * (2 if bayer else 1) / float(meta["grossissement"]) * 1e-6
        source_pixel = "json"
    else:
        pixel_eff, source_pixel = PIXEL_EFF, "constantes du script"

    # Temps de chaque image : horodatages du capteur, sinon fps
    stamps = meta.get("timestamps_ns")
    if not stamps and chemin.is_dir() and (chemin / "timestamps_ns.npy").exists():
        stamps = np.load(chemin / "timestamps_ns.npy").tolist()
    if stamps and len(stamps) == len(frames):
        t = (np.asarray(stamps, dtype=np.int64) - stamps[0]) / 1e9
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / FPS_DEFAUT
        source_temps = "horodatages du json"
    else:
        fps = fps_fichier if fps_fichier and fps_fichier > 0 else FPS_DEFAUT
        dt = 1 / fps
        t = np.arange(len(frames)) * dt
        source_temps = f"fps = {fps:g} ({'fichier vidéo' if fps_fichier else 'valeur par défaut'})"
    indices = np.round(t / dt).astype(int)

    return dict(frames=frames, temps_s=t, indices=indices, dt=dt, pixel_eff=pixel_eff, meta=meta,
                bayer=bayer, source=f"pixel : {source_pixel} · temps : {source_temps}")


def parametres_pixels(pixel_eff):
    """Convertit les paramètres en µm en pixels pour ce fichier."""
    um = pixel_eff * 1e6
    return dict(sigma=LISSAGE_UM / um, k=SEUIL_K, distance_min=max(1, round(DISTANCE_MIN_UM / um))), \
        DEPLACEMENT_MAX_UM / um, max(4, round(DEMI_REGION_UM / um))


def contraste(img):
    """Bornes d'affichage robustes (les données gardent leur profondeur d'origine)."""
    bas, haut = np.percentile(img[::2, ::2], (0.5, 99.9))
    return bas, max(haut, bas + 1)


def afficher_frame(frames, indice, temps=None):
    """Affiche la frame `indice` (0 = première) avec une barre d'intensité."""
    n = len(frames)
    if not 0 <= indice < n:
        raise IndexError(f"Indice {indice} hors de [0, {n - 1}]")

    gris = frames[indice]
    titre = f"Frame {indice}/{n - 1}"
    if temps is not None:
        titre += f"  (t = {temps:.3f} s)"

    fig, ax = plt.subplots()
    bas, haut = contraste(gris)
    im = ax.imshow(gris, cmap="gray", vmin=bas, vmax=haut)
    ax.set_title(titre)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    fig.colorbar(im, ax=ax, label="Intensité (comptes)")
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
    bas, haut = contraste(image)
    ax.imshow(image, cmap="gray", vmin=bas, vmax=haut)
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


def estimer_sigma_psf(frames, n_frames_max=30, demi=8, **kw_detection):
    """σ de la PSF (pixels) de CETTE vidéo: médiane des σ ajustés librement sur les pics détectés.
    Retourne (σ médian, dispersion MAD, nombre de PSF ajustées)."""
    indices = np.linspace(0, len(frames) - 1, min(n_frames_max, len(frames))).astype(int)
    sigmas = []
    for i in indices:
        img = frames[i].astype(float)
        fond = np.median(img)
        for y, x, _ in detecter_particules(frames[i], **kw_detection):
            r = fit_gauss_local(img, x, y, fond, sigma_demi_hauteur(img, x, y, fond, demi))
            if r is not None:
                sigmas.append(r[4])
    if not sigmas:
        raise RuntimeError("Aucune PSF ajustée: revoir les paramètres de détection.")
    sigmas = np.array(sigmas)
    med = np.median(sigmas)
    return med, 1.4826 * np.median(np.abs(sigmas - med)), len(sigmas)


def localiser_toutes_frames(frames, sigma_psf, **kw_detection):
    detections = []
    for n, im in enumerate(frames, 1):
        img = im.astype(float)
        fond = np.median(img)
        liste = [r for y, x, _ in detecter_particules(im, **kw_detection)
                 if (r := fit_gauss_local(img, x, y, fond, sigma_psf)) is not None]
        detections.append(np.array(liste).reshape(-1, 5))
        if n % max(1, len(frames) // 10) == 0:
            print(f"  localisation : {n}/{len(frames)} images", flush=True)
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


def lier_trajectoires(detections, indices=None, dist_max_px=3.0, trou_max=2):
    """detections : liste (une entrée par image du fichier);
    indices      : numéro de créneau temporel de chaque image (les images perdues laissent des trous).
                   None = images consécutives.
    dist_max_px  : déplacement max entre deux créneaux consécutifs (≈ 4·√(2 D dt)/pixel_eff)
    trou_max     : nb de créneaux consécutifs où une particule peut manquer (portée × √écart)."""
    if indices is None:
        indices = range(len(detections))
    pistes, actives = [], []
    for f, det in zip(indices, detections):
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
    """Garde les pistes d'au moins n_min positions. pos_px: (n, 2) avec NaN aux créneaux manquants;
    sig_px: σ de la gaussienne ajustée à chaque créneau (taille apparente)."""
    sorties = []
    for p in pistes:
        if len(p["frames"]) < n_min:
            continue
        f0, f1 = p["frames"][0], p["frames"][-1]
        pos = np.full((f1 - f0 + 1, 2), np.nan)
        inc = pos.copy()
        sig = np.full(f1 - f0 + 1, np.nan)
        for f, d in zip(p["frames"], p["data"]):
            pos[f - f0] = d[:2]
            inc[f - f0] = d[2:4]
            sig[f - f0] = d[4]
        sorties.append({"f0": f0, "pos_px": pos, "inc_px": inc, "sig_px": sig})
    return sorties


def analyser_pistes(pistes_tab, dt, **kw):
    """Une entrée par piste d'au moins N_MIN positions. AUCUNE piste n'est écartée automatiquement :
    une bille immobile (MSD qui ne croît pas, D <= 0) est gardée et affichée avec la mention « immobile »;
    son rayon n'est pas défini (a = ∞) : si elle est sélectionnée, rayon_final passe par la moyenne des D.
    Les exclusions se font ensuite À LA MAIN (choisir_exclusions, ou clic dans l'analyse 2)."""
    resultats = []
    for num, p in enumerate(pistes_tab):
        r = rayon_piste(p["pos_px"], p["inc_px"], dt, **kw)
        if r is None:
            pos = p["pos_px"] * kw.get("pixel_eff", PIXEL_EFF)
            D, offset = ajuste_msd(pos, dt, kw.get("lag_max", 3))
            sig = np.sqrt(offset / 4) if np.isfinite(offset) and offset > 0 else np.nan
            r = dict(a=np.inf, D=D, sigma_exp=sig, inc_psf=np.nan, inc_stat=np.nan, inc=np.nan, immobile=True)
        else:
            r["immobile"] = False
        r.update(piste=num, f0=p["f0"], pos_px=p["pos_px"], sig_px=p["sig_px"],
                 n_pts=int(np.isfinite(p["pos_px"][:, 0]).sum()))
        resultats.append(r)
    return resultats


def statut(r, exclues=()):
    if r["piste"] in exclues:
        return "exclue"
    return "immobile" if r["immobile"] else "mesurée"


def pistes_moyenne(resultats, exclues=()):
    """Pistes qui entrent dans le rayon moyen : toutes celles qui ne sont pas exclues à la main,
    immobiles comprises."""
    return [r for r in resultats if statut(r, exclues) != "exclue"]


def rayon_final(resultats):
    """Rayon moyen des pistes données. None s'il n'y en a aucune.

    Sans piste immobile : moyenne pondérée des rayons, poids = 1/(incertitude RELATIVE)² (pondérer par 1/σ²
    biaiserait vers les petits rayons, car σ est proportionnelle à a). σ_PSF et σ_stat sont propagées
    séparément puis sommées en quadrature.

    Avec au moins une piste immobile (D <= 0, donc a = ∞, qui ne se moyenne pas) : on moyenne les
    coefficients de diffusion D de toutes les pistes, puis a = kT / (6πη D_moyen) (Stokes-Einstein).
    Incertitude : dispersion des D (écart-type / √n), propagée par δa/a = δD/D. Si D_moyen <= 0, a = ∞."""
    resultats = list(resultats)
    if not resultats:
        return None
    if any(r["immobile"] for r in resultats if "immobile" in r):
        D = np.array([r["D"] for r in resultats], dtype=float)
        n = len(D)
        D_moy = D.mean()
        inc_D = D.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        if D_moy > 0:
            a = kB * T / (6 * np.pi * ETA * D_moy)
            inc = a * inc_D / D_moy
        else:
            a, inc = np.inf, np.nan
        return dict(a=a, inc_psf=np.nan, inc_stat=inc, inc=inc, chi2_red=np.nan, n=n, D_moy=D_moy,
                    n_immobiles=sum(r["immobile"] for r in resultats), methode="D")
    resultats = [r for r in resultats if np.isfinite(r["a"]) and np.isfinite(r["inc"]) and r["inc"] > 0]
    if not resultats:
        return None
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
                chi2_red=chi2_red, n=n, methode="a")


def texte_rayon(final):
    if not np.isfinite(final["a"]):
        return "a = ∞ (D moyen <= 0)"
    inc = f"{final['inc'] * 1e6:.3f}" if np.isfinite(final["inc"]) else "?"
    return f"a = {final['a'] * 1e6:.3f} ± {inc} µm"


def texte_final(final):
    if final is None:
        return "Rayon final : aucune piste dans la moyenne."
    if final.get("methode") == "D":
        return (f"Rayon final : {texte_rayon(final)}  (moyenne des D sur {final['n']} pistes, dont "
                f"{final['n_immobiles']} immobile(s) : D moyen = {final['D_moy'] * 1e12:.4f} µm²/s)")
    return (f"Rayon final : {final['a']*1e6:.3f} µm  ± {final['inc']*1e6:.3f} µm "
            f"(σ_PSF = {final['inc_psf']*1e6:.3f} ⊕ σ_stat = {final['inc_stat']*1e6:.3f}), "
            f"{final['n']} particules, χ²_red = {final['chi2_red']:.2f}")


def imprimer_resultats(resultats, final, exclues=()):
    print(f"{'piste':>5s} {'n_pts':>5s} {'D (µm²/s)':>10s} {'σ_exp (nm)':>10s} {'a (µm)':>9s} "
          f"{'σ_PSF':>7s} {'σ_stat':>7s} {'σ_tot':>7s}  statut")
    for r in resultats:
        a = f"{r['a']*1e6:9.3f}" if np.isfinite(r["a"]) else f"{'∞':>9s}"
        inc = [f"{r[c]*1e6:7.3f}" if np.isfinite(r[c]) else f"{'—':>7s}" for c in ("inc_psf", "inc_stat", "inc")]
        print(f"{r['piste']:5d} {r['n_pts']:5d} {r['D']*1e12:10.3f} {r['sigma_exp']*1e9:10.0f} {a} "
              f"{' '.join(inc)}  {statut(r, exclues)}")
    print("\n" + texte_final(final))


def choisir_exclusions(resultats, exclues=None):
    """Après l'analyse : on tape les numéros des pistes à exclure (ou à réintégrer) de la moyenne,
    autant de fois qu'on veut; Entrée seule pour terminer. Retourne (exclues, final)."""
    exclues = set(exclues or ())
    ids = {r["piste"] for r in resultats}
    while True:
        final = rayon_final(pistes_moyenne(resultats, exclues))
        print()
        imprimer_resultats(resultats, final, exclues)
        rep = input("\nPistes à exclure ou à réintégrer (ex. 0 4 8) · Entrée = terminé : ").strip()
        if not rep:
            if exclues:
                print(f"Pistes exclues à la main : {sorted(exclues)}")
            return exclues, final
        for t in rep.replace(",", " ").split():
            if t.isdigit() and int(t) in ids:
                exclues ^= {int(t)}
            else:
                print(f"  « {t} » n'est pas un numéro de piste ({sorted(ids)}).")


def tracer_rayons(ax, resultats, final, exclues=()):
    """Rayon par piste : en couleur les pistes de la moyenne, en gris (×) les pistes exclues à la main.
    Les pistes immobiles (a = ∞) ne peuvent pas être placées sur l'axe : elles sont nommées dans le titre."""
    finis = [r for r in resultats if np.isfinite(r["a"])]
    for r in finis:
        exclue = r["piste"] in exclues
        ax.errorbar(r["piste"], r["a"] * 1e6, yerr=r["inc"] * 1e6, fmt="x" if exclue else "o", capsize=4,
                    color="0.65" if exclue else "tab:blue", alpha=0.7 if exclue else 1)
    if final and np.isfinite(final["a"]):
        ax.axhline(final["a"] * 1e6, color="tab:green")
        ax.axhspan((final["a"] - final["inc"]) * 1e6, (final["a"] + final["inc"]) * 1e6, color="tab:green",
                   alpha=0.2, label="final : " + texte_rayon(final)
                   + (" (moyenne des D)" if final.get("methode") == "D" else ""))
        ax.legend()
    a = np.array([r["a"] for r in finis])
    if len(a) and a.max() / a.min() > 20:                    # une bille quasi immobile écrase l'échelle
        ax.set_yscale("log")
    immobiles = [r["piste"] for r in resultats if r["immobile"]]
    ax.set(xlabel="piste", ylabel="rayon (µm)",
           title="Rayon par particule (× gris = exclue)" + (f"\nimmobiles (a = ∞) : {immobiles}" if immobiles else ""))


def afficher_resultats(frames, resultats, final, exclues=()):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    proj = frames.max(axis=0)
    bas, haut = contraste(proj)
    ax[0].imshow(proj, cmap="gray", vmin=bas, vmax=haut)
    for r in resultats:
        st = statut(r, exclues)
        ax[0].plot(r["pos_px"][:, 0], r["pos_px"][:, 1], "--" if st == "exclue" else "-", lw=1.2,
                   color="0.6" if st == "exclue" else None)
        i0 = np.flatnonzero(np.isfinite(r["pos_px"][:, 0]))[0]
        ax[0].text(*r["pos_px"][i0], str(r["piste"]) + ("" if st == "mesurée" else f" ({st})"),
                   color="0.7" if st == "exclue" else "yellow", fontsize=9)
    ax[0].set(xlabel="x (pixels)", ylabel="y (pixels)", title="Trajectoires (max. sur toutes les frames)")
    tracer_rayons(ax[1], resultats, final, exclues)
    plt.tight_layout()
    plt.show()


#############################################################################################################
#                                       ANALYSE DYNAMIQUE (option 2)
#    Représentation reprise de interface_suivi.py : images annotées (temps, échelle, conditions, barre
#    d'échelle), trajectoires en couleur, carte « rayon moyen », MSD avec ajustements, rayon par piste, tableau.
#############################################################################################################

PALETTE = [(255, 165, 0), (0, 200, 255), (0, 230, 0), (255, 0, 255), (255, 255, 0),
           (110, 110, 255), (160, 255, 110), (255, 90, 90), (0, 255, 255), (255, 130, 210)]   # RVB
ROUGE_DET = (255, 60, 60)
GRIS_EXCLUE = (150, 150, 150)
LARGEUR_AFFICHAGE_MAX = 1400            # au-delà, la vidéo est réduite pour l'affichage (pas pour l'export)


def couleur_piste(pid):
    return PALETTE[pid % len(PALETTE)]


def couleur_mpl(pid):
    return tuple(v / 255 for v in couleur_piste(pid))


@lru_cache(maxsize=8)
def _police(taille):
    for nom in ("DejaVuSans.ttf", "arial.ttf", "segoeui.ttf", "Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(nom, taille)
        except OSError:
            continue
    return ImageFont.load_default()


def _longueur_ronde(cible_um):
    """Plus grande longueur « ronde » (1, 2 ou 5 × 10^n) inférieure ou égale à la cible."""
    e = 10 ** math.floor(math.log10(cible_um))
    for m in (5, 2, 1):
        if m * e <= cible_um:
            return m * e
    return e


def conditions_acquisition(acq):
    """Conditions incrustées sur chaque image, tirées du .json (sinon des constantes du script)."""
    meta = acq["meta"]
    expo = meta.get("exposure_us") or []
    expo = [e for e in expo if e]
    if "somme" in meta.get("contenu", "").lower() or acq["bayer"]:
        bits = "12 bits (somme 2x2 : 0–16 380)"
    elif meta:
        bits = "12 bits"
    else:
        bits = "8 bits" if acq["frames"].dtype == np.uint8 else "16 bits"
    return dict(echelle_um=acq["pixel_eff"] * 1e6,
                grossissement=meta.get("grossissement", GROSSISSEMENT),
                exposition_ms=np.median(expo) / 1e3 if expo else None,
                fps=1 / acq["dt"], bits=bits)


def composer_image(acq, k, resultats, detections, bornes, s=1.0, exclues=()):
    """Image k -> image RVB 8 bits annotée, à l'échelle s (1 = résolution native, pour l'export).
    Trajectoires colorées, cercle de 2σ (taille apparente) + étiquette « piste : a » pour chaque bille suivie
    (« immobile » si D <= 0; en gris avec « exclue » si exclue à la main), cercles rouges pour les autres
    détections, conditions d'acquisition et barre d'échelle."""
    bas, haut = bornes
    img = acq["frames"][k]
    v8 = np.clip((img.astype(np.float32) - bas) * (255.0 / (haut - bas)), 0, 255).astype(np.uint8)
    if s != 1.0:
        v8 = cv2.resize(v8, (round(v8.shape[1] * s), round(v8.shape[0] * s)), interpolation=cv2.INTER_AREA)
    rvb = cv2.cvtColor(v8, cv2.COLOR_GRAY2RGB)
    h, w = rvb.shape[:2]
    ep = max(1, round(w / 700))
    slot = acq["indices"][k]

    etiquettes, suivis = [], []
    for r in resultats:
        j = slot - r["f0"]
        if not 0 <= j < len(r["pos_px"]):
            continue
        st = statut(r, exclues)
        c = GRIS_EXCLUE if st == "exclue" else couleur_piste(r["piste"])
        pts = r["pos_px"][:j + 1]
        pts = pts[np.isfinite(pts[:, 0])] * s
        if len(pts) > 1:
            cv2.polylines(rvb, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, c, ep, cv2.LINE_AA)
        if np.isfinite(r["pos_px"][j, 0]):
            x, y = r["pos_px"][j]
            rayon = max(3.0, 2 * r["sig_px"][j] * s)
            cv2.circle(rvb, (round(x * s), round(y * s)), round(rayon), c, ep + 1, cv2.LINE_AA)
            texte = f"{r['a'] * 1e6:.2f} µm" if np.isfinite(r["a"]) else "immobile"
            if st == "exclue":
                texte += " (exclue)"
            etiquettes.append(((x * s + rayon + 3, y * s - rayon - 2), f"{r['piste']} : {texte}", c))
            suivis.append((x, y))
    det = detections[k]
    for x, y in det[:, :2] if len(det) else []:
        if not suivis or min(np.hypot(x - a, y - b) for a, b in suivis) > 1:
            cv2.circle(rvb, (round(x * s), round(y * s)), 7 * ep, ROUGE_DET, ep, cv2.LINE_AA)
    return annoter(rvb, acq, k, s, etiquettes)


def annoter(rvb, acq, k, s, etiquettes=()):
    """Incruste le temps, l'échelle, les conditions d'acquisition et une barre d'échelle (cf. interface_suivi.py)."""
    cond = conditions_acquisition(acq)
    echelle = cond["echelle_um"] / s                       # µm par pixel de l'image affichée
    h, w = rvb.shape[:2]
    taille = max(12, round(w / 46))
    police, police_pt = _police(taille), _police(max(10, round(taille * 0.8)))
    marge = max(4, taille // 3)
    expo = "n/d" if cond["exposition_ms"] is None else f"{cond['exposition_ms']:.2f} ms"
    lignes = [f"t = {acq['temps_s'][k]:.3f} s",
              f"Échelle : {cond['echelle_um']:.4f} µm/pixel",
              f"Grossissement : {cond['grossissement']}X",
              f"Temps d'exposition : {expo}",
              f"Fréquence d'acquisition : {cond['fps']:.1f} images/s",
              f"Profondeur de bit : {cond['bits']}"]
    mesure = ImageDraw.Draw(Image.new("L", (1, 1)))
    larg = max(mesure.textlength(l, font=police) for l in lignes)
    interl = int(taille * 1.3)
    bx, by = int(larg + 2 * marge), len(lignes) * interl + 2 * marge
    rvb[:by, :bx] = (rvb[:by, :bx] * 0.35).astype(np.uint8)

    L_um = _longueur_ronde(0.2 * w * echelle)               # barre d'échelle ≈ 20 % de la largeur
    L_px = max(2, int(round(L_um / echelle)))
    texte_barre = f"{L_um:g} µm" if L_um < 1000 else f"{L_um / 1000:g} mm"
    epb = max(3, taille // 4)
    x0, y0 = marge, h - marge - epb
    zone = int(max(L_px, mesure.textlength(texte_barre, font=police)) + 2 * marge)
    zy = max(0, y0 - taille - 2 * marge)
    rvb[zy:h, :zone] = (rvb[zy:h, :zone] * 0.35).astype(np.uint8)

    pil = Image.fromarray(rvb)
    d = ImageDraw.Draw(pil)
    for i, l in enumerate(lignes):
        d.text((marge, marge + i * interl), l, font=police, fill=(255, 255, 255))
    d.rectangle([x0, y0, x0 + L_px, y0 + epb], fill=(255, 255, 255))
    d.text((x0, y0 - taille - marge // 2), texte_barre, font=police, fill=(255, 255, 255))
    for (x, y), texte, c in etiquettes:
        d.text((x, y), texte, font=police_pt, fill=c, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(pil)


def exporter(acq, resultats, final, detections, bornes, chemin, exclues=()):
    """Écrit à côté du fichier analysé : <nom>_annote.mp4 (chaque image annotée, résolution native,
    au rythme réel) et <nom>_analyse.csv (résultats par piste, statut, rayon final avec les exclusions)."""
    base = chemin.with_suffix("") if chemin.is_file() else chemin
    mp4 = base.with_name(base.name + "_annote.mp4")
    h, w = acq["frames"].shape[1:]
    video = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 1 / acq["dt"], (w, h))
    if not video.isOpened():
        print("Impossible de créer le MP4 (codec mp4v indisponible).")
        return
    n = len(acq["frames"])
    for k in range(n):
        video.write(cv2.cvtColor(composer_image(acq, k, resultats, detections, bornes, 1.0, exclues),
                                 cv2.COLOR_RGB2BGR))
        if (k + 1) % max(1, n // 10) == 0:
            print(f"  export : {k + 1}/{n} images", flush=True)
    video.release()

    fmt = lambda v, f, e: f % (v * e) if np.isfinite(v) else ""
    fichier_csv = base.with_name(base.name + "_analyse.csv")
    with open(fichier_csv, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow(["piste", "statut", "images", "D_um2_s", "sigma_exp_nm", "a_um", "sigma_PSF_um", "sigma_stat_um",
                     "sigma_tot_um"])
        for r in resultats:
            wr.writerow([r["piste"], statut(r, exclues), r["n_pts"], fmt(r["D"], "%.5f", 1e12),
                         fmt(r["sigma_exp"], "%.1f", 1e9), fmt(r["a"], "%.5f", 1e6) or "inf",
                         fmt(r["inc_psf"], "%.5f", 1e6), fmt(r["inc_stat"], "%.5f", 1e6), fmt(r["inc"], "%.5f", 1e6)])
        wr.writerow([])
        if final:
            wr.writerow(["rayon_final_um", f"{final['a'] * 1e6:.5f}", "incertitude_um", f"{final['inc'] * 1e6:.5f}",
                         "chi2_red", f"{final['chi2_red']:.3f}", "n_particules", final["n"],
                         "methode", "moyenne des D (immobiles inclus)" if final.get("methode") == "D"
                         else "moyenne ponderee des rayons"])
        wr.writerow(["pistes_exclues_a_la_main", " ".join(map(str, sorted(exclues))) or "aucune"])
        cond = conditions_acquisition(acq)
        wr.writerow(["echelle_um_px", f"{cond['echelle_um']:.5f}", "dt_s", f"{acq['dt']:.5f}",
                     "grossissement", cond["grossissement"]])
    print(f"Exporté : {mp4.name} et {fichier_csv.name} (dans {base.parent})")


def animer_resultats(acq, resultats, detections, chemin, sigma_psf=None):
    """Vidéo annotée en boucle au rythme réel + panneau de résultats.
    Espace = pause · CLIC sur une bille (vidéo) ou sur un point (graphique des rayons) = exclure / réintégrer
    la piste · r = réintégrer toutes les pistes · e = exporter MP4 + CSV. Retourne les pistes exclues."""
    frames, n = acq["frames"], len(acq["frames"])
    bornes = contraste(frames[n // 2])
    s = min(1.0, LARGEUR_AFFICHAGE_MAX / frames.shape[2])
    pe, dt = acq["pixel_eff"], acq["dt"]
    par_id = {r["piste"]: r for r in resultats}
    etat = dict(k=0, exclues=set(), final=rayon_final(pistes_moyenne(resultats)), pause=False)

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.1, 1], height_ratios=[0.6, 1.6, 0.8])
    ax_v = fig.add_subplot(gs[0:2, 0])
    ax_t = fig.add_subplot(gs[2, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_m = fig.add_subplot(gs[1, 1])
    ax_r = fig.add_subplot(gs[2, 1])
    im = ax_v.imshow(composer_image(acq, 0, resultats, detections, bornes, s))
    ax_v.set_axis_off()

    # --- MSD et ajustement linéaire, une couleur par piste (dessiné une fois; style mis à jour)
    lignes_msd = {}
    for r in resultats:
        col = couleur_mpl(r["piste"])
        pos = r["pos_px"] * pe
        lags, msd = msd_temporel(pos, LAG_MAX)
        ok = np.isfinite(msd)
        D, off = ajuste_msd(pos, dt, LAG_MAX)
        tt = np.linspace(0, lags[-1] * dt, 20)
        pts, = ax_m.plot(lags[ok] * dt, msd[ok] * 1e12, "o", color=col, ms=4)
        fit, = ax_m.plot(tt, (4 * D * tt + off) * 1e12, "-", color=col, lw=1,
                         label=str(r["piste"]) + (" (immobile)" if r["immobile"] else ""))
        lignes_msd[r["piste"]] = (pts, fit, col)
    ax_m.set(xlabel="Délai τ (s)", ylabel="MSD (µm²)", title="MSD et ajustement linéaire (Éq. 7)")
    ax_m.grid(alpha=0.3)
    if resultats:
        ax_m.legend(title="piste", fontsize=7, ncol=2)

    def dessiner_panneaux():
        """Carte, rayons et tableau : redessinés après chaque exclusion."""
        final, exclues = etat["final"], etat["exclues"]
        ax_c.clear()
        ax_c.set_axis_off()
        ax_c.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_c.transAxes, color="#eef3f8", zorder=0))
        ax_c.text(0.04, 0.85, "RAYON MOYEN DES PARTICULES", fontsize=9, fontweight="bold", transform=ax_c.transAxes)
        if final:
            ax_c.text(0.04, 0.48, texte_rayon(final), fontsize=19,
                      fontweight="bold", color="#0b3d6b", transform=ax_c.transAxes)
            if final.get("methode") == "D":
                details = (f"{final['n']} particule(s), dont {final['n_immobiles']} immobile(s) : "
                           f"moyenne des D = {final['D_moy'] * 1e12:.4f} µm²/s, puis Stokes-Einstein"
                           f"\nincertitude = dispersion des D (écart-type / √n)")
            else:
                chi = f"  ·  χ²réd = {final['chi2_red']:.2f}" if np.isfinite(final["chi2_red"]) else ""
                details = (f"{final['n']} particule(s){chi}"
                           + (f"  ·  σ_PSF = {sigma_psf:.2f} px" if sigma_psf else "")
                           + f"\nσ_PSF (localisation) = {final['inc_psf'] * 1e6:.3f} µm ⊕ "
                             f"σ_stat = {final['inc_stat'] * 1e6:.3f} µm")
        else:
            ax_c.text(0.04, 0.48, "Aucune piste dans la moyenne", fontsize=14, transform=ax_c.transAxes)
            details = ""
        details += f"\nExclues à la main : {sorted(exclues) if exclues else 'aucune'}"
        ax_c.text(0.04, 0.05, details, fontsize=8.5, transform=ax_c.transAxes)

        ax_r.clear()
        tracer_rayons(ax_r, resultats, final, exclues)
        ax_r.set_title(ax_r.get_title() + "  —  clic = exclure / réintégrer", fontsize=9)
        ax_r.grid(alpha=0.3)
        etat["actifs"] = ax_r.scatter([], [], s=200, facecolors="none", edgecolors="red", lw=1.5, zorder=5)

        for p, (pts, fit, col) in lignes_msd.items():
            ex = p in exclues
            for l in (pts, fit):
                l.set_color("0.7" if ex else col)
            fit.set_linestyle("--" if ex else "-")

        ax_t.clear()
        ax_t.set_axis_off()
        if resultats:
            tetes = ["Piste", "Statut", "Images", "D (µm²/s)", "σ_exp (nm)", "a (µm)", "σ_PSF (µm)", "σ_stat (µm)",
                     "σ_tot (µm)"]
            f = lambda v, e, d: f"{v * e:.{d}f}" if np.isfinite(v) else "—"
            cellules = [[r["piste"], statut(r, exclues), r["n_pts"], f(r["D"], 1e12, 3), f(r["sigma_exp"], 1e9, 0),
                         f(r["a"], 1e6, 3) if np.isfinite(r["a"]) else "∞", f(r["inc_psf"], 1e6, 3),
                         f(r["inc_stat"], 1e6, 3), f(r["inc"], 1e6, 3)] for r in resultats]
            tab = ax_t.table(cellText=cellules, colLabels=tetes, loc="upper center", cellLoc="center")
            tab.auto_set_font_size(False)
            tab.set_fontsize(8)
            for i, r in enumerate(resultats, 1):
                if r["piste"] in exclues:
                    for j in range(len(tetes)):
                        tab[i, j].get_text().set_color("0.6")
                else:
                    tab[i, 0].set_facecolor(couleur_mpl(r["piste"]) + (0.45,))

    def maj(k):
        etat["k"] = k
        exclues = etat["exclues"]
        im.set_data(composer_image(acq, k, resultats, detections, bornes, s, exclues))
        slot = acq["indices"][k]
        presents = [r for r in resultats
                    if 0 <= slot - r["f0"] < len(r["pos_px"]) and np.isfinite(r["pos_px"][slot - r["f0"], 0])]
        ids_presents = {r["piste"] for r in presents}
        for p, (pts, fit, col) in lignes_msd.items():
            for l in (pts, fit):
                l.set_alpha(1.0 if p in ids_presents else 0.2)
        finis = [r for r in presents if np.isfinite(r["a"])]
        etat["actifs"].set_offsets(np.column_stack([[r["piste"] for r in finis], [r["a"] * 1e6 for r in finis]])
                                   if finis else np.empty((0, 2)))
        perdues = int(acq["indices"][-1] + 1 - n)
        fig.suptitle(f"{chemin.name} — image {k + 1}/{n}   {len(presents)} bille(s) suivie(s)"
                     + (f"   ({perdues} images perdues à l'acquisition)" if perdues else "")
                     + "      [espace = pause · clic = exclure/réintégrer · r = tout réintégrer · e = exporter]",
                     fontsize=10)
        return []

    def basculer(pids):
        for p in pids:
            etat["exclues"] ^= {p}
            print(f"Piste {p}{' (immobile)' if par_id[p]['immobile'] else ''} "
                  f"{'exclue' if p in etat['exclues'] else 'réintégrée'}")
        etat["final"] = rayon_final(pistes_moyenne(resultats, etat["exclues"]))
        print("  -> " + texte_final(etat["final"]), flush=True)
        dessiner_panneaux()
        maj(etat["k"])
        fig.canvas.draw_idle()

    def clic(evt):
        if evt.xdata is None or not resultats or fig.canvas.toolbar is not None and fig.canvas.toolbar.mode:
            return
        if evt.inaxes is ax_v:                           # bille la plus proche dans l'image affichée
            slot = acq["indices"][etat["k"]]
            x, y = evt.xdata / s, evt.ydata / s
            meilleur, d_min = None, 25 / s
            for r in resultats:
                j = slot - r["f0"]
                if 0 <= j < len(r["pos_px"]) and np.isfinite(r["pos_px"][j, 0]):
                    d = np.hypot(r["pos_px"][j, 0] - x, r["pos_px"][j, 1] - y)
                    if d < d_min:
                        meilleur, d_min = r["piste"], d
            if meilleur is not None:
                basculer([meilleur])
        elif evt.inaxes is ax_r:                         # point le plus proche (en x = numéro de piste)
            finis = [r["piste"] for r in resultats if np.isfinite(r["a"])]
            if finis:
                p = min(finis, key=lambda q: abs(q - evt.xdata))
                if abs(p - evt.xdata) < 0.5:
                    basculer([p])

    def touche(evt):
        if evt.key == " ":
            (anim.resume if etat["pause"] else anim.pause)()
            etat["pause"] = not etat["pause"]
        elif evt.key == "r" and etat["exclues"]:
            basculer(sorted(etat["exclues"]))
        elif evt.key == "e":
            anim.pause()
            print("\nExport en cours...", flush=True)
            exporter(acq, resultats, etat["final"], detections, bornes, chemin, etat["exclues"])
            if not etat["pause"]:
                anim.resume()

    dessiner_panneaux()
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    anim = FuncAnimation(fig, maj, frames=n, interval=1000 * dt, repeat=True)
    fig.canvas.mpl_connect("key_press_event", touche)
    fig.canvas.mpl_connect("button_press_event", clic)
    fig._anim = anim                # garde une référence, sinon l'animation s'arrête
    plt.show()
    if etat["exclues"]:
        print(f"\nPistes exclues à la main : {sorted(etat['exclues'])}")
    print(texte_final(etat["final"]))
    return etat["exclues"]


#############################################################################################################
#                                       CHOIX DANS LE TERMINAL
#############################################################################################################

def chemin_json(p):
    return p.with_suffix(".json") if p.is_file() else p.parent / (p.name + ".json")


def lettre_option(p):
    """Option d'acquisition (A à E) : lue dans le .json, sinon dans le nom (…_B_20X…); « - » si inconnue."""
    try:
        lettre = json.loads(chemin_json(p).read_text(encoding="utf-8")).get("option")
        if lettre:
            return lettre
    except (OSError, ValueError):
        pass
    m = re.search(r"_([A-E])_[\d.]+X", p.name)
    return m.group(1) if m else "-"


def choisir_fichier(dossier=DOSSIER_DONNEES):
    candidats = sorted(p for p in dossier.iterdir()
                       if not p.stem.endswith("_annote")                 # vidéos exportées par l'analyse 2
                       and p.suffix.lower() in EXTENSIONS_TIFF | EXTENSIONS_VIDEO
                       or (p.is_dir() and any(q.suffix.lower() in EXTENSIONS_TIFF | {".png"}
                                              for q in p.iterdir())))
    if not candidats:
        raise FileNotFoundError(f"Aucune acquisition dans {dossier}")
    print(f"Acquisitions dans {dossier} :")
    for k, p in enumerate(candidats, 1):
        taille = (sum(q.stat().st_size for q in p.iterdir()) if p.is_dir() else p.stat().st_size) / 1e6
        print(f"  {k}. [{lettre_option(p)}] {p.name}   ({taille:.0f} Mo)"
              + ("" if chemin_json(p).exists() else "   (pas de .json)"))
    while True:
        rep = input(f"Numéro du fichier (1-{len(candidats)}) : ").strip()
        if rep.isdigit() and 1 <= int(rep) <= len(candidats):
            return candidats[int(rep) - 1]
        print("  Choix invalide.")


def choisir_analyse():
    print("Analyses :\n  1. classique (une image + détections, puis trajectoires et rayons)\n"
          "  2. dynamique (vidéo en boucle avec les billes suivies, cercles et rayons)")
    while True:
        rep = input("Analyse (1 ou 2) : ").strip()
        if rep in ("1", "2"):
            return int(rep)
        print("  Taper 1 ou 2.")


#############################################################################################################
#                                                 MAIN
#############################################################################################################

if __name__ == "__main__":
    chemin = choisir_fichier()
    analyse = choisir_analyse()

    print(f"\nChargement de {chemin.name}...", flush=True)
    acq = charger_acquisition(chemin)
    frames, dt, pixel_eff, meta = acq["frames"], acq["dt"], acq["pixel_eff"], acq["meta"]
    n, h, w = frames.shape
    perdues = int(acq["indices"][-1] + 1 - n)

    print(f"Fichier          : {chemin}   [option {lettre_option(chemin)}]")
    print(f"Nombre de frames : {n}" + (f"  ({perdues} images perdues à l'acquisition)" if perdues else ""))
    print(f"Résolution       : {w} x {h} pixels" + ("  (après somme 2x2 du Bayer)" if acq["bayer"] else ""))
    print(f"Δt               : {dt * 1e3:.1f} ms  ({1 / dt:.2f} fps)")
    print(f"Durée            : {acq['temps_s'][-1]:.2f} s")
    print(f"Pixel (objet)    : {pixel_eff * 1e6:.4f} µm/px" + (f"  (grossissement {meta['grossissement']}X)"
                                                             if meta.get("grossissement") else ""))
    print(f"Source           : {acq['source']}")
    if not meta:
        print("ATTENTION : pas de .json -> grossissement, taille de pixel et fps tirés des constantes du script.")

    kw_det, dist_max_px, demi = parametres_pixels(pixel_eff)

    if analyse == 1:
        INDICE_FRAME = n // 2   # à modifier
        gris = afficher_frame(frames, INDICE_FRAME, acq["temps_s"][INDICE_FRAME])
        particules = compter_particules(gris, indice=INDICE_FRAME, **kw_det)
        afficher_detection(gris, particules, indice=INDICE_FRAME)

    sigma_psf, disp_psf, n_fit = estimer_sigma_psf(frames, demi=demi, **kw_det)
    print(f"σ_PSF mesuré : {sigma_psf:.2f} ± {disp_psf:.2f} px ({sigma_psf*pixel_eff*1e6:.2f} µm), {n_fit} PSF")

    detections = localiser_toutes_frames(frames, sigma_psf, **kw_det)
    pistes = lier_trajectoires(detections, acq["indices"], dist_max_px=dist_max_px, trou_max=TROU_MAX)
    pistes_tab = pistes_en_tableaux(pistes, n_min=N_MIN)
    resultats = analyser_pistes(pistes_tab, dt, pixel_eff=pixel_eff, lag_max=LAG_MAX)
    if not resultats:
        print(f"Aucune piste d'au moins {N_MIN} positions : impossible de calculer un rayon.")
    imprimer_resultats(resultats, rayon_final(pistes_moyenne(resultats)))

    if analyse == 1:
        if resultats:
            afficher_resultats(frames, resultats, rayon_final(pistes_moyenne(resultats)))
            # Exclusion à la main, une fois la figure regardée (et fermée)
            exclues, final = choisir_exclusions(resultats)
            if exclues:
                afficher_resultats(frames, resultats, final, exclues)
    else:
        animer_resultats(acq, resultats, detections, chemin, sigma_psf)
