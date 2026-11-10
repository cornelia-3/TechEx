"""Traitement des acquisitions du microscope : rayon des billes par suivi de particules (MSD + Stokes-Einstein).

Au lancement, deux choix dans le terminal :
  1. le FICHIER, parmi ceux du dossier M3_Microscope/techex_M3_videos (piles .tif de acquisition_raspberry.py,
     dossiers de PNG de l'option E, ou anciennes vidéos .mp4);
  2. l'ANALYSE :
     1 = analyse « classique » : une image avec ses détections, puis trajectoires et rayon par particule;
     2 = analyse dynamique, représentation reprise de interface_suivi.py : la vidéo joue en boucle,
         chaque image annotée (temps, échelle µm/pixel, grossissement, exposition, fréquence, profondeur
         de bit, barre d'échelle); chaque bille suivie a sa trajectoire en couleur, un cercle de 2σ (taille
         apparente) et son rayon estimé; cercles rouges = autres détections. À droite : carte « rayon moyen »,
         MSD avec ajustements et rayon par piste (pistes visibles mises en évidence); dessous : tableau.
         Espace = pause · e = exporter <nom>_annote.mp4 + <nom>_analyse.csv à côté du fichier.
         Flèches ◀ ▶ au-dessus de la vidéo (ou touches ← →) : zoom sur la trajectoire de chaque bille
         sélectionnée, à tour de rôle (vidéo agrandie autour de la trajectoire, vrais pixels visibles),
         puis retour à la vue d'ensemble.

Exclusion À LA MAIN, une fois l'analyse faite (rien n'est exclu automatiquement) :
  - analyse 1 : après la figure, taper dans le terminal les numéros des pistes à exclure / réintégrer;
  - analyse 2 : cliquer sur une bille dans la vidéo, ou sur son point dans le graphique des rayons
    ou sur sa ligne dans le tableau (r = tout réintégrer). Le rayon moyen, la carte, le tableau et l'export se mettent à jour.
Toutes les pistes sont traitées de la même façon et toutes les valeurs du tableau sont calculées pour
chacune (D, σ_exp, a, σ_PSF, σ_stat, σ_tot), même quand D <= 0 (a est alors négatif). Aucune piste n'est
qualifiée ni écartée automatiquement : c'est l'utilisateur qui choisit. Rayon moyen des pistes sélectionnées :
moyenne des D, puis a = kT / (6πη·D_moyen) (moyenner les rayons serait biaisé de ≈ +6 %, car a ∝ 1/D).

Option au lancement : retirer le MOUVEMENT COMMUN (vibration, dérive) mesuré sur les billes immobiles avant le
calcul du MSD (les billes immobiles servent de référence; chacune n'est corrigée qu'avec les AUTRES). Sans cette
correction, une vibration d'allers-retours fait baisser le MSD entre Δt et 3Δt : D négatif pour les billes
collées, D sous-estimé pour les billes libres.

Les paramètres d'acquisition sont lus dans le .json écrit à côté de chaque pile par acquisition_raspberry.py :
  - le temps de chaque image (horodatages du capteur) -> Δt réel, et les images perdues deviennent des trous
    dans les trajectoires au lieu de fausser le MSD;
  - la taille de pixel = pixel_effectif_capteur_um / grossissement RÉEL. Le grossissement réel est mesuré
    sur une photo de règle du même objectif (regle<G>x….png, dans le dossier des acquisitions) par la
    méthode de grossissement_regle.py; résultat et figure mis en cache (<photo>_etalonnage.json / .png).
    Sans photo, on garde le grossissement nominal du .json. Comme a ∝ (pixel)², c'est crucial;
  - le contenu : une mosaïque de Bayer brute (options D, E) est d'abord sommée 2x2 pour donner une image grise.
Les données restent en 16 bits (aucune conversion en 8 bits, sauf pour l'affichage).
Sans .json (mp4, TIFF externe), on retombe sur les constantes ci-dessous, avec un avertissement.
"""
import csv
import json
import math
import re
import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile                      # pip install tifffile
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.optimize import curve_fit, linear_sum_assignment
from scipy.signal import find_peaks
from scipy.special import erf

DOSSIER_DONNEES = Path(__file__).resolve().parent / "techex_M3_videos"   # TechEx/M3_Microscope/techex_M3_videos
# pour dense_up.mp4, on ne détecte aucune particule
# pour gab.mp4, on ne détecte aucune particule
# pour 20X on détecte bien les 6 particules : a = 1.257 µm  ± 0.088 µm (σ_PSF = 0.008 ⊕ σ_stat = 0.088), 6 particules, χ²_red = 1.76
# dense_up_contraste.mp4, grossissement 4X, on détecte 15 particules : a = 0.400 µm  ± 0.022 µm (σ_PSF = 0.008 ⊕ σ_stat = 0.021), 17 particules, χ²_red = 349.89
# dense_up_contraste.mp4, grossissement 10X, on détecte 15 particules : 2.499 µm  ± 0.140 µm (σ_PSF = 0.052 ⊕ σ_stat = 0.130), 17 particules, χ²_red = 349.89
# constantes
kB = 1.380649e-23          # J/K
T = 22 + 273.15            # K (eau à 22 °C)
ETA = 0.95e-3              # Pa·s (eau à 22 °C)

# Valeurs par défaut, utilisées SEULEMENT quand il n'y a pas de .json (anciens mp4, TIFF externes)
GROSSISSEMENT = 20         # X
TAILLE_PIXEL = 1.55e-6     # m
REDUCTION = 4056 / 640     # nbr pixels capteur / nbr pixels image (mp4 de 640 px de large)
PIXEL_EFF = TAILLE_PIXEL * REDUCTION / GROSSISSEMENT        # m/pixel dans le plan objet
FPS_DEFAUT = 30.0
MODE_TEST = False          # analyse 3 (test en conditions réelles) : carte réduite au rayon, tableau défilable
INFO_VIBRATION = {}        # option « retirer le mouvement commun » et pistes de référence (affichage, export)

# Paramètres de détection et de suivi, en µm dans le plan objet (convertis en pixels selon le fichier).
# Avec les anciens mp4 (0,49 µm/px), ils redonnent exactement les valeurs en pixels d'avant (1,5 / 5 / 3 / 8 px).
LISSAGE_UM = 0.74          # lissage gaussien avant la détection
DISTANCE_MIN_UM = 2.45     # distance minimale entre deux particules
DEPLACEMENT_MAX_UM = 1.47  # déplacement maximal entre deux images consécutives
DEMI_REGION_UM = 3.9       # demi-largeur de la région utilisée pour la 1re estimation de σ
SEUIL_K = 5.0              # seuil de détection = fond + k * bruit
TROU_MAX = 5               # nb d'images manquantes tolérées dans une trajectoire (images perdues comprises)
N_MIN = 20                 # nb minimal de positions pour garder une trajectoire
# Indication « immobile ? » (colonne du tableau de l'analyse 2; n'exclut RIEN, c'est une information) :
SEUIL_IMMOBILE_SIGMA = 2.0     # immobile si D < SEUIL × σ_D0 (σ_D0 : dispersion de D d'une bille parfaitement immobile)
A_MAX_IMMOBILE_UM = 5.0        # ... ou si a > 5 µm : plus gros que les particules du mandat (10 µm de diamètre au plus)
N_SIMULATIONS_IMMOBILE = 300   # tirages pour estimer σ_D0
LAG_MAX = 3                # nb de décalages L utilisés dans l'ajustement du MSD (valeur proposée au lancement)

EXTENSIONS_TIFF = {".tif", ".tiff"}
EXTENSIONS_VIDEO = {".mp4", ".avi", ".h264", ".mov"}


#############################################################################################################
#                       ÉTALONNAGE DU GROSSISSEMENT AVEC UNE RÈGLE (repris de grossissement_regle.py)
#   Principe : on mesure le pas p (pixels) entre les graduations d'une photo de règle prise avec le même
#   objectif. Pixel dans le plan objet = PAS_GRADUATION / p ; grossissement réel M = pixel capteur / pixel objet.
#   Le résultat est mis en cache à côté de la photo (<photo>_etalonnage.json) avec sa figure
#   (<photo>_etalonnage.png) : l'étalonnage (≈ 20 s) n'est refait que si la photo ou le pas changent.
#############################################################################################################

ETALONNER = True              # False = grossissement nominal du .json, sans règle
PAS_GRADUATION = 100e-6       # m entre deux graduations consécutives de la règle (confirmé)
F_TUBE = 150e-3               # m, focale de la lentille de tube (sert seulement à l'affichage)
EXTENSIONS_REGLE = {".png", ".tif", ".tiff", ".jpg"}
VERSION_ETALONNAGE = 2        # à incrémenter si la méthode change : les caches existants sont alors refaits


def charger_image_regle(chemin):
    """Photo de la règle en niveaux de gris (float), profondeur du fichier conservée."""
    img = cv2.imread(str(chemin), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Impossible d'ouvrir l'image : {chemin}")
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    return img.astype(float)


def angles_candidats(img, reduction=4, demi_plage_deg=4.0, pas_deg=0.1):
    """Normales des deux familles de traits d'une règle quadrillée (≈ 0° : traits verticaux, ≈ 90° : traits
    horizontaux), en radians. Chacune est affinée (±4°) en maximisant la netteté du profil sur l'image réduite.
    (Le tenseur de structure de grossissement_regle.py donne ≈ 45° quand les deux familles pèsent autant,
    ce qui arrive au 40X où l'on ne voit que 2 traits verticaux et 1 horizontal.)"""
    petit = cv2.resize(img, None, fx=1 / reduction, fy=1 / reduction, interpolation=cv2.INTER_AREA)
    angles = []
    for base in (0.0, 90.0):
        meilleur = (-np.inf, np.radians(base))
        for d in np.arange(-demi_plage_deg, demi_plage_deg + pas_deg / 2, pas_deg):
            th = np.radians(base + d)
            _, pr = profil_normal(petit, th)
            nettete = np.mean(np.diff(ndi.gaussian_filter1d(pr, 1.0))**2)
            if nettete > meilleur[0]:
                meilleur = (nettete, th)
        angles.append(meilleur[1])
    return angles


def profil_normal(img, theta, marge=0.03):
    """Intensité moyenne en fonction de la distance s (pixels) mesurée perpendiculairement aux traits."""
    h, w = img.shape
    y0, y1 = int(h * marge), int(h * (1 - marge))
    x0, x1 = int(w * marge), int(w * (1 - marge))
    Y, X = np.mgrid[y0:y1, x0:x1]
    s = X * np.cos(theta) + Y * np.sin(theta)
    s0 = s.min()
    cases = np.round(s - s0).astype(int).ravel()
    somme = np.bincount(cases, weights=img[y0:y1, x0:x1].ravel())
    n = np.bincount(cases)
    ok = n >= 0.5 * n.max()                      # écarte les bords peu peuplés
    positions = np.arange(len(n)) + s0
    profil = np.where(n > 0, somme / np.maximum(n, 1), 0.0)
    return positions[ok], profil[ok]


def _gauss1d(x, A, x0, sigma, b):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b


def _creneau(x, A, x0, w, sig, b):
    """Trait de largeur w (plateau) aux bords adoucis par un flou gaussien sig."""
    r2 = np.sqrt(2) * sig
    return b + 0.5 * A * (erf((x - x0 + w / 2) / r2) - erf((x - x0 - w / 2) / r2))


def _ajuster_trait(s, signal, i, periode_attendue):
    """Position sub-pixel du trait dont le maximum approximatif est à l'indice i.
    Modèle principal : créneau flouté (traits larges à fond plat). Repli : gaussienne (traits fins)."""
    A_est = signal[i]
    g, d = i, i
    while g > 0 and signal[g - 1] > 0.5 * A_est:
        g -= 1
    while d < len(s) - 1 and signal[d + 1] > 0.5 * A_est:
        d += 1
    w0 = max(s[d] - s[g], 3.0)
    x00 = 0.5 * (s[g] + s[d])
    demi = int(min(w0 + 30, 0.45 * periode_attendue))
    lo, hi = max(np.searchsorted(s, x00 - demi), 0), min(np.searchsorted(s, x00 + demi), len(s))
    if hi - lo < 9:
        return None
    xs, ys = s[lo:hi], signal[lo:hi]
    try:
        p, cov = curve_fit(_creneau, xs, ys, p0=[A_est, x00, w0, 2.0, np.median(ys)],
                           bounds=([0, x00 - w0, 0.5, 0.3, -np.inf],
                                   [np.inf, x00 + w0, 2 * w0 + 10, w0 + 10, np.inf]))
        err = np.sqrt(cov[1, 1])
        if np.isfinite(err) and err < w0:
            return p[1], err
    except (RuntimeError, ValueError):
        pass
    try:
        p, cov = curve_fit(_gauss1d, xs, ys, p0=[A_est, x00, max(w0 / 2.355, 1.0), np.median(ys)],
                           bounds=([0, xs[0], 0.3, -np.inf], [np.inf, xs[-1], 2 * demi, np.inf]))
    except (RuntimeError, ValueError):
        return None
    err = np.sqrt(cov[1, 1])
    return (p[1], err) if np.isfinite(err) else None


def detecter_traits(s, profil, periode_attendue):
    """Position sub-pixel (pixels, le long de la normale) de chaque trait."""
    fenetre = min(int(0.6 * periode_attendue), len(profil) - 1) | 1     # > largeur d'un trait, < pas
    fond = ndi.median_filter(profil, size=fenetre, mode="nearest")
    brut = profil - fond

    def amplitude(sig):                          # polarité : l'écart le plus ample (creux ou bosses)
        return ndi.gaussian_filter1d(sig, 2).max() - np.median(sig)
    signal = brut if amplitude(brut) >= amplitude(-brut) else -brut
    bruit = 1.4826 * np.median(np.abs(signal - np.median(signal)))
    pics, _ = find_peaks(ndi.gaussian_filter1d(signal, 2), distance=max(1, int(0.5 * periode_attendue)),
                         prominence=max(0.25 * amplitude(signal), 5 * bruit))
    positions, erreurs = [], []
    for i in pics:
        r = _ajuster_trait(s, signal, i, periode_attendue)
        if r is not None:
            positions.append(r[0])
            erreurs.append(r[1])
    return np.array(positions), np.array(erreurs), signal


def pas_en_pixels(positions, erreurs, periode_attendue):
    """Régression position = x0 + p·n sur l'indice entier n de chaque trait (traits manquants tolérés).
    Incertitude : la plus grande entre celle des résidus et celle des ajustements de chaque trait.
    Avec 2 traits seulement : p = écart / n (n = nombre de pas entre eux, estimé avec le pas attendu),
    incertitude = celles des 2 ajustements combinées (aucun résidu possible)."""
    ordre = np.argsort(positions)
    positions, erreurs = positions[ordre], erreurs[ordre]
    if len(positions) == 2:
        ecart = positions[1] - positions[0]
        k = max(1, int(round(ecart / periode_attendue)))
        p = ecart / k
        return p, np.hypot(*erreurs) / k, np.array([0, k]), np.zeros(2), positions, erreurs
    pas_approx = np.median(np.diff(positions))
    n = np.round((positions - positions[0]) / pas_approx).astype(int)
    (p, x0), cov = np.polyfit(n, positions, 1, cov=True)
    _, cov_f = np.polyfit(n, positions, 1, w=1 / erreurs, cov="unscaled")
    inc_p = max(np.sqrt(cov[0, 0]), np.sqrt(cov_f[0, 0]))
    return p, inc_p, n, positions - (p * n + x0), positions, erreurs


def etalonner_regle(chemin, M_nominal):
    """Grossissement réel à partir de la photo de règle `chemin`. Retourne un dict (M, inc_M, pixel_eff
    de la photo, pas en pixels...) et les données de la figure."""
    img = charger_image_regle(chemin)
    h, w = img.shape
    pixel_capteur = TAILLE_PIXEL * 4056 / w          # taille d'un pixel de CETTE photo sur le capteur
    periode_attendue = PAS_GRADUATION / (pixel_capteur / M_nominal)
    if periode_attendue < 6:
        raise ValueError("pas attendu < 6 pixels : vérifier PAS_GRADUATION et le grossissement nominal")
    # Les deux familles de traits de la grille : on garde celle qui montre le plus de traits
    meilleur = None
    for theta in angles_candidats(img):
        s, profil = profil_normal(img, theta)
        positions, erreurs, signal = detecter_traits(s, profil, periode_attendue)
        if meilleur is None or len(positions) > len(meilleur[3]):
            meilleur = (theta, s, signal, positions, erreurs)
    theta, s, signal, positions, erreurs = meilleur
    if len(positions) < 2:
        raise RuntimeError(f"seulement {len(positions)} trait(s) détecté(s) sur la règle (il en faut 2)")
    p, inc_p, n, residus, positions, erreurs = pas_en_pixels(positions, erreurs, periode_attendue)
    pixel_eff = PAS_GRADUATION / p
    M = pixel_capteur / pixel_eff
    res = dict(photo=Path(chemin).name, M_nominal=M_nominal, M=M, inc_M=M * inc_p / p,
               pixel_eff_photo_um=pixel_eff * 1e6, inc_pixel_eff_photo_um=pixel_eff * inc_p / p * 1e6,
               pas_px=p, inc_pas_px=inc_p, n_traits=len(positions), theta_deg=float(np.degrees(theta)),
               pas_graduation_um=PAS_GRADUATION * 1e6, largeur_photo_px=w,
               date=datetime.now().isoformat(timespec="seconds"))
    figure = dict(img=img, theta=theta, s=s, signal=signal, positions=positions, erreurs=erreurs, n=n,
                  residus=residus)
    return res, figure


def figure_etalonnage(res, fig_data, chemin_png):
    """Photo de la règle avec les traits détectés, profil perpendiculaire, résidus de la régression."""
    img, th = fig_data["img"], fig_data["theta"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1.3, 1.2, 0.8]})
    bas, haut = np.percentile(img[::4, ::4], (1, 99.5))
    ax[0].imshow(img, cmap="gray", vmin=bas, vmax=haut)
    h, w = img.shape
    for pos in fig_data["positions"]:                  # trait : x cosθ + y sinθ = pos
        if abs(np.cos(th)) > abs(np.sin(th)):
            y = np.array([0, h - 1])
            x = (pos - y * np.sin(th)) / np.cos(th)
        else:
            x = np.array([0, w - 1])
            y = (pos - x * np.cos(th)) / np.sin(th)
        ax[0].plot(x, y, "r-", lw=1)
    ax[0].set(xlim=(0, w), ylim=(h, 0), title=f"{res['photo']} : {res['n_traits']} traits détectés",
              xlabel="x (pixels)", ylabel="y (pixels)")
    ax[1].plot(fig_data["s"], fig_data["signal"], lw=0.8)
    ax[1].plot(fig_data["positions"], np.interp(fig_data["positions"], fig_data["s"], fig_data["signal"]), "rv",
               label="traits détectés")
    ax[1].set(xlabel="position le long de la normale aux traits (pixels)", ylabel="signal (sans fond)",
              title=f"Profil · pas = {res['pas_px']:.2f} ± {res['inc_pas_px']:.2f} px")
    ax[1].legend()
    ax[2].errorbar(fig_data["n"], fig_data["residus"], yerr=fig_data["erreurs"], fmt="o", capsize=3)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set(xlabel="indice du trait", ylabel="résidu (pixels)",
              title="Résidus de la régression" if res["n_traits"] > 2
              else "2 traits : pas de résidu\n(barres = incertitude de position de chaque trait)")
    fig.suptitle(f"Étalonnage du grossissement — M = {res['M']:.3f} ± {res['inc_M']:.3f} (nominal "
                 f"{res['M_nominal']}X) · pixel objet (photo pleine résolution) = "
                 f"{res['pixel_eff_photo_um']:.5f} µm · graduation = {res['pas_graduation_um']:g} µm · "
                 f"objectif équivalent f = {F_TUBE / res['M'] * 1e3:.2f} mm (tube {F_TUBE * 1e3:.0f} mm)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(chemin_png, dpi=110)
    plt.close(fig)


def trouver_regle(grossissement, dossiers):
    """Photo de règle pour ce grossissement : nom commençant par « regle<G>x » (ex. regle20x_100mm.png)."""
    motif = re.compile(rf"^r[eè]gle0*{int(round(float(grossissement)))}x", re.IGNORECASE)
    for d in dossiers:
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in EXTENSIONS_REGLE and motif.match(p.stem) and "_etalonnage" not in p.stem:
                    return p
    return None


def grossissement_reel(grossissement, dossiers):
    """(M, inc_M, texte_source). Lit le cache <photo>_etalonnage.json s'il correspond à la photo et au pas
    de graduation actuels; sinon refait l'étalonnage et écrit le cache et la figure. Sans photo (ou si
    l'étalonnage échoue) : grossissement nominal."""
    if not ETALONNER:
        return float(grossissement), 0.0, "nominal (ETALONNER = False)"
    photo = trouver_regle(grossissement, dossiers)
    if photo is None:
        return float(grossissement), 0.0, f"nominal (aucune photo regle{grossissement}x… trouvée)"
    cache = photo.with_name(photo.stem + "_etalonnage.json")
    signature = dict(taille_octets=photo.stat().st_size, mtime=photo.stat().st_mtime,
                     pas_graduation_um=PAS_GRADUATION * 1e6, M_nominal=float(grossissement),
                     version_methode=VERSION_ETALONNAGE)
    try:
        res = json.loads(cache.read_text(encoding="utf-8"))
        if res.get("signature") == signature:
            return res["M"], res["inc_M"], f"étalonné ({photo.name}, cache)"
    except (OSError, ValueError):
        pass
    print(f"Étalonnage du grossissement sur {photo.name} (≈ 20 s, une seule fois)...", flush=True)
    try:
        res, fig_data = etalonner_regle(photo, float(grossissement))
    except (RuntimeError, ValueError) as ex:
        print(f"  ÉCHEC : {ex}. On garde le grossissement nominal {grossissement}X.")
        return float(grossissement), 0.0, f"nominal (étalonnage impossible sur {photo.name})"
    res["signature"] = signature
    try:
        cache.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        figure_etalonnage(res, fig_data, photo.with_name(photo.stem + "_etalonnage.png"))
        print(f"  M = {res['M']:.3f} ± {res['inc_M']:.3f} (nominal {grossissement}X) · figure : "
              f"{photo.stem}_etalonnage.png")
    except OSError as ex:
        print(f"  (cache non écrit : {ex})")
    return res["M"], res["inc_M"], f"étalonné ({photo.name})"


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


# Pixel effectif au capteur (µm) et durée d'image (µs) de chaque option de acquisition_raspberry.py :
# servent quand le .json manque (la lettre et le grossissement sont alors lus dans le nom du fichier).
PIXEL_CAPTEUR_OPTION_UM = {"A": 3.1, "B": 3.1, "C": 6.2, "D": 1.55, "E": 1.55}
DUREE_IMAGE_OPTION_US = {"A": 100_000, "B": 100_000, "C": 92_000, "D": 100_000, "E": 100_000}


def infos_depuis_nom(chemin):
    """(option, grossissement) lus dans un nom du type …_C_40X.tif ; None si absent."""
    m = re.search(r"_([A-E])_([\d.]+)X", Path(chemin).name)
    return (m.group(1), float(m.group(2))) if m else (None, None)


def fps_par_defaut(chemin):
    """fps de l'acquisition déduit de l'option (exposition de 90 ms du script d'acquisition)."""
    option, _ = infos_depuis_nom(chemin)
    return 1e6 / DUREE_IMAGE_OPTION_US[option] if option else FPS_DEFAUT


def charger_acquisition(chemin, fps_sans_json=None):
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

    # Sans .json : option et grossissement tirés du nom du fichier (…_C_40X.tif)
    option_nom, G_nom = infos_depuis_nom(chemin)
    option = meta.get("option") or option_nom

    # Mosaïque de Bayer brute (options D et E) -> somme 2x2 en gris, pixel 2x plus grand
    if meta:
        bayer = "bayer" in meta.get("contenu", "").lower() and "somme" not in meta.get("contenu", "").lower()
    else:
        bayer = option in ("D", "E")
    if bayer:
        frames = np.stack([somme_bayer(f) for f in frames])

    # Taille de pixel dans le plan objet = pixel effectif au capteur / grossissement RÉEL (étalonné sur la règle)
    G_nominal = meta.get("grossissement") or (int(G_nom) if G_nom and G_nom.is_integer() else G_nom) or GROSSISSEMENT
    M, inc_M, source_M = grossissement_reel(G_nominal, [chemin.parent, DOSSIER_DONNEES])
    if meta.get("pixel_effectif_capteur_um"):
        pixel_capteur_um = meta["pixel_effectif_capteur_um"] * (2 if bayer else 1)
        source_pixel = "json"
    elif option in PIXEL_CAPTEUR_OPTION_UM:
        pixel_capteur_um = PIXEL_CAPTEUR_OPTION_UM[option] * (2 if bayer else 1)
        source_pixel = f"option {option} lue dans le nom du fichier"
    else:
        pixel_capteur_um = TAILLE_PIXEL * REDUCTION * 1e6
        source_pixel = "constantes du script"
    pixel_eff = pixel_capteur_um / M * 1e-6

    # Temps de chaque image : horodatages du capteur, sinon fps
    stamps = meta.get("timestamps_ns")
    if not stamps and chemin.is_dir() and (chemin / "timestamps_ns.npy").exists():
        stamps = np.load(chemin / "timestamps_ns.npy").tolist()
    if stamps and len(stamps) == len(frames):
        t = (np.asarray(stamps, dtype=np.int64) - stamps[0]) / 1e9
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / FPS_DEFAUT
        source_temps = "horodatages du json"
    else:
        if fps_fichier and fps_fichier > 0:
            fps, origine = fps_fichier, "fichier vidéo"
        elif fps_sans_json:
            fps, origine = fps_sans_json, "saisi au lancement, pas d'horodatages"
        else:
            fps, origine = FPS_DEFAUT, "valeur par défaut"
        dt = 1 / fps
        t = np.arange(len(frames)) * dt
        source_temps = f"fps = {fps:g} ({origine})"
    indices = np.round(t / dt).astype(int)

    expo = [e for e in meta.get("exposure_us", []) if e]
    t_exp = (np.median(expo) * 1e-6 if expo else
             (min(0.090, dt) if option else 0.0))           # 90 ms : exposition du script d'acquisition
    return dict(frames=frames, temps_s=t, indices=indices, dt=dt, pixel_eff=pixel_eff, meta=meta, t_exp=t_exp,
                bayer=bayer, option=option, grossissement_nominal=G_nominal, grossissement_reel=M, inc_M=inc_M,
                source=f"pixel capteur : {source_pixel} · grossissement : {source_M} · temps : {source_temps}")


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
#                       MOUVEMENT COMMUN (VIBRATION) — option posée au lancement
#   Toutes les billes bougent ensemble d'une image à l'autre (vibration de la platine ou du support, dérive) :
#   ce mouvement s'ajoute au MSD de chaque bille. Il est mesuré sur les billes IMMOBILES (colonne « Immobile ? »)
#   et retiré des positions avant le calcul du MSD. Les trajectoires affichées restent les positions réelles.
#############################################################################################################

def mouvement_commun(pistes_tab, refs, n_creneaux, sauf=None):
    """Mouvement commun à toutes les billes (vibration + dérive), par créneau : médiane, sur les pistes de
    RÉFÉRENCE (billes immobiles), de l'écart à leur position moyenne. sauf : piste à exclure des références
    (pour corriger une référence, on n'utilise que les AUTRES, sinon elle se corrige par elle-même).
    Créneaux avec moins de 2 références : interpolés. Retourne (n_creneaux, 2) en pixels."""
    refs = [p for p in refs if p != sauf]
    M = np.full((max(len(refs), 1), n_creneaux, 2), np.nan)
    for i, p in enumerate(refs):
        q = pistes_tab[p]["pos_px"]
        M[i, pistes_tab[p]["f0"]:pistes_tab[p]["f0"] + len(q)] = q - np.nanmean(q, axis=0)
    n = np.sum(np.isfinite(M[:, :, 0]), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # créneau sans aucune référence
        c = np.nanmedian(M, axis=0)
    c[n < 2] = np.nan
    for a in range(2):
        ok = np.isfinite(c[:, a])
        c[:, a] = np.interp(np.arange(n_creneaux), np.flatnonzero(ok), c[ok, a]) if ok.any() else 0.0
    return c


def corriger_vibration(pistes_tab, resultats, n_creneaux, analyser=None, iter_max=10):
    """Retire de chaque piste le mouvement commun mesuré sur les billes immobiles (colonne « Immobile ? »).
    Choix ITÉRATIF des références : une bille marquée immobile avant correction peut se révéler mobile une fois
    la vibration retirée (sa lente diffusion était masquée). On recalcule alors le mouvement commun avec les
    seules billes encore immobiles, et on répète jusqu'à ce que la liste ne change plus (ou iter_max; si elle
    tourne en rond, on garde la dernière). analyser(pistes) -> résultats (sans lui : une seule passe).
    Retourne (pistes corrigées, références (numéros), écart type du mouvement commun en px, résultats corrigés
    ou None, nombre de passes) ; pistes corrigées = None s'il y a moins de 2 billes immobiles."""
    refs = [i for i, r in enumerate(resultats) if r["immobile"]]           # indices dans pistes_tab
    vus, corr, n_passes = set(), None, 0
    while True:
        if len(refs) < 2:
            return None, [resultats[i]["piste"] for i in refs], np.nan, None, n_passes
        vus.add(tuple(refs))
        c_tous = mouvement_commun(pistes_tab, refs, n_creneaux)
        corrigees = []
        for i, t in enumerate(pistes_tab):
            c = mouvement_commun(pistes_tab, refs, n_creneaux, sauf=i) if i in refs else c_tous
            corrigees.append(dict(t, pos_px=t["pos_px"] - c[t["f0"]:t["f0"] + len(t["pos_px"])]))
        n_passes += 1
        if analyser is None:
            break
        corr = analyser(corrigees)
        nouveaux = [i for i, r in enumerate(corr) if r["immobile"]]
        if nouveaux == refs or tuple(nouveaux) in vus or n_passes >= iter_max or len(nouveaux) < 2:
            break
        refs = nouveaux
    return corrigees, [resultats[i]["piste"] for i in refs], float(np.std(np.diff(c_tous, axis=0))), corr, n_passes


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
    """MSD = 2N D t + 2N σ_exp² (Éq. 7). Retourne D (m²/s) et l'offset (m²).
    Avec un seul décalage (L = 1), une droite à 2 paramètres n'est pas définie : D = MSD(Δt) / (2N Δt),
    c.-à-d. une droite forcée par l'origine (σ_exp non estimé, offset = NaN)."""
    lags, msd = msd_temporel(pos, lag_max)
    ok = np.isfinite(msd)
    if ok.sum() == 1:
        k = lags[ok][0]
        return msd[ok][0] / (k * dt) / (2 * pos.shape[1]), np.nan
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


def retirer_fond_lent(frames, taille_px):
    """Retire, image par image, le fond lisse à grande échelle (éclairage non uniforme) : ouverture morphologique
    (taille > tache d'une bille) puis moyenne glissante, calculées sur l'image sous-échantillonnée ×4.
    Le niveau moyen du fond est conservé (même type entier). Modifie `frames` en place."""
    q = 4
    t = max(3, int(taille_px) // q)
    h, w = frames.shape[1:]
    for i in range(len(frames)):
        im = frames[i].astype(np.float32)
        p = ndi.uniform_filter(ndi.grey_opening(im[::q, ::q], size=(t, t)), t)
        fond = ndi.zoom(p, (h / p.shape[0], w / p.shape[1]), order=1)[:h, :w]
        frames[i] = np.clip(im - fond + np.median(fond), 0, np.iinfo(frames.dtype).max if frames.dtype.kind in "ui"
                            else None).astype(frames.dtype)
    return frames


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
    """Pente de la régression linéaire = Σ c_k · MSD_k (un seul décalage : pente = MSD_k / t_k)."""
    if len(ks) == 1:
        return np.array([1.0 / (ks[0] * dt)])
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


def rayon_piste_complet(pos_px, inc_px, dt, pixel_eff=PIXEL_EFF, lag_max=3, t_exp=0.0):
    """Comme rayon_piste, mais calcule TOUT pour toutes les pistes, y compris D <= 0 :
    a = kT / (6πηD) (négatif si D < 0, infini si D = 0), incertitudes propagées par δa = |a|·δD/|D|.
    σ_stat utilise |D| comme variance d'un pas (une variance ne peut pas être négative).
    σ_exp : l'exposition t_exp moyenne la position pendant le mouvement (flou de mouvement), ce qui abaisse
    l'ordonnée du MSD : MSD = 2nDτ + 2nσ_exp² − (2n/3)·D·t_exp (Savin & Doyle, 2005). La pente, donc D,
    n'est pas touchée; on corrige seulement σ_exp² = ordonnée/(2n) + D·t_exp/3."""
    pos, inc = pos_px * pixel_eff, inc_px * pixel_eff
    n_dim = pos.shape[1]
    D, offset = ajuste_msd(pos, dt, lag_max)
    nan = dict(a=np.nan, D=D, sigma_exp=np.nan, inc_psf=np.nan, inc_stat=np.nan, inc=np.nan)
    if not np.isfinite(D):
        return nan
    s2 = offset / (2 * n_dim) + max(D, 0) * t_exp / 3 if np.isfinite(offset) else np.nan
    sigma_exp = np.sqrt(s2) if np.isfinite(s2) and s2 > 0 else (0.0 if np.isfinite(s2) else np.nan)
    if D == 0:
        return {**nan, "a": np.inf, "sigma_exp": sigma_exp}
    a = kB * T / (6 * np.pi * ETA * D)
    inc_psf = abs(a) * incertitude_D_psf(pos, inc, dt, lag_max) / abs(D)
    inc_stat = abs(a) * incertitude_D_stat(pos, abs(D), dt, lag_max) / abs(D)
    return dict(a=a, D=D, sigma_exp=sigma_exp, inc_psf=inc_psf, inc_stat=inc_stat, inc=np.hypot(inc_psf, inc_stat))


def analyser_pistes(pistes_tab, dt, **kw):
    """Une entrée par piste d'au moins N_MIN positions, toutes les valeurs calculées pour chacune.
    Aucune piste n'est qualifiée ni écartée automatiquement : les exclusions se font À LA MAIN
    (choisir_exclusions, ou clic dans l'analyse 2)."""
    resultats = []
    rng = np.random.default_rng(0)                   # graine fixe : la colonne « immobile ? » est reproductible
    for num, p in enumerate(pistes_tab):
        r = rayon_piste_complet(p["pos_px"], p["inc_px"], dt, **kw)          # kw : pixel_eff, lag_max, t_exp
        r.update(piste=num, f0=p["f0"], pos_px=p["pos_px"], sig_px=p["sig_px"],
                 n_pts=int(np.isfinite(p["pos_px"][:, 0]).sum()))
        r.update(test_immobile(r, p, dt, kw.get("pixel_eff", PIXEL_EFF), kw.get("lag_max", 3), rng,
                               kw.get("t_exp", 0.0)))
        resultats.append(r)
    return resultats


def test_immobile(r, piste, dt, pixel_eff, lag_max, rng, t_exp=0.0):
    """Critère « immobile ? » (indicatif seulement). Une bille est jugée immobile si :
      1) son D n'est pas significativement plus grand que celui qu'on mesurerait sur une bille parfaitement
         immobile : D < SEUIL_IMMOBILE_SIGMA · σ_D0, où σ_D0 est la dispersion de D obtenue en simulant
         N_SIMULATIONS_IMMOBILE fois une bille fixe avec la MÊME erreur de localisation (σ_exp de la piste),
         les mêmes images manquantes et le même ajustement du MSD (L décalages). D <= 0 tombe dans ce cas;
      2) ou si son rayon dépasse A_MAX_IMMOBILE_UM (5 µm) : aucune particule du mandat (10 µm de diamètre au plus)
         ne peut diffuser aussi lentement; une bille plus lente est collée ou freinée par la paroi."""
    pos = piste["pos_px"] * pixel_eff
    ok = np.isfinite(pos[:, 0])
    sig = r["sigma_exp"]
    if not (np.isfinite(sig) and sig > 0):                    # L = 1, ou offset négatif : erreur des ajustements
        sig = np.nanmedian(piste["inc_px"]) * pixel_eff
    D_sim = [ajuste_msd(np.where(ok[:, None], rng.normal(0, sig, pos.shape), np.nan), dt, lag_max)[0]
             for _ in range(N_SIMULATIONS_IMMOBILE)]
    s0 = float(np.nanstd(D_sim))
    if not np.isfinite(r["D"]) or r["D"] < SEUIL_IMMOBILE_SIGMA * s0:
        raison = "oui (D ≤ 0)" if not r["D"] > 0 else f"oui (D < {SEUIL_IMMOBILE_SIGMA:g}σ₀)"
        return dict(immobile=True, raison_immobile=raison, sigma_D0=s0)
    if np.isfinite(r["a"]) and r["a"] * 1e6 > A_MAX_IMMOBILE_UM:
        return dict(immobile=True, raison_immobile=f"oui (a > {A_MAX_IMMOBILE_UM:g} µm)", sigma_D0=s0)
    return dict(immobile=False, raison_immobile="non", sigma_D0=s0)


def statut(r, exclues=()):
    return "exclue" if r["piste"] in exclues else "sélectionnée"


def pistes_moyenne(resultats, exclues=()):
    """Pistes qui entrent dans le rayon moyen : toutes celles qui ne sont pas exclues à la main."""
    return [r for r in resultats if statut(r, exclues) != "exclue"]


def rayon_final(resultats, birge=False):
    """Rayon moyen des pistes données (None s'il n'y en a aucune).

    On moyenne les coefficients de diffusion, PAS les rayons : D_moy = (1/n) Σ D_i, puis Stokes-Einstein
    a = kT / (6πη·D_moy). Raison : a ∝ 1/D, et la moyenne de 1/D_i est biaisée vers le haut d'environ
    (σ_D/D)² ≈ +6 % pour des pistes de 67 images (vérifié par simulation), alors que D_i est sans biais.
    Ça permet aussi d'inclure les pistes à D <= 0 sans traitement spécial.

    Incertitudes : σ_PSF et σ_stat de chaque D_i propagées à la moyenne (σ_D_moy = √Σσ_i² / n), puis
    δa/a = δD_moy/D_moy. χ²_red compare la dispersion des D_i à leurs incertitudes (≈ 1 : cohérentes;
    >> 1 : billes hétérogènes — collées, freinées par la paroi... — ou incertitudes sous-estimées).
    Il est TOUJOURS calculé et affiché, mais l'incertitude n'est multipliée par le facteur de Birge
    max(1, √χ²_red) que si birge=True (bascule au choix de l'utilisateur). « inc_birge » donne
    l'incertitude ajustée dans tous les cas, pour comparaison. Si D_moy <= 0 : a = ∞."""
    resultats = [r for r in resultats if np.isfinite(r["D"])]
    if not resultats:
        return None
    D = np.array([r["D"] for r in resultats])
    # incertitudes sur D de chaque piste (δD = |D|·δa/|a|)
    rel = lambda r, cle: r[cle] / abs(r["a"]) * abs(r["D"]) if np.isfinite(r[cle]) and np.isfinite(r["a"]) \
        and r["a"] != 0 else np.nan
    sD_psf = np.array([rel(r, "inc_psf") for r in resultats])
    sD_stat = np.array([rel(r, "inc_stat") for r in resultats])
    n = len(D)
    D_moy = D.mean()
    sDm_psf = np.sqrt(np.nansum(sD_psf**2)) / n
    sDm_stat = np.sqrt(np.nansum(sD_stat**2)) / n
    sD = np.hypot(sD_psf, sD_stat)
    ok = np.isfinite(sD) & (sD > 0)
    chi2_red = np.sum((D[ok] - D_moy)**2 / sD[ok]**2) / (ok.sum() - 1) if ok.sum() > 1 else np.nan
    facteur = max(1.0, np.sqrt(chi2_red)) if np.isfinite(chi2_red) else 1.0
    applique = facteur if birge else 1.0
    commun = dict(chi2_red=chi2_red, n=n, D_moy=D_moy, inc_D=np.hypot(sDm_psf, sDm_stat) * applique,
                  birge=facteur, birge_applique=bool(birge), methode="D")
    if D_moy <= 0:
        return dict(a=np.inf, inc_psf=np.nan, inc_stat=np.nan, inc=np.nan, inc_birge=np.nan, **commun)
    a = kB * T / (6 * np.pi * ETA * D_moy)
    inc_psf, inc_stat = a * sDm_psf / D_moy, a * sDm_stat / D_moy
    inc_brute = np.hypot(inc_psf, inc_stat)
    return dict(a=a, inc_psf=inc_psf, inc_stat=inc_stat, inc=inc_brute * applique, inc_birge=inc_brute * facteur,
                **commun)


def texte_rayon(final):
    if not np.isfinite(final["a"]):
        return "a = ∞ (D moyen <= 0)"
    inc = f"{final['inc'] * 1e6:.3f}" if np.isfinite(final["inc"]) else "?"
    return f"a = {final['a'] * 1e6:.3f} ± {inc} µm"


def texte_birge(final):
    """χ²_red et effet du facteur de Birge, appliqué ou non."""
    if not np.isfinite(final["chi2_red"]):
        return "χ²_red non défini (1 seule particule)"
    t = f"χ²_red = {final['chi2_red']:.2f}"
    if not np.isfinite(final["a"]) or final["birge"] <= 1:
        return t + (" (≤ 1 : Birge sans effet)" if final["birge"] <= 1 else "")
    if final["birge_applique"]:
        return t + f", Birge APPLIQUÉ (× {final['birge']:.2f})"
    return t + f", Birge non appliqué (avec : ± {final['inc_birge'] * 1e6:.3f} µm)"


def texte_final(final):
    if final is None:
        return "Rayon final : aucune piste dans la moyenne."
    morceaux = [f"{final['n']} particules", f"D moyen = {final['D_moy'] * 1e12:.4f} µm²/s", texte_birge(final)]
    if np.isfinite(final["a"]):
        morceaux.insert(0, f"σ_PSF = {final['inc_psf'] * 1e6:.3f} ⊕ σ_stat = {final['inc_stat'] * 1e6:.3f} µm")
    return f"Rayon final : {texte_rayon(final)}  (" + " ; ".join(morceaux) + ")"


def imprimer_resultats(resultats, final, exclues=()):
    print(f"{'piste':>5s} {'n_pts':>5s} {'D (µm²/s)':>10s} {'σ_exp (nm)':>10s} {'a (µm)':>9s} "
          f"{'σ_PSF':>7s} {'σ_stat':>7s} {'σ_tot':>7s}  statut")
    for r in resultats:
        a = f"{r['a']*1e6:9.3f}" if np.isfinite(r["a"]) else f"{'∞':>9s}"
        inc = [f"{r[c]*1e6:7.3f}" if np.isfinite(r[c]) else f"{'—':>7s}" for c in ("inc_psf", "inc_stat", "inc")]
        print(f"{r['piste']:5d} {r['n_pts']:5d} {r['D']*1e12:10.3f} {r['sigma_exp']*1e9:10.0f} {a} "
              f"{' '.join(inc)}  {statut(r, exclues)}")
    print("\n" + texte_final(final))


def choisir_exclusions(resultats, exclues=None, birge=False):
    """Après l'analyse : on tape les numéros des pistes à exclure (ou à réintégrer) de la moyenne,
    autant de fois qu'on veut; « b » active/désactive le facteur de Birge; Entrée seule pour terminer.
    Retourne (exclues, final, birge)."""
    exclues = set(exclues or ())
    ids = {r["piste"] for r in resultats}
    while True:
        final = rayon_final(pistes_moyenne(resultats, exclues), birge)
        print()
        imprimer_resultats(resultats, final, exclues)
        rep = input("\nPistes à exclure ou à réintégrer (ex. 0 4 8) · b = Birge oui/non · Entrée = terminé : ").strip()
        if not rep:
            if exclues:
                print(f"Pistes exclues à la main : {sorted(exclues)}")
            return exclues, final, birge
        for t in rep.replace(",", " ").split():
            if t.lower() == "b":
                birge = not birge
                print(f"  Facteur de Birge {'appliqué' if birge else 'non appliqué'}.")
            elif t.isdigit() and int(t) in ids:
                exclues ^= {int(t)}
            else:
                print(f"  « {t} » n'est pas un numéro de piste ({sorted(ids)}).")


def tracer_rayons(ax, resultats, final, exclues=()):
    """Rayon par piste (toutes, y compris a < 0) : en couleur les pistes sélectionnées, en gris (×) les exclues."""
    finis = [r for r in resultats if np.isfinite(r["a"])]
    for r in finis:
        exclue = r["piste"] in exclues
        ax.errorbar(r["piste"], r["a"] * 1e6, yerr=r["inc"] * 1e6, fmt="x" if exclue else "o", capsize=4,
                    color="0.65" if exclue else "tab:blue", alpha=0.7 if exclue else 1)
    if final and np.isfinite(final["a"]):
        ax.axhline(final["a"] * 1e6, color="tab:green")
        ax.axhspan((final["a"] - final["inc"]) * 1e6, (final["a"] + final["inc"]) * 1e6, color="tab:green",
                   alpha=0.2, label="final : " + texte_rayon(final)
                   + " (via D moyen)")
        ax.legend()
    a = np.array([r["a"] for r in finis])
    if len(a) and a.min() > 0 and a.max() / a.min() > 20:   # grand écart (tous positifs) : échelle log
        ax.set_yscale("log")
    elif len(a) and a.min() < 0:
        ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="piste", ylabel="rayon (µm)", title="Rayon par particule (× gris = exclue)")


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
    if "somme" in meta.get("contenu", "").lower() or acq["bayer"] or acq.get("option") in ("A", "B", "C"):
        bits = "12 bits (somme 2x2 : 0–16 380)"
    elif meta or acq.get("option"):
        bits = "12 bits"
    else:
        bits = "8 bits" if acq["frames"].dtype == np.uint8 else "16 bits"
    return dict(echelle_um=acq["pixel_eff"] * 1e6,
                grossissement=(f"{acq['grossissement_reel']:.2f}X (étalonné ; nominal {acq['grossissement_nominal']}X)"
                               if acq["inc_M"] > 0 else f"{acq['grossissement_nominal']}X (nominal)"),
                exposition_ms=np.median(expo) / 1e3 if expo else None,
                fps=1 / acq["dt"], bits=bits)


def composer_image(acq, k, resultats, detections, bornes, s=1.0, exclues=(), region=None, cible=None, focus=None):
    """Image k -> image RVB 8 bits annotée.
    Vue d'ensemble : toute l'image à l'échelle s (1 = résolution native, pour l'export).
    Zoom : region = (x0, y0, x1, y1) en pixels natifs, agrandie pour remplir exactement cible = (largeur, hauteur);
    focus = n° de la piste zoomée (trait plus épais; seule la partie déjà parcourue est dessinée).
    Trajectoires colorées, cercle de 2σ (taille apparente) + étiquette « piste : a » pour chaque bille suivie
    (en gris avec « exclue » si exclue à la main), cercles rouges pour les autres détections,
    conditions d'acquisition et barre d'échelle."""
    bas, haut = bornes
    img = acq["frames"][k]
    x0 = y0 = 0
    if region is not None:
        x0, y0, x1, y1 = region
        img = img[y0:y1, x0:x1]
    v8 = np.clip((img.astype(np.float32) - bas) * (255.0 / (haut - bas)), 0, 255).astype(np.uint8)
    if region is not None:
        cw, ch = cible
        sx, sy = cw / v8.shape[1], ch / v8.shape[0]
        # agrandissement : plus proche voisin, pour voir les vrais pixels de la caméra
        v8 = cv2.resize(v8, (cw, ch), interpolation=cv2.INTER_NEAREST if sx >= 2 else cv2.INTER_LINEAR)
    else:
        sx = sy = s
        if s != 1.0:
            v8 = cv2.resize(v8, (round(v8.shape[1] * s), round(v8.shape[0] * s)), interpolation=cv2.INTER_AREA)
    rvb = cv2.cvtColor(v8, cv2.COLOR_GRAY2RGB)
    h, w = rvb.shape[:2]
    ep = max(1, round(w / 700))
    slot = acq["indices"][k]

    # pixel natif (x, y) -> pixel affiché : le centre du pixel i couvre [i - 0.5, i + 0.5]
    def vers_ecran(xy):
        xy = np.asarray(xy, dtype=float)
        return np.stack([(xy[..., 0] - x0 + 0.5) * sx - 0.5, (xy[..., 1] - y0 + 0.5) * sy - 0.5], axis=-1)

    etiquettes, suivis = [], []
    for r in resultats:
        j = slot - r["f0"]
        if not 0 <= j < len(r["pos_px"]):
            continue
        st = statut(r, exclues)
        c = GRIS_EXCLUE if st == "exclue" else couleur_piste(r["piste"])
        epp = ep * (2 if r["piste"] == focus else 1)
        pts = r["pos_px"][:j + 1]
        pts = vers_ecran(pts[np.isfinite(pts[:, 0])])
        if len(pts) > 1:
            cv2.polylines(rvb, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, c, epp, cv2.LINE_AA)
        if np.isfinite(r["pos_px"][j, 0]):
            x, y = r["pos_px"][j]
            X, Y = vers_ecran((x, y))
            rayon = max(3.0, 2 * r["sig_px"][j] * sx)
            cv2.circle(rvb, (round(X), round(Y)), round(rayon), c, epp + 1, cv2.LINE_AA)
            texte = f"{r['a'] * 1e6:.2f} µm" if np.isfinite(r["a"]) else "a = ∞"
            if st == "exclue":
                texte += " (exclue)"
            etiquettes.append(((X + rayon + 3, Y - rayon - 2), f"{r['piste']} : {texte}", c))
            suivis.append((x, y))
    det = detections[k]
    for x, y in det[:, :2] if len(det) else []:
        if not suivis or min(np.hypot(x - a, y - b) for a, b in suivis) > 1:
            X, Y = vers_ecran((x, y))
            cv2.circle(rvb, (round(X), round(Y)), 7 * ep, ROUGE_DET, ep, cv2.LINE_AA)
    return annoter(rvb, acq, k, sx, etiquettes)


def contraste_region(img, region):
    """Bornes d'affichage calculées sur une région (x0, y0, x1, y1) seulement."""
    x0, y0, x1, y1 = region
    bas, haut = np.percentile(img[y0:y1, x0:x1], (0.5, 99.9))
    return bas, max(haut, bas + 1)


def region_zoom(r, forme, rapport, marge_px):
    """Région (x0, y0, x1, y1), en pixels natifs, qui contient toute la trajectoire de la piste r avec une marge,
    au même rapport largeur/hauteur que la vue d'ensemble (la vidéo garde sa taille à l'écran)."""
    H, W = forme
    pos = r["pos_px"][np.isfinite(r["pos_px"][:, 0])]
    cx, cy = (pos.min(axis=0) + pos.max(axis=0)) / 2
    larg = max(np.ptp(pos[:, 0]) + 2 * marge_px, (np.ptp(pos[:, 1]) + 2 * marge_px) * rapport, 24)
    larg = min(larg, W)
    haut = min(larg / rapport, H)
    larg = haut * rapport
    x0 = int(round(min(max(cx - larg / 2, 0), W - larg)))
    y0 = int(round(min(max(cy - haut / 2, 0), H - haut)))
    return x0, y0, x0 + int(round(larg)), y0 + int(round(haut))


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
              f"Grossissement : {cond['grossissement']}",
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
    if MODE_TEST:     # vidéo du mandat : chaque image avec temps, échelle et conditions d'acquisition, sans annotations
        mp4c = mp4.with_name(mp4.name.replace("_annote.mp4", "_conditions.mp4"))
        video = cv2.VideoWriter(str(mp4c), cv2.VideoWriter_fourcc(*"mp4v"), 1 / acq["dt"], (w, h))
        vides = [np.empty((0, 2))] * n
        for k in range(n):
            video.write(cv2.cvtColor(composer_image(acq, k, [], vides, bornes, 1.0), cv2.COLOR_RGB2BGR))
        video.release()
        print(f"  vidéo des conditions d'acquisition : {mp4c.name}", flush=True)

    fmt = lambda v, f, e: f % (v * e) if np.isfinite(v) else ""
    fichier_csv = base.with_name(base.name + "_analyse.csv")
    with open(fichier_csv, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow(["piste", "statut", "immobile", "images", "D_um2_s", "D_sur_sigma0", "sigma_exp_nm", "a_um",
                     "sigma_PSF_um", "sigma_stat_um", "sigma_tot_um"])
        for r in resultats:
            wr.writerow([r["piste"], statut(r, exclues), r["raison_immobile"], r["n_pts"], fmt(r["D"], "%.5f", 1e12),
                         fmt(r["D"] / r["sigma_D0"], "%.2f", 1) if r["sigma_D0"] > 0 else "",
                         fmt(r["sigma_exp"], "%.1f", 1e9), fmt(r["a"], "%.5f", 1e6) or "inf",
                         fmt(r["inc_psf"], "%.5f", 1e6), fmt(r["inc_stat"], "%.5f", 1e6), fmt(r["inc"], "%.5f", 1e6)])
        wr.writerow([])
        if final:
            wr.writerow(["rayon_final_um", f"{final['a'] * 1e6:.5f}", "incertitude_um", f"{final['inc'] * 1e6:.5f}",
                         "chi2_red", f"{final['chi2_red']:.3f}", "facteur_birge", f"{final['birge']:.3f}",
                         "birge_applique", "oui" if final["birge_applique"] else "non",
                         "incertitude_avec_birge_um", f"{final['inc_birge'] * 1e6:.5f}", "n_particules", final["n"],
                         "methode", "moyenne des D" if final.get("methode") == "D"
                         else "moyenne ponderee des rayons"])
        wr.writerow(["pistes_exclues_a_la_main", " ".join(map(str, sorted(exclues))) or "aucune"])
        cond = conditions_acquisition(acq)
        wr.writerow(["decalages_L_msd", LAG_MAX])
        wr.writerow(["vibration_retiree", "oui" if INFO_VIBRATION.get("vibration") else "non",
                     "references_vibration", " ".join(map(str, INFO_VIBRATION.get("refs", []))) or "-"])
        wr.writerow(["echelle_um_px", f"{cond['echelle_um']:.5f}", "dt_s", f"{acq['dt']:.5f}",
                     "grossissement", cond["grossissement"]])
    print(f"Exporté : {mp4.name} et {fichier_csv.name} (dans {base.parent})")


def agrandir_fenetre(fig):
    """Ouvre la fenêtre de l'analyse en plein écran (maximisée) quand le système le permet : vidéo plus grande."""
    try:
        fen = fig.canvas.manager.window
    except AttributeError:
        return
    for essai in (lambda: fen.state("zoomed"),                 # TkAgg (Windows)
                  lambda: fen.attributes("-zoomed", True),     # TkAgg (Linux)
                  lambda: fen.showMaximized()):                # QtAgg
        try:
            essai()
            return
        except Exception:
            pass


def animer_resultats(acq, resultats, detections, chemin, sigma_psf=None):
    """Vidéo annotée en boucle au rythme réel + panneau de résultats.
    Espace = pause · CLIC sur une bille (vidéo), un point (graphique des rayons) ou une ligne (tableau) = exclure /
    réintégrer la piste · r = réintégrer toutes les pistes · e = exporter MP4 + CSV ·
    flèches ◀ ▶ (boutons au-dessus de la vidéo, ou touches ← →) = zoom sur la trajectoire de chaque bille
    SÉLECTIONNÉE, à tour de rôle, puis retour à la vue d'ensemble. Retourne les pistes exclues."""
    for cle, touches in (("keymap.back", ("left",)), ("keymap.forward", ("right",))):   # ← → : zoom, pas
        plt.rcParams[cle] = [t for t in plt.rcParams[cle] if t not in touches]          # l'historique de vue
    frames, n = acq["frames"], len(acq["frames"])
    bornes = contraste(frames[n // 2])
    H, W = frames.shape[1:]
    s = min(1.0, LARGEUR_AFFICHAGE_MAX / W)
    cible = (round(W * s), round(H * s))                    # taille de la vidéo à l'écran (vue d'ensemble et zoom)
    pe, dt = acq["pixel_eff"], acq["dt"]
    par_id = {r["piste"]: r for r in resultats}
    marge = max(12.0, 4 * (sigma_psf or 3.0))               # px natifs autour de la trajectoire zoomée
    regions = {r["piste"]: region_zoom(r, (H, W), W / H, marge) for r in resultats}
    # contraste propre à chaque zoom (sinon une bille brillante sature en blanc une fois agrandie)
    bornes_zoom = {p: contraste_region(frames[n // 2], reg) for p, reg in regions.items()}
    etat = dict(k=0, exclues=set(), final=rayon_final(pistes_moyenne(resultats)), pause=False, vue=None, birge=False)

    etat["epure"] = False                   # vue épurée : ni trajectoires, ni cercles, ni numéros
    vides = [np.empty((0, 2))] * n

    def image(k):
        p = etat["vue"]
        res, det = ([], vides) if etat["epure"] else (resultats, detections)
        if p is None:
            return composer_image(acq, k, res, det, bornes, s, etat["exclues"])
        return composer_image(acq, k, res, det, bornes_zoom[p], exclues=etat["exclues"],
                              region=regions[p], cible=cible, focus=p)

    def ecran_vers_natif(X, Y):
        """Clic dans la vidéo affichée -> pixel natif; et l'échelle (px affichés par px natif)."""
        if etat["vue"] is None:
            return (X + 0.5) / s - 0.5, (Y + 0.5) / s - 0.5, s
        x0, y0, x1, y1 = regions[etat["vue"]]
        sx, sy = cible[0] / (x1 - x0), cible[1] / (y1 - y0)
        return x0 + (X + 0.5) / sx - 0.5, y0 + (Y + 0.5) / sy - 0.5, sx

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "legend.title_fontsize": 7})
    fig = plt.figure(figsize=(17, 9.5))
    agrandir_fenetre(fig)
    if MODE_TEST:                           # tableau de hauteur fixe (défilable) et carte réduite : vidéo plus grande
        h_tableau, h_carte = 0.55, 0.4
    else:                                   # la rangée du tableau grandit avec le nombre de pistes (plafonnée)
        h_tableau, h_carte = min(1.2, 0.7 * max(1.0, (len(resultats) + 1) / 12)), 0.75
    gs = fig.add_gridspec(3, 2, width_ratios=[2.4, 1], height_ratios=[h_carte, 1.9, h_tableau])
    ax_v = fig.add_subplot(gs[0:2, 0])
    ax_t = fig.add_subplot(gs[2, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_m = fig.add_subplot(gs[1, 1])
    ax_r = fig.add_subplot(gs[2, 1])
    im = ax_v.imshow(image(0))
    ax_v.set_axis_off()

    # --- MSD et ajustement linéaire, une couleur par piste (dessiné une fois; style mis à jour)
    lignes_msd = {}
    for r in resultats:
        col = couleur_mpl(r["piste"])
        pos = r.get("pos_msd", r["pos_px"]) * pe        # positions du calcul (vibration retirée si demandé)
        lags, msd = msd_temporel(pos, LAG_MAX)
        ok = np.isfinite(msd)
        D, off = ajuste_msd(pos, dt, LAG_MAX)
        tt = np.linspace(0, lags[-1] * dt, 20)
        pts, = ax_m.plot(lags[ok] * dt, msd[ok] * 1e12, "o", color=col, ms=4)
        fit, = ax_m.plot(tt, (4 * D * tt + (off if np.isfinite(off) else 0)) * 1e12, "-", color=col, lw=1,
                         label=str(r["piste"]))
        lignes_msd[r["piste"]] = (pts, fit, col)
    ax_m.set(xlabel="Délai τ (s)", ylabel="MSD (µm²)",
             title=f"MSD et ajustement · L = {LAG_MAX}")
    ax_m.grid(alpha=0.3)
    if resultats:
        ax_m.legend(title="piste", fontsize=7, ncol=2)

    def dessiner_panneaux():
        """Carte, rayons et tableau : redessinés après chaque exclusion."""
        final, exclues = etat["final"], etat["exclues"]
        ax_c.clear()
        ax_c.set_axis_off()
        ax_c.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_c.transAxes, color="#eef3f8", zorder=0))
        ax_c.text(0.04, 0.95, "RAYON MOYEN DES PARTICULES", fontsize=9, fontweight="bold", va="top",
                  transform=ax_c.transAxes)
        if final and MODE_TEST:                 # analyse 3 : seulement la valeur de a et son incertitude
            ax_c.text(0.04, 0.42, texte_rayon(final), fontsize=20, va="center",
                      fontweight="bold", color="#0b3d6b", transform=ax_c.transAxes)
            details = ""
        elif final:
            ax_c.text(0.04, 0.70, texte_rayon(final), fontsize=17, va="center",
                      fontweight="bold", color="#0b3d6b", transform=ax_c.transAxes)
            details = (f"{final['n']} particule(s)  ·  D moyen = {final['D_moy'] * 1e12:.4f} µm²/s → Stokes-Einstein"
                       + (f"\nσ_PSF (localisation) = {final['inc_psf'] * 1e6:.3f} µm ⊕ σ_stat = "
                          f"{final['inc_stat'] * 1e6:.3f} µm" if np.isfinite(final["a"]) else "")
                       + "\n" + texte_birge(final) + "  [b = Birge oui/non]")
        else:
            ax_c.text(0.04, 0.70, "Aucune piste dans la moyenne", fontsize=14, va="center", transform=ax_c.transAxes)
            details = ""
        details += f"\nMSD sur L = {LAG_MAX} décalage(s) · exclues à la main : {sorted(exclues) if exclues else 'aucune'}"
        if not MODE_TEST:
            ax_c.text(0.04, 0.04, details, fontsize=8, va="bottom", linespacing=1.25, transform=ax_c.transAxes)

        ax_r.clear()
        tracer_rayons(ax_r, resultats, final, exclues)
        ax_r.set_title(ax_r.get_title() + " · clic = exclure", fontsize=8)
        ax_r.grid(alpha=0.3)
        etat["actifs"] = ax_r.scatter([], [], s=200, facecolors="none", edgecolors="red", lw=1.5, zorder=5)

        for p, (pts, fit, col) in lignes_msd.items():
            ex = p in exclues
            for l in (pts, fit):
                l.set_color("0.7" if ex else col)
            fit.set_linestyle("--" if ex else "-")

        ax_t.clear()
        ax_t.set_axis_off()
        etat["tab"] = None
        if resultats:
            tetes = ["Piste", "Statut", "Immobile ?", "Images", "D (µm²/s)", "D / σ₀", "σ_exp (nm)", "a (µm)",
                     "σ_PSF (µm)", "σ_stat (µm)", "σ_tot (µm)"]
            f = lambda v, e, d: f"{v * e:.{d}f}" if np.isfinite(v) else "—"
            cellules = [[r["piste"], statut(r, exclues), r["raison_immobile"], r["n_pts"], f(r["D"], 1e12, 3),
                         f(r["D"] / r["sigma_D0"], 1, 1) if r["sigma_D0"] > 0 else "—", f(r["sigma_exp"], 1e9, 0),
                         f(r["a"], 1e6, 3) if np.isfinite(r["a"]) else "∞", f(r["inc_psf"], 1e6, 3),
                         f(r["inc_stat"], 1e6, 3), f(r["inc"], 1e6, 3)] for r in resultats]
            hauteur_pt = ax_t.get_position().height * fig.get_figheight() * 72    # hauteur de la zone (points)
            largeur_pt = ax_t.get_position().width * fig.get_figwidth() * 72
            # largeur de chaque colonne selon son texte le plus long; police assez petite pour que tout tienne
            n_car = [max([len(str(tetes[j]))] + [len(str(c[j])) for c in cellules]) + 2 for j in range(len(tetes))]
            police = min(8.5 if MODE_TEST else 8.0, largeur_pt / (0.58 * sum(n_car)))
            deb, visibles = 0, resultats
            if MODE_TEST:                       # au moins 5 lignes visibles (police réduite au besoin), le reste défile
                police = max(4.0, min(police, 0.6 * hauteur_pt / 6))
                n_vis = max(5, int(0.6 * hauteur_pt / police) - 1)
                deb = int(np.clip(etat.get("deb", 0), 0, max(0, len(resultats) - n_vis)))
                etat["deb"] = deb
                visibles = resultats[deb:deb + n_vis]
                cellules = cellules[deb:deb + n_vis]
            else:
                police = max(4.0, min(police, 0.6 * hauteur_pt / (len(resultats) + 1)))
            etat["tab_deb"] = deb
            # bbox = toute la zone du tableau : les lignes se partagent la hauteur disponible, rien ne déborde
            tab = ax_t.table(cellText=cellules, colLabels=tetes, bbox=[0, 0, 1, 1], cellLoc="center")
            etat["tab"] = tab
            tab.auto_set_font_size(False)
            tab.set_fontsize(police)
            for (_, j), case in tab.get_celld().items():
                case.set_width(n_car[j])           # relatif : le tableau est remis à l'échelle dans sa zone
            for i, r in enumerate(visibles, 1):
                if r["piste"] in exclues:
                    for j in range(len(tetes)):
                        tab[i, j].get_text().set_color("0.6")
                else:
                    tab[i, 0].set_facecolor(couleur_mpl(r["piste"]) + (0.45,))
            if len(visibles) < len(resultats):
                ax_t.text(1.0, 1.01, f"lignes {deb + 1}–{deb + len(visibles)} sur {len(resultats)}  ·  molette ou ↑ ↓ "
                          "pour défiler", transform=ax_t.transAxes, ha="right", va="bottom", fontsize=8, color="0.4")

    def maj(k):
        etat["k"] = k
        exclues = etat["exclues"]
        im.set_data(image(k))
        slot = acq["indices"][k]
        presents = [r for r in resultats
                    if 0 <= slot - r["f0"] < len(r["pos_px"]) and np.isfinite(r["pos_px"][slot - r["f0"], 0])]
        ids_presents = {r["piste"] for r in presents}
        vue = etat["vue"]
        for p, (pts, fit, col) in lignes_msd.items():
            for l in (pts, fit):
                l.set_alpha((1.0 if p == vue else 0.12) if vue is not None else (1.0 if p in ids_presents else 0.2))
        finis = [r for r in presents if np.isfinite(r["a"]) and (vue is None or r["piste"] == vue)]
        etat["actifs"].set_offsets(np.column_stack([[r["piste"] for r in finis], [r["a"] * 1e6 for r in finis]])
                                   if finis else np.empty((0, 2)))
        perdues = int(acq["indices"][-1] + 1 - n)
        fig.suptitle(y=0.995, t=f"{chemin.name} — image {k + 1}/{n}   "
                     + (f"ZOOM sur la piste {vue}" if vue is not None else f"{len(presents)} bille(s) suivie(s)")
                     + (f"   ({perdues} images perdues à l'acquisition)" if perdues else "")
                     + ("   [vibration retirée]" if INFO_VIBRATION.get("vibration") else "")
                     + ("" if MODE_TEST else "\n[espace = pause · ◀ ▶ ou ← → = zoom · a = épuré · clic ou rectangle (vidéo), clic (rayons, tableau) = exclure/réintégrer"
                       " · b = Birge · r = tout réintégrer · e = exporter]"),
                     fontsize=8.5)
        return []

    def basculer(pids):
        for p in pids:
            etat["exclues"] ^= {p}
            print(f"Piste {p} {'exclue' if p in etat['exclues'] else 'réintégrée'}")
        recalculer()

    def recalculer():
        etat["final"] = rayon_final(pistes_moyenne(resultats, etat["exclues"]), etat["birge"])
        print("  -> " + texte_final(etat["final"]), flush=True)
        dessiner_panneaux()
        maj(etat["k"])
        fig.canvas.draw_idle()

    def ligne_du_tableau(evt):
        """Piste de la ligne du tableau sous le clic (n'importe quelle case de la ligne), sinon None."""
        tab = etat.get("tab")
        if tab is None:
            return None
        for (ligne, _), case in tab.get_celld().items():
            if ligne >= 1 and case.contains(evt)[0]:
                return resultats[etat.get("tab_deb", 0) + ligne - 1]["piste"]
        return None

    def bille_proche(X, Y):
        """Piste la plus proche d'un clic dans la vidéo affichée (None si aucune assez près)."""
        slot = acq["indices"][etat["k"]]
        x, y, echelle = ecran_vers_natif(X, Y)
        meilleur, d_min = None, np.inf
        for r in resultats:
            j = slot - r["f0"]
            if 0 <= j < len(r["pos_px"]) and np.isfinite(r["pos_px"][j, 0]):
                d = np.hypot(r["pos_px"][j, 0] - x, r["pos_px"][j, 1] - y)
                # accepté à moins de 25 px d'écran OU à l'intérieur du cercle de 2σ (utile en zoom)
                if d < max(25 / echelle, 2.5 * r["sig_px"][j]) and d < d_min:
                    meilleur, d_min = r["piste"], d
        return meilleur

    def billes_dans(X0, Y0, X1, Y1):
        """Pistes présentes à l'image affichée dont la position est dans le rectangle (coordonnées d'écran)."""
        slot = acq["indices"][etat["k"]]
        xa, ya, _ = ecran_vers_natif(min(X0, X1), min(Y0, Y1))
        xb, yb, _ = ecran_vers_natif(max(X0, X1), max(Y0, Y1))
        dedans = []
        for r in resultats:
            j = slot - r["f0"]
            if 0 <= j < len(r["pos_px"]) and np.isfinite(r["pos_px"][j, 0]):
                x, y = r["pos_px"][j, :2]
                if xa <= x <= xb and ya <= y <= yb:
                    dedans.append(r["piste"])
        return dedans

    def glisser(evt):
        if etat.get("glisse") is None or evt.inaxes is not ax_v or evt.xdata is None:
            return
        _, _, X0, Y0 = etat["glisse"]
        etat["rect"].set_bounds(min(X0, evt.xdata), min(Y0, evt.ydata), abs(evt.xdata - X0), abs(evt.ydata - Y0))
        etat["rect"].set_visible(True)
        fig.canvas.draw_idle()

    def relacher(evt):
        g = etat.pop("glisse", None)
        etat["rect"].set_visible(False)
        if g is None:
            return
        x0, y0, X0, Y0 = g
        if np.hypot(evt.x - x0, evt.y - y0) < 6:          # pas de glissement : clic simple
            p = bille_proche(X0, Y0)
            if p is not None:
                basculer([p])
            fig.canvas.draw_idle()
            return
        X1 = evt.xdata if evt.inaxes is ax_v and evt.xdata is not None else X0 + etat["rect"].get_width()
        Y1 = evt.ydata if evt.inaxes is ax_v and evt.ydata is not None else Y0 + etat["rect"].get_height()
        dedans = billes_dans(X0, Y0, X1, Y1)
        if not dedans:
            print("Rectangle : aucune bille à l'image affichée.")
            fig.canvas.draw_idle()
            return
        # toutes déjà exclues -> on les réintègre toutes; sinon on les exclut toutes
        cibles = dedans if all(p in etat["exclues"] for p in dedans) else [p for p in dedans if p not in etat["exclues"]]
        print(f"Rectangle : {len(dedans)} bille(s) {dedans}")
        basculer(cibles)

    def clic(evt):
        if not resultats or fig.canvas.toolbar is not None and fig.canvas.toolbar.mode:
            return
        p = ligne_du_tableau(evt)                        # tableau : clic sur le numéro (ou la ligne)
        if p is not None:
            basculer([p])
            return
        if evt.xdata is None:
            return
        if evt.inaxes is ax_v:                           # vidéo : clic simple OU rectangle (au relâchement)
            if evt.button == 1:
                etat["glisse"] = (evt.x, evt.y, evt.xdata, evt.ydata)
                etat["rect"].set_bounds(evt.xdata, evt.ydata, 0, 0)
            return
        elif evt.inaxes is ax_r:                         # point le plus proche (en x = numéro de piste)
            finis = [r["piste"] for r in resultats if np.isfinite(r["a"])]
            if finis:
                p = min(finis, key=lambda q: abs(q - evt.xdata))
                if abs(p - evt.xdata) < 0.5:
                    basculer([p])

    def naviguer(sens):
        """Vue d'ensemble -> zoom sur chaque bille sélectionnée (dans l'ordre des numéros) -> vue d'ensemble."""
        ordre = [None] + [r["piste"] for r in resultats if r["piste"] not in etat["exclues"]]
        vue = etat["vue"]
        if vue in ordre:
            i = ordre.index(vue)
        else:                                           # bille zoomée exclue entre-temps : on repart d'à côté
            i = max([j for j, q in enumerate(ordre) if q is not None and q < vue] or [0])
        etat["vue"] = ordre[(i + sens) % len(ordre)]
        n_sel = len(ordre) - 1
        etat["texte_vue"].set_text("Vue d'ensemble" if etat["vue"] is None else
                                   f"Zoom : piste {etat['vue']} ({ordre.index(etat['vue'])}/{n_sel} sélectionnées)")
        maj(etat["k"])
        fig.canvas.draw_idle()

    def defiler_tableau(pas):
        etat["deb"] = etat.get("deb", 0) + pas
        dessiner_panneaux()
        fig.canvas.draw_idle()

    def molette(evt):
        if MODE_TEST and evt.inaxes is ax_t:
            defiler_tableau(-1 if evt.button == "up" else 1)

    def touche(evt):
        if evt.key == "a":
            etat["basculer_epure"]()
        elif evt.key in ("up", "down") and MODE_TEST:
            defiler_tableau(-1 if evt.key == "up" else 1)
        elif evt.key == "right":
            naviguer(+1)
        elif evt.key == "left":
            naviguer(-1)
        elif evt.key == " ":
            (anim.resume if etat["pause"] else anim.pause)()
            etat["pause"] = not etat["pause"]
        elif evt.key == "b":
            etat["birge"] = not etat["birge"]
            print(f"Facteur de Birge {'appliqué' if etat['birge'] else 'non appliqué'}")
            recalculer()
        elif evt.key == "r" and etat["exclues"]:
            basculer(sorted(etat["exclues"]))
        elif evt.key == "e":
            anim.pause()
            print("\nExport en cours...", flush=True)
            exporter(acq, resultats, etat["final"], detections, bornes, chemin, etat["exclues"])
            if not etat["pause"]:
                anim.resume()

    dessiner_panneaux()
    ax_g, ax_d, ax_e = fig.add_axes([0, 0, 0.01, 0.01]), fig.add_axes([0, 0, 0.01, 0.01]), fig.add_axes([0, 0, 0.01, 0.01])
    bouton_g, bouton_d = Button(ax_g, "◀"), Button(ax_d, "▶")
    bouton_e = Button(ax_e, "Épuré : non")
    bouton_g.on_clicked(lambda _: naviguer(-1))
    bouton_d.on_clicked(lambda _: naviguer(+1))

    def basculer_epure(_=None):
        etat["epure"] = not etat["epure"]
        bouton_e.label.set_text("Épuré : oui" if etat["epure"] else "Épuré : non")
        maj(etat["k"])
        fig.canvas.draw_idle()

    bouton_e.on_clicked(basculer_epure)
    etat["basculer_epure"] = basculer_epure
    etat["texte_vue"] = fig.text(0, 0, "Vue d'ensemble", va="center", fontsize=10, fontweight="bold")

    def mise_en_page(_=None):
        """Positions calculées une fois, à la taille nominale de la figure : en plein écran, les proportions
        sont gardées (la vidéo grandit avec la fenêtre). Le tableau, lui, est recalculé à chaque redimensionnement."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout(rect=(0, 0.02, 1, 0.91), h_pad=2.2, w_pad=2.5)
        bb = ax_v.get_position()
        hb = 0.03
        ax_g.set_position([bb.x0, bb.y1 + 0.004, 0.028, hb])
        ax_d.set_position([bb.x0 + 0.031, bb.y1 + 0.004, 0.028, hb])
        ax_e.set_position([bb.x1 - 0.085, bb.y1 + 0.004, 0.085, hb])
        etat["texte_vue"].set_position((bb.x0 + 0.066, bb.y1 + 0.02))
        dessiner_panneaux()

    mise_en_page()
    fig.canvas.mpl_connect("resize_event", lambda e: (dessiner_panneaux(), fig.canvas.draw_idle()))
    fig._boutons = (bouton_g, bouton_d, bouton_e)     # garde une référence, sinon les boutons ne répondent plus
    anim = FuncAnimation(fig, maj, frames=n, interval=1000 * dt, repeat=True)
    fig.canvas.mpl_connect("key_press_event", touche)
    fig.canvas.mpl_connect("button_press_event", clic)
    fig.canvas.mpl_connect("motion_notify_event", glisser)
    fig.canvas.mpl_connect("button_release_event", relacher)
    etat["rect"] = ax_v.add_patch(plt.Rectangle((0, 0), 0, 0, fill=True, facecolor="yellow", alpha=0.15,
                                                edgecolor="yellow", lw=1.2, ls="--", visible=False, zorder=10))
    fig.canvas.mpl_connect("scroll_event", molette)
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


def _videos_du_dossier(d):
    return sorted(p for p in d.iterdir() if p.is_file() and not p.stem.endswith(("_annote", "_conditions"))
                  and p.suffix.lower() in EXTENSIONS_TIFF | EXTENSIONS_VIDEO)      # pas les vidéos exportées


def _sous_dossier_de_videos(d):
    """Sous-dossier qui contient des vidéos complètes avec leur .json (ex. _avant_premier_trou) : on liste ses
    fichiers un par un. (Un dossier d'images TIFF séparées, sans .json, reste UNE acquisition.)"""
    return d.is_dir() and any(q.suffix.lower() == ".json" for q in d.iterdir()) and bool(_videos_du_dossier(d))


def choisir_fichier(dossier=DOSSIER_DONNEES):
    sous = [d for d in sorted(dossier.iterdir()) if _sous_dossier_de_videos(d)]
    candidats = _videos_du_dossier(dossier) + sorted(
        p for p in dossier.iterdir()
        if p.is_dir() and p not in sous and any(q.suffix.lower() in EXTENSIONS_TIFF | {".png"} for q in p.iterdir()))
    for d in sous:
        candidats += _videos_du_dossier(d)
    if not candidats:
        raise FileNotFoundError(f"Aucune acquisition dans {dossier}")
    print(f"Acquisitions dans {dossier} :")
    for k, p in enumerate(candidats, 1):
        taille = (sum(q.stat().st_size for q in p.iterdir()) if p.is_dir() else p.stat().st_size) / 1e6
        nom = p.relative_to(dossier).as_posix()
        if p.parent != dossier and (k == 1 or candidats[k - 2].parent != p.parent):
            print(f"  --- dans {p.parent.name}/ ---")
        print(f"  {k}. [{lettre_option(p)}] {nom}   ({taille:.0f} Mo)"
              + ("" if chemin_json(p).exists() else "   (pas de .json)"))
    # .json d'acquisition dont les images n'ont pas été copiées : signalés, mais pas sélectionnables
    orphelins = []
    for j in sorted(dossier.glob("acq_*.json")):
        try:
            images = json.loads(j.read_text(encoding="utf-8")).get("fichier_images")
        except (OSError, ValueError):
            continue
        if images and not (dossier / images).exists():
            orphelins.append((j, images))
    if orphelins:
        print("  Non disponibles (le .json est là, mais pas les images : copier depuis la Pi) :")
        for j, images in orphelins:
            print(f"     [{lettre_option(j)}] {images}")
    while True:
        rep = input(f"Numéro du fichier (1-{len(candidats)}) : ").strip()
        if rep.isdigit() and 1 <= int(rep) <= len(candidats):
            return candidats[int(rep) - 1]
        print("  Choix invalide.")


def choisir_lag():
    """Nombre de décalages L (1 à 5) utilisés pour ajuster le MSD."""
    while True:
        rep = input(f"Nombre de décalages L pour le MSD (1 à 5) [Entrée = {LAG_MAX}] : ").strip()
        if not rep:
            return LAG_MAX
        if rep in ("1", "2", "3", "4", "5"):
            L = int(rep)
            if L == 1:
                print("  L = 1 : droite forcée par l'origine. L'erreur de localisation n'est pas retirée :\n"
                      "  D est surestimé d'environ σ_exp²/Δt (σ_exp n'est pas estimé).")
            return L
        print("  Taper un entier de 1 à 5.")


def choisir_vibration():
    """Retirer (o) ou non (N, défaut) le mouvement commun mesuré sur les billes immobiles."""
    print("Vibration : le mouvement commun à toutes les billes (vibration de la platine, dérive) peut être mesuré sur\n"
          "  les billes immobiles et retiré avant le MSD. Les billes immobiles servent alors de référence.")
    while True:
        rep = input("Retirer le mouvement commun ? (o/N) : ").strip().lower()
        if rep in ("", "n", "non"):
            return False
        if rep in ("o", "oui", "y"):
            return True
        print("  Taper o ou n.")


def choisir_analyse():
    print("Analyses :\n  1. classique (une image + détections, puis trajectoires et rayons)\n"
          "  2. dynamique (vidéo en boucle avec les billes suivies, cercles et rayons)\n"
          "  3. test en conditions réelles (comme 2, présentation épurée : rayon moyen seul, tableau défilable)")
    while True:
        rep = input("Analyse (1, 2 ou 3) : ").strip()
        if rep in ("1", "2", "3"):
            return int(rep)
        print("  Taper 1, 2 ou 3.")


#############################################################################################################
#                                                 MAIN
#############################################################################################################

if __name__ == "__main__":
    chemin = choisir_fichier()
    analyse = choisir_analyse()
    MODE_TEST = analyse == 3       # global du module : présentation de l'analyse 3
    LAG_MAX = choisir_lag()          # global du module : utilisé par l'ajustement, l'affichage et l'export
    INFO_VIBRATION["vibration"] = choisir_vibration()

    fps_sans_json = None
    if not chemin_json(chemin).exists() and chemin.suffix.lower() not in EXTENSIONS_VIDEO:
        option, G = infos_depuis_nom(chemin)
        print(f"\nATTENTION : pas de .json pour {chemin.name} (pas d'horodatages : les images perdues ne seront "
              f"pas détectées). Copier le .json depuis la Pi si possible.")
        if option:
            print(f"  Option {option} et grossissement {G:g}X lus dans le nom du fichier.")
        defaut = fps_par_defaut(chemin)
        while True:
            rep = input(f"  fps de l'acquisition [Entrée = {defaut:.2f}] : ").strip().replace(",", ".")
            try:
                fps_sans_json = float(rep) if rep else defaut
                if fps_sans_json > 0:
                    break
            except ValueError:
                pass
            print("  Entrer un nombre positif.")

    print(f"\nChargement de {chemin.name}...", flush=True)
    acq = charger_acquisition(chemin, fps_sans_json)
    frames, dt, pixel_eff, meta = acq["frames"], acq["dt"], acq["pixel_eff"], acq["meta"]
    n, h, w = frames.shape
    perdues = int(acq["indices"][-1] + 1 - n)

    print(f"Fichier          : {chemin}   [option {lettre_option(chemin)}]")
    print(f"Nombre de frames : {n}" + (f"  ({perdues} images perdues à l'acquisition)" if perdues else ""))
    print(f"Résolution       : {w} x {h} pixels" + ("  (après somme 2x2 du Bayer)" if acq["bayer"] else ""))
    print(f"Δt               : {dt * 1e3:.1f} ms  ({1 / dt:.2f} fps)")
    print(f"MSD              : ajusté sur L = {LAG_MAX} décalage(s), de Δt à {LAG_MAX}·Δt")
    print(f"Durée            : {acq['temps_s'][-1]:.2f} s")
    M, inc_M = acq["grossissement_reel"], acq["inc_M"]
    print(f"Grossissement    : {M:.3f} ± {inc_M:.3f}  (nominal {acq['grossissement_nominal']}X)"
          + (f"  -> incertitude d'étalonnage sur a : 2·δM/M = {200 * inc_M / M:.2f} %" if inc_M else ""))
    print(f"Pixel (objet)    : {pixel_eff * 1e6:.4f} µm/px")
    print(f"Source           : {acq['source']}")
    if not meta and not acq["option"]:
        print("ATTENTION : pas de .json ni d'option dans le nom -> grossissement, pixel et fps tirés des constantes.")

    kw_det, dist_max_px, demi = parametres_pixels(pixel_eff)

    if analyse == 1:
        INDICE_FRAME = n // 2   # à modifier
        gris = afficher_frame(frames, INDICE_FRAME, acq["temps_s"][INDICE_FRAME])
        particules = compter_particules(gris, indice=INDICE_FRAME, **kw_det)
        afficher_detection(gris, particules, indice=INDICE_FRAME)

    try:
        sigma_psf, disp_psf, n_fit = estimer_sigma_psf(frames, demi=demi, **kw_det)
        if n_fit < 5:
            raise RuntimeError
    except RuntimeError:
        # Éclairage très non uniforme (taches faibles sur un fond en gradient) : le seuil global ne voit rien.
        print("Peu ou pas de taches détectées : retrait du fond lisse (éclairage non uniforme), image par image...")
        retirer_fond_lent(frames, 2 * demi + 1)
        sigma_psf, disp_psf, n_fit = estimer_sigma_psf(frames, demi=demi, **kw_det)
    print(f"σ_PSF mesuré : {sigma_psf:.2f} ± {disp_psf:.2f} px ({sigma_psf*pixel_eff*1e6:.2f} µm), {n_fit} PSF")
    # Garde-fous tirés de la taille MESURÉE des taches (indépendants de l'échelle supposée) :
    # - deux maxima à moins de 2σ appartiennent à la même bille (évite les doubles détections);
    # - une bille peut se déplacer d'au moins 1,5σ entre deux images sans changer de numéro.
    kw_det["distance_min"] = max(kw_det["distance_min"], int(np.ceil(2 * sigma_psf)))
    dist_max_px = max(dist_max_px, 1.5 * sigma_psf)
    print(f"Détection : distance min. entre billes = {kw_det['distance_min']} px · "
          f"déplacement max. entre images = {dist_max_px:.1f} px")

    detections = localiser_toutes_frames(frames, sigma_psf, **kw_det)
    pistes = lier_trajectoires(detections, acq["indices"], dist_max_px=dist_max_px, trou_max=TROU_MAX)
    pistes_tab = pistes_en_tableaux(pistes, n_min=N_MIN)
    resultats = analyser_pistes(pistes_tab, dt, pixel_eff=pixel_eff, lag_max=LAG_MAX, t_exp=acq["t_exp"])
    if INFO_VIBRATION["vibration"] and resultats:
        corrigees, refs, ec, corr, n_passes = corriger_vibration(
            pistes_tab, resultats, int(acq["indices"][-1]) + 1,
            analyser=lambda p: analyser_pistes(p, dt, pixel_eff=pixel_eff, lag_max=LAG_MAX, t_exp=acq["t_exp"]))
        if corrigees is None:
            print(f"Vibration NON retirée : il faut au moins 2 billes immobiles de référence ({len(refs)} trouvée(s)).")
            INFO_VIBRATION["vibration"] = False
        else:
            INFO_VIBRATION["refs"] = refs
            print(f"Vibration retirée : références = pistes {refs} (billes immobiles, choisies en {n_passes} passe(s)) "
                  f"· mouvement commun entre images : {ec * pixel_eff * 1e9:.0f} nm (écart type)")
            for r, rc, t in zip(resultats, corr, corrigees):
                rc.update(pos_px=r["pos_px"], pos_msd=t["pos_px"])     # affichage : positions réelles dans l'image
            resultats = corr
    if not resultats:
        print(f"Aucune piste d'au moins {N_MIN} positions : impossible de calculer un rayon.")
    imprimer_resultats(resultats, rayon_final(pistes_moyenne(resultats)))

    if analyse == 1:
        if resultats:
            afficher_resultats(frames, resultats, rayon_final(pistes_moyenne(resultats)))
            # Exclusion à la main, une fois la figure regardée (et fermée)
            exclues, final, _ = choisir_exclusions(resultats)
            if exclues:
                afficher_resultats(frames, resultats, final, exclues)
    else:
        animer_resultats(acq, resultats, detections, chemin, sigma_psf)
