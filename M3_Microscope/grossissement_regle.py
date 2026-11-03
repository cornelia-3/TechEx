"""Calibration du grossissement réel à partir d'une photo d'une règle graduée.

Principe
    1. On mesure la distance p (en pixels) entre deux graduations consécutives de la règle.
    2. Taille d'un pixel dans le plan objet :  PIXEL_EFF = PAS_GRADUATION / p
    3. Grossissement réel :                    M = (taille d'un pixel sur le capteur) / PIXEL_EFF

Utilisation :  python calibration_grossissement.py [chemin_image.png]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.special import erf

# ---------------------------------------------------------------------------------------------
#                                       PARAMÈTRES
# ---------------------------------------------------------------------------------------------
CHEMIN_IMAGE = Path(r"C:\Users\corne\OneDrive - polymtlus\Bureau\data_2nov\regle20x_100mm.png")      # photo PNG de la règle (à modifier ou passer en argument)

PAS_GRADUATION = 100e-6    # m entre deux graduations consécutives  <-- À CONFIRMER (voir mise en garde)

TAILLE_PIXEL = 1.55e-6    # m, pixel natif du capteur (caméra HQ)
X_PIXELS = 4056           # largeur du capteur en pixels natifs
M_NOMINAL = 20            # grossissement gravé sur l'objectif (sert de vérification et d'échelle de départ)
F_TUBE = 150e-3           # m, focale de la lentille de tube

# pour l'objectif 20X :
# Inclinaison des traits : 1.39° (angle de la normale / axe x)
# Traits détectés  : 3 (indices 0 à 2)
# Pas mesuré       : 1067.992 ± 0.431 pixels
# Écart-type des résidus : 0.610 pixel
# PIXEL_EFF        = 0.09363 ± 0.00004 µm/pixel  (pour cette image)
# Grossissement réel M = 16.554 ± 0.007   (nominal : 20X




# ---------------------------------------------------------------------------------------------
#                                       CHARGEMENT
# ---------------------------------------------------------------------------------------------.    
def charger_image(chemin):
    """Charge l'image en niveaux de gris (float), en gardant la profondeur de bits du fichier."""
    img = cv2.imread(str(chemin), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Impossible d'ouvrir l'image : {chemin}")
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    return img.astype(float)


# ---------------------------------------------------------------------------------------------
#                       PROFIL PERPENDICULAIRE AUX TRAITS (corrige l'inclinaison)
# ---------------------------------------------------------------------------------------------
def angle_normale_aux_traits(img, reduction=4):
    """Direction dominante du gradient (normale aux traits) par rapport à l'axe x, en radians.
    Tenseur de structure sur l'image réduite : déterministe, sans paramètre à régler."""
    petit = cv2.resize(img, None, fx=1 / reduction, fy=1 / reduction, interpolation=cv2.INTER_AREA)
    petit = ndi.gaussian_filter(petit, 1.0)
    gx = ndi.sobel(petit, axis=1)
    gy = ndi.sobel(petit, axis=0)
    return 0.5 * np.arctan2(2 * np.sum(gx * gy), np.sum(gx**2 - gy**2))


def profil_normal(img, theta, marge=0.03):
    """Intensité moyenne en fonction de la distance s (pixels) mesurée perpendiculairement aux traits.
    Marge faible (3 %) : avec un pas de 100 µm il n'y a que 3 ou 4 traits dans le champ."""
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


# ---------------------------------------------------------------------------------------------
#                                    DÉTECTION DES TRAITS
# ---------------------------------------------------------------------------------------------
def _gauss1d(x, A, x0, sigma, b):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b


def _creneau(x, A, x0, w, sig, b):
    """Trait de largeur w (plateau) aux bords adoucis par un flou gaussien sig."""
    r2 = np.sqrt(2) * sig
    return b + 0.5 * A * (erf((x - x0 + w / 2) / r2) - erf((x - x0 - w / 2) / r2))


def _ajuster_trait(s, signal, i, periode_attendue):
    """Position sub-pixel du trait dont le maximum approximatif est à l'indice i.
    Modèle principal : créneau flouté (traits larges à fond plat). Repli : gaussienne (traits fins)."""
    # Largeur à mi-hauteur autour du pic -> valeurs de départ
    A_est = signal[i]
    g, d = i, i
    while g > 0 and signal[g - 1] > 0.5 * A_est:
        g -= 1
    while d < len(s) - 1 and signal[d + 1] < 1e9 and signal[d + 1] > 0.5 * A_est:
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
    """Position sub-pixel (en pixels, le long de la normale) de chaque trait.
    periode_attendue : espacement approximatif en pixels (sert d'échelle, pas de valeur finale)."""
    fenetre = min(int(2.5 * periode_attendue), len(profil) - 1) | 1
    fond = ndi.median_filter(profil, size=fenetre, mode="nearest")
    brut = profil - fond

    # Polarité : les traits sont l'écart le plus ample (creux ou bosses), pas le signe qui a le plus de pics
    def amplitude(sig):
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


# ---------------------------------------------------------------------------------------------
#                                 PAS EN PIXELS ET GROSSISSEMENT
# ---------------------------------------------------------------------------------------------
def pas_en_pixels(positions, erreurs=None):
    """Régression linéaire position = x0 + p·n sur l'indice entier n de chaque trait.
    Les traits manquants (non détectés) sont repérés par un écart de plusieurs pas."""
    positions_brutes = positions
    positions = np.sort(positions)
    pas_approx = np.median(np.diff(positions))
    n = np.round((positions - positions[0]) / pas_approx).astype(int)
    (p, x0), cov = np.polyfit(n, positions, 1, cov=True)
    inc_p = np.sqrt(cov[0, 0])                    # basée sur les résidus (peu fiable si 3 ou 4 traits)
    if erreurs is not None:
        w = 1 / erreurs[np.argsort(positions_brutes)]
        _, cov_f = np.polyfit(n, positions, 1, w=w, cov="unscaled")
        inc_p = max(inc_p, np.sqrt(cov_f[0, 0]))  # incertitude formelle des ajustements de chaque trait
    residus = positions - (p * n + x0)
    return p, inc_p, n, residus


def calibrer(chemin_image=CHEMIN_IMAGE, afficher=True):
    img = charger_image(chemin_image)
    h, w = img.shape
    pixel_capteur = TAILLE_PIXEL * X_PIXELS / w          # taille d'un pixel de CETTE image sur le capteur
    periode_attendue = PAS_GRADUATION / (pixel_capteur / M_NOMINAL)

    print(f"Image            : {chemin_image}  ({w} x {h} pixels)")
    print(f"Pixel sur capteur: {pixel_capteur * 1e6:.2f} µm")
    print(f"Pas attendu si M = {M_NOMINAL}X : {periode_attendue:.1f} pixels entre traits")
    if periode_attendue < 6:
        raise ValueError("Pas attendu < 6 pixels: les traits ne seraient pas résolus. "
                         "Vérifie PAS_GRADUATION (unité !) et le grossissement nominal.")

    theta = angle_normale_aux_traits(img)
    s, profil = profil_normal(img, theta)
    positions, erreurs, signal = detecter_traits(s, profil, periode_attendue)
    if len(positions) < 3 and afficher:
        plt.figure(figsize=(10, 4))
        plt.plot(s, profil, lw=0.8, label="profil")
        plt.plot(s, signal, lw=0.8, label="signal (sans fond)")
        plt.plot(positions, np.interp(positions, s, signal), "rv", label="traits détectés")
        plt.xlabel("position le long de la normale (pixels)")
        plt.legend()
        plt.show()
    if len(positions) < 3:
        raise RuntimeError(f"Seulement {len(positions)} trait(s) détecté(s): vérifier l'image et PAS_GRADUATION.")
    if len(positions) < 5:
        print(f"ATTENTION : seulement {len(positions)} traits, l'incertitude sera grande.")

    p, inc_p, n, residus = pas_en_pixels(positions, erreurs)

    pixel_eff = PAS_GRADUATION / p
    inc_pixel_eff = pixel_eff * inc_p / p
    M = pixel_capteur / pixel_eff
    inc_M = M * inc_p / p

    print(f"\nInclinaison des traits : {np.degrees(theta):.2f}° (angle de la normale / axe x)")
    print(f"Traits détectés  : {len(positions)} (indices {n.min()} à {n.max()})")
    print(f"Pas mesuré       : {p:.3f} ± {inc_p:.3f} pixels")
    print(f"Écart-type des résidus : {residus.std(ddof=2) if len(residus) > 2 else float('nan'):.3f} pixel")
    print(f"\nPIXEL_EFF        = {pixel_eff * 1e6:.5f} ± {inc_pixel_eff * 1e6:.5f} µm/pixel  (pour cette image)")
    print(f"Grossissement réel M = {M:.3f} ± {inc_M:.3f}   (nominal : {M_NOMINAL}X)")
    print(f"Focale d'objectif équivalente = {F_TUBE / M * 1e3:.2f} mm (avec f_tube = {F_TUBE * 1e3:.0f} mm)")
    print("\nPour une vidéo 640 de large couvrant le même champ (capteur réduit) :")
    print(f"  PIXEL_EFF = {pixel_eff * w / 640 * 1e6:.4f} µm/pixel")

    if abs(p - periode_attendue) / periode_attendue > 0.4:
        print("\nATTENTION : le pas mesuré s'écarte de plus de 40 % du pas attendu pour le grossissement "
              "nominal. Vérifie PAS_GRADUATION (unité, valeur) et la focale de la lentille de tube.")

    if afficher:
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
        ax[0].plot(s, signal, lw=0.8)
        ax[0].plot(positions, np.interp(positions, s, signal), "rv", label="traits détectés")
        ax[0].set(xlabel="position le long de la normale (pixels)", ylabel="signal (sans fond)",
                  title="Profil et traits détectés")
        ax[0].legend()
        ax[1].errorbar(n, residus, yerr=erreurs[np.argsort(positions)], fmt="o", capsize=3)
        ax[1].axhline(0, color="k", lw=0.8)
        ax[1].set(xlabel="indice du trait", ylabel="résidu (pixels)", title="Résidus de la régression linéaire")
        plt.tight_layout()
        plt.show()

    return dict(pas_px=p, inc_pas_px=inc_p, pixel_eff=pixel_eff, inc_pixel_eff=inc_pixel_eff, M=M, inc_M=inc_M)


if __name__ == "__main__":
    chemin = Path(sys.argv[1]) if len(sys.argv) > 1 else CHEMIN_IMAGE
    calibrer(chemin)