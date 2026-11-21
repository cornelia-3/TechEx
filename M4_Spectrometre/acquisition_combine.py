#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHS3910 - Mandat 4 : Spectromètre (acquisition + analyse + calibration)
Raspberry Pi 4 + caméra HQ (SC1220, Sony IMX477R)

Trois modes :
  1) Source / laser : acquisition -> spectre d'intensité (position, largeur du pic)
  2) Filtre en transmission : référence (sans filtre) + mesure (avec filtre)
     -> T = I_mes / I_réf + longueurs d'onde de coupure avec incertitudes
  3) Analyse d'images existantes (sans caméra) : spectre d'intensité à partir
     d'un dossier de PNG déjà enregistrés (utilisable sur un PC)

Calibration (choisie au démarrage) :
  - aucune    : axe en pixels (non calibré)
  - theorique : modèle réseau + lentille f2, calé sur 0 ou 1 laser connu
  - lineaire  : ajustement pixel -> lambda sur >= 2 lasers connus
Si les paramètres/lasers requis manquent, retour automatique à l'axe en pixels.

Après chaque acquisition, une ligne demande des notes, enregistrées dans
notes.txt dans le dossier des images de cette acquisition.

Dépendances : sudo apt install python3-picamera2 python3-opencv python3-matplotlib
"""

import os
import glob
import time
import shutil
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from picamera2 import Picamera2, Preview
except ImportError:  # permet le mode 3 (analyse) sur un PC sans picamera2
    Picamera2 = None

# ----------------------------------------------------------------------------
# PARAMÈTRES À AJUSTER
# ----------------------------------------------------------------------------
DOSSIER_SORTIE = "acquisitions"
N_IMAGES_DEFAUT = 10
FPS_DEFAUT = 10.0            # 10 fps = maximum à 4056 x 3040
EXPOSITION_MS_DEFAUT = 20.0  # doit être < 1000/fps (ms)
GAIN_ANALOGIQUE = 1.0        # gain minimal = meilleur rapport signal/bruit

DEMI_HAUTEUR_BANDE = 50      # demi-hauteur (px) de la bande de lignes sommées
LISSAGE_PX = 5               # lissage du spectre (fenêtre en pixels)
SEUIL_REF = 0.10             # T ignorée où I_réf < 10 % du max de I_réf
FRACTION_COUPURE = 0.50      # coupure définie à 50 % de la transmission max
PLAGE_NM = (400, 700)        # plage affichée en mode source (None = tout)

# --- Calibration ---
MODE_CALIBRATION = "aucune"  # défaut proposé : "aucune", "theorique" ou "lineaire"

# Montage optique (requis pour "theorique")
LBD_CENTRALE = 550e-9   # longueur d'onde centrale supposée si aucun laser (m)
F2 = None               # focale de la 2e lentille (m)
LAMBDA_RESEAU = None    # pas du réseau (m)
L_OUVERTURE = None      # taille de l'ouverture (m) -> résolution théorique
ORDRE = 1               # ordre de diffraction
THETA_I = 0.0           # angle d'incidence sur le réseau (rad)
PIXEL_M = 1.55e-6       # pas des pixels de l'IMX477 (m)
SENS = 1                # +1 si lambda augmente avec l'indice de pixel, -1 sinon

# Lasers de calibration : soit un dossier/motif d'images, soit une position connue
# LASERS_CALIBRATION = [
#     {"images": "acquisitions/20250101_120000/images", "lbd_nm": 532.0},
#     {"pixel": 2900.0, "lbd_nm": 650.0},
# ]
LASERS_CALIBRATION = []

# Incertitude de calibration (nm) ajoutée en quadrature aux coupures
U_CALIBRATION_NM = 0.0
# ----------------------------------------------------------------------------


# ============================== Utilitaires ==================================
def demander(texte, defaut, type_=float):
    rep = input(f"{texte} [{defaut}] : ").strip()
    if rep == "":
        return type_(defaut)
    try:
        return type_(rep)
    except ValueError:
        print("Valeur invalide, valeur par défaut utilisée.")
        return type_(defaut)


def lisser(y, n=LISSAGE_PX):
    if n <= 1:
        return y
    return np.convolve(y, np.ones(n) / n, mode="same")


def demander_notes(dossier):
    """Demande une ligne de notes et l'ajoute à notes.txt dans `dossier`."""
    notes = input("Notes pour cette acquisition (Entrée pour passer) : ").strip()
    if notes:
        chemin = os.path.join(dossier, "notes.txt")
        with open(chemin, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {notes}\n")
        print(f"Notes enregistrées dans {chemin}")


def verifier_espace(dossier, n_images, taille):
    """Avertit si l'espace disque libre semble insuffisant (évite 'libpng write error')."""
    libre = shutil.disk_usage(dossier).free
    brut = n_images * taille[0] * taille[1] * 3  # borne haute (sans compression)
    print(f"Espace disque libre : {libre / 1e6:.0f} Mo "
          f"(jusqu'à ~{brut / 1e6:.0f} Mo nécessaires au maximum)")
    if libre < brut:
        print("ATTENTION : espace possiblement insuffisant. Libérez de la place "
              "(df -h) ou réduisez le nombre d'images.")


def wavelength_to_rgb(wavelength, gamma=0.8):
    """Longueur d'onde (nm, 380-750) -> couleur RGB approximative (Dan Bruton)."""
    w = float(wavelength)
    if 380 <= w <= 440:
        a = 0.3 + 0.7 * (w - 380) / 60
        R, G, B = (-(w - 440) / 60 * a) ** gamma, 0.0, a ** gamma
    elif 440 <= w <= 490:
        R, G, B = 0.0, ((w - 440) / 50) ** gamma, 1.0
    elif 490 <= w <= 510:
        R, G, B = 0.0, 1.0, (-(w - 510) / 20) ** gamma
    elif 510 <= w <= 580:
        R, G, B = ((w - 510) / 70) ** gamma, 1.0, 0.0
    elif 580 <= w <= 645:
        R, G, B = 1.0, (-(w - 645) / 65) ** gamma, 0.0
    elif 645 <= w <= 750:
        a = 0.3 + 0.7 * (750 - w) / 105
        R, G, B = a ** gamma, 0.0, 0.0
    else:
        R = G = B = 0.0
    return int(R * 255), int(G * 255), int(B * 255)


def tracer_spectre(ax, x, y, unite):
    """Spectre ; sous la courbe, couleur réelle de chaque longueur d'onde (si nm)."""
    ax.plot(x, y, "k", lw=0.8)
    if unite == "nm":
        pas = max(1, len(x) // 600)
        for i in range(0, len(x) - pas, pas):
            r, g, b = wavelength_to_rgb(x[i])
            ax.fill_between(x[i:i + pas + 1], y[i:i + pas + 1],
                            color=(r / 255, g / 255, b / 255), lw=0)


def stats_pic(x, y):
    """Position du max, centroïde et FWHM du pic principal."""
    i = int(np.argmax(y))
    demi = y[i] / 2.0
    g = i
    while g > 0 and y[g] > demi:
        g -= 1
    d = i
    while d < len(y) - 1 and y[d] > demi:
        d += 1
    f = slice(g, d + 1)
    return x[i], np.sum(x[f] * y[f]) / np.sum(y[f]), abs(x[d] - x[g])


def coupures(x, T):
    """
    Longueurs d'onde où T traverse FRACTION_COUPURE * T_max.
    Retourne (montante, descendante) ; None si absente.
    - passe-bande : montante = coupure basse, descendante = coupure haute
    - passe-long  : seulement la montante
    - passe-court : seulement la descendante
    """
    ordre = np.argsort(x)
    x, T = x[ordre], T[ordre]
    if np.all(np.isnan(T)):
        return None, None
    s = T - FRACTION_COUPURE * np.nanmax(T)
    mont, desc = [], []
    for i in range(len(x) - 1):
        if np.isnan(s[i]) or np.isnan(s[i + 1]):
            continue
        xc = x[i] - s[i] * (x[i + 1] - x[i]) / (s[i + 1] - s[i]) \
            if s[i + 1] != s[i] else None
        if xc is None:
            continue
        if s[i] < 0 <= s[i + 1]:
            mont.append(xc)
        elif s[i] >= 0 > s[i + 1]:
            desc.append(xc)
    return (mont[0] if mont else None), (desc[-1] if desc else None)


# ============================== Caméra =======================================
def demarrer_camera(fps, expo_ms):
    cam = Picamera2()
    taille = cam.sensor_resolution  # (4056, 3040) pour l'IMX477
    duree_us = int(1e6 / fps)
    controles = {"FrameDurationLimits": (duree_us, duree_us)}
    if expo_ms > 0:  # exposition manuelle
        controles.update({
            "ExposureTime": int(expo_ms * 1000),
            "AnalogueGain": GAIN_ANALOGIQUE,
            "AwbEnable": False,
            "ColourGains": (1.0, 1.0),
        })
    else:            # aperçu automatique
        controles.update({"AeEnable": True, "AwbEnable": True})
    config = cam.create_still_configuration(
        main={"size": taille, "format": "RGB888"},
        lores={"size": (taille[0] // 4, taille[1] // 4)},  # flux léger pour l'aperçu
        display="lores",  # sans ça, une config "still" n'affiche rien
        buffer_count=4,
        controls=controles,
    )
    cam.configure(config)
    print(f"Résolution : {taille[0]} x {taille[1]}")
    try:
        cam.start_preview(Preview.QTGL)
    except Exception:
        try:
            cam.start_preview(Preview.QT)
        except Exception:
            print("Aperçu indisponible (pas d'affichage) -> acquisition sans aperçu.")
            cam.start_preview(Preview.NULL)
    cam.start()
    return cam, taille


def verrouiller(cam, expo_ms):
    """
    En mode automatique (expo_ms = 0), fige exposition, gain et balance des
    blancs sur les valeurs de l'aperçu pour que toutes les images (et la
    référence vs la mesure) soient acquises avec les mêmes réglages.
    Retourne l'exposition utilisée (ms).
    """
    if expo_ms > 0:
        return expo_ms
    meta = cam.capture_metadata()
    expo_us, gain = meta["ExposureTime"], meta["AnalogueGain"]
    cam.set_controls({"AeEnable": False, "AwbEnable": False,
                      "ExposureTime": expo_us, "AnalogueGain": gain,
                      "ColourGains": meta["ColourGains"]})
    for _ in range(4):  # laisser les nouveaux réglages s'appliquer
        cam.capture_metadata()
    print(f"Réglages verrouillés : exposition {expo_us / 1000:.1f} ms, "
          f"gain {gain:.2f}")
    return expo_us / 1000.0


def acquerir(cam, n, dossier):
    """Capte n images, les enregistre en PNG, demande des notes.
    Retourne (images, fps_reel)."""
    os.makedirs(dossier, exist_ok=True)
    images, instants = [], []
    t0 = time.time()
    for i in range(n):
        images.append(cam.capture_array("main"))  # tableau BGR (convention OpenCV)
        instants.append(time.time() - t0)
        print(f"  image {i + 1}/{n}", end="\r")
    print()
    fps_reel = (n - 1) / (instants[-1] - instants[0]) if n > 1 else float("nan")
    print(f"Cadence réelle : {fps_reel:.2f} fps")

    for i, img in enumerate(images):
        chemin = os.path.join(dossier, f"image_{i:03d}.png")
        try:
            ok = cv2.imwrite(chemin, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        except cv2.error:
            ok = False
        if not ok:
            print(f"ERREUR d'écriture : {chemin}\n"
                  "  -> disque plein, carte SD défaillante ou dossier non "
                  "accessible ? Vérifier `df -h` et `dmesg | tail`.\n"
                  "  Les images restent en mémoire pour l'analyse, mais ne "
                  "sont pas toutes sauvegardées.")
            break
    else:
        print(f"Images enregistrées dans {dossier}/")

    maxi = max(int(im.max()) for im in images)
    if maxi >= 250:
        print(f"ATTENTION : saturation détectée (max = {maxi}/255). "
              "Réduire l'exposition ou l'intensité de la source.")
    demander_notes(dossier)
    return images, fps_reel


# ============================== Traitement ===================================
def image_somme(img):
    """Somme des canaux, fond (médiane) soustrait."""
    g = img.sum(axis=2, dtype=np.float32)
    g -= np.median(g[::8, ::8])
    np.clip(g, 0, None, out=g)
    return g


def traiter(images, bande=None):
    """
    Retourne (profils[n, W], bande(y0, y1, y2)).
    Si bande est None, elle est centrée sur la ligne la plus intense.
    """
    if bande is None:
        lignes = None
        for img in images:
            l = image_somme(img).sum(axis=1)
            lignes = l if lignes is None else lignes + l
        y0 = int(np.argmax(lisser(lignes, 15)))
        h = images[0].shape[0]
        bande = (y0, max(0, y0 - DEMI_HAUTEUR_BANDE), min(h, y0 + DEMI_HAUTEUR_BANDE))
    _, y1, y2 = bande
    profils = [image_somme(img)[y1:y2, :].mean(axis=0) for img in images]
    return np.array(profils), bande


def lister_images(chemin):
    """Dossier ou motif glob -> liste triée d'images."""
    if os.path.isdir(chemin):
        fichiers = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            fichiers += glob.glob(os.path.join(chemin, ext))
    else:
        fichiers = glob.glob(chemin)
    fichiers = sorted(fichiers)
    if not fichiers:
        raise FileNotFoundError(f"Aucune image trouvée : {chemin}")
    return fichiers


def profil_dossier(chemin):
    """Charge les images d'un dossier/motif (une à la fois, économe en RAM).
    Retourne (profil, image_moyenne, bande, n_images)."""
    fichiers = lister_images(chemin)
    moy, n = None, 0
    for f in fichiers:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Image illisible ignorée : {f}")
            continue
        g = image_somme(img)
        moy = g if moy is None else moy + g
        n += 1
    if n == 0:
        raise FileNotFoundError(f"Aucune image lisible : {chemin}")
    moy /= n
    y0 = int(np.argmax(lisser(moy.sum(axis=1), 15)))
    y1, y2 = max(0, y0 - DEMI_HAUTEUR_BANDE), min(moy.shape[0], y0 + DEMI_HAUTEUR_BANDE)
    return moy[y1:y2].mean(axis=0), moy, (y0, y1, y2), n


# ============================== Calibration ==================================
def choisir_calibration():
    noms = {"1": "aucune", "2": "theorique", "3": "lineaire"}
    defaut = {v: k for k, v in noms.items()}.get(MODE_CALIBRATION, "1")
    rep = input("Calibration : [1] aucune (axe en pixels)  [2] théorique  "
                f"[3] linéaire (>= 2 lasers)  [{defaut}] : ").strip()
    return noms.get(rep or defaut, "aucune")


def pixel_vers_lbd_theorique(px, px_ref, lbd_ref):
    """
    Longueur d'onde (m) à chaque pixel selon le modèle réseau + lentille f2 :
        sin(theta) = ordre*lbd/Lambda + sin(theta_i),   x = f2*tan(theta)
    calé sur un point connu (px_ref, lbd_ref).
    """
    th_ref = np.arcsin(ORDRE * lbd_ref / LAMBDA_RESEAU + np.sin(THETA_I))
    th = np.arctan(np.tan(th_ref) + SENS * (px - px_ref) * PIXEL_M / F2)
    return LAMBDA_RESEAU * (np.sin(th) - np.sin(THETA_I)) / ORDRE


def positions_lasers(n_pixels):
    """Position (pixels) et longueur d'onde (nm) de chaque laser de calibration."""
    px = np.arange(n_pixels, dtype=float)
    pos, lam = [], []
    for c in LASERS_CALIBRATION:
        if "pixel" in c:
            p = float(c["pixel"])
            print(f"Laser {c['lbd_nm']:.1f} nm : pixel {p:.2f} (donné)")
        else:
            try:
                prof, _, _, n = profil_dossier(c["images"])
            except FileNotFoundError as e:
                print(f"Laser {c['lbd_nm']} nm ignoré : {e}")
                continue
            if len(prof) != n_pixels:
                print(f"Laser {c['lbd_nm']} nm ignoré : résolution différente.")
                continue
            _, p, fwhm = stats_pic(px, lisser(prof))
            print(f"Laser {c['lbd_nm']:.1f} nm : pixel {p:.2f} "
                  f"(FWHM {fwhm:.2f} px, {n} images)")
        pos.append(p)
        lam.append(c["lbd_nm"])
    return np.array(pos), np.array(lam)


def calibrer(n_pixels, mode):
    """Retourne (axe, étiquette, unité, description)."""
    px = np.arange(n_pixels, dtype=float)
    en_pixels = (px, "Position (pixels)", "px", "aucune (axe en pixels)")

    if mode == "aucune":
        print("Mode non calibré : axe en pixels.")
        return en_pixels

    pos, lam = positions_lasers(n_pixels)

    if mode == "lineaire":
        if len(pos) < 2:
            print("ATTENTION : calibration linéaire impossible (< 2 lasers) "
                  "-> axe en pixels.")
            return en_pixels
        a, b = np.polyfit(pos, lam, 1)
        print(f"Calibration linéaire : lbd = {a:.5f} nm/px * pixel + {b:.3f} nm")
        if len(pos) > 2:
            print("Résidus (nm) :", np.round(lam - (a * pos + b), 3))
        return (a * px + b, "Longueur d'onde (nm)", "nm",
                f"lineaire : {a:.5f} nm/px * pixel + {b:.3f} nm ({len(pos)} lasers)")

    if mode == "theorique":
        if LAMBDA_RESEAU is None or F2 is None:
            print("ATTENTION : LAMBDA_RESEAU et F2 non définis -> axe en pixels.")
            return en_pixels
        if len(pos) == 0:
            print("ATTENTION : aucun laser -> lbd supposée au centre du capteur.")
            px_ref, lbd_ref = n_pixels / 2, LBD_CENTRALE
        else:
            px_ref, lbd_ref = pos[0], lam[0] * 1e-9
            for p, l in zip(pos[1:], lam[1:]):  # vérification du modèle
                l_mod = pixel_vers_lbd_theorique(p, px_ref, lbd_ref) * 1e9
                print(f"Vérification {l:.1f} nm : modèle {l_mod:.2f} nm "
                      f"(écart {l_mod - l:+.2f} nm)")
        x = pixel_vers_lbd_theorique(px, px_ref, lbd_ref) * 1e9
        return (x, "Longueur d'onde (nm)", "nm",
                f"theorique (pixel de référence {px_ref:.1f} -> {lbd_ref * 1e9:.1f} nm)")

    print("Mode de calibration inconnu -> axe en pixels.")
    return en_pixels


def ecrire_params(chemin, lignes):
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes) + "\n")


# ============================== Spectre d'intensité ==========================
def resultats_source(intensite, apercu, bande, dossier, mode_cal, infos):
    """Calibre, calcule le pic, enregistre CSV/paramètres/figure (modes 1 et 3)."""
    y0, y1, y2 = bande
    x, etiq, unite, desc = calibrer(len(intensite), mode_cal)

    masque = np.ones_like(x, bool)
    if unite == "nm" and PLAGE_NM:
        m = (x >= PLAGE_NM[0]) & (x <= PLAGE_NM[1])
        if m.any():
            masque = m
    xs = x[masque]
    ys = lisser(intensite)[masque]
    norm = ys / ys.max()

    pos, cent, fwhm = stats_pic(xs, ys)
    print(f"Pic : max = {pos:.2f} {unite}, centroïde = {cent:.2f} {unite}, "
          f"FWHM = {fwhm:.2f} {unite}")
    if unite == "nm":
        print(f"Pas par pixel : {np.median(np.abs(np.diff(x))):.4f} nm/px")
        if LAMBDA_RESEAU is not None and L_OUVERTURE is not None:
            print("Résolution théorique (limite d'ouverture) : "
                  f"{LBD_CENTRALE * LAMBDA_RESEAU / (ORDRE * L_OUVERTURE) * 1e9:.3f} nm")

    np.savetxt(os.path.join(dossier, "spectre.csv"),
               np.column_stack([np.arange(len(x))[masque], xs,
                                intensite[masque], ys, norm]),
               delimiter=",",
               header=f"pixel,x_{unite},intensite,intensite_lissee,intensite_norm",
               comments="")
    ecrire_params(os.path.join(dossier, "parametres.txt"), infos + [
        f"Bande lignes : {y1}-{y2} (y0={y0})",
        f"Calibration : {desc}",
        f"Pic : max={pos:.3f} {unite}, centroide={cent:.3f} {unite}, "
        f"FWHM={fwhm:.3f} {unite}",
    ])

    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5))
    ax[0].imshow(apercu, cmap="gray" if apercu.ndim == 2 else None)
    ax[0].axhline(y1 / 8, color="r", lw=0.8, ls="--")
    ax[0].axhline(y2 / 8, color="r", lw=0.8, ls="--")
    ax[0].set_title("Image et bande d'intégration")
    ax[0].axis("off")
    tracer_spectre(ax[1], xs, norm, unite)
    ax[1].set_xlabel(etiq)
    ax[1].set_ylabel("Intensité normalisée")
    ax[1].set_title(f"Spectre — pic {cent:.1f} {unite}, FWHM {fwhm:.2f} {unite}")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(dossier, "spectre.png"), dpi=150)
    print(f"Résultats enregistrés dans {dossier}/")
    plt.show()


def mode_source(cam, taille, n, fps, expo_ms, dossier, mode_cal):
    input("\nAperçu en cours. Ajustez le montage, puis Entrée pour acquérir...")
    expo_ms = verrouiller(cam, expo_ms)
    images, fps_reel = acquerir(cam, n, os.path.join(dossier, "images"))
    cam.stop()
    cam.close()

    profils, bande = traiter(images)
    derniere = cv2.cvtColor(images[-1], cv2.COLOR_BGR2RGB)[::8, ::8]
    del images

    infos = [
        "Mode : source / laser",
        f"Resolution : {taille[0]} x {taille[1]}",
        f"N images : {n} | fps demande : {fps} | fps reel : {fps_reel:.2f}",
        f"Exposition (ms) : {expo_ms} | Gain : {GAIN_ANALOGIQUE}",
    ]
    resultats_source(profils.mean(axis=0), derniere, bande, dossier, mode_cal, infos)


def mode_analyse(dossier, mode_cal):
    """Mode 3 : analyse d'images déjà enregistrées (sans caméra)."""
    chemin = input("Dossier (ou motif, ex. acquisitions/.../images/*.png) "
                   "des images à analyser : ").strip()
    try:
        prof, moy, bande, n = profil_dossier(chemin)
    except FileNotFoundError as e:
        print(e)
        return
    infos = ["Mode : analyse d'images existantes",
             f"Images : {chemin} ({n} images)"]
    resultats_source(prof, moy[::8, ::8], bande, dossier, mode_cal, infos)


# ============================== Filtre en transmission =======================
def charger_reference(chemin):
    """Lit I_réf depuis un CSV produit par ce script (mode 1 ou 2)."""
    try:
        d = np.genfromtxt(chemin, delimiter=",", names=True)
        champ = "I_ref" if "I_ref" in d.dtype.names else "intensite"
        return np.asarray(d[champ], dtype=float)
    except Exception as e:
        print(f"Impossible de lire la référence ({e}).")
        return None


def mode_filtre(cam, taille, n, fps, expo_ms, dossier, mode_cal):
    print("\n--- Caractérisation du filtre en transmission ---")
    print("Source large bande (ex. lampe blanche). Gardez la MÊME exposition "
          "pour la référence et la mesure.")
    choix = input("Référence : [1] acquérir maintenant (sans filtre)  "
                  "[2] charger un CSV  [1] : ").strip()

    bande = None
    if choix == "2":
        chemin = input("Chemin du CSV de référence : ").strip()
        i_ref = charger_reference(chemin)
        if i_ref is None:
            cam.stop()
            cam.close()
            return
    else:
        input("\nRetirez le filtre, ajustez l'aperçu, puis Entrée pour acquérir "
              "la RÉFÉRENCE...")
        expo_ms = verrouiller(cam, expo_ms)
        imgs, _ = acquerir(cam, n, os.path.join(dossier, "reference"))
        profils_ref, bande = traiter(imgs)
        del imgs
        i_ref = profils_ref.mean(axis=0)

    input("\nInsérez le filtre à caractériser, puis Entrée pour acquérir la MESURE...")
    if choix == "2":  # référence chargée : fixer l'exposition ici
        if expo_ms == 0:
            print("ATTENTION : exposition auto avec référence chargée -> utilisez "
                  "plutôt la même exposition manuelle que pour la référence.")
        expo_ms = verrouiller(cam, expo_ms)
    imgs, fps_mes = acquerir(cam, n, os.path.join(dossier, "mesure"))
    cam.stop()
    cam.close()
    profils_mes, bande = traiter(imgs, bande)
    derniere = cv2.cvtColor(imgs[-1], cv2.COLOR_BGR2RGB)[::8, ::8]
    del imgs
    y0, y1, y2 = bande

    if len(i_ref) != profils_mes.shape[1]:
        print("Référence incompatible (nombre de pixels différent). Abandon.")
        return

    x, etiq, unite, desc = calibrer(len(i_ref), mode_cal)
    u_cal = U_CALIBRATION_NM if unite == "nm" else 0.0
    ref_s = lisser(i_ref)
    valide = ref_s > SEUIL_REF * ref_s.max()

    def transmission(i_mes):
        T = np.full(len(ref_s), np.nan)
        T[valide] = lisser(i_mes)[valide] / ref_s[valide]
        return T

    i_mes = profils_mes.mean(axis=0)
    T = transmission(i_mes)

    # --- Coupures : moyenne sur les images + incertitude ---------------------
    par_image = [coupures(x, transmission(p)) for p in profils_mes]
    pas = float(np.median(np.abs(np.diff(x))))  # largeur d'un pixel dans l'unité de x
    u_res = pas / np.sqrt(12)                   # lecture sur la grille de pixels
    resultats = {}
    for k, nom in enumerate(["montante (basse)", "descendante (haute)"]):
        vals = np.array([c[k] for c in par_image if c[k] is not None])
        if len(vals) == 0:
            resultats[nom] = None
            continue
        moy = vals.mean()
        u_stat = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        u = np.sqrt(u_stat**2 + u_res**2 + u_cal**2)
        resultats[nom] = (moy, u, u_stat, len(vals))
        print(f"Coupure {nom} : {moy:.2f} ± {u:.2f} {unite} "
              f"(stat {u_stat:.3f}, pixel {u_res:.3f}, {len(vals)} images)")
    if all(v is None for v in resultats.values()):
        print("Aucune coupure détectée (vérifier la référence et l'exposition).")
    if len(profils_mes) < 2:
        print("Note : 1 seule image -> incertitude statistique non estimable.")

    # --- Sauvegardes ---------------------------------------------------------
    np.savetxt(os.path.join(dossier, "spectre.csv"),
               np.column_stack([np.arange(len(x)), x, i_ref, i_mes, T]),
               delimiter=",", header=f"pixel,x_{unite},I_ref,I_mes,transmission",
               comments="")
    lignes = [
        "Mode : filtre en transmission",
        f"Resolution : {taille[0]} x {taille[1]}",
        f"N images : {n} | fps demande : {fps} | fps reel (mesure) : {fps_mes:.2f}",
        f"Exposition (ms) : {expo_ms} | Gain : {GAIN_ANALOGIQUE}",
        f"Bande lignes : {y1}-{y2} (y0={y0})",
        f"Calibration : {desc} | U_calibration : {u_cal}",
        f"Definition coupure : T = {FRACTION_COUPURE:.0%} du max, incertitude k=1",
    ]
    for nom, v in resultats.items():
        lignes.append(f"Coupure {nom} : " + (
            f"{v[0]:.3f} +/- {v[1]:.3f} {unite} (stat {v[2]:.3f}, N={v[3]})"
            if v else "non detectee"))
    ecrire_params(os.path.join(dossier, "parametres.txt"), lignes)

    # --- Figure --------------------------------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(10, 9.5))
    ax[0].imshow(derniere)
    ax[0].axhline(y1 / 8, color="w", lw=0.8, ls="--")
    ax[0].axhline(y2 / 8, color="w", lw=0.8, ls="--")
    ax[0].set_title("Image avec filtre (dernière) et bande d'intégration")
    ax[0].axis("off")

    ax[1].plot(x, i_ref / i_ref.max(), "k", label="Référence (sans filtre)")
    ax[1].plot(x, i_mes / i_ref.max(), "b", label="Avec filtre")
    ax[1].set_xlabel(etiq)
    ax[1].set_ylabel("Intensité (u.a.)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    ax[2].plot(x, T, "r")
    for v in resultats.values():
        if v:
            ax[2].axvline(v[0], color="g", ls="--", lw=0.8)
            ax[2].errorbar(v[0], FRACTION_COUPURE * np.nanmax(T), xerr=v[1],
                           fmt="go", capsize=4)
    ax[2].set_xlabel(etiq)
    ax[2].set_ylabel("Transmission")
    ax[2].set_title("Spectre de transmission et longueurs d'onde de coupure")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(dossier, "transmission.png"), dpi=150)
    print(f"Résultats enregistrés dans {dossier}/")
    plt.show()


# ============================== Programme ====================================
def main():
    mode = input("Mode : [1] source / laser   [2] filtre en transmission   "
                 "[3] analyser des images existantes  [1] : ").strip()
    mode = {"2": 2, "3": 3}.get(mode, 1)
    mode_cal = choisir_calibration()

    dossier = os.path.join(DOSSIER_SORTIE, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dossier, exist_ok=True)

    if mode == 3:
        mode_analyse(dossier, mode_cal)
        return

    if Picamera2 is None:
        print("picamera2 introuvable : modes 1 et 2 réservés au Raspberry Pi "
              "(sudo apt install python3-picamera2). Le mode 3 fonctionne sans caméra.")
        return

    n = demander("Nombre d'images à capter (par acquisition)", N_IMAGES_DEFAUT, int)
    fps = demander("Images par seconde (max 10 en pleine résolution)", FPS_DEFAUT, float)
    if fps > 10:
        print("fps > 10 impossible à pleine résolution -> fps = 10")
        fps = 10.0
    expo_ms = demander("Temps d'exposition en ms (0 = automatique)", 0.0, float)
    if expo_ms > 1000.0 / fps:
        expo_ms = 1000.0 / fps * 0.95
        print(f"Exposition limitée à {expo_ms:.1f} ms (période d'image).")
    if n > 40:
        print("Note : les images d'une acquisition sont gardées en RAM "
              "(~37 Mo/image) ; au-delà de ~40, risque de manquer de mémoire.")

    cam, taille = demarrer_camera(fps, expo_ms)
    verifier_espace(dossier, n * (2 if mode == 2 else 1), taille)
    if mode == 1:
        mode_source(cam, taille, n, fps, expo_ms, dossier, mode_cal)
    else:
        mode_filtre(cam, taille, n, fps, expo_ms, dossier, mode_cal)


if __name__ == "__main__":
    main()