#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHS3910 - Mandat 4 : Spectromètre
Acquisition avec Raspberry Pi 4 + caméra HQ (SC1220, Sony IMX477R)

Deux modes :
  1) Source / laser : spectre d'intensité (position et largeur du pic)
  2) Filtre en transmission : acquisition d'une référence (sans filtre) puis
     d'une mesure (avec filtre) -> spectre de transmission T = I_mes / I_réf
     + longueurs d'onde de coupure (T = 50 % du max) avec incertitudes

Dans les deux modes : aperçu en direct, résolution maximale (4056 x 3040),
nombre d'images (défaut 10) et cadence (défaut 10 fps) au choix, images
enregistrées en PNG.

Dépendances : sudo apt install python3-picamera2 python3-opencv python3-matplotlib
"""

import os
import time
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
from picamera2 import Picamera2, Preview

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

# Calibration pixel -> longueur d'onde : liste de (position_pixel, lambda_nm)
# mesurées avec des lasers connus (ex. [(1200, 532.0), (2900, 650.0)]).
# Avec >= 2 points : ajustement linéaire. Sinon : axe en pixels.
CALIBRATION = []

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


def axe_longueur_onde(n_pixels):
    px = np.arange(n_pixels)
    if len(CALIBRATION) >= 2:
        p = np.array([c[0] for c in CALIBRATION], dtype=float)
        l = np.array([c[1] for c in CALIBRATION], dtype=float)
        a, b = np.polyfit(p, l, 1)
        return a * px + b, "Longueur d'onde (nm)", "nm"
    print("ATTENTION : pas de calibration -> axe en pixels.")
    return px.astype(float), "Position (pixels)", "px"


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
    else:            # aperçu automatique (comme dans le mandat précédent)
        controles.update({"AeEnable": True, "AwbEnable": True})
    config = cam.create_still_configuration(
        main={"size": taille, "format": "RGB888"},
        lores={"size": (taille[0] // 4, taille[1] // 4)},  # flux léger pour l'aperçu
        display="lores",  # sans ça, une config "still" n'affiche rien (fenêtre grise)
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
    """Capte n images, les enregistre en PNG. Retourne (images, fps_reel)."""
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
        cv2.imwrite(os.path.join(dossier, f"image_{i:03d}.png"), img,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
    maxi = max(int(im.max()) for im in images)
    if maxi >= 250:
        print(f"ATTENTION : saturation détectée (max = {maxi}/255). "
              "Réduire l'exposition ou l'intensité de la source.")
    print(f"Images enregistrées dans {dossier}/")
    return images, fps_reel


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


def ecrire_params(chemin, lignes):
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes) + "\n")


# ============================== Mode 1 : source ==============================
def mode_source(cam, taille, n, fps, expo_ms, dossier):
    input("\nAperçu en cours. Ajustez le montage, puis Entrée pour acquérir...")
    expo_ms = verrouiller(cam, expo_ms)
    images, fps_reel = acquerir(cam, n, os.path.join(dossier, "images"))
    cam.stop()
    cam.close()

    profils, (y0, y1, y2) = traiter(images)
    derniere = cv2.cvtColor(images[-1], cv2.COLOR_BGR2RGB)[::8, ::8]
    del images

    intensite = profils.mean(axis=0)
    norm = intensite / intensite.max()
    x, etiq, unite = axe_longueur_onde(len(intensite))
    pos, cent, fwhm = stats_pic(x, lisser(intensite))
    print(f"Pic : max = {pos:.2f} {unite}, centroïde = {cent:.2f} {unite}, "
          f"FWHM = {fwhm:.2f} {unite}")

    np.savetxt(os.path.join(dossier, "spectre.csv"),
               np.column_stack([np.arange(len(x)), x, intensite, norm]),
               delimiter=",", header=f"pixel,x_{unite},intensite,intensite_norm",
               comments="")
    ecrire_params(os.path.join(dossier, "parametres.txt"), [
        "Mode : source / laser",
        f"Resolution : {taille[0]} x {taille[1]}",
        f"N images : {n} | fps demande : {fps} | fps reel : {fps_reel:.2f}",
        f"Exposition (ms) : {expo_ms} | Gain : {GAIN_ANALOGIQUE}",
        f"Bande lignes : {y1}-{y2} (y0={y0})",
        f"Calibration : {CALIBRATION}",
        f"Pic : max={pos:.3f} {unite}, centroide={cent:.3f} {unite}, FWHM={fwhm:.3f} {unite}",
    ])

    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5))
    ax[0].imshow(derniere)
    ax[0].axhline(y1 / 8, color="w", lw=0.8, ls="--")
    ax[0].axhline(y2 / 8, color="w", lw=0.8, ls="--")
    ax[0].set_title("Image acquise (dernière) et bande d'intégration")
    ax[0].axis("off")
    ax[1].plot(x, norm, "b")
    ax[1].set_xlabel(etiq)
    ax[1].set_ylabel("Intensité normalisée")
    ax[1].set_title(f"Spectre ({n} images) — pic {cent:.1f} {unite}, "
                    f"FWHM {fwhm:.2f} {unite}")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(dossier, "spectre.png"), dpi=150)
    print(f"Résultats enregistrés dans {dossier}/")
    plt.show()


# ============================== Mode 2 : filtre ==============================
def charger_reference(chemin):
    """Lit I_réf depuis un CSV produit par ce script (mode 1 ou 2)."""
    try:
        d = np.genfromtxt(chemin, delimiter=",", names=True)
        champ = "I_ref" if "I_ref" in d.dtype.names else "intensite"
        return np.asarray(d[champ], dtype=float)
    except Exception as e:
        print(f"Impossible de lire la référence ({e}).")
        return None


def mode_filtre(cam, taille, n, fps, expo_ms, dossier):
    print("\n--- Caractérisation du filtre en transmission ---")
    print("Source large bande (ex. lampe blanche). Gardez la MÊME exposition "
          "pour la référence et la mesure.")
    choix = input("Référence : [1] acquérir maintenant (sans filtre)  "
                  "[2] charger un CSV  [1] : ").strip()

    bande = None
    profils_ref, fps_ref = None, float("nan")
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
        imgs, fps_ref = acquerir(cam, n, os.path.join(dossier, "reference"))
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

    x, etiq, unite = axe_longueur_onde(len(i_ref))
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
        u = np.sqrt(u_stat**2 + u_res**2 + U_CALIBRATION_NM**2)
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
        f"Calibration : {CALIBRATION} | U_calibration : {U_CALIBRATION_NM}",
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
    mode = input("Mode : [1] source / laser   [2] filtre en transmission  [1] : ").strip()
    mode = 2 if mode == "2" else 1

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

    dossier = os.path.join(DOSSIER_SORTIE, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dossier, exist_ok=True)

    cam, taille = demarrer_camera(fps, expo_ms)
    if mode == 1:
        mode_source(cam, taille, n, fps, expo_ms, dossier)
    else:
        mode_filtre(cam, taille, n, fps, expo_ms, dossier)


if __name__ == "__main__":
    main()