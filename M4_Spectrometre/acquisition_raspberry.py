#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHS3910 - Mandat 4 : Spectromètre
Acquisition avec Raspberry Pi 4 + caméra HQ (SC1220, Sony IMX477R)

Trois modes :
  1) Source / laser : spectre d'intensité (position et largeur du pic)
  2) Filtre en transmission : acquisition d'une référence (sans filtre) puis
     d'une mesure (avec filtre) -> spectre de transmission T = I_mes / I_réf
     + longueurs d'onde de coupure (T = 50 % du max) avec incertitudes
  3) Réglage de l'exposition (sur la saturation RAW) + image avec filtre
     optionnelle + image dark

Pour chaque image acquise, trois fichiers sont enregistrés :
  - raw_XXX.npy   : RAW 12 bits non traité (tableau uint16, valeurs 0..4095)
  - image_XXX.dng : RAW au format DNG (s'ouvre dans RawTherapee, darktable...)
  - image_XXX.png : image traitée 8 bits, PNG NON compressé

La saturation est calculée sur le RAW (par canal R, V, B du motif de Bayer).
Les spectres affichés par ce script viennent encore des PNG 8 bits (aperçu
rapide) ; pour des intensités quantitatives, analyser les .npy dans le notebook.

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

# Saturation RAW : un pixel est saturé s'il est à moins de MARGE_SAT du blanc
MARGE_SAT = 5                # (4095 - 5 = 4090 en 12 bits)
SEUIL_SAT_PCT = 0.1          # avertissement si > 0,1 % de pixels saturés dans la bande

# Focales (mm), optionnel : permet d'estimer l'image de la fente sur le capteur
F1_MM, F2_MM = None, None
LARGEUR_FENTE_UM = 0.0       # rempli par main()

# Calibration pixel -> longueur d'onde : liste de (position_pixel, lambda_nm)
# mesurées avec des lasers connus (ex. [(1200, 532.0), (2900, 650.0)]).
# Avec >= 2 points : ajustement linéaire. Sinon : axe en pixels.
CALIBRATION = []

# Incertitude de calibration (nm) ajoutée en quadrature aux coupures
U_CALIBRATION_NM = 0.0

# Infos RAW remplies par demarrer_camera()
RAW = {"bits": 12, "blanc": 4095, "motif": "RGGB", "w": 4056, "h": 3040}
_dng_erreur = [False]
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
    taille = tuple(cam.sensor_resolution)  # (4056, 3040) pour l'IMX477

    # Mode capteur pleine résolution, avec la plus grande profondeur de bits (12)
    modes = [m for m in cam.sensor_modes if tuple(m["size"]) == taille]
    if not modes:
        modes = list(cam.sensor_modes)
    mode = max(modes, key=lambda m: m["bit_depth"])
    fmt = mode.get("unpacked") or mode["format"].replace("_CSI2P", "")  # ex. SRGGB12
    RAW.update({"bits": mode["bit_depth"], "blanc": (1 << mode["bit_depth"]) - 1,
                "motif": fmt[1:5], "w": taille[0], "h": taille[1]})

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
        raw={"size": tuple(mode["size"]), "format": fmt},   # RAW non compacté (16 bits)
        sensor={"output_size": tuple(mode["size"]), "bit_depth": mode["bit_depth"]},
        display="lores",  # sans ça, une config "still" n'affiche rien (fenêtre grise)
        buffer_count=3,   # si erreur d'allocation mémoire (CMA) : mettre 2
        controls=controles,
    )
    cam.configure(config)
    print(f"Résolution : {taille[0]} x {taille[1]} | RAW {fmt}, "
          f"{RAW['bits']} bits (blanc = {RAW['blanc']})")
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


# ------------------------------- RAW / saturation ----------------------------
def capturer(cam):
    """Une capture : (image main BGR, buffer RAW, métadonnées)."""
    req = cam.capture_request()
    try:
        img = req.make_array("main")     # tableau BGR (convention OpenCV)
        buf = req.make_buffer("raw")     # copie du RAW brut
        meta = req.get_metadata()
    finally:
        req.release()
    return img, buf, meta


def buffer_vers_raw(cam, buf):
    """Buffer RAW -> tableau uint16 (h, w), valeurs 0..blanc."""
    cfg = cam.camera_config["raw"]
    w, h = cfg["size"]
    a = np.ascontiguousarray(cam.helpers.make_array(buf, cfg))
    if a.dtype == np.uint8:
        a = a.view(np.uint16)
    return a[:h, :w].copy()


def sauver_image(cam, dossier, i, img, raw, buf, meta, prefixe="image"):
    """Enregistre .png (non compressé), .npy (RAW) et .dng pour une image."""
    os.makedirs(dossier, exist_ok=True)
    cv2.imwrite(os.path.join(dossier, f"{prefixe}_{i:03d}.png"), img,
                [cv2.IMWRITE_PNG_COMPRESSION, 0])           # 0 = non compressé
    np.save(os.path.join(dossier, f"raw_{i:03d}.npy"), raw)
    try:
        cam.helpers.save_dng(buf, meta, cam.camera_config,
                             os.path.join(dossier, f"{prefixe}_{i:03d}.dng"))
    except Exception as e:
        if not _dng_erreur[0]:
            print(f"  ATTENTION : DNG non enregistré ({e}). .npy et .png sont enregistrés.")
            _dng_erreur[0] = True


def compte_sat_lignes(raw):
    """Nombre de pixels saturés par ligne, par canal (R, V, B) -> (3, h)."""
    seuil = RAW["blanc"] - MARGE_SAT
    out = np.zeros((3, raw.shape[0]), dtype=np.int32)
    for k, lettre in enumerate(RAW["motif"]):
        i, j = divmod(k, 2)
        out["RGB".index(lettre), i::2] += (raw[i::2, j::2] >= seuil).sum(axis=1)
    return out


def pct_saturation(sat_lignes, y1, y2):
    """% de pixels saturés dans la bande [y1, y2), par canal -> [R, V, B]."""
    frac = {"R": 0.25, "G": 0.5, "B": 0.25}   # part de chaque canal dans le motif de Bayer
    n = sat_lignes.shape[0]
    pct = []
    for k, c in enumerate("RGB"):
        tot = n * (y2 - y1) * RAW["w"] * frac[c]
        pct.append(100.0 * sat_lignes[:, k, y1:y2].sum() / max(tot, 1))
    return np.array(pct)


def rapport_saturation(sat, bande, etiquette=""):
    """Affiche la saturation RAW dans la bande et avertit si nécessaire."""
    sat_l, raw_max = sat
    _, y1, y2 = bande
    p = pct_saturation(sat_l, y1, y2)
    print(f"Saturation RAW {etiquette}(bande {y1}-{y2}) : R/V/B = "
          f"{p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f} % | max RAW = {max(raw_max)}/{RAW['blanc']}")
    if p.max() > SEUIL_SAT_PCT:
        print("ATTENTION : saturation RAW -> réduire l'exposition ou l'intensité "
              "de la source (mode 3 pour ajuster).")
    return p


def acquerir(cam, n, dossier):
    """
    Capte n images ; enregistre pour chacune .png, .npy et .dng.
    Retourne (images PNG, fps_reel, (saturation par ligne, max RAW)).
    """
    os.makedirs(dossier, exist_ok=True)
    images, bufs, metas, instants = [], [], [], []
    t0 = time.time()
    for i in range(n):
        img, buf, meta = capturer(cam)
        images.append(img)
        bufs.append(buf)
        metas.append(meta)
        instants.append(time.time() - t0)
        print(f"  image {i + 1}/{n}", end="\r")
    print()
    fps_reel = (n - 1) / (instants[-1] - instants[0]) if n > 1 else float("nan")
    print(f"Cadence réelle : {fps_reel:.2f} fps")

    print("Enregistrement (.png, .npy, .dng)...")
    sat_lignes, raw_max = [], []
    for i in range(n):
        raw = buffer_vers_raw(cam, bufs[i])
        sauver_image(cam, dossier, i, images[i], raw, bufs[i], metas[i])
        sat_lignes.append(compte_sat_lignes(raw))
        raw_max.append(int(raw.max()))
        bufs[i] = None   # libère la mémoire au fur et à mesure
    print(f"Images enregistrées dans {dossier}/")
    return images, fps_reel, (np.array(sat_lignes), raw_max)


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
        f.write("\n".join(lignes + [
            f"Largeur de la fente (um) : {LARGEUR_FENTE_UM}",
            f"RAW : {RAW['bits']} bits, blanc {RAW['blanc']}, motif {RAW['motif']}",
        ]) + "\n")


# ============================== Mode 1 : source ==============================
def mode_source(cam, taille, n, fps, expo_ms, dossier):
    input("\nAperçu en cours. Ajustez le montage, puis Entrée pour acquérir...")
    expo_ms = verrouiller(cam, expo_ms)
    images, fps_reel, sat = acquerir(cam, n, os.path.join(dossier, "images"))
    cam.stop()
    cam.close()

    profils, (y0, y1, y2) = traiter(images)
    rapport_saturation(sat, (y0, y1, y2))
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
        imgs, fps_ref, sat_ref = acquerir(cam, n, os.path.join(dossier, "reference"))
        profils_ref, bande = traiter(imgs)
        rapport_saturation(sat_ref, bande, "référence ")
        del imgs
        i_ref = profils_ref.mean(axis=0)

    input("\nInsérez le filtre à caractériser, puis Entrée pour acquérir la MESURE...")
    if choix == "2":  # référence chargée : fixer l'exposition ici
        if expo_ms == 0:
            print("ATTENTION : exposition auto avec référence chargée -> utilisez "
                  "plutôt la même exposition manuelle que pour la référence.")
        expo_ms = verrouiller(cam, expo_ms)
    imgs, fps_mes, sat_mes = acquerir(cam, n, os.path.join(dossier, "mesure"))
    cam.stop()
    cam.close()
    profils_mes, bande = traiter(imgs, bande)
    rapport_saturation(sat_mes, bande, "mesure ")
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


# ============================== Mode 3 : réglage exposition + dark ===========
def bande_lumineuse(img):
    lignes = lisser(image_somme(img).sum(axis=1), 15)
    y0 = int(np.argmax(lignes))
    return y0, max(0, y0 - DEMI_HAUTEUR_BANDE), min(img.shape[0], y0 + DEMI_HAUTEUR_BANDE)


def appliquer_exposition(cam, expo_ms):
    cam.set_controls({"ExposureTime": int(expo_ms * 1000)})
    for _ in range(4):  # laisser le réglage s'appliquer
        cam.capture_metadata()


def stats_bande_raw(raw, y1, y2):
    """(% saturés [R, V, B], max RAW, percentile 99,9) dans la bande."""
    pct = pct_saturation(compte_sat_lignes(raw)[None], y1, y2)
    sub = raw[y1:y2]
    return pct, int(sub.max()), float(np.percentile(sub, 99.9))


def mode_reglage(cam, taille, fps, expo_ms, dossier):
    input("\nAperçu en cours. Placez la source, puis Entrée...")
    expo_ms = verrouiller(cam, expo_ms)   # fige l'exposition si elle était auto
    expo_max = 1000.0 / fps * 0.95
    blanc = RAW["blanc"]

    for essai in range(10):
        img, buf, _ = capturer(cam)
        raw = buffer_vers_raw(cam, buf)
        _, y1, y2 = bande_lumineuse(img)
        pct, mx, p = stats_bande_raw(raw, y1, y2)
        print(f"Essai {essai + 1} : {expo_ms:.2f} ms | saturés R/V/B = "
              f"{pct[0]:.3f}/{pct[1]:.3f}/{pct[2]:.3f} % | max RAW = {mx}/{blanc} "
              f"| p99.9 = {p:.0f}")
        if pct.max() > SEUIL_SAT_PCT:        # saturé -> on réduit
            nouv = expo_ms * 0.7
        elif p < 0.5 * blanc:                # trop sombre -> on augmente
            nouv = min(expo_ms * 0.75 * blanc / max(p, 1), expo_ms * 2, expo_max)
        else:
            break                            # bien exposé
        if abs(nouv - expo_ms) < 0.01 or nouv < 0.1:
            break
        expo_ms = nouv
        appliquer_exposition(cam, expo_ms)

    # Image retenue (une seule), enregistrée dans images/ pour le notebook
    img, buf, meta = capturer(cam)
    raw = buffer_vers_raw(cam, buf)
    _, y1, y2 = bande_lumineuse(img)
    sauver_image(cam, os.path.join(dossier, "images"), 0, img, raw, buf, meta)
    pct, mx, _ = stats_bande_raw(raw, y1, y2)
    print(f"\nImage enregistrée (.png, .npy, .dng). Saturation RAW R/V/B = "
          f"{pct[0]:.3f}/{pct[1]:.3f}/{pct[2]:.3f} % | max RAW = {mx}/{blanc}")
    if pct.max() > SEUIL_SAT_PCT:
        print("ATTENTION : encore saturé -> relancer le mode 3 avec une exposition plus courte.")
    print(f"=> Exposition retenue : {expo_ms:.2f} ms")

    # Optionnel : une image AVEC le filtre (même exposition, même gain)
    rep = input("\nAjouter le filtre et prendre une image avec filtre ? [o/N] : ")
    avec_filtre = rep.strip().lower() in ("o", "oui", "y")
    if avec_filtre:
        input("Insérez le filtre, puis Entrée...")
        img_f, buf_f, meta_f = capturer(cam)
        raw_f = buffer_vers_raw(cam, buf_f)
        sauver_image(cam, os.path.join(dossier, "mesure"), 0, img_f, raw_f, buf_f, meta_f)
        pct_f, mx_f, _ = stats_bande_raw(raw_f, y1, y2)
        print(f"Image avec filtre enregistrée. Saturation RAW R/V/B = "
              f"{pct_f[0]:.3f}/{pct_f[1]:.3f}/{pct_f[2]:.3f} % | max RAW = {mx_f}/{blanc}")

    # Image dark : MÊME exposition et MÊME gain, sans lumière
    input("\nBloquez la lumière (lampe éteinte ou entrée du spectromètre couverte), "
          "puis Entrée pour l'image DARK...")
    img_d, buf_d, meta_d = capturer(cam)
    raw_d = buffer_vers_raw(cam, buf_d)
    sauver_image(cam, os.path.join(dossier, "dark"), 0, img_d, raw_d, buf_d, meta_d,
                 prefixe="dark")
    noir = meta_d.get("SensorBlackLevels")
    print(f"Dark enregistré. RAW : moyenne {raw_d.mean():.1f}, max {int(raw_d.max())} "
          f"| niveaux de noir capteur : {noir}")

    ecrire_params(os.path.join(dossier, "parametres.txt"), [
        "Mode : reglage exposition + dark",
        f"Exposition retenue (ms) : {expo_ms:.3f} | Gain : {GAIN_ANALOGIQUE}",
        f"Bande lignes : {y1}-{y2}",
        f"Image avec filtre : {'oui' if avec_filtre else 'non'}",
        f"Niveaux de noir capteur : {noir}",
    ])
    cam.stop()
    cam.close()


# ============================== Programme ====================================
def main():
    global LARGEUR_FENTE_UM
    LARGEUR_FENTE_UM = demander("Largeur de la fente (µm, 0 = inconnue)", 50.0, float)
    if LARGEUR_FENTE_UM > 0 and F1_MM and F2_MM:
        l_px = LARGEUR_FENTE_UM * (F2_MM / F1_MM) / 1.55
        print(f"Image attendue de la fente : ~{l_px:.0f} px (largeur d'une raie)")

    mode = input("Mode : [1] source/laser  [2] filtre en transmission  "
                 "[3] réglage exposition + dark  [1] : ").strip()
    mode = int(mode) if mode in ("2", "3") else 1

    n = demander("Nombre d'images à capter (par acquisition)", N_IMAGES_DEFAUT, int)
    fps = demander("Images par seconde (max 10 en pleine résolution)", FPS_DEFAUT, float)
    if fps > 10:
        print("fps > 10 impossible à pleine résolution -> fps = 10")
        fps = 10.0
    expo_ms = demander("Temps d'exposition en ms (0 = automatique)", 0.0, float)
    if expo_ms > 1000.0 / fps:
        expo_ms = 1000.0 / fps * 0.95
        print(f"Exposition limitée à {expo_ms:.1f} ms (période d'image).")
    if n > 30:
        print("Note : les images d'une acquisition sont gardées en RAM "
              "(~60 Mo/image) ; au-delà de ~30, risque de manquer de mémoire. "
              "Espace disque : ~90 Mo/image (.png + .npy + .dng).")

    dossier = os.path.join(DOSSIER_SORTIE, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dossier, exist_ok=True)

    cam, taille = demarrer_camera(fps, expo_ms)
    if mode == 1:
        mode_source(cam, taille, n, fps, expo_ms, dossier)
    elif mode == 2:
        mode_filtre(cam, taille, n, fps, expo_ms, dossier)
    else:
        mode_reglage(cam, taille, fps, expo_ms, dossier)


if __name__ == "__main__":
    main()