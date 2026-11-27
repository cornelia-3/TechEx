#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHS3910 M4 — Spectromètre sur Raspberry Pi 4 et caméra HQ."""

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText

import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    from picamera2 import Picamera2
except ImportError:  # ordinateur sans Raspberry Pi : mode relecture seulement
    Picamera2 = None


DOSSIER_SORTIE = Path(__file__).resolve().parent / "acquisitions"
DOSSIER_LASERS_PROFS = DOSSIER_SORTIE / "calibration_lasers"
DOSSIER_TXT_PROFS = DOSSIER_SORTIE.parent / "references_profs"

# Ancien format (dossiers des profs sans laser.json) : dossier -> (TXT, vérification?)
ASSOCIATIONS_LASERS = {
    "532": ("532nm Laser (battery powered).txt", False),
    "555": ("555nm laser (battery powered).txt", False),
    "571": ("571nm laser (battery powered).txt", False),
    "593": ("593nm laser (battery powered).txt", False),
    "650": ("650nm_laser_pointer_cat.txt", False),
    # Ajoute les quatre autres lasers sur le même modèle.
    # "nom_du_dossier": ("nom_exact_du_fichier.txt", False),
}
FPS, EXPO_MS, GAIN = 10.0, 20.0, 1.0

# Réseau en réflexion, focale de sortie et pixels après binning 2 × 2.
LAMBDA, ORDRE = 1 / 600e3, 1
F2, PIXEL = 30e-3, 2 * 1.55e-6
LBD_C, SENS = 550e-9, -1

BLANC, NOIR = 4095, 256
DEMI_H, LISS = 50, 5
SEUIL_REF, FRAC = 0.10, 0.50
# Pic laser saturé : flancs gardés au-dessus de 10 % du sommet écrêté,
# alerte si le centre gaussien s'écarte du centroïde de plus de 0,5 px.
FRAC_FLANC, ECART_MAX_PX = 0.10, 0.5

# Incertitude sur le pic du spectromètre de référence (à ajuster selon sa résolution)
U_NM_REF = 0.133      # nm
# Plancher sur le centre d'un pic (méthode de centroïde, quantification)
U_PX_MIN = 0.1      # px binné


@dataclass(frozen=True)
class Laser:
    px: float
    nm: float
    largeur: int
    verification: bool
    source: str
    alerte: str = ""  # vide si le centre du pic est fiable
    u_px: float = U_PX_MIN
    u_nm: float = U_NM_REF
    n_acq: int = 1


def lisser(y, n=LISS):
    return np.convolve(y, np.ones(n) / n, mode="same") if n > 1 else y


def binner(r):
    h, w = r.shape[0] // 2 * 2, r.shape[1] // 2 * 2
    r = r[:h, :w]
    return r[::2, ::2] + r[::2, 1::2] + r[1::2, ::2] + r[1::2, 1::2]


def profil(raw, noir, bande):
    """Moyenne du RAW corrigé du noir sur une bande de lignes binnées."""
    r = binner(np.clip(raw.astype(np.float32) - noir, 0, None))
    if bande is None:
        y0 = int(np.argmax(lisser(r.sum(axis=1), 15)))
        bande = (max(0, y0 - DEMI_H), min(r.shape[0], y0 + DEMI_H))
    return r[bande[0]:bande[1]].mean(axis=0), bande


def pic(x, y):
    """Renvoie le centroïde au-dessus de la demi-hauteur et ses bornes."""
    y = y - np.median(y)
    i = int(np.argmax(y))
    if y[i] <= 0:
        raise ValueError("Aucun pic détecté.")

    g, d = i, i
    while g > 0 and y[g - 1] >= y[i] / 2:
        g -= 1
    while d < len(y) - 1 and y[d + 1] >= y[i] / 2:
        d += 1
    if g == 0 or d == len(y) - 1:
        raise ValueError("Le pic touche le bord de la plage.")

    s = slice(g, d + 1)
    return float(np.average(x[s], weights=y[s])), g, d


def longueur_txt(chemin):
    """Extrait le centroïde du pic dominant d’un TXT au format fourni."""
    lignes = Path(chemin).read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines()
    debut = next(
        (i for i, ligne in enumerate(lignes) if "Begin Spectral Data" in ligne),
        None,
    )
    if debut is None:
        raise ValueError("Marqueur « Begin Spectral Data » absent.")

    bloc = []
    for ligne in lignes[debut + 1:]:
        if "End Spectral Data" in ligne:
            break
        bloc.append(ligne)

    donnees = np.loadtxt(bloc, ndmin=2)
    if (donnees.shape[0] < 3 or donnees.shape[1] != 2
            or not np.isfinite(donnees).all()
            or np.any(np.diff(donnees[:, 0]) <= 0)):
        raise ValueError(
            "Deux colonnes numériques attendues, avec les longueurs d’onde croissantes."
        )
    return pic(*donnees.T)[0]


def profil_laser(images):
    profils, bande, satures = [], None, None
    for raw in images:
        p, bande = profil(raw, NOIR, bande)
        zone = raw[2 * bande[0]:2 * bande[1]]
        colonnes = np.any(zone >= BLANC - 5, axis=0)
        colonnes = colonnes[:2 * len(p)].reshape(-1, 2).any(axis=1)
        satures = colonnes if satures is None else satures | colonnes
        profils.append(p)

    if not profils:
        raise ValueError("Aucune image RAW du laser.")

    intensite = lisser(np.mean(profils, axis=0))
    px, g, d = pic(np.arange(intensite.size), intensite)
    demi_lissage = LISS // 2
    if not satures[max(0, g - demi_lissage):d + demi_lissage + 1].any():
        return px, intensite, ""

    # Pic saturé : sommet écrêté. On ajuste une gaussienne sur les flancs
    # non saturés (parabole sur le log), ce qui reconstruit le sommet.
    try:
        centre = centre_gaussien(np.mean(profils, axis=0), satures)
    except ValueError as exc:
        return px, intensite, (
            f"pic saturé, ajustement gaussien impossible ({exc}). "
            f"Centroïde utilisé : {px:.2f} px (biais possible)."
        )
    ecart = centre - px
    alerte = ""
    if abs(ecart) > ECART_MAX_PX:
        alerte = (
            f"pic saturé, centre gaussien {centre:.2f} px vs centroïde {px:.2f} px "
            f"(écart {ecart:+.2f} px > {ECART_MAX_PX} px). Centre gaussien utilisé."
        )
    return centre, intensite, alerte


def profils_dossier(dossier, noir, bande):
    """Profils de tous les raw_*.npy d'un dossier, corrigés du noir."""
    fichiers = sorted(Path(dossier).glob("raw_*.npy"))
    if not fichiers:
        raise ValueError(f"Aucun raw_*.npy dans {dossier}")
    profs = []
    for f in fichiers:
        p, bande = profil(np.load(f), noir, bande)
        profs.append(p)
    return np.array(profs), bande


def centre_gaussien(y, satures):
    """Centre d'une gaussienne ajustée sur les colonnes non saturées du pic.

    ln(y) est une parabole pour une gaussienne : on l'ajuste par moindres
    carrés pondérés par y (les points faibles, bruités en log, comptent moins).
    """
    y = y - np.median(y)
    i = int(np.argmax(y))
    seuil = FRAC_FLANC * y[i]
    g, d = i, i
    while g > 0 and y[g - 1] >= seuil:
        g -= 1
    while d < len(y) - 1 and y[d + 1] >= seuil:
        d += 1

    cols = np.arange(g, d + 1)
    bons = cols[~satures[g:d + 1] & (y[g:d + 1] > 0)]
    zone_sat = cols[satures[g:d + 1]]
    if zone_sat.size and not (
        (bons < zone_sat.min()).any() and (bons > zone_sat.max()).any()
    ):
        raise ValueError("pas de flanc non saturé des deux côtés")
    if bons.size < 3:
        raise ValueError(f"seulement {bons.size} colonne(s) non saturée(s)")

    a, b, _ = np.polyfit(bons, np.log(y[bons]), 2, w=y[bons])
    if a >= 0:
        raise ValueError("forme non gaussienne")
    centre = -b / (2 * a)
    if not g <= centre <= d:
        raise ValueError("centre hors du pic")
    return float(centre)


def avertissements_pic(lasers):
    return [f"ATTENTION : laser {p.nm:g} nm : {p.alerte}" for p in lasers if p.alerte]


def axe_lbd(n, px_ref, lbd_ref_nm):
    th_i = np.arcsin(ORDRE * LBD_C / (2 * LAMBDA))
    th_ref = np.arcsin(ORDRE * lbd_ref_nm * 1e-9 / LAMBDA - np.sin(th_i))
    th = np.arctan(np.tan(th_ref) + SENS * (np.arange(n) - px_ref) * PIXEL / F2)
    return LAMBDA * (np.sin(th) + np.sin(th_i)) / ORDRE * 1e9


def verifier_lasers(lasers):
    if len({p.largeur for p in lasers}) > 1:
        raise ValueError("Les captures laser doivent avoir la même largeur RAW binnée.")

    for i, p in enumerate(lasers):
        if not np.isfinite(p.nm) or not 0 < p.nm < LAMBDA * 1e9:
            raise ValueError("Longueur d’onde invalide dans les données laser.")
        if any(abs(p.px - q.px) < 1 or abs(p.nm - q.nm) < 0.001
               for q in lasers[:i]):
            raise ValueError("Raie déjà enregistrée ou deux pics séparés de moins d’un pixel.")

    reglage = sorted((p for p in lasers if not p.verification), key=lambda p: p.px)
    ecarts = np.diff([p.nm for p in reglage])
    if ecarts.size and not (np.all(ecarts > 0) or np.all(ecarts < 0)):
        raise ValueError("Les raies d’ajustement doivent être ordonnées le long du capteur.")


# ---------------------------------------------------------------------------
# Calibration : dossiers de lasers, acquisitions multiples
# ---------------------------------------------------------------------------

def nom_dossier_laser(nm):
    return f"{nm:.0f}"          # 532.03 -> "532"


def acquisitions_laser(dossier):
    """Liste de listes de fichiers RAW : une liste par acquisition (acq_*).
    Ancien format (raw_*.npy directement dans le dossier) = 1 acquisition."""
    d = Path(dossier)
    acqs = sorted(p for p in d.glob("acq_*") if p.is_dir()) or [d]
    res = []
    for a in acqs:
        fichiers = sorted(a.glob("raw_*.npy"))
        if not fichiers and (a / "raw.npy").is_file():
            fichiers = [a / "raw.npy"]
        if fichiers:
            res.append(fichiers)
    return res


def laser_depuis_dossier(d):
    """Construit un Laser (moyenne de toutes les acquisitions du dossier).
    Renvoie (Laser, intensite) ou None si le dossier n'est pas un laser."""
    d = Path(d)
    meta = d / "laser.json"
    if meta.is_file():
        infos = json.loads(meta.read_text(encoding="utf-8"))
        nm, verif = float(infos["nm"]), bool(infos["verification"])
        source = f"{d.name} / {infos.get('txt', 'saisie manuelle')}"
    elif d.name in ASSOCIATIONS_LASERS:            # ancien format des profs
        nom_txt, verif = ASSOCIATIONS_LASERS[d.name]
        nm = longueur_txt(DOSSIER_TXT_PROFS / nom_txt)
        source = f"{d.name} / {nom_txt}"
    else:
        return None

    acqs = acquisitions_laser(d)
    if not acqs:
        raise ValueError(f"aucun raw_*.npy dans {d}")

    centres, alertes, intensite = [], [], None
    for i, fichiers in enumerate(acqs, 1):
        px, intensite, alerte = profil_laser(np.load(f) for f in fichiers)
        centres.append(px)
        if alerte:
            alertes.append(f"acq {i} : {alerte}")

    n = len(centres)
    u_stat = np.std(centres, ddof=1) / np.sqrt(n) if n >= 2 else 0.0
    u_px = float(np.hypot(u_stat, U_PX_MIN))
    laser = Laser(float(np.mean(centres)), nm, intensite.size, verif, source,
                  " | ".join(alertes), u_px, U_NM_REF, n)
    return laser, intensite


def charger_calibration_dossier(racine):
    """Charge tous les sous-dossiers laser d'un dossier de calibration."""
    racine = Path(racine)
    if not racine.is_dir():
        raise FileNotFoundError(f"Dossier de calibration introuvable :\n{racine}")
    lasers, dernier = [], None
    for d in sorted(p for p in racine.iterdir() if p.is_dir()):
        try:
            r = laser_depuis_dossier(d)
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"Laser {d.name} : {exc}") from exc
        if r is not None:
            lasers.append(r[0])
            dernier = r
    verifier_lasers(lasers)
    return lasers, dernier


def charger_lasers_profs(racine=None):
    """Version « stricte » utilisée au démarrage et avant chaque mesure."""
    lasers, dernier = charger_calibration_dossier(racine or DOSSIER_LASERS_PROFS)
    if not lasers:
        raise ValueError("Aucun laser reconnu dans le dossier de calibration.")
    if all(p.verification for p in lasers):
        raise ValueError("Affectez au moins un laser à l’ajustement (vérification décochée).")
    return lasers, dernier


def ajustement(lasers):
    """nm = a·px + b, moindres carrés pondérés. -> (a, b, cov, chi2_red) ou None.

    Chaque raie a une variance effective σ² = u_nm² + (a·u_px)² (incertitudes
    en px et en nm ramenées sur nm). cov est la covariance de (a, b),
    mise à l'échelle par χ²_red si χ²_red > 1.
    """
    pts = [p for p in lasers if not p.verification]
    if len(pts) < 2:
        return None
    px = np.array([p.px for p in pts])
    nm = np.array([p.nm for p in pts])
    upx = np.array([p.u_px for p in pts])
    unm = np.array([p.u_nm for p in pts])
    X = np.column_stack([px, np.ones_like(px)])
    a = np.polyfit(px, nm, 1)[0]
    for _ in range(5):                                  # variance effective itérée
        sig = np.maximum(np.sqrt(unm**2 + (a * upx)**2), 1e-9)
        W = 1 / sig**2
        cov = np.linalg.inv(X.T @ (X * W[:, None]))
        a, b = cov @ (X.T @ (W * nm))
    dof = len(pts) - 2
    chi2r = float(np.sum(((nm - X @ np.array([a, b])) / sig)**2) / dof) if dof > 0 else None
    if chi2r is not None and chi2r > 1:
        cov = cov * chi2r
    return a, b, cov, chi2r


def calibrer(n, lasers):
    fit = ajustement(lasers)
    if fit is not None:
        a, b, _, chi2r = fit
        k = sum(not p.verification for p in lasers)
        extra = f", χ²red = {chi2r:.2f}" if chi2r is not None else ""
        return a * np.arange(n) + b, f"linéaire pondérée sur {k} raies ({a:.4f} nm/px{extra})"
    pts = [(p.px, p.nm) for p in lasers if not p.verification]
    if len(pts) == 1:
        return axe_lbd(n, *pts[0]), f"modèle calé sur {pts[0][1]:g} nm"
    return axe_lbd(n, n / 2, LBD_C * 1e9), "axe nominal (550 nm au centre)"


def sigma_axe(n, lasers):
    """Incertitude-type (nm) de la droite d'étalonnage en chaque pixel.

    Renvoie None s'il y a moins de 2 raies d'ajustement.
    """
    fit = ajustement(lasers)
    if fit is None:
        return None
    cov = fit[2]
    G = np.column_stack([np.arange(n), np.ones(n)])
    return np.sqrt(np.einsum("ij,jk,ik->i", G, cov, G))


def ecarts_verification(x, lasers):
    if not any(not p.verification for p in lasers):
        return []
    px = np.arange(len(x))
    return [(p.nm, np.interp(p.px, px, x) - p.nm) for p in lasers if p.verification]


def incertitude_calibration(ecarts):
    return np.sqrt(np.mean([e**2 for _, e in ecarts])) if ecarts else None


def formater(c, u):
    """Valeur et incertitude avec 1 chiffre significatif sur u.

    Les décimales de c sont alignées sur celles de u.
    """
    if u is None or not np.isfinite(u) or u <= 0:
        return f"{c:.2f} nm"
    exposant = int(np.floor(np.log10(u)))
    u1 = round(u, -exposant)          # arrondi à 1 chiffre significatif
    if u1 >= 10 ** (exposant + 1):    # ex. 0.96 -> 1.0 : exposant augmenté
        exposant += 1
    decimales = max(0, -exposant)
    return f"({c:.{decimales}f} ± {u1:.{decimales}f}) nm"


def resume_verification(ecarts, u_cal):
    if u_cal is None:
        return ["u_cal indisponible : il faut un axe ajusté et une raie de vérification."]
    return [f"Vérification à {nm:g} nm : écart {e:+.3f} nm" for nm, e in ecarts] + [
        f"u_cal estimée = {u_cal:.3g} nm (écart quadratique moyen sur {len(ecarts)} raie(s))."
    ]


def transmission(ref, mes):
    i_ref = lisser(ref.mean(axis=0))
    T = np.full(i_ref.shape, np.nan)
    valide = i_ref > SEUIL_REF * i_ref.max()
    T[valide] = lisser(mes.mean(axis=0))[valide] / i_ref[valide]
    return i_ref, T


def coupures(x, T):
    """L’axe est croissant. Renvoie les deux coupures et le niveau utilisé."""
    valides = T[np.isfinite(T)]
    niveau = FRAC * np.percentile(valides, 99) if valides.size else np.nan
    if not niveau > 0:
        return (None, None), niveau

    s = T - niveau
    g, d = s[:-1], s[1:]
    montantes = np.flatnonzero((g < 0) & (d >= 0))
    descendantes = np.flatnonzero((g >= 0) & (d < 0))

    def interpole(i):
        return x[i] - g[i] * (x[i + 1] - x[i]) / (d[i] - g[i])

    return (
        interpole(montantes[0]) if montantes.size else None,
        interpole(descendantes[-1]) if descendantes.size else None,
    ), niveau


def repetabilite(x, ordre, ref, mes):
    """Incertitude-type (nm) sur la moyenne des coupures, paire par paire."""
    n = min(len(ref), len(mes))
    if n < 2:
        return [(None, None), (None, None)]

    xc = x[ordre]
    valeurs = [[], []]
    for k in range(n):
        _, T = transmission(ref[k:k + 1], mes[k:k + 1])
        centres, _ = coupures(xc, T[ordre])
        for serie, c in zip(valeurs, centres):
            if c is not None:
                serie.append(c)

    return [(np.std(v, ddof=1) / np.sqrt(n) if len(v) == n else None, len(v))
            for v in valeurs]


class App:
    def __init__(self, root):
        self.root, self.lasers, self.occupe = root, [], False
        # Vrai seulement pendant la prise d'images : l'aperçu se fige alors.
        # Le reste du temps (y compris pendant les boîtes de dialogue entre
        # obscurité, lumière blanche sans filtre et avec filtre), il reste en direct.
        self.en_capture = False
        self.expo = tk.DoubleVar(value=EXPO_MS)
        self.n_img = tk.IntVar(value=1)
        self.n_ang = tk.IntVar(value=1)
        self.coupure_theo = tk.StringVar()
        self.u_coupure_theo = tk.StringVar()
        self.calib_txt = tk.StringVar(value="Aucune raie enregistrée")
        self.laser_nm = tk.StringVar(value="532")
        self.laser_verif = tk.BooleanVar()
        self.txt_reference = None      # (nom du TXT, texte du champ, nm numérique)
        self.racine_calib = DOSSIER_LASERS_PROFS
        self.commandes = []

        root.title("Spectromètre — acquisition")
        self.u_ang = tk.DoubleVar(value=0.5)   # incertitude sur l'angle (°)
        self.vue = tk.StringVar(value="spectres")
        self.pts_angle = []                    # (angle, nom, coupure, u)
        self.theorie = None
        self.construire()
        self.demarrer_camera()
        root.after_idle(self.calibration_demarrage)
        self.apercu()
        root.protocol("WM_DELETE_WINDOW", self.fermer)

    def construire(self):
        g = ttk.Frame(self.root)
        g.pack(side="left", fill="y", padx=6, pady=6)

        def bouton(parent, texte, action, **position):
            b = ttk.Button(parent, text=texte, command=action)
            b.pack(fill="x", **position)
            self.commandes.append(b)

        def reglage(parent, texte, variable, minimum, maximum):
            ttk.Label(parent, text=texte).pack(anchor="w")
            ttk.Spinbox(
                parent, from_=minimum, to=maximum, textvariable=variable, width=8
            ).pack(anchor="w", pady=(0, 4))

        choix = ttk.Notebook(g)
        choix.pack(fill="x")
        blanc = ttk.Frame(choix, padding=8)
        laser = ttk.Frame(choix, padding=8)
        choix.add(blanc, text="Lumière blanche")
        choix.add(laser, text="Laser")

        ttk.Label(
            blanc,
            text="Mesure du filtre\nObscurité, puis référence sans filtre, puis mesure avec filtre.",
            wraplength=225,
        ).pack(anchor="w", pady=(0, 6))
        reglage(blanc, "Nombre d’angles", self.n_ang, 1, 20)
        ttk.Label(blanc, text="Incertitude angle ± (°)").pack(anchor="w")
        ttk.Spinbox(
            blanc, from_=0, to=10, increment=0.1, textvariable=self.u_ang,
            width=8, command=self.tracer_angle,
        ).pack(anchor="w", pady=(0, 4))
        ttk.Label(blanc, text="Coupure théorique (nm, facultative)").pack(anchor="w")
        ttk.Entry(blanc, textvariable=self.coupure_theo, width=12).pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(blanc, text="Incertitude ± (nm, facultative)").pack(anchor="w")
        ttk.Entry(blanc, textvariable=self.u_coupure_theo, width=12).pack(
            anchor="w", pady=(0, 6)
        )
        bouton(blanc, "Mesurer la transmission", self.sequence)
        bouton(blanc, "Charger une acquisition (dossier)…", self.charger_acquisition)

        ttk.Label(laser, text="Longueur d’onde du laser (nm)").pack(anchor="w")
        ttk.Entry(laser, textvariable=self.laser_nm, width=12).pack(
            anchor="w", pady=(0, 4)
        )
        bouton(laser, "Longueur d’onde depuis un TXT…", self.choisir_txt)
        ttk.Checkbutton(
            laser, text="Raie de vérification", variable=self.laser_verif
        ).pack(anchor="w", pady=4)
        bouton(laser, "Nouvelle calibration (choisir dossier)…", self.nouvelle_calib, pady=(0, 6))
        bouton(laser, "Capturer le laser (RAW)", self.ajouter_laser)
        bouton(laser, "Importer des captures laser…", self.choisir_calib, pady=(6, 0))
        bouton(laser, "Désactiver l’étalonnage", self.sans_calib)

        commun = ttk.LabelFrame(g, text="Réglages communs", padding=8)
        commun.pack(fill="x", pady=(8, 0))
        reglage(commun, "Exposition (ms)", self.expo, 0.1, 95)
        reglage(commun, "Images par acquisition", self.n_img, 1, 50)
        bouton(commun, "Appliquer l’exposition", self.appliquer_expo)

        ttk.Label(g, textvariable=self.calib_txt, wraplength=225).pack(
            anchor="w", pady=(8, 2)
        )
        self.log_w = ScrolledText(g, width=31, height=4, state="disabled")
        self.log_w.pack(fill="both", expand=True)

        # ---------------- Onglets de droite ----------------
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(side="right", fill="both", expand=True)
        t1, t2, t3 = (ttk.Frame(self.nb) for _ in range(3))
        self.nb.add(t1, text="Aperçu")
        self.nb.add(t2, text="Transmission")
        self.nb.add(t3, text="Étalonnage")

        # Onglet Aperçu
        self.lbl = ttk.Label(t1)
        self.lbl.pack()

        # Onglet Transmission : boutons de vue en haut, texte en bas,
        # graphique (avec barre de zoom) au milieu
        barre_vue = ttk.Frame(t2)
        barre_vue.pack(side="top", fill="x")
        for texte, valeur in (("Spectres", "spectres"), ("Coupure vs angle", "angle")):
            ttk.Radiobutton(
                barre_vue, text=texte, value=valeur,
                variable=self.vue, command=self.changer_vue,
            ).pack(side="left", padx=4)

        self.res = ScrolledText(t2, height=8, wrap="word")
        self.res.pack(side="bottom", fill="x")

        # Vue « Spectres »
        self.zone_spec = ttk.Frame(t2)
        self.zone_spec.pack(fill="both", expand=True)
        self.fig = Figure(figsize=(7, 4.2), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.zone_spec)
        self.barre = NavigationToolbar2Tk(
            self.canvas, self.zone_spec, pack_toolbar=False
        )
        self.barre.update()
        self.barre.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Vue « Coupure vs angle » (cachée au départ)
        self.zone_ang = ttk.Frame(t2)
        self.fig_ang = Figure(figsize=(7, 4.2), dpi=100)
        self.canvas_ang = FigureCanvasTkAgg(self.fig_ang, master=self.zone_ang)
        self.canvas_ang.get_tk_widget().pack(fill="both", expand=True)

        # Onglet Étalonnage
        self.fig_cal = Figure(figsize=(7, 4.2), dpi=100)
        self.canvas_cal = FigureCanvasTkAgg(self.fig_cal, master=t3)
        self.canvas_cal.get_tk_widget().pack(fill="both", expand=True)
        self.res_cal = ScrolledText(t3, height=6, wrap="word")
        self.res_cal.pack(fill="x")
        self.afficher_calib()

    def log(self, s):
        self.log_w.config(state="normal")
        self.log_w.insert("end", s + "\n")
        self.log_w.see("end")
        self.log_w.config(state="disabled")
        self.root.update_idletasks()

    def occupation(self, active):
        self.occupe = active
        for bouton in self.commandes:
            bouton.state(["disabled" if active else "!disabled"])

    def choisir_txt(self):
        chemin = filedialog.askopenfilename(
            parent=self.root,
            title="Spectre de référence",
            filetypes=[("Fichier spectral TXT", "*.txt")],
        )
        if not chemin:
            return
        try:
            nm = longueur_txt(chemin)
            self.laser_nm.set(f"{nm:.3f}")
            self.txt_reference = (Path(chemin).name, self.laser_nm.get(), nm)
            self.log(f"Pic du TXT : {nm:.3f} nm. Vérifiez qu’il s’agit du même laser.")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Spectre de référence", str(exc), parent=self.root)

    def nouvelle_calib(self):
        chemin = filedialog.askdirectory(
            parent=self.root, mustexist=False,
            title="Dossier de la calibration (créez-le au besoin)",
        )
        if not chemin:
            return
        self.racine_calib = Path(chemin)
        self.racine_calib.mkdir(parents=True, exist_ok=True)
        try:   # reprend les raies déjà présentes, sinon repart à vide
            self.lasers, dernier = charger_calibration_dossier(self.racine_calib)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Calibration", str(exc), parent=self.root)
            return
        self.afficher_calib(dernier)
        self.log(f"Calibration : {self.racine_calib} ({len(self.lasers)} laser(s)).")

    def ajouter_laser(self):
        if self.cam is None:
            messagebox.showinfo("Caméra", "Pas de caméra : utilisez « Charger une acquisition ».", parent=self.root)
            return
        try:
            texte = self.laser_nm.get()
            nm = float(texte.strip().replace(",", "."))
            n = self.n_img.get()
            if not np.isfinite(nm) or not 0 < nm < LAMBDA * 1e9:
                raise ValueError("Saisissez une longueur d’onde laser valide en nm.")
            if not 1 <= n <= 50:
                raise ValueError("Saisissez de 1 à 50 images.")
            verif = self.laser_verif.get()
            source = (
                self.txt_reference[0]
                if self.txt_reference and texte == self.txt_reference[1]
                else "saisie manuelle"
            )
            # nm de référence = pic du TXT si c'est lui qui a rempli le champ
            if self.txt_reference and texte == self.txt_reference[1]:
                nm = self.txt_reference[2]
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Capture laser", str(exc), parent=self.root)
            return

        self.occupation(True)
        self.nb.select(0)
        try:
            if not messagebox.askokcancel(
                "Capture laser",
                f"Retirez le filtre et présentez la raie à {nm:g} nm.\n"
                "Vérifiez l’alignement et réglez une exposition sans saturation.",
                parent=self.root,
            ):
                return
            if not self.appliquer_expo():
                return

            dossier_laser = Path(self.racine_calib) / nom_dossier_laser(nm)
            dossier_laser.mkdir(parents=True, exist_ok=True)
            meta = dossier_laser / "laser.json"
            if meta.is_file():
                nm_old = float(json.loads(meta.read_text(encoding="utf-8"))["nm"])
                if abs(nm_old - nm) > 0.5:
                    raise ValueError(
                        f"{dossier_laser.name}/ contient déjà un laser à {nm_old:g} nm."
                    )
            dossier = dossier_laser / datetime.now().strftime("acq_%Y%m%d_%H%M%S")

            for _ in self.images(str(dossier), n):   # sauvegarde sans tout garder en RAM
                pass
            meta.write_text(
                json.dumps({"nm": nm, "verification": verif, "txt": source},
                           ensure_ascii=False),
                encoding="utf-8",
            )

            # relit toutes les acquisitions de ce laser et remplace la raie
            laser, intensite = laser_depuis_dossier(dossier_laser)
            nouveaux = [p for p in self.lasers if abs(p.nm - laser.nm) > 0.5] + [laser]
            verifier_lasers(nouveaux)
            self.lasers = nouveaux
            self.afficher_calib((laser, intensite))
            self.nb.select(2)
            self.log(
                f"Laser {nm:g} nm : acquisition {laser.n_acq} enregistrée, "
                f"pixel moyen {laser.px:.2f} ± {laser.u_px:.2f}."
            )
            if laser.n_acq == 1:
                self.log("Une seule acquisition : refaites-en au moins une pour estimer u_px.")
            alerte = laser.alerte
            if alerte:
                self.log_pic([laser])
                messagebox.showwarning(
                    "Capture laser",
                    f"{alerte}\nRaie enregistrée quand même. "
                    "Pour un centre plus fiable, réduisez l’exposition et recommencez.",
                    parent=self.root,
                )
        except (OSError, RuntimeError, ValueError, KeyError, tk.TclError) as exc:
            messagebox.showerror("Capture laser", str(exc), parent=self.root)
        finally:
            self.occupation(False)

    def choisir_calib(self):
        chemin = filedialog.askdirectory(parent=self.root, title="Dossier de calibration")
        if not chemin:
            return
        self.occupation(True)
        try:
            lasers, dernier = charger_calibration_dossier(chemin)
            if not lasers:
                raise ValueError("Aucun dossier laser reconnu (laser.json absent).")
            self.racine_calib, self.lasers = Path(chemin), lasers
            self.afficher_calib(dernier)
            self.nb.select(2)
            self.log(f"{len(lasers)} lasers chargés depuis {chemin}.")
            self.log_pic(self.lasers)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            messagebox.showerror("Dossier de calibration", str(exc), parent=self.root)
        finally:
            self.occupation(False)

    def charger_calibration(self):
        """Recharge la calibration des lasers (RAW + TXT).

        Renvoie ((lasers, dernier), None) si tout va bien,
        sinon (None, description de l'erreur).
        """
        try:
            return charger_lasers_profs(self.racine_calib), None
        except Exception as exc:  # on veut voir toute erreur, pas la masquer
            return None, f"{type(exc).__name__} : {exc}"

    def calibration_demarrage(self):
        resultat, erreur = self.charger_calibration()
        if erreur is not None:
            self.log("ERREUR : calibration non chargée au démarrage.")
            messagebox.showerror(
                "Calibration introuvable",
                "Impossible de charger la calibration des lasers au démarrage.\n\n"
                f"{erreur}\n\n"
                "Elle sera retentée au début de chaque mesure en lumière blanche.",
                parent=self.root,
            )
            return
        self.lasers, dernier = resultat
        self.afficher_calib(dernier)
        self.log(f"Calibration chargée : {len(self.lasers)} lasers.")
        self.log_pic(self.lasers)

    def calibration_pour_mesure(self):
        """Recharge la calibration avant une mesure en lumière blanche.

        Renvoie la liste des lasers à utiliser ([] = sans calibration),
        ou None si l'utilisateur annule la mesure.
        """
        resultat, erreur = self.charger_calibration()
        if erreur is None:
            self.lasers, dernier = resultat
            self.afficher_calib(dernier)
            self.log(f"Calibration rechargée : {len(self.lasers)} lasers.")
            self.log_pic(self.lasers)
            return list(self.lasers)

        self.log("ERREUR : calibration non chargée.")
        if not messagebox.askyesno(
            "Calibration introuvable",
            "Impossible de charger la calibration des lasers.\n\n"
            f"{erreur}\n\n"
            "Continuer SANS calibration (axe nominal, 550 nm au centre)?",
            icon="error",
            parent=self.root,
        ):
            return None
        self.log("ATTENTION : mesure lancée sans calibration.")
        return []

    def log_pic(self, lasers):
        for ligne in avertissements_pic(lasers):
            self.log(ligne)

    def sans_calib(self):
        self.lasers = []
        self.afficher_calib()

    def afficher_calib(self, mesure=None):
        n_verif = sum(p.verification for p in self.lasers)
        self.calib_txt.set(
            f"Raies : {len(self.lasers) - n_verif} ajustement, {n_verif} vérification\n"
            f"Dossier : {Path(self.racine_calib).name}"
        )
        self.fig_cal.clear()
        bas = self.fig_cal.subplots()
        lignes = []

        if self.lasers:
            x, desc = calibrer(self.lasers[0].largeur, self.lasers)
            bas.plot(x, color="0.5", label=desc)

            s = sigma_axe(self.lasers[0].largeur, self.lasers)
            if s is not None:
                bas.fill_between(
                    np.arange(len(x)), x - s, x + s,
                    color="tab:blue", alpha=0.2, label="Incertitude régression (±1σ)",
                )
                lignes.append(
                    f"Incertitude de régression : ±{s.min():.2f} à ±{s.max():.2f} nm (k=1)."
                )

            for verif, couleur, nom in (
                (False, "tab:blue", "Ajustement"),
                (True, "tab:orange", "Vérification"),
            ):
                pts = [p for p in self.lasers if p.verification == verif]
                if pts:
                    bas.errorbar(
                        [p.px for p in pts], [p.nm for p in pts],
                        xerr=[p.u_px for p in pts], yerr=[p.u_nm for p in pts],
                        fmt="o", color=couleur, capsize=3, label=nom,
                    )
                    for p in pts:
                        bas.annotate(
                            f"{p.nm:g} nm",
                            (p.px, p.nm),
                            textcoords="offset points",
                            xytext=(6, -12),
                            fontsize=8,
                            color=couleur,
                        )
                    lignes.extend(
                        f"{nom}  {p.nm:g} nm  |  {p.px:.2f} ± {p.u_px:.2f} px "
                        f"({p.n_acq} acq)  |  {p.source}"
                        for p in pts
                    )
            bas.legend(fontsize=8)
            ecarts = ecarts_verification(x, self.lasers)
            lignes.extend(resume_verification(ecarts, incertitude_calibration(ecarts)))
            lignes.extend(avertissements_pic(self.lasers))

        bas.set(xlabel="Pixel binné", ylabel="Longueur d’onde (nm)")
        self.fig_cal.tight_layout()
        self.canvas_cal.draw_idle()
        self.res_cal.delete("1.0", "end")
        self.res_cal.insert("end", "\n".join(lignes))

    def demarrer_camera(self):
        if Picamera2 is None:
            self.cam, self.taille = None, None
            self.log("Mode sans caméra : seul « Charger une acquisition » fonctionne.")
            self.lbl.config(text="Aucune caméra détectée", font=("TkDefaultFont", 14))
            return
        cam = Picamera2()
        taille = tuple(cam.sensor_resolution)
        modes = [m for m in cam.sensor_modes if tuple(m["size"]) == taille] or list(cam.sensor_modes)
        mode = max(modes, key=lambda m: m["bit_depth"])
        fmt = mode.get("unpacked") or mode["format"].replace("_CSI2P", "")
        d = int(1e6 / FPS)
        config = cam.create_still_configuration(
            main={"size": taille, "format": "RGB888"},
            lores={"size": (1024, 768)},
            raw={"size": tuple(mode["size"]), "format": fmt},
            sensor={"output_size": tuple(mode["size"]), "bit_depth": mode["bit_depth"]},
            buffer_count=3,
            controls={
                "FrameDurationLimits": (d, d),
                "ExposureTime": int(self.expo.get() * 1000),
                "AnalogueGain": GAIN,
                "AwbEnable": False,
                "ColourGains": (1.0, 1.0),
            },
        )
        cam.configure(config)
        cam.start()
        self.cam, self.taille = cam, taille

    def appliquer_expo(self):
        try:
            ms = self.expo.get()
            if not np.isfinite(ms):
                raise ValueError("Saisissez une exposition finie.")
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Exposition", str(exc), parent=self.root)
            return False

        ms = min(max(ms, 0.1), 1000 / FPS * 0.95)
        self.expo.set(ms)
        if self.cam is not None:
            self.cam.set_controls({"ExposureTime": int(ms * 1000)})
            for _ in range(4):
                self.cam.capture_metadata()
        self.log(f"Exposition : {ms:.1f} ms")
        return True

    def apercu(self):
        if self.cam is None:
            return
        if not self.en_capture:
            try:
                rgb = cv2.cvtColor(
                    self.cam.capture_array("lores"),
                    cv2.COLOR_YUV2RGB_I420,
                )[:, :1024]
                self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.resize(rgb, (640, 480))))
                self.lbl.config(image=self.photo)
            except Exception:
                pass
        self.root.after(150, self.apercu)

    def capturer(self):
        req = self.cam.capture_request()
        try:
            return req.make_array("main"), req.make_buffer("raw")
        finally:
            req.release()

    def buffer_vers_raw(self, buf):
        cfg = self.cam.camera_config["raw"]
        w, h = cfg["size"]
        a = np.ascontiguousarray(self.cam.helpers.make_array(buf, cfg))
        if a.dtype == np.uint8:
            a = a.view(np.uint16)
        return a[:h, :w].copy()

    def images(self, sous, n):
        os.makedirs(sous, exist_ok=True)
        self.en_capture = True
        try:
            for i in range(n):
                img, buf = self.capturer()
                raw = self.buffer_vers_raw(buf)
                cv2.imwrite(
                    os.path.join(sous, f"image_{i:03d}.png"),
                    img, [cv2.IMWRITE_PNG_COMPRESSION, 0],
                )
                np.save(os.path.join(sous, f"raw_{i:03d}.npy"), raw)
                self.log(f"  {os.path.basename(sous)} : image {i + 1}/{n}")
                yield raw
        finally:
            self.en_capture = False

    def mesurer(self, sous, n, noir, bande=None):
        profs, sat = [], 0.0
        for raw in self.images(sous, n):
            p, bande = profil(raw, noir, bande)
            profs.append(p)
            zone = raw[2 * bande[0]:2 * bande[1]]
            sat = max(sat, 100 * (zone >= BLANC - 5).mean())

        if sat > 0.1:
            self.log(f"ATTENTION : {sat:.2f} % de photosites saturés. Réduisez l’exposition.")
        return np.array(profs), bande

    def lire_coupure_theorique(self):
        texte = self.coupure_theo.get().strip()
        incertitude = self.u_coupure_theo.get().strip()
        if not texte:
            if incertitude:
                raise ValueError("Saisissez la coupure théorique pour indiquer son incertitude.")
            return None

        try:
            nm = float(texte.replace(",", "."))
            u = float(incertitude.replace(",", ".")) if incertitude else None
        except ValueError:
            raise ValueError("La coupure théorique et son incertitude doivent être en nm.") from None

        if not np.isfinite(nm) or nm <= 0:
            raise ValueError("La coupure théorique doit être une longueur d’onde positive.")
        if u is not None and (not np.isfinite(u) or u < 0 or not np.isfinite(nm + u)):
            raise ValueError("L’incertitude théorique doit être finie, positive ou nulle.")
        return nm, u

    def sequence(self):
        if self.cam is None:
            messagebox.showinfo("Caméra", "Pas de caméra : utilisez « Charger une acquisition ».", parent=self.root)
            return
        try:
            n, na = self.n_img.get(), self.n_ang.get()
            if not (1 <= n <= 50 and 1 <= na <= 20):
                raise ValueError("Saisissez de 1 à 50 images et de 1 à 20 angles.")
            theorie = self.lire_coupure_theorique()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Paramètres", str(exc), parent=self.root)
            return

        dossier = os.path.join(
            DOSSIER_SORTIE, datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.occupation(True)
        self.nb.select(0)  # onglet Aperçu visible pendant toute la séquence
        try:
            lasers = self.calibration_pour_mesure()
            if lasers is None:
                return
            self.nb.select(0)

            if not self.appliquer_expo():
                return

            if not messagebox.askokcancel(
                "Obscurité", "Bloquez la lumière, puis validez.", parent=self.root
            ):
                return

            noir = sum(
                raw.astype(np.float32)
                for raw in self.images(os.path.join(dossier, "dark"), n)
            ) / n

            if not messagebox.askokcancel(
                "Référence",
                "Allumez la lumière blanche et retirez le filtre, puis validez.",
                parent=self.root,
            ):
                return

            ref, bande = self.mesurer(os.path.join(dossier, "reference"), n, noir)

            if lasers and lasers[0].largeur != ref.shape[1]:
                self.log("ERREUR : largeur des RAW lasers ≠ largeur de la mesure.")
                if not messagebox.askyesno(
                    "Calibration incompatible",
                    f"Les RAW lasers font {lasers[0].largeur} pixels binnés de large, "
                    f"la mesure en fait {ref.shape[1]}.\n"
                    "Les lasers ont sans doute été pris avec une autre résolution "
                    "ou un autre mode du capteur.\n\n"
                    "Continuer SANS calibration (axe nominal, 550 nm au centre)?",
                    icon="error",
                    parent=self.root,
                ):
                    return
                lasers = []
                self.log("ATTENTION : mesure poursuivie sans calibration.")

            mesures = []
            for k in range(na):
                self.nb.select(0)
                angle = simpledialog.askfloat(
                    "Angle",
                    f"Mesure {k + 1}/{na}\n"
                    "Insérez le filtre, réglez l’angle d’incidence (°), puis validez.",
                    parent=self.root,
                    initialvalue=0.0,
                    minvalue=-90,
                    maxvalue=90,
                )
                if angle is None:
                    break

                sous = os.path.join(dossier, f"mesure_{k + 1:02d}_{angle:g}deg")
                profils, _ = self.mesurer(sous, n, noir, bande)
                mesures.append((angle, profils))

            if mesures:
                self.analyser(dossier, ref, mesures, n, theorie, lasers)
        except (OSError, RuntimeError, ValueError) as exc:
            messagebox.showerror("Séquence", str(exc), parent=self.root)
        finally:
            self.occupation(False)

    def analyser(self, dossier, ref, mesures, n_img, theorie=None, lasers=None):
        npx = ref.shape[1]
        lasers = self.lasers if lasers is None else lasers
        avertissement = []
        if lasers and lasers[0].largeur != npx:
            lasers = []
            avertissement = [
                "Largeur RAW incompatible avec les captures laser. Axe nominal utilisé."
            ]

        x, desc = calibrer(npx, lasers)
        calibre = any(not p.verification for p in lasers)
        ordre = np.argsort(x)
        ecarts = ecarts_verification(x, lasers)
        u_cal = incertitude_calibration(ecarts)
        s_reg = sigma_axe(npx, lasers)
        raies = [nm for nm, _ in ecarts]
        lignes = (
            [f"Étalonnage : {desc}"] + avertissement
            + avertissements_pic(lasers) + resume_verification(ecarts, u_cal)
        )
        lignes.append(
            "Barres : incertitude-type (k=1) combinant répétabilité, vérification "
            "(u_cal) et régression (u_reg), quand disponibles."
        )
        if len(raies) == 1:
            lignes.append("Une seule raie de vérification. L’estimation de u_cal est indicative.")

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        i_ref, _ = transmission(ref, mesures[0][1])
        colonnes, entete = [x, i_ref], "lbd_nm,I_ref"

        pts_ang = []

        for a, profs in mesures:
            _, T = transmission(ref, profs)
            centres, niveau = coupures(x[ordre], T[ordre])
            ligne, = ax.plot(x, T)
            lignes.append(f"θ = {a:g}°")
            txt = []
            incertitudes = (
                repetabilite(x, ordre, ref, profs) if any(c is not None for c in centres) else []
            )

            for nom, c, (u_stat, nb) in zip(
                    ("montante", "descendante"), centres, incertitudes):
                if c is None:
                    continue

                # u_reg à la position de la coupure (interpolé sur l'axe croissant)
                u_reg = (float(np.interp(c, x[ordre], s_reg[ordre]))
                         if s_reg is not None else None)
                parts = [v for v in (u_stat, u_cal, u_reg) if v is not None]
                u = float(np.sqrt(sum(v**2 for v in parts))) if parts else None
                pts_ang.append((a, nom, c, u))

                valeur = formater(c, u)

                detail = []
                n_paires = min(len(ref), len(profs))
                if u_stat is not None:
                    detail.append(f"u_rép = {u_stat:.3g} nm (σ/√{n_paires}, {n_paires} paires)")
                elif nb is None:
                    detail.append("répétabilité indisponible : au moins 2 images par série")
                else:
                    detail.append(f"coupure introuvable sur {n_paires - nb}/{n_paires} paires")
                if u_reg is not None:
                    detail.append(f"u_reg = {u_reg:.3g} nm")

                txt.append(valeur)
                lignes.append(f"    {nom} : {valeur} ({', '.join(detail)})")
                ax.axvline(c, color=ligne.get_color(), ls="--", lw=0.7)
                if u is not None:
                    ax.axvspan(c - u, c + u, color=ligne.get_color(), alpha=0.15, zorder=0)
                ax.errorbar(c, niveau, xerr=u, fmt="o", color=ligne.get_color(), capsize=4)

            if not txt:
                lignes.append("    Aucune coupure détectée.")
            ligne.set_label(f"{a:g}° : " + (" / ".join(txt) if txt else "aucune coupure"))
            colonnes.append(T)
            entete += f",T_{a:g}deg"

        if theorie is not None:
            nm, u_theo = theorie
            valeur = f"{nm:g} nm" if u_theo is None else f"({nm:g} ± {u_theo:g}) nm"
            label = f"Coupure théorique : {valeur}"
            lignes.append(label)

            demi_largeur = u_theo if u_theo is not None else 0
            debut_theo, fin_theo = nm - demi_largeur, nm + demi_largeur
            if demi_largeur > 0:
                ax.axvspan(debut_theo, fin_theo, color="0.2", alpha=0.13, zorder=0)
            ax.axvline(nm, color="0.2", ls=":", lw=1.5, label=label)

        ax.set_xlim(400, 700)
        ax.set_xlabel("Longueur d’onde (nm)" + ("" if calibre else " — NON CALIBRÉ"))
        ax.set_ylabel("Transmission (u.a.)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

        self.barre.update()

        self.pts_angle, self.theorie = pts_ang, theorie
        self.log(f"Points coupure vs angle : {len(pts_ang)}")
        self.tracer_angle()

        self.fig_ang.savefig(os.path.join(dossier, "coupure_vs_angle.png"), dpi=150)

        self.fig.savefig(os.path.join(dossier, "transmission.png"), dpi=150)
        np.savetxt(
            os.path.join(dossier, "transmission.csv"), np.column_stack(colonnes),
            delimiter=",", header=entete, comments="",
        )
        parametres = [
            f"Exposition (ms) : {self.expo.get()} | Gain : {GAIN}",
            f"Images par acquisition : {n_img}",
            f"RAW : {BLANC + 1} niveaux, noir = dark",
            f"Coupure : {FRAC:.0%} du 99e percentile de T valide",
            "Répétabilité : écart-type des coupures par paire d'images, divisé par √N",
            "Barres horizontales : incertitude-type (k=1)",
            f"Dossier de calibration : {self.racine_calib}",
        ]
        Path(dossier, "parametres.txt").write_text(
            "\n".join(parametres + lignes) + "\n", encoding="utf-8"
        )
        self.res.delete("1.0", "end")
        self.res.insert("end", "\n".join(lignes))
        self.nb.select(1)
        self.log(f"Résultats enregistrés dans {dossier}/")

    def fermer(self):
        if self.occupe:
            return
        try:
            if self.cam is not None:
                self.cam.stop()
                self.cam.close()
        finally:
            self.root.destroy()

    def charger_acquisition(self):
        """Relit un dossier d'acquisition (dark/, reference/, mesure_XX_YYdeg/)
        et relance l'analyse sans caméra."""
        chemin = filedialog.askdirectory(
            parent=self.root, title="Dossier d'acquisition (contient dark/, reference/, mesure_*)"
        )
        if not chemin:
            return
        dossier = Path(chemin)

        self.occupation(True)
        try:
            theorie = self.lire_coupure_theorique()

            dark = sorted((dossier / "dark").glob("raw_*.npy"))
            if not dark:
                raise ValueError("Sous-dossier dark/ absent ou sans raw_*.npy.")
            noir = sum(np.load(f).astype(np.float32) for f in dark) / len(dark)

            ref, bande = profils_dossier(dossier / "reference", noir, None)

            mesures = []
            for d in sorted(dossier.glob("mesure_*")):
                m = re.search(r"_(-?\d+(?:\.\d+)?)deg$", d.name)
                if not d.is_dir() or not m:
                    continue
                profs, _ = profils_dossier(d, noir, bande)
                mesures.append((float(m.group(1)), profs))
            if not mesures:
                raise ValueError("Aucun dossier mesure_XX_YYdeg trouvé.")

            lasers = self.calibration_pour_mesure()
            if lasers is None:
                return

            # Résultats écrits dans un sous-dossier pour ne rien écraser
            sortie = dossier / "relecture"
            sortie.mkdir(exist_ok=True)
            self.analyser(str(sortie), ref, mesures, len(ref), theorie, lasers)
            self.log(f"Acquisition relue : {len(mesures)} angle(s), {len(ref)} image(s).")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Chargement", f"{type(exc).__name__} : {exc}", parent=self.root)
        finally:
            self.occupation(False)

    def changer_vue(self):
        spectres = self.vue.get() == "spectres"
        (self.zone_ang if spectres else self.zone_spec).pack_forget()
        (self.zone_spec if spectres else self.zone_ang).pack(fill="both", expand=True)
        if not spectres:
            self.tracer_angle()

    def tracer_angle(self):
        try:
            ua = float(self.u_ang.get())
        except (ValueError, tk.TclError):
            ua = 0.5
        self.fig_ang.clear()
        ax = self.fig_ang.add_subplot(111)

        for nom, couleur in (("montante", "tab:blue"), ("descendante", "tab:red")):
            pts = sorted((a, c, 0 if u is None else u)
                         for a, n, c, u in self.pts_angle if n == nom)
            if not pts:
                continue
            a, c, u = (np.array(v) for v in zip(*pts))
            ax.errorbar(a, c, xerr=ua, yerr=u, fmt="o", color=couleur,
                        capsize=4, label=f"Coupure {nom}")
            ax.fill_between(a, c - u, c + u, color=couleur, alpha=0.12)

        if self.theorie is not None:
            nm, u_theo = self.theorie
            label = (f"Coupure théorique : {nm:g} nm" if not u_theo
                     else f"Coupure théorique : ({nm:g} ± {u_theo:g}) nm")
            if u_theo:
                ax.axhspan(nm - u_theo, nm + u_theo, color="0.2", alpha=0.13, zorder=0)
            ax.axhline(nm, color="0.2", ls=":", lw=1.5, label=label)

        if not self.pts_angle:
            ax.text(0.5, 0.5, "Aucune coupure mesurée", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_xlabel(f"Angle d’incidence (°)  —  incertitude ± {ua:g}°")
        ax.set_ylabel("Longueur d’onde de coupure (nm)")
        ax.grid(alpha=0.3)
        if self.pts_angle or self.theorie is not None:
            ax.legend(fontsize=8)
        self.fig_ang.tight_layout()
        self.canvas_ang.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()