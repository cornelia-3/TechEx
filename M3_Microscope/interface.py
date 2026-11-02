#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface graphique – Mandat 3 (PHS3910) : suivi de particules et mesure du rayon
==================================================================================

Version « fichier TIFF » : l'acquisition ne passe plus par la caméra du Raspberry Pi. Les images proviennent
d'un fichier TIFF (multipage ou dossier d'un TIFF par image), d'une vidéo, ou de la simulation (test sans fichier).

Ce programme enveloppe l'analyse de `traitement_video_tiff.py` (détection, ajustement gaussien,
liaison des trajectoires, MSD, Stokes-Einstein, incertitudes) dans une interface graphique :

  * réglage des conditions (grossissement, exposition, fréquence, résolution, profondeur de bit) ;
  * affichage des images avec l'échelle spatiale (µm/pixel), une barre d'échelle et les conditions,
    plus les détections et les trajectoires en temps réel ;
  * enregistrement de séquences vidéo (MP4) où CHAQUE image affiche : temps écoulé (s), échelle (µm/pixel),
    grossissement, temps d'exposition, fréquence d'acquisition et profondeur de bit ;
  * analyse affichée dans l'interface : MSD, rayon par particule, rayon moyen ± incertitude.

Installation
------------
  pip install numpy scipy matplotlib opencv-python pillow tifffile

Placer ce fichier dans le même dossier que `traitement_video_tiff.py`, puis :  python interface_suivi.py
"""
from __future__ import annotations

import csv
import json
import math
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont, ImageTk
from scipy.optimize import linear_sum_assignment

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tkinter.font as tkfont

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:                                    # le module d'analyse (même dossier)
    import traitement_video_tiff as tp
except ImportError:
    import suivi_particules as tp

NOM_APP = "Suivi de particules – PHS3910"
LARGEUR_CAPTEUR_PX = 4056                         # IMX477 (caméra HQ) : 4056 × 3040, pixels de 1,55 µm
GROSSISSEMENTS = [2, 4, 10, 20, 40, 60]
OUVERTURE_NUM = {2: 0.06, 4: 0.10, 10: 0.25, 20: 0.40, 40: 0.65, 60: 0.85}   # valeurs typiques (simulation)
RESOLUTIONS = {"640 × 480": (640, 480), "1014 × 760": (1014, 760), "2028 × 1520": (2028, 1520)}
EXTENSIONS_FICHIERS = "*.mp4 *.avi *.mov *.mkv *.tif *.tiff"
PALETTE = [(0, 165, 255), (255, 200, 0), (0, 230, 0), (255, 0, 255), (0, 255, 255),
           (255, 110, 110), (110, 255, 160), (90, 90, 255), (255, 255, 0), (210, 130, 255)]   # BGR
GRIS_COURT = (170, 170, 170)


def couleur_piste(pid):
    return PALETTE[pid % len(PALETTE)]


def viscosite_eau(T_C):
    """Viscosité dynamique de l'eau (Pa·s), relation de Vogel. ≈ 0,95 mPa·s à 22 °C."""
    return 2.414e-5 * 10 ** (247.8 / (T_C + 273.15 - 140.0))


#############################################################################################################
#                                       DONNÉES : CONDITIONS ET PARAMÈTRES
#############################################################################################################

@dataclass
class Conditions:
    grossissement: int = 20
    exposition_ms: Optional[float] = 10.0     # None = inconnue (fichier)
    fps: float = 30.0
    gain: float = 1.0
    bits: int = 8
    largeur: int = 640
    hauteur: int = 480
    echelle_manuelle_um: Optional[float] = None   # µm/pixel dans le plan objet, si imposée par l'utilisateur
    temperature_C: float = 22.0

    def echelle_um_px(self):
        """Échelle spatiale dans le plan objet (µm/pixel) : même modèle que PIXEL_EFF du module d'analyse."""
        if self.echelle_manuelle_um:
            return self.echelle_manuelle_um
        return tp.TAILLE_PIXEL * 1e6 * (LARGEUR_CAPTEUR_PX / self.largeur) / self.grossissement

    @property
    def pixel_eff_m(self):
        return self.echelle_um_px() * 1e-6


@dataclass
class ParamsAnalyse:
    actif: bool = True
    sigma: float = 1.5            # lissage gaussien de la détection (pixels)
    k: float = 5.0                # seuil de détection = fond + k × bruit
    distance_min: int = 5         # distance minimale entre deux pics (pixels)
    dist_max_px: float = 3.0      # déplacement maximal entre deux images (pixels)
    trou_max: int = 2             # images consécutives où une particule peut disparaître
    n_min: int = 20               # nombre minimal d'images par trajectoire utilisée
    lag_max: int = 3              # délais utilisés pour l'ajustement du MSD
    fenetre: int = 300            # nombre maximal d'images par trajectoire dans l'analyse (borne le temps de calcul)


@dataclass
class InfoImage:
    """Ce qui est incrusté sur chaque image."""
    t_s: float
    echelle_um_px: float
    grossissement: int
    exposition_ms: Optional[float]
    fps: float
    bits: int


@dataclass
class Instantane:
    num: int
    img: np.ndarray
    t_s: float
    idx: int
    md: dict


#############################################################################################################
#                                       AFFICHAGE : INCRUSTATIONS SUR L'IMAGE
#############################################################################################################

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


def vers_8bits(img, bits=8, etirer=False):
    """Image 8 bits pour l'affichage. `etirer` : étirement du contraste (affichage seulement)."""
    if etirer:
        lo, hi = np.percentile(img[::4, ::4], (1, 99.8))
        if hi <= lo:
            hi = lo + 1
        return np.clip((img.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    if img.dtype == np.uint8:
        return img
    return (img >> max(bits - 8, 0)).clip(0, 255).astype(np.uint8)


def dessiner_pistes(bgr, pistes, dets=None):
    ep = max(1, bgr.shape[1] // 640)
    if dets is not None:
        for d in dets:
            cv2.circle(bgr, (int(round(d[0])), int(round(d[1]))), 7 * ep, (0, 0, 255), ep, cv2.LINE_AA)
    for p in pistes or []:
        pts = np.round(p["pts"]).astype(np.int32).reshape(-1, 1, 2)
        c = couleur_piste(p["id"]) if p["longue"] else GRIS_COURT
        cv2.polylines(bgr, [pts], False, c, ep, cv2.LINE_AA)
        x, y = pts[-1, 0]
        cv2.circle(bgr, (int(x), int(y)), 4 * ep, c, ep, cv2.LINE_AA)
        if p["longue"]:
            cv2.putText(bgr, str(p["id"]), (int(x) + 6 * ep, int(y) - 6 * ep),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * ep, c, ep, cv2.LINE_AA)


def annoter(bgr, info):
    """Incruste, sur l'image BGR, le temps, l'échelle, les conditions d'acquisition et une barre d'échelle.
    C'est cette fonction qui garantit que chaque image de la vidéo respecte les exigences du mandat."""
    h, w = bgr.shape[:2]
    taille = max(12, round(w / 46))
    police = _police(taille)
    marge = max(4, taille // 3)
    expo = "n/d" if info.exposition_ms is None else f"{info.exposition_ms:.2f} ms"
    lignes = [f"t = {info.t_s:.3f} s",
              f"Échelle : {info.echelle_um_px:.4f} µm/pixel",
              f"Grossissement : {info.grossissement}X",
              f"Temps d'exposition : {expo}",
              f"Fréquence d'acquisition : {info.fps:.1f} images/s",
              f"Profondeur de bit : {info.bits} bits"]
    mesure = ImageDraw.Draw(Image.new("L", (1, 1)))
    larg = max(mesure.textlength(l, font=police) for l in lignes)
    interl = int(taille * 1.3)
    bx, by = int(larg + 2 * marge), len(lignes) * interl + 2 * marge
    bgr[:by, :bx] = (bgr[:by, :bx] * 0.35).astype(np.uint8)

    # barre d'échelle (≈ 20 % de la largeur), en bas à gauche
    L_um = _longueur_ronde(0.2 * w * info.echelle_um_px)
    L_px = max(2, int(round(L_um / info.echelle_um_px)))
    etiquette = f"{L_um:g} µm" if L_um < 1000 else f"{L_um / 1000:g} mm"
    ep = max(3, taille // 4)
    x0, y0 = marge, h - marge - ep
    zone_larg = int(max(L_px, mesure.textlength(etiquette, font=police)) + 2 * marge)
    zy = max(0, y0 - taille - 2 * marge)
    bgr[zy:h, :zone_larg] = (bgr[zy:h, :zone_larg] * 0.35).astype(np.uint8)

    pil = Image.fromarray(bgr)                      # on traite le BGR comme du RGB : on n'utilise que du blanc
    d = ImageDraw.Draw(pil)
    for i, l in enumerate(lignes):
        d.text((marge, marge + i * interl), l, font=police, fill=(255, 255, 255))
    d.rectangle([x0, y0, x0 + L_px, y0 + ep], fill=(255, 255, 255))
    d.text((x0, y0 - taille - marge // 2), etiquette, font=police, fill=(255, 255, 255))
    return np.array(pil)


def composer(img, info, pistes=None, dets=None, etirer=False):
    """Image brute -> image BGR 8 bits annotée (trajectoires, détections, conditions, échelle)."""
    bgr = cv2.cvtColor(vers_8bits(img, info.bits, etirer), cv2.COLOR_GRAY2BGR)
    if pistes or dets is not None:
        dessiner_pistes(bgr, pistes, dets)
    return annoter(bgr, info)


#############################################################################################################
#                                       SOURCES D'IMAGES
#############################################################################################################

class Source:
    nom = ""
    temps_reel = True                 # False : lecture sans perte (fichier) -> le traitement ralentit la lecture
    parametres_modifiables = ()       # champs de Conditions réglables pendant l'acquisition

    def ouvrir(self, cond):           # peut compléter/modifier `cond` (taille, fps, bits...)
        raise NotImplementedError

    def lire(self):                   # -> (image, t_s, dict) ou None à la fin
        raise NotImplementedError

    def appliquer(self, cond):
        pass

    def fermer(self):
        pass


class SourceFichier(Source):
    """Vidéo (mp4, avi...) ou TIFF (fichier multipage ou dossier d'un TIFF par image). Profondeur native conservée."""
    nom = "Fichier"
    temps_reel = False

    def __init__(self, chemin, vitesse=1.0):
        self.chemin = Path(chemin)
        self.vitesse = vitesse            # 1.0 = temps réel, None = le plus vite possible
        self.meta = {}

    @staticmethod
    def _gris(stack):
        stack = np.asarray(stack)
        if stack.ndim == 2:
            stack = stack[None]
        elif stack.ndim == 3 and stack.shape[-1] in (3, 4) and stack.shape[1] > 4:
            stack = stack[None]                                   # une seule image couleur
        if stack.ndim == 4:
            if stack.shape[-1] >= 3:
                stack = (stack[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)).astype(stack.dtype)
            else:
                stack = stack[..., 0]
        if stack.ndim != 3:
            raise ValueError(f"Forme d'image non gérée : {stack.shape}")
        if stack.dtype not in (np.uint8, np.uint16):              # float, int16... -> uint16 par facteur global
            s = stack.astype(np.float64)
            lo, hi = s.min(), s.max()
            stack = np.round(65535 * (s - lo) / max(hi - lo, 1e-12)).astype(np.uint16)
        return stack

    def ouvrir(self, cond):
        ch = self.chemin
        if not ch.exists():
            raise FileNotFoundError(f"Fichier ou dossier introuvable : {ch}")
        fps, bits = cond.fps, 8
        if ch.is_dir() or ch.suffix.lower() in tp.EXTENSIONS_TIFF:
            if ch.is_dir():
                fichiers = sorted(p for p in ch.iterdir() if p.suffix.lower() in tp.EXTENSIONS_TIFF)
                if not fichiers:
                    raise FileNotFoundError(f"Aucun TIFF dans : {ch}")
                stack = np.stack([tifffile.imread(str(f)) for f in fichiers])
            else:
                with tifffile.TiffFile(str(ch)) as tif:
                    stack = tif.asarray()
                    ij = tif.imagej_metadata or {}
                    if ij.get("finterval"):
                        fps = 1.0 / float(ij["finterval"])
                    elif ij.get("fps"):
                        fps = float(ij["fps"])
            frames = self._gris(stack)
            if frames.dtype == np.uint16:
                mx = int(frames.max())
                bits = next((b for b in (10, 12, 14, 16) if mx < 2 ** b), 16)
        else:
            frames, fps = tp.charger_video(ch)
            if not fps or fps <= 0:
                fps = cond.fps
        # conditions éventuellement enregistrées à côté du fichier par cette application
        for cand in (ch.with_suffix(".json"), ch.with_name(ch.stem.replace("_brut", "") + ".json")):
            if cand.is_file():
                try:
                    self.meta = json.loads(cand.read_text(encoding="utf-8")).get("conditions", {})
                except (OSError, ValueError):
                    pass
                break
        if self.meta.get("bits"):
            bits = int(self.meta["bits"])
        self.frames, self.fps = frames, float(fps)
        self.i, self.t_debut = 0, None
        cond.fps, cond.bits = self.fps, bits
        cond.hauteur, cond.largeur = frames.shape[1:3]
        # garde la valeur saisie par l'utilisateur si le fichier n'en contient pas
        cond.exposition_ms = self.meta.get("exposition_ms", cond.exposition_ms)
        if self.meta.get("grossissement"):
            cond.grossissement = int(self.meta["grossissement"])
        if self.meta.get("echelle_um_px"):
            cond.echelle_manuelle_um = float(self.meta["echelle_um_px"])

    def lire(self):
        if self.i >= len(self.frames):
            return None
        if self.vitesse:
            if self.t_debut is None:
                self.t_debut = time.perf_counter()
            attente = self.t_debut + self.i / (self.fps * self.vitesse) - time.perf_counter()
            if attente > 0:
                time.sleep(attente)
        img, t = self.frames[self.i], self.i / self.fps
        self.i += 1
        return img, t, {}


class SourceSimulation(Source):
    """Particules sphériques en mouvement brownien (Stokes-Einstein), PSF gaussienne, bruit réaliste.
    L'exposition et le gain modifient la luminosité (la saturation est donc observable)."""
    nom = "Simulation"
    parametres_modifiables = ("exposition_ms", "fps", "gain")

    def __init__(self, rayon_um=1.0, n=6):
        self.rayon_um, self.n = rayon_um, n

    def ouvrir(self, cond):
        self.cond = cond                      # référence : les changements de réglages agissent immédiatement
        self.rng = np.random.default_rng()
        w, h = cond.largeur, cond.hauteur
        self.pos = self.rng.uniform([0.15 * w, 0.15 * h], [0.85 * w, 0.85 * h], size=(self.n, 2))
        self.rayons = self.rayon_um * (1 + 0.05 * self.rng.standard_normal(self.n))
        self.i, self.t_debut = 0, None

    def lire(self):
        c = self.cond
        if self.t_debut is None:
            self.t_debut = time.perf_counter()
        attente = self.t_debut + self.i / c.fps - time.perf_counter()
        if attente > 0:
            time.sleep(attente)
        w, h = c.largeur, c.hauteur
        pe_um = c.echelle_um_px()
        D_um2 = tp.kB * (c.temperature_C + 273.15) / (6 * math.pi * viscosite_eau(c.temperature_C) * self.rayons * 1e-6) * 1e12
        pas = np.sqrt(2 * D_um2 / c.fps)[:, None] / pe_um                      # écart-type d'un pas (px)
        self.pos += self.rng.normal(size=self.pos.shape) * pas
        self.pos[:, 0] = np.abs(self.pos[:, 0]) % (2 * (w - 1))                # réflexion sur les bords
        self.pos[:, 1] = np.abs(self.pos[:, 1]) % (2 * (h - 1))
        self.pos[:, 0] = np.where(self.pos[:, 0] > w - 1, 2 * (w - 1) - self.pos[:, 0], self.pos[:, 0])
        self.pos[:, 1] = np.where(self.pos[:, 1] > h - 1, 2 * (h - 1) - self.pos[:, 1], self.pos[:, 1])

        sig = max(0.21 * 0.58 / OUVERTURE_NUM.get(c.grossissement, 0.4) / pe_um, 1.2)   # σ de la PSF (px)
        S = np.full((h, w), 12.0, np.float32)
        ech = (c.exposition_ms or 10.0) / 10.0 * c.gain
        r = int(math.ceil(5 * sig))
        for (x, y), a in zip(self.pos, self.rayons):
            x0, x1 = max(int(x) - r, 0), min(int(x) + r + 1, w)
            y0, y1 = max(int(y) - r, 0), min(int(y) + r + 1, h)
            X, Y = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            S[y0:y1, x0:x1] += 110 * ech * min(max(a, 0.3), 3.0) * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sig ** 2))
        S = S + self.rng.standard_normal(S.shape).astype(np.float32) * np.sqrt(S + 4.0)
        t, self.i = self.i / c.fps, self.i + 1
        if c.bits > 8:
            return np.clip(S * 2 ** (c.bits - 8), 0, 2 ** c.bits - 1).astype(np.uint16), t, {}
        return np.clip(S, 0, 255).astype(np.uint8), t, {}


#############################################################################################################
#                                       ENREGISTREMENT
#############################################################################################################

class Enregistreur:
    """Écrit (1) la vidéo MP4 annotée – obligatoire pour le mandat – et (2) optionnellement les images
    brutes en TIFF (profondeur native, fps dans les métadonnées ImageJ), plus un fichier JSON de conditions."""

    def __init__(self, dossier, cond, taille, duree_s, avec_pistes, brut, source_nom, pa, sigma_psf):
        dossier.mkdir(parents=True, exist_ok=True)
        self.base = dossier / ("sequence_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.fps = float(cond.fps)
        self.duree, self.avec_pistes = duree_s, avec_pistes
        self.t0, self.n, self.ferme = None, 0, False
        self.verrou = threading.Lock()
        self.chemin_video = self.base.with_suffix(".mp4")
        self.video = cv2.VideoWriter(str(self.chemin_video), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, taille)
        if not self.video.isOpened():
            raise RuntimeError("Impossible de créer la vidéo MP4 (codec mp4v indisponible).")
        self.tif, self.chemin_tif = None, None
        if brut:
            self.chemin_tif = self.base.with_name(self.base.name + "_brut.tif")
            self.tif = tifffile.TiffWriter(str(self.chemin_tif), imagej=True)
        self.meta = {"conditions": {**asdict(cond), "echelle_um_px": cond.echelle_um_px(), "source": source_nom},
                     "analyse": asdict(pa), "sigma_psf_px": sigma_psf, "debut": datetime.now().isoformat(timespec="seconds")}

    def ecrire(self, brut, annotee):
        with self.verrou:
            if self.ferme:
                return
            self.video.write(annotee)
            if self.tif is not None:
                self.tif.write(brut, contiguous=True, metadata={"finterval": 1.0 / self.fps, "fps": self.fps})
            self.n += 1

    def fermer(self):
        with self.verrou:
            if self.ferme:
                return
            self.ferme = True
            self.video.release()
            if self.tif is not None:
                self.tif.close()
        self.meta["images"] = self.n
        self.base.with_suffix(".json").write_text(json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def chemins(self):
        return [p for p in (self.chemin_video, self.chemin_tif, self.base.with_suffix(".json")) if p]


#############################################################################################################
#                                       SUIVI EN TEMPS RÉEL (liaison incrémentale)
#############################################################################################################

class Suivi:
    """Même algorithme de liaison que `lier_trajectoires` (affectation optimale de Hongrie, déplacement maximal,
    trous autorisés), mais image par image pour fonctionner en direct. Les images sont repérées par leur indice
    réel : une image perdue devient un trou dans la trajectoire, sans décaler le temps."""

    def __init__(self):
        self.verrou = threading.Lock()
        self.pistes, self.actives, self.prochain = {}, [], 0

    def ajouter(self, f, det, dist_max_px, trou_max):
        with self.verrou:
            pistes = self.pistes
            self.actives = actives = [k for k in self.actives if f - pistes[k]["frames"][-1] <= trou_max + 1]
            liees = set()
            if actives and len(det):
                derniere = np.array([pistes[k]["data"][-1][:2] for k in actives])
                ecart = np.array([f - pistes[k]["frames"][-1] for k in actives], dtype=float)
                d = np.linalg.norm(derniere[:, None, :] - det[None, :, :2], axis=2)
                cout = np.where(d <= dist_max_px * np.sqrt(ecart)[:, None], d, 1e6)
                for i, j in zip(*linear_sum_assignment(cout)):
                    if cout[i, j] < 1e6:
                        p = pistes[actives[i]]
                        p["frames"].append(f)
                        p["data"].append(det[j])
                        liees.add(j)
            for j in range(len(det)):
                if j not in liees:
                    pistes[self.prochain] = {"frames": [f], "data": [det[j]]}
                    actives.append(self.prochain)
                    self.prochain += 1
            if f % 200 == 0:                                   # élagage : pistes mortes et très courtes
                for k in [k for k, p in pistes.items() if k not in actives and len(p["frames"]) < 5]:
                    del pistes[k]
            return len(actives)

    def instantane(self, f, trou_max, n_min, n_aff=200):
        """Pistes à afficher : [{id, pts (k,2), longue}] pour les trajectoires encore actives."""
        sortie = []
        with self.verrou:
            for k in self.actives:
                p = self.pistes.get(k)
                if p and len(p["frames"]) >= 2 and f - p["frames"][-1] <= trou_max + 1:
                    pts = np.array([d[:2] for d in p["data"][-n_aff:]])
                    sortie.append({"id": k, "pts": pts, "longue": len(p["frames"]) >= n_min})
        return sortie

    def copie(self, n_min, fenetre):
        """Trajectoires d'au moins n_min images, tronquées aux `fenetre` dernières : [(id, frames, data)]."""
        with self.verrou:
            return [(k, list(p["frames"][-fenetre:]), list(p["data"][-fenetre:]))
                    for k, p in self.pistes.items() if len(p["frames"]) >= n_min]

    def nombres(self, n_min):
        with self.verrou:
            return len(self.actives), sum(len(p["frames"]) >= n_min for p in self.pistes.values())


#############################################################################################################
#                                       MOTEUR : ACQUISITION + ANALYSE + ENREGISTREMENT
#############################################################################################################

class Moteur:
    """Deux fils d'exécution : acquisition (lecture, enregistrement) et analyse (détection, ajustement, suivi,
    rayon). L'interface ne fait que lire l'état ; elle n'est jamais bloquée par le calcul."""

    def __init__(self, pa):
        self.pa, self.cond = pa, Conditions()
        self.source, self.etat, self.message = None, "arret", ""
        self._stop = threading.Event()
        self._verrou = threading.Lock()
        self._dernier, self._num = None, 0
        self.suivi, self._gen = Suivi(), 0
        self._q = queue.Queue(maxsize=8)
        self.enreg, self.dernier_enreg = None, []
        self._res = ([], None, 0)
        self._dets = np.empty((0, 5))
        self.sigma_psf, self._sigmas = None, deque(maxlen=300)
        self._temps = deque(maxlen=30)
        self.stats = dict(analysees=0, perdues=0, n_det=0, fps_mesure=0.0)
        self._t_res = 0.0
        self._fils = []

    # ---------- commandes ----------
    def demarrer(self, source, cond):
        if self.etat == "marche":
            raise RuntimeError("Une acquisition est déjà en cours.")
        self.cond = replace(cond)
        source.ouvrir(self.cond)                    # lève une exception si la source est inutilisable
        self.source = source
        self.reinitialiser()
        with self._verrou:
            self._dernier = None
        self.message = ""
        self._temps.clear()
        self._stop.clear()
        self.etat = "marche"
        self._fils = [threading.Thread(target=self._boucle_acquisition, daemon=True),
                      threading.Thread(target=self._boucle_analyse, daemon=True)]
        for t in self._fils:
            t.start()

    def arreter(self):
        self._stop.set()
        for t in self._fils:
            t.join(timeout=3)
        if self.etat == "marche":
            self.etat = "arret"

    def appliquer(self, c):
        """Applique les réglages modifiés pendant l'acquisition (en place : les sources gardent la référence)."""
        cur = self.cond
        reinit = (c.grossissement != cur.grossissement or c.echelle_manuelle_um != cur.echelle_manuelle_um
                  or ("fps" in self.source.parametres_modifiables and abs(c.fps - cur.fps) > 1e-6))
        for nom in ("grossissement", "echelle_manuelle_um", "temperature_C") + tuple(self.source.parametres_modifiables):
            setattr(cur, nom, getattr(c, nom))
        self.source.appliquer(cur)
        if reinit:
            self.reinitialiser()

    def reinitialiser(self):
        with self._verrou:
            self._gen += 1
            self.suivi = Suivi()
            self._res = ([], None, self._res[2] + 1)
            self._dets = np.empty((0, 5))
            self.sigma_psf = None
            self._sigmas.clear()
            self.stats.update(analysees=0, perdues=0, n_det=0)
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def demarrer_enregistrement(self, dossier, duree_s=0.0, avec_pistes=True, brut=True):
        if self.etat != "marche":
            raise RuntimeError("Démarrez d'abord l'acquisition.")
        if self.enreg is not None:
            return
        snap = self.dernier()
        if snap is None:
            raise RuntimeError("Aucune image reçue pour l'instant.")
        h, w = snap.img.shape[:2]
        self.enreg = Enregistreur(Path(dossier), self.cond, (w, h), duree_s, avec_pistes, brut,
                                  self.source.nom, self.pa, self.sigma_psf)

    def arreter_enregistrement(self):
        e, self.enreg = self.enreg, None
        if e is not None:
            e.fermer()
            self.dernier_enreg = e.chemins()
        return self.dernier_enreg

    def capturer(self, dossier):
        snap = self.dernier()
        if snap is None:
            raise RuntimeError("Aucune image à capturer.")
        dossier = Path(dossier)
        dossier.mkdir(parents=True, exist_ok=True)
        base = dossier / ("capture_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        tifffile.imwrite(str(base.with_suffix(".tif")), snap.img)
        cv2.imwrite(str(base.with_suffix(".png")), composer(snap.img, self.info_pour(snap.t_s, snap.md)))
        return base

    # ---------- lecture de l'état (appelée par l'interface) ----------
    def dernier(self):
        with self._verrou:
            return self._dernier

    def detections(self):
        return self._dets

    def pistes_affichage(self, idx):
        return self.suivi.instantane(idx, self.pa.trou_max, self.pa.n_min)

    def resultats(self):
        return self._res

    def info_pour(self, t_s, md=None):
        c = self.cond
        expo = c.exposition_ms
        if md and md.get("exposition_ms") is not None:
            expo = md["exposition_ms"]
        return InfoImage(t_s, c.echelle_um_px(), c.grossissement, expo, c.fps, c.bits)

    # ---------- fil d'acquisition ----------
    def _boucle_acquisition(self):
        src, dernier_idx = self.source, -1
        try:
            while not self._stop.is_set():
                r = src.lire()
                if r is None:
                    self.message = "Fin du fichier : analyse terminée."
                    self.etat = "termine"
                    break
                img, t, md = r
                cond = self.cond
                f_eff = md.get("fps")
                if f_eff and abs(f_eff - cond.fps) > 0.01 * cond.fps:
                    cond.fps = f_eff
                idx = max(int(round(t * cond.fps)), dernier_idx + 1)
                dernier_idx = idx
                self._temps.append(time.perf_counter())
                if len(self._temps) > 2:
                    self.stats["fps_mesure"] = (len(self._temps) - 1) / (self._temps[-1] - self._temps[0])
                self._num += 1
                with self._verrou:
                    self._dernier = Instantane(self._num, img, t, idx, md)

                e = self.enreg
                if e is not None:
                    if e.t0 is None:
                        e.t0 = t
                    t_rel = t - e.t0
                    pistes = self.pistes_affichage(idx) if e.avec_pistes else None
                    e.ecrire(img, composer(img, self.info_pour(t_rel, md), pistes))
                    if e.duree and t_rel >= e.duree:
                        self.arreter_enregistrement()
                        self.message = "Enregistrement terminé : " + e.chemin_video.name

                if self.pa.actif:
                    item = (self._gen, idx, img)
                    if src.temps_reel:
                        try:
                            self._q.put_nowait(item)
                        except queue.Full:
                            self.stats["perdues"] += 1            # trop lent : l'image devient un « trou »
                    else:
                        while not self._stop.is_set():
                            try:
                                self._q.put(item, timeout=0.1)
                                break
                            except queue.Full:
                                continue
        except Exception as ex:                                    # erreur de source : on la signale, on ne plante pas
            self.message = f"Erreur d'acquisition : {ex}"
            self.etat = "arret"
        finally:
            src.fermer()
            self.arreter_enregistrement()
            if self.etat == "marche":
                self.etat = "arret"
            self._q.put(None)

    # ---------- fil d'analyse ----------
    def _boucle_analyse(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            gen, idx, img = item
            if gen != self._gen:
                continue
            try:
                self._traiter(gen, idx, img)
                if time.perf_counter() - self._t_res > 1.0:
                    self._calculer_resultats()
            except Exception as ex:                                # une image problématique ne doit pas tuer le suivi
                self.message = f"Avertissement analyse : {ex}"
        self._calculer_resultats()

    def _traiter(self, gen, idx, img):
        pa = self.pa
        suivi = self.suivi
        imgf = img.astype(float)
        fond = float(np.median(imgf))
        lignes = []
        for y, x, _ in tp.detecter_particules(img, sigma=pa.sigma, k=pa.k, distance_min=pa.distance_min):
            s0 = self.sigma_psf or tp.sigma_demi_hauteur(imgf, x, y, fond)
            r = tp.fit_gauss_local(imgf, x, y, fond, s0)
            if r is not None:
                lignes.append(r)
        det = np.array(lignes).reshape(-1, 5)
        if len(det):
            self._sigmas.extend(det[:, 4])
            self.sigma_psf = float(np.median(self._sigmas))
        suivi.ajouter(idx, det, pa.dist_max_px, pa.trou_max)
        if gen == self._gen:
            self._dets = det
            self.stats["analysees"] += 1
            self.stats["n_det"] = len(det)

    def _calculer_resultats(self):
        self._t_res = time.perf_counter()
        gen, suivi, cond, pa = self._gen, self.suivi, self.cond, self.pa
        tp.T = cond.temperature_C + 273.15                         # le module d'analyse lit ces constantes
        tp.ETA = viscosite_eau(cond.temperature_C)
        dt = 1.0 / cond.fps
        liste = []
        for pid, frames, data in suivi.copie(pa.n_min, pa.fenetre):
            tab = tp.pistes_en_tableaux([{"frames": frames, "data": data}], n_min=1)[0]
            try:
                with np.errstate(all="ignore"):
                    r = tp.rayon_piste(tab["pos_px"], tab["inc_px"], dt, pixel_eff=cond.pixel_eff_m, lag_max=pa.lag_max)
            except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
                continue
            if r is None or not np.isfinite(r["inc"]) or r["inc"] <= 0:
                continue
            r.update(piste=pid, f0=tab["f0"], pos_px=tab["pos_px"], n_pts=int(np.isfinite(tab["pos_px"][:, 0]).sum()))
            liste.append(r)
        final = None
        if liste:
            with np.errstate(all="ignore"):
                final = tp.rayon_final(liste)
            final["sigma_psf_px"] = self.sigma_psf
            final["n_suivies"] = suivi.nombres(pa.n_min)[1]
        with self._verrou:
            if gen == self._gen:
                self._res = (liste, final, self._res[2] + 1)


#############################################################################################################
#                                       INTERFACE GRAPHIQUE
#############################################################################################################

class Curseur(ttk.Frame):
    """Curseur et champ numérique synchronisés (échelle linéaire ou logarithmique)."""

    def __init__(self, parent, texte, mini, maxi, valeur, unite="", log=False, dec=1, commande=None):
        super().__init__(parent)
        self.mini, self.maxi, self.log, self.dec, self.commande = mini, maxi, log, dec, commande
        self._maj, self._val, self._valide = False, valeur, None
        self.var = tk.StringVar()
        ttk.Label(self, text=texte).grid(row=0, column=0, sticky="w")
        self.entree = ttk.Entry(self, textvariable=self.var, width=8, justify="right")
        self.entree.grid(row=0, column=1, sticky="e", padx=4)
        ttk.Label(self, text=unite, width=8).grid(row=0, column=2, sticky="w")
        self.echelle = ttk.Scale(self, from_=0, to=1000, command=self._glisse)
        self.echelle.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.columnconfigure(0, weight=1)
        self.echelle.bind("<ButtonRelease-1>", lambda e: self._valider())
        self.entree.bind("<Return>", self._depuis_entree)
        self.entree.bind("<FocusOut>", self._depuis_entree)
        self.fixer(valeur)
        self._valide = self._val

    def _vers_pos(self, v):
        if self.log:
            return 1000 * math.log(v / self.mini) / math.log(self.maxi / self.mini)
        return 1000 * (v - self.mini) / (self.maxi - self.mini)

    def _depuis_pos(self, p):
        if self.log:
            return self.mini * (self.maxi / self.mini) ** (p / 1000)
        return self.mini + (self.maxi - self.mini) * p / 1000

    def _glisse(self, p):
        if self._maj:
            return
        self._val = self._depuis_pos(float(p))
        self.var.set(f"{self._val:.{self.dec}f}")

    def _depuis_entree(self, _=None):
        try:
            v = float(self.var.get().replace(",", "."))
        except ValueError:
            v = self._val
        self.fixer(v)
        self._valider()

    def _valider(self):
        if self._valide is None or abs(self._val - self._valide) > 1e-12:
            self._valide = self._val
            if self.commande:
                self.commande()

    def fixer(self, v):
        v = min(max(float(v), self.mini), self.maxi)
        self._val = v
        self._maj = True
        self.echelle.set(self._vers_pos(v))
        self._maj = False
        self.var.set(f"{v:.{self.dec}f}")
        self._valide = v if self._valide is not None else None

    def valeur(self):
        return self._val

    def actif(self, oui):
        self.entree.configure(state="normal" if oui else "disabled")
        self.echelle.state(["!disabled"] if oui else ["disabled"])


class Application(ttk.Frame):
    def __init__(self, racine):
        super().__init__(racine)
        self.racine = racine
        racine.title(NOM_APP)
        racine.geometry("1380x860")
        racine.minsize(1100, 700)
        self.pa = ParamsAnalyse()
        self.moteur = Moteur(self.pa)
        self._photo = self._item = None
        self._dernier_num, self._version_res, self._fig_a_jour = -1, -1, False
        self._t_stats, self._map, self._img_courante = 0.0, (0, 0, 1.0), None
        self._style()
        self._construire()
        self._maj_etats()
        self.pack(fill="both", expand=True)
        racine.protocol("WM_DELETE_WINDOW", self._quitter)
        self._rafraichir()

    # ------------------------------------------------------------------ style
    def _style(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        tkfont.nametofont("TkDefaultFont").configure(size=10)
        tkfont.nametofont("TkTextFont").configure(size=10)
        s.configure("TLabelframe.Label", font=("TkDefaultFont", 10, "bold"))
        for nom, bg, bga in (("Go", "#2e7d32", "#388e3c"), ("Stop", "#c62828", "#d32f2f"), ("Rec", "#ad1457", "#c2185b")):
            s.configure(f"{nom}.TButton", background=bg, foreground="white", font=("TkDefaultFont", 11, "bold"), padding=7)
            s.map(f"{nom}.TButton", background=[("active", bga), ("disabled", "#9e9e9e")], foreground=[("disabled", "#eeeeee")])
        s.configure("Gros.TLabel", font=("TkDefaultFont", 20, "bold"))
        s.configure("Carte.TFrame", background="#eef3f8")
        s.configure("Carte.TLabel", background="#eef3f8")
        s.configure("GrosCarte.TLabel", background="#eef3f8", font=("TkDefaultFont", 20, "bold"), foreground="#0b3d6b")

    # ------------------------------------------------------------------ construction
    def _construire(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # ----- colonne de gauche : vue en direct / analyse
        gauche = ttk.Frame(self)
        gauche.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        gauche.rowconfigure(0, weight=1)
        gauche.columnconfigure(0, weight=1)
        self.onglets = ttk.Notebook(gauche)
        self.onglets.grid(row=0, column=0, sticky="nsew")
        vue = ttk.Frame(self.onglets)
        ana = ttk.Frame(self.onglets)
        self.onglets.add(vue, text="  Vue en direct  ")
        self.onglets.add(ana, text="  Analyse  ")
        self.onglets.bind("<<NotebookTabChanged>>", lambda e: self._maj_figure())

        self.canvas = tk.Canvas(vue, bg="#101010", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._placeholder())
        self.canvas.bind("<Motion>", self._survol)

        self.fig = Figure(figsize=(7, 4.6), dpi=100, constrained_layout=True)
        self.ax_msd = self.fig.add_subplot(1, 2, 1)
        self.ax_a = self.fig.add_subplot(1, 2, 2)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=ana)
        self.canvas_fig.get_tk_widget().pack(fill="both", expand=True)
        cols = ("piste", "n_pts", "D", "sigma_exp", "a", "s_psf", "s_stat", "s_tot")
        tetes = ("Piste", "Images", "D (µm²/s)", "σ_exp (nm)", "a (µm)", "σ_PSF (µm)", "σ_stat (µm)", "σ_tot (µm)")
        self.tableau = ttk.Treeview(ana, columns=cols, show="headings", height=6)
        for c, t in zip(cols, tetes):
            self.tableau.heading(c, text=t)
            self.tableau.column(c, width=90, anchor="center")
        self.tableau.pack(fill="x", padx=4, pady=4)

        self.var_etat = tk.StringVar(value="Prêt.")
        self.var_stats = tk.StringVar(value="")
        self.var_survol = tk.StringVar(value="")
        barre = ttk.Frame(gauche)
        barre.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(barre, textvariable=self.var_stats).pack(side="left")
        self.lbl_sat = ttk.Label(barre, text="")
        self.lbl_sat.pack(side="left", padx=12)
        ttk.Label(barre, textvariable=self.var_survol, foreground="#555").pack(side="right")
        ttk.Label(gauche, textvariable=self.var_etat, foreground="#0b3d6b").grid(row=2, column=0, sticky="w")

        # ----- colonne de droite
        droite = ttk.Frame(self, width=390)
        droite.grid(row=0, column=1, sticky="ns", padx=(4, 8), pady=8)
        droite.grid_propagate(False)
        self._carte_resultat(droite).pack(fill="x")
        self.ctl = ttk.Notebook(droite)
        self.ctl.pack(fill="both", expand=True, pady=(8, 0))
        t1, t2, t3 = ttk.Frame(self.ctl, padding=6), ttk.Frame(self.ctl, padding=6), ttk.Frame(self.ctl, padding=6)
        self.ctl.add(t1, text="Acquisition")
        self.ctl.add(t2, text="Enregistrement")
        self.ctl.add(t3, text="Analyse")
        self._onglet_acquisition(t1)
        self._onglet_enregistrement(t2)
        self._onglet_analyse(t3)
        self._sur_source()          # seulement maintenant : tous les widgets (température, boutons...) existent
        self._sur_changement()

    def _carte_resultat(self, parent):
        cadre = ttk.Frame(parent, style="Carte.TFrame", padding=10)
        ttk.Label(cadre, text="RAYON MOYEN DES PARTICULES", style="Carte.TLabel",
                  font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        self.var_rayon = tk.StringVar(value="—")
        self.var_rayon_detail = tk.StringVar(value="En attente de trajectoires…")
        ttk.Label(cadre, textvariable=self.var_rayon, style="GrosCarte.TLabel").pack(anchor="w")
        ttk.Label(cadre, textvariable=self.var_rayon_detail, style="Carte.TLabel", wraplength=350,
                  justify="left").pack(anchor="w")
        return cadre

    def _onglet_acquisition(self, p):
        # --- source
        src = ttk.LabelFrame(p, text="1. Source des images", padding=6)
        src.pack(fill="x")
        self.var_source = tk.StringVar(value="fichier")
        self.rb = {}
        for val, txt in (("fichier", "Fichier TIFF (ou vidéo)"), ("simulation", "Simulation (test sans fichier)")):
            self.rb[val] = ttk.Radiobutton(src, text=txt, value=val, variable=self.var_source, command=self._sur_source)
            self.rb[val].pack(anchor="w")
        self.f_fichier = ttk.Frame(src)
        self.var_fichier = tk.StringVar()
        ttk.Entry(self.f_fichier, textvariable=self.var_fichier).grid(row=0, column=0, sticky="ew")
        ttk.Button(self.f_fichier, text="Parcourir…", command=self._parcourir).grid(row=0, column=1, padx=(4, 0))
        ttk.Button(self.f_fichier, text="Dossier…", command=self._parcourir_dossier).grid(row=0, column=2, padx=(4, 0))
        ttk.Label(self.f_fichier, text="Lecture :").grid(row=1, column=0, sticky="e", pady=(4, 0))
        self.cb_vitesse = ttk.Combobox(self.f_fichier, values=["temps réel", "×4", "maximale"], state="readonly", width=12)
        self.cb_vitesse.set("temps réel")
        self.cb_vitesse.grid(row=1, column=1, columnspan=2, pady=(4, 0))
        self.f_fichier.columnconfigure(0, weight=1)
        self.f_sim = ttk.Frame(src)
        self.var_rayon_sim = tk.DoubleVar(value=1.0)
        self.var_n_sim = tk.IntVar(value=6)
        ttk.Label(self.f_sim, text="Rayon simulé (µm)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(self.f_sim, from_=0.05, to=10, increment=0.05, textvariable=self.var_rayon_sim, width=7).grid(row=0, column=1, padx=6)
        ttk.Label(self.f_sim, text="Particules").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(self.f_sim, from_=1, to=30, textvariable=self.var_n_sim, width=4).grid(row=0, column=3, padx=6)

        # --- conditions
        cond = ttk.LabelFrame(p, text="2. Conditions d'acquisition", padding=6)
        cond.pack(fill="x", pady=6)
        g = ttk.Frame(cond)
        g.pack(fill="x")
        ttk.Label(g, text="Grossissement").grid(row=0, column=0, sticky="w")
        self.cb_gross = ttk.Combobox(g, values=[f"{x}X" for x in GROSSISSEMENTS], state="readonly", width=8)
        self.cb_gross.set("20X")
        self.cb_gross.grid(row=0, column=1, sticky="e", pady=2)
        ttk.Label(g, text="Résolution (px)").grid(row=1, column=0, sticky="w")
        self.cb_res = ttk.Combobox(g, values=list(RESOLUTIONS), state="readonly", width=12)
        self.cb_res.set("640 × 480")
        self.cb_res.grid(row=1, column=1, sticky="e", pady=2)
        ttk.Label(g, text="Profondeur de bit").grid(row=2, column=0, sticky="w")
        self.cb_bits = ttk.Combobox(g, values=["8 bits", "12 bits (RAW)"], state="readonly", width=12)
        self.cb_bits.set("8 bits")
        self.cb_bits.grid(row=2, column=1, sticky="e", pady=2)
        g.columnconfigure(0, weight=1)
        self.cb_gross.bind("<<ComboboxSelected>>", self._sur_changement)
        self.cb_res.bind("<<ComboboxSelected>>", self._sur_structure)
        self.cb_bits.bind("<<ComboboxSelected>>", self._sur_structure)
        self.c_expo = Curseur(cond, "Temps d'exposition", 0.05, 1000, 10.0, "ms", log=True, dec=2, commande=self._sur_changement)
        self.c_fps = Curseur(cond, "Fréquence d'acquisition", 1, 120, 30.0, "images/s", log=True, dec=1, commande=self._sur_changement)
        self.c_gain = Curseur(cond, "Gain analogique", 1, 16, 1.0, "×", dec=2, commande=self._sur_changement)
        for c in (self.c_expo, self.c_fps, self.c_gain):
            c.pack(fill="x", pady=2)
        self.var_echelle = tk.StringVar()
        ttk.Label(cond, textvariable=self.var_echelle, font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(4, 0))
        m = ttk.Frame(cond)
        m.pack(fill="x")
        self.var_manuelle = tk.BooleanVar(value=False)
        self.var_echelle_man = tk.StringVar(value="0.500")
        ttk.Checkbutton(m, text="Échelle imposée (calibrée) :", variable=self.var_manuelle, command=self._sur_changement).pack(side="left")
        e = ttk.Entry(m, textvariable=self.var_echelle_man, width=7, justify="right")
        e.pack(side="left", padx=4)
        e.bind("<Return>", self._sur_changement)
        e.bind("<FocusOut>", self._sur_changement)
        ttk.Label(m, text="µm/px").pack(side="left")

        # --- affichage
        aff = ttk.LabelFrame(p, text="3. Affichage", padding=6)
        aff.pack(fill="x")
        self.var_etirer = tk.BooleanVar(value=False)
        self.var_pistes = tk.BooleanVar(value=True)
        self.var_dets = tk.BooleanVar(value=True)
        ttk.Checkbutton(aff, text="Trajectoires", variable=self.var_pistes).pack(anchor="w")
        ttk.Checkbutton(aff, text="Détections (cercles rouges)", variable=self.var_dets).pack(anchor="w")
        ttk.Checkbutton(aff, text="Étirer le contraste (affichage seulement)", variable=self.var_etirer).pack(anchor="w")

        # --- boutons
        self.btn_acq = ttk.Button(p, text="▶  Démarrer l'acquisition", style="Go.TButton", command=self._basculer_acq)
        self.btn_acq.pack(fill="x", pady=(10, 4))
        self.btn_capture = ttk.Button(p, text="Capturer une image", command=self._capturer)
        self.btn_capture.pack(fill="x")

    def _onglet_enregistrement(self, p):
        self.var_dossier = tk.StringVar(value=str(Path(__file__).resolve().parent / "enregistrements"))
        d = ttk.LabelFrame(p, text="Destination", padding=6)
        d.pack(fill="x")
        ttk.Entry(d, textvariable=self.var_dossier).grid(row=0, column=0, sticky="ew")
        ttk.Button(d, text="Parcourir…", command=self._choisir_dossier).grid(row=0, column=1, padx=(4, 0))
        d.columnconfigure(0, weight=1)
        o = ttk.LabelFrame(p, text="Options de la séquence", padding=6)
        o.pack(fill="x", pady=8)
        ttk.Label(o, text="Durée (s) — 0 = arrêt manuel").grid(row=0, column=0, sticky="w")
        self.var_duree = tk.DoubleVar(value=30.0)
        ttk.Spinbox(o, from_=0, to=3600, increment=5, textvariable=self.var_duree, width=7).grid(row=0, column=1, padx=6)
        self.var_enr_pistes = tk.BooleanVar(value=True)
        self.var_enr_brut = tk.BooleanVar(value=True)
        ttk.Checkbutton(o, text="Inclure les trajectoires dans la vidéo", variable=self.var_enr_pistes).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(o, text="Enregistrer aussi les images brutes (TIFF)", variable=self.var_enr_brut).grid(row=2, column=0, columnspan=2, sticky="w")
        o.columnconfigure(0, weight=1)
        ttk.Label(p, wraplength=350, justify="left", foreground="#444",
                  text="Chaque image de la vidéo MP4 affiche le temps écoulé (s), l'échelle (µm/pixel), le grossissement, "
                       "le temps d'exposition, la fréquence d'acquisition et la profondeur de bit. Le TIFF brut (sans "
                       "incrustation) sert à ré-analyser la séquence avec le module d'analyse.").pack(fill="x")
        self.btn_enr = ttk.Button(p, text="●  Démarrer l'enregistrement", style="Rec.TButton", command=self._basculer_enreg)
        self.btn_enr.pack(fill="x", pady=(14, 4))
        self.var_enr = tk.StringVar(value="")
        ttk.Label(p, textvariable=self.var_enr, wraplength=350, justify="left").pack(fill="x")
        self.prog = ttk.Progressbar(p, maximum=100)
        self.prog.pack(fill="x", pady=4)

    def _onglet_analyse(self, p):
        a = ttk.LabelFrame(p, text="Suivi et mesure du rayon", padding=6)
        a.pack(fill="x")
        self.var_actif = tk.BooleanVar(value=True)
        ttk.Checkbutton(a, text="Activer le suivi en temps réel", variable=self.var_actif, command=self._sur_pa).pack(anchor="w")
        t = ttk.Frame(a)
        t.pack(fill="x", pady=4)
        ttk.Label(t, text="Température de l'eau (°C)").grid(row=0, column=0, sticky="w")
        self.var_temp = tk.StringVar(value="22.0")
        sp = ttk.Spinbox(t, from_=0, to=60, increment=0.5, textvariable=self.var_temp, width=6, command=self._sur_changement)
        sp.grid(row=0, column=1, padx=6)
        sp.bind("<Return>", self._sur_changement)
        sp.bind("<FocusOut>", self._sur_changement)
        self.var_visc = tk.StringVar()
        ttk.Label(t, textvariable=self.var_visc, foreground="#444").grid(row=1, column=0, columnspan=2, sticky="w")
        t.columnconfigure(0, weight=1)

        av = ttk.LabelFrame(p, text="Paramètres de l'algorithme", padding=6)
        av.pack(fill="x", pady=8)
        self.vars_pa = {}
        champs = [("Seuil de détection k (× bruit)", "k", 1, 30, 0.5), ("Lissage gaussien σ (px)", "sigma", 0.5, 6, 0.1),
                  ("Distance min. entre pics (px)", "distance_min", 2, 40, 1),
                  ("Déplacement max. entre images (px)", "dist_max_px", 1, 40, 0.5),
                  ("Trou max. (images)", "trou_max", 0, 10, 1), ("Images min. par trajectoire", "n_min", 8, 3000, 1),
                  ("Délais MSD utilisés (lag max)", "lag_max", 2, 10, 1), ("Fenêtre d'analyse (images)", "fenetre", 30, 3000, 10)]
        for i, (txt, nom, lo, hi, pas) in enumerate(champs):
            ttk.Label(av, text=txt).grid(row=i, column=0, sticky="w", pady=1)
            v = tk.StringVar(value=str(getattr(self.pa, nom)))
            self.vars_pa[nom] = v
            sp = ttk.Spinbox(av, from_=lo, to=hi, increment=pas, textvariable=v, width=7, command=self._sur_pa)
            sp.grid(row=i, column=1, padx=6, pady=1)
            sp.bind("<Return>", self._sur_pa)
            sp.bind("<FocusOut>", self._sur_pa)
        av.columnconfigure(0, weight=1)
        ttk.Button(p, text="Réinitialiser le suivi", command=self._reinit).pack(fill="x", pady=(4, 2))
        ttk.Button(p, text="Exporter l'analyse (CSV + figure)", command=self._exporter).pack(fill="x")
        self._sur_pa()

    # ------------------------------------------------------------------ lecture des widgets
    @staticmethod
    def _flottant(texte, defaut):
        try:
            return float(str(texte).replace(",", "."))
        except ValueError:
            return defaut

    def _lire_cond(self):
        larg, haut = RESOLUTIONS[self.cb_res.get()]
        man = None
        if self.var_manuelle.get():
            v = self._flottant(self.var_echelle_man.get(), 0)
            man = v if v > 0 else None
        return Conditions(grossissement=int(self.cb_gross.get().rstrip("X")), exposition_ms=self.c_expo.valeur(),
                          fps=self.c_fps.valeur(), gain=self.c_gain.valeur(),
                          bits=12 if self.cb_bits.get().startswith("12") else 8, largeur=larg, hauteur=haut,
                          echelle_manuelle_um=man, temperature_C=self._flottant(self.var_temp.get(), 22.0))

    def _sur_pa(self, *_):
        pa = self.pa
        pa.actif = self.var_actif.get()
        for nom, v in self.vars_pa.items():
            x = self._flottant(v.get(), None)
            if x is None:
                continue
            setattr(pa, nom, type(getattr(pa, nom))(round(x) if isinstance(getattr(pa, nom), int) else x))

    def _sur_changement(self, *_):
        c = self._lire_cond()
        expo_max = 950.0 / c.fps                       # l'exposition ne peut pas dépasser la durée d'une image
        if c.exposition_ms > expo_max:
            c.exposition_ms = expo_max
            self.c_expo.fixer(expo_max)
        self.var_visc.set(f"Viscosité de l'eau : {viscosite_eau(c.temperature_C) * 1e3:.3f} mPa·s")
        if self.moteur.etat == "marche":
            self.moteur.appliquer(c)
            c = self.moteur.cond
        self.var_echelle.set(f"Échelle spatiale : {c.echelle_um_px():.4f} µm/pixel")

    def _sur_structure(self, *_):
        """Résolution ou profondeur de bit (simulation) : la source doit être reconfigurée."""
        if self.cb_bits.get().startswith("12"):
            self.cb_res.set("1014 × 760")
        self._sur_changement()
        if self.moteur.etat == "marche" and self.var_source.get() != "fichier":
            self._arreter()
            self.racine.after(300, self._demarrer)

    def _sur_source(self):
        s = self.var_source.get()
        self.f_fichier.pack_forget()
        self.f_sim.pack_forget()
        if s == "fichier":
            self.f_fichier.pack(fill="x", pady=(4, 0))
        elif s == "simulation":
            self.f_sim.pack(fill="x", pady=(4, 0))
        self._maj_etats()

    def _maj_etats(self):
        marche = self.moteur.etat == "marche"
        fichier = self.var_source.get() == "fichier"
        enr = self.moteur.enreg is not None
        for v in self.rb.values():
            v.state(["disabled"] if marche else ["!disabled"])
        # fichier : exposition et fréquence réglables seulement avant le démarrage (elles sont incrustées sur la vidéo)
        for c in (self.c_expo, self.c_fps):
            c.actif(not enr and (not marche if fichier else True))
        self.c_gain.actif(not fichier and not enr)     # le gain n'a aucun sens pour un TIFF déjà acquis
        self.cb_res.state(["disabled"] if fichier else ["!disabled"])
        self.cb_bits.state(["disabled"] if fichier else ["!disabled"])
        self.btn_acq.configure(text="■  Arrêter l'acquisition" if marche else "▶  Démarrer l'acquisition",
                               style="Stop.TButton" if marche else "Go.TButton")
        self.btn_capture.state(["!disabled"] if marche else ["disabled"])
        if not hasattr(self, "btn_enr"):               # appelé pendant la construction, avant l'onglet Enregistrement
            return
        self.btn_enr.state(["!disabled"] if marche else ["disabled"])
        self.btn_enr.configure(text="■  Arrêter l'enregistrement" if enr else "●  Démarrer l'enregistrement",
                               style="Stop.TButton" if enr else "Rec.TButton")

    # ------------------------------------------------------------------ actions
    def _parcourir(self):
        ch = filedialog.askopenfilename(title="Choisir un fichier TIFF",
                                        filetypes=[("TIFF", "*.tif *.tiff"), ("Vidéos", "*.mp4 *.avi *.mov *.mkv"),
                                                   ("Tous les fichiers", "*.*")])
        if ch:
            self.var_fichier.set(ch)

    def _parcourir_dossier(self):
        d = filedialog.askdirectory(title="Dossier contenant un TIFF par image")
        if d:
            self.var_fichier.set(d)

    def _choisir_dossier(self):
        d = filedialog.askdirectory(title="Dossier de sortie")
        if d:
            self.var_dossier.set(d)

    def _basculer_acq(self):
        if self.moteur.etat == "marche":
            self._arreter()
        else:
            self._demarrer()

    def _construire_source(self):
        if self.var_source.get() == "fichier":
            ch = self.var_fichier.get().strip().strip('"')
            if not ch:
                raise ValueError("Choisissez d'abord un fichier TIFF (ou un dossier de TIFF).")
            vit = {"temps réel": 1.0, "×4": 4.0, "maximale": None}[self.cb_vitesse.get()]
            return SourceFichier(ch, vit)
        return SourceSimulation(max(self._flottant(self.var_rayon_sim.get(), 1.0), 0.05),
                                max(int(self._flottant(self.var_n_sim.get(), 6)), 1))

    def _demarrer(self):
        try:
            self._sur_pa()
            self.moteur.demarrer(self._construire_source(), self._lire_cond())
        except Exception as ex:
            messagebox.showerror("Impossible de démarrer", str(ex))
            return
        c = self.moteur.cond                            # la source a pu imposer taille, fps, profondeur (fichier)
        if self.var_source.get() == "fichier":
            self.c_fps.fixer(c.fps)
            if c.exposition_ms is not None:
                self.c_expo.fixer(c.exposition_ms)
            self.cb_gross.set(f"{c.grossissement}X")
            if c.echelle_manuelle_um:
                self.var_manuelle.set(True)
                self.var_echelle_man.set(f"{c.echelle_manuelle_um:.4f}")
            self.cb_bits.set("8 bits" if c.bits <= 8 else "12 bits (RAW)")
        self._version_res, self._dernier_num, self._fig_a_jour = -1, -1, False
        self._maj_etats()
        self._sur_changement()
        self.var_etat.set(f"Acquisition en cours — {self.moteur.source.nom}")

    def _arreter(self):
        if self.moteur.enreg is not None:
            self._basculer_enreg()
        self.moteur.arreter()
        self._maj_etats()
        self.var_etat.set("Acquisition arrêtée. Les résultats restent affichés ; ils sont remis à zéro au prochain démarrage.")

    def _capturer(self):
        try:
            base = self.moteur.capturer(self.var_dossier.get())
            self.var_etat.set(f"Image enregistrée : {base.name}.tif / .png")
        except Exception as ex:
            messagebox.showerror("Capture impossible", str(ex))

    def _basculer_enreg(self):
        m = self.moteur
        if m.enreg is None:
            try:
                self._sur_pa()
                m.demarrer_enregistrement(self.var_dossier.get(), max(self._flottant(self.var_duree.get(), 0), 0),
                                          self.var_enr_pistes.get(), self.var_enr_brut.get())
            except Exception as ex:
                messagebox.showerror("Enregistrement impossible", str(ex))
                return
            self.var_enr.set("Enregistrement en cours…")      # les conditions sont figées pendant la séquence
        else:
            chemins = m.arreter_enregistrement()
            self.var_enr.set("Enregistré :\n" + "\n".join(str(c) for c in chemins))
            self.prog["value"] = 0
        self._maj_etats()

    def _reinit(self):
        self.moteur.reinitialiser()
        self._version_res = -1
        self.var_etat.set("Suivi réinitialisé.")

    def _exporter(self):
        liste, final, _ = self.moteur.resultats()
        if not liste:
            messagebox.showinfo("Rien à exporter", "Aucune trajectoire analysée pour l'instant.")
            return
        dossier = Path(self.var_dossier.get())
        dossier.mkdir(parents=True, exist_ok=True)
        base = dossier / ("analyse_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        with open(base.with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["piste", "images", "D_um2_s", "sigma_exp_nm", "a_um", "sigma_PSF_um", "sigma_stat_um", "sigma_tot_um"])
            for r in liste:
                w.writerow([r["piste"], r["n_pts"], f"{r['D'] * 1e12:.5f}", f"{r['sigma_exp'] * 1e9:.1f}", f"{r['a'] * 1e6:.5f}",
                            f"{r['inc_psf'] * 1e6:.5f}", f"{r['inc_stat'] * 1e6:.5f}", f"{r['inc'] * 1e6:.5f}"])
            w.writerow([])
            w.writerow(["rayon_final_um", f"{final['a'] * 1e6:.5f}", "incertitude_um", f"{final['inc'] * 1e6:.5f}",
                        "chi2_red", f"{final['chi2_red']:.3f}", "n_particules", final["n"]])
        self._dessiner_figure()
        self.fig.savefig(base.with_suffix(".png"), dpi=150)
        c = self.moteur.cond
        base.with_suffix(".json").write_text(json.dumps(
            {"conditions": {**asdict(c), "echelle_um_px": c.echelle_um_px()}, "analyse": asdict(self.pa),
             "rayon_um": final["a"] * 1e6, "incertitude_um": final["inc"] * 1e6, "chi2_red": final["chi2_red"],
             "n_particules": final["n"], "sigma_psf_px": final.get("sigma_psf_px")}, indent=2, ensure_ascii=False), encoding="utf-8")
        self.var_etat.set(f"Analyse exportée : {base.name}.csv / .png / .json")

    def _quitter(self):
        try:
            self._arreter()
        finally:
            self.racine.destroy()

    # ------------------------------------------------------------------ rafraîchissement
    def _rafraichir(self):
        try:
            self._maj_vue()
            self._maj_resultats()
            if time.time() - self._t_stats > 0.4:
                self._t_stats = time.time()
                self._maj_stats()
        except Exception as ex:                          # l'interface ne doit jamais s'arrêter sur une erreur d'affichage
            self.var_etat.set(f"Erreur d'affichage : {ex}")
        self.after(33, self._rafraichir)

    def _maj_vue(self):
        m = self.moteur
        snap = m.dernier()
        if snap is None:
            return
        if snap.num != self._dernier_num:
            self._dernier_num = snap.num
            self._img_courante = snap.img
            pistes = m.pistes_affichage(snap.idx) if self.var_pistes.get() and self.pa.actif else None
            dets = m.detections() if self.var_dets.get() and self.pa.actif else None
            bgr = composer(snap.img, m.info_pour(snap.t_s, snap.md), pistes, dets, self.var_etirer.get())
            e = m.enreg
            if e is not None and e.t0 is not None:
                cv2.putText(bgr, f"REC {snap.t_s - e.t0:.1f} s", (bgr.shape[1] - 150, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            self._poser_image(bgr)
        if m.etat != "marche" and self.btn_acq.cget("text").startswith("■"):     # fin de fichier ou erreur
            self._maj_etats()
            self.var_etat.set(m.message or "Acquisition terminée.")

    def _poser_image(self, bgr):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 20 or ch < 20:
            return
        h, w = bgr.shape[:2]
        s = min(cw / w, ch / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        if (nw, nh) != (w, h):
            bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        self.canvas.delete("ph")
        if self._item is None:
            self._item = self.canvas.create_image(ox, oy, anchor="nw", image=self._photo)
        else:
            self.canvas.itemconfig(self._item, image=self._photo)
            self.canvas.coords(self._item, ox, oy)
        self._map = (ox, oy, s)

    def _placeholder(self):
        if self.moteur.dernier() is None:
            self.canvas.delete("ph")
            self.canvas.create_text(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, tag="ph",
                                    fill="#8a8a8a", font=("TkDefaultFont", 14), justify="center",
                                    text="Aucune image.\nChoisissez un fichier TIFF, réglez les conditions,\npuis cliquez sur « Démarrer l'acquisition ».")

    def _survol(self, ev):
        img = self._img_courante
        if img is None:
            return
        ox, oy, s = self._map
        x, y = int((ev.x - ox) / s), int((ev.y - oy) / s)
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            pe = self.moteur.cond.echelle_um_px()
            self.var_survol.set(f"x = {x * pe:.1f} µm, y = {y * pe:.1f} µm  |  pixel ({x}, {y})  |  intensité = {int(img[y, x])}")
        else:
            self.var_survol.set("")

    def _maj_stats(self):
        m = self.moteur
        st = m.stats
        n_act, n_longues = m.suivi.nombres(self.pa.n_min)
        txt = f"{st['fps_mesure']:.1f} images/s mesurées"
        if m.etat != "arret" or m.dernier() is not None:
            txt += f"  |  détections : {st['n_det']}  |  pistes actives : {n_act} (≥ {self.pa.n_min} images : {n_longues})"
            if st["perdues"]:
                txt += f"  |  images non analysées : {st['perdues']}"
        self.var_stats.set(txt)
        snap = m.dernier()
        if snap is not None:
            img = snap.img
            sous = img[::2, ::2]
            niv = 2 ** m.cond.bits - 1
            sat = 100.0 * np.count_nonzero(sous >= niv) / sous.size
            mx = int(sous.max())
            self.lbl_sat.configure(text=f"Saturation : {sat:.2f} %   max : {mx}/{niv}",
                                   foreground="#c62828" if sat > 0.05 else "#2e7d32")
        if m.etat == "marche" and m.message:
            self.var_etat.set(m.message)
        e = m.enreg
        if e is not None and e.t0 is not None and snap is not None:
            ecoule = snap.t_s - e.t0
            self.var_enr.set(f"● Enregistrement : {ecoule:.1f} s" + (f" / {e.duree:.0f} s" if e.duree else "") + f"  ({e.n} images)")
            self.prog["value"] = 100 * ecoule / e.duree if e.duree else 0
        elif e is None and self.btn_enr.cget("text").startswith("■"):         # arrêt automatique à la fin de la durée
            self.var_enr.set("Enregistré :\n" + "\n".join(str(c) for c in m.dernier_enreg))
            self.prog["value"] = 0
            self._maj_etats()

    def _maj_resultats(self):
        liste, final, version = self.moteur.resultats()
        if version == self._version_res:
            return
        self._version_res = version
        if final is None:
            self.var_rayon.set("—")
            self.var_rayon_detail.set(f"En attente de trajectoires d'au moins {self.pa.n_min} images…"
                                      if self.pa.actif else "Suivi désactivé.")
        else:
            self.var_rayon.set(f"a = {final['a'] * 1e6:.3f} ± {final['inc'] * 1e6:.3f} µm")
            chi = "" if not np.isfinite(final["chi2_red"]) else f"  ·  χ²réd = {final['chi2_red']:.2f}"
            sp = final.get("sigma_psf_px")
            self.var_rayon_detail.set(f"{final['n']} particule(s){chi}"
                                      + (f"  ·  σ_PSF = {sp:.2f} px" if sp else "")
                                      + f"\nσ_PSF (localisation) = {final['inc_psf'] * 1e6:.3f} µm ⊕ σ_stat = {final['inc_stat'] * 1e6:.3f} µm")
        self.tableau.delete(*self.tableau.get_children())
        for r in liste:
            self.tableau.insert("", "end", values=(r["piste"], r["n_pts"], f"{r['D'] * 1e12:.3f}", f"{r['sigma_exp'] * 1e9:.0f}",
                                                   f"{r['a'] * 1e6:.3f}", f"{r['inc_psf'] * 1e6:.3f}",
                                                   f"{r['inc_stat'] * 1e6:.3f}", f"{r['inc'] * 1e6:.3f}"))
        self._fig_a_jour = False
        self._maj_figure()

    def _maj_figure(self):
        if not self._fig_a_jour and self.onglets.index("current") == 1:
            self._dessiner_figure()

    def _dessiner_figure(self):
        liste, final, _ = self.moteur.resultats()
        c = self.moteur.cond
        dt, pe = 1.0 / c.fps, c.pixel_eff_m
        self.ax_msd.clear()
        self.ax_a.clear()
        for r in liste:
            col = tuple(v / 255 for v in couleur_piste(r["piste"])[::-1])
            pos = r["pos_px"] * pe
            lags, msd = tp.msd_temporel(pos, self.pa.lag_max)
            ok = np.isfinite(msd)
            self.ax_msd.plot(lags[ok] * dt, msd[ok] * 1e12, "o", color=col, ms=4)
            D, off = tp.ajuste_msd(pos, dt, self.pa.lag_max)
            tt = np.linspace(0, lags[-1] * dt, 20)
            self.ax_msd.plot(tt, (4 * D * tt + off) * 1e12, "-", color=col, lw=1, label=f"{r['piste']}")
        self.ax_msd.set(xlabel="Délai τ (s)", ylabel="MSD (µm²)", title="MSD et ajustement linéaire (Éq. 7)")
        self.ax_msd.grid(alpha=0.3)
        if liste:
            self.ax_msd.legend(title="piste", fontsize=7, ncol=2)
            ids = [r["piste"] for r in liste]
            a = np.array([r["a"] for r in liste]) * 1e6
            self.ax_a.errorbar(range(len(ids)), a, yerr=[r["inc"] * 1e6 for r in liste], fmt="o", capsize=4, label="a ± σ_tot")
            self.ax_a.set_xticks(range(len(ids)))
            self.ax_a.set_xticklabels(ids)
            self.ax_a.axhline(final["a"] * 1e6, color="tab:green")
            self.ax_a.axhspan((final["a"] - final["inc"]) * 1e6, (final["a"] + final["inc"]) * 1e6, color="tab:green", alpha=0.2,
                              label=f"final = {final['a'] * 1e6:.3f} ± {final['inc'] * 1e6:.3f} µm")
            self.ax_a.legend(fontsize=8)
        else:
            for ax in (self.ax_msd, self.ax_a):
                ax.text(0.5, 0.5, "En attente de trajectoires…", ha="center", va="center", transform=ax.transAxes, color="#888")
        self.ax_a.set(xlabel="Piste", ylabel="Rayon (µm)", title="Rayon par particule")
        self.ax_a.grid(alpha=0.3)
        self.canvas_fig.draw_idle()
        self._fig_a_jour = True


def main():
    try:                                                  # rendu net sur écrans Windows à haute densité
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    racine = tk.Tk()
    Application(racine)
    racine.mainloop()


if __name__ == "__main__":
    main()