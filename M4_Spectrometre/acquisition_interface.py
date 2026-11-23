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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from picamera2 import Picamera2


DOSSIER_SORTIE = Path(__file__).resolve().parent / "acquisitions"
DOSSIER_LASERS_PROFS = DOSSIER_SORTIE / "calibration" / "lasers"
DOSSIER_TXT_PROFS = DOSSIER_SORTIE.parent / "references_profs"

ASSOCIATIONS_LASERS = {
    "532": ("532nm Laser (battery powered).txt", False),
    "555": ("555nm Laser (battery powered).txt", False),
    "571": ("571nm Laser (battery powered).txt", False),
    "593": ("593nm Laser (battery powered).txt", False),
    "650": ("635nm_laser_pointer_cat.txt", False),
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
N_BOOT, TAUX_MIN = 2000, 0.95


@dataclass(frozen=True)
class Laser:
    px: float
    nm: float
    largeur: int
    verification: bool
    source: str


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
    if satures[max(0, g - demi_lissage):d + demi_lissage + 1].any():
        raise ValueError("Pic laser saturé. Réduisez l’exposition puis recommencez.")
    return px, intensite


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

def charger_lasers_profs():
    if not ASSOCIATIONS_LASERS:
        raise ValueError("Renseignez ASSOCIATIONS_LASERS dans le script.")

    lasers, dernier = [], None
    for nom, (nom_txt, verification) in ASSOCIATIONS_LASERS.items():
        dossier = DOSSIER_LASERS_PROFS / nom
        txt = DOSSIER_TXT_PROFS / nom_txt

        fichiers = sorted(dossier.glob("raw_*.npy"))
        if not fichiers and (dossier / "raw.npy").is_file():
            fichiers = [dossier / "raw.npy"]
        if not fichiers:
            raise ValueError(f"{dossier} : aucun raw.npy ou raw_*.npy.")

        try:
            nm = longueur_txt(txt)
            px, intensite = profil_laser(np.load(f) for f in fichiers)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Laser {nom} · {txt.name} : {exc}") from exc

        point = Laser(px, nm, intensite.size, verification, f"{nom} / {txt.name}")
        lasers.append(point)
        dernier = (point, intensite)

    verifier_lasers(lasers)
    if all(point.verification for point in lasers):
        raise ValueError("Affectez au moins un laser à l’ajustement avec False.")
    return lasers, dernier

def calibrer(n, lasers):
    pts = [(p.px, p.nm) for p in lasers if not p.verification]
    if len(pts) >= 2:
        a, b = np.polyfit(*zip(*pts), 1)
        return a * np.arange(n) + b, f"linéaire sur {len(pts)} raies ({a:.4f} nm/px)"
    if len(pts) == 1:
        return axe_lbd(n, *pts[0]), f"modèle calé sur {pts[0][1]:g} nm"
    return axe_lbd(n, n / 2, LBD_C * 1e9), "axe nominal (550 nm au centre)"


def ecarts_verification(x, lasers):
    if not any(not p.verification for p in lasers):
        return []
    px = np.arange(len(x))
    return [(p.nm, np.interp(p.px, px, x) - p.nm) for p in lasers if p.verification]


def incertitude_calibration(ecarts):
    return np.sqrt(np.mean([e**2 for _, e in ecarts])) if ecarts else None


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


def bootstrap(x, ordre, ref, mes):
    """Écarts-types des coupures, sous hypothèse d’images indépendantes."""
    nr, nm = len(ref), len(mes)
    if min(nr, nm) < 2:
        return [(None, None), (None, None)]

    rng = np.random.default_rng(3910)
    valeurs = [[], []]
    x_croissant = x[ordre]
    for _ in range(N_BOOT):
        ref_b = ref[rng.integers(nr, size=nr)]
        mes_b = mes[rng.integers(nm, size=nm)]
        _, T = transmission(ref_b, mes_b)
        centres, _ = coupures(x_croissant, T[ordre])
        for serie, c in zip(valeurs, centres):
            if c is not None:
                serie.append(c)

    return [(np.std(v, ddof=1) if len(v) >= TAUX_MIN * N_BOOT else None, len(v))
            for v in valeurs]


class App:
    def __init__(self, root):
        self.root, self.lasers, self.occupe = root, [], False
        self.expo = tk.DoubleVar(value=EXPO_MS)
        self.n_img = tk.IntVar(value=5)
        self.n_ang = tk.IntVar(value=3)
        self.coupure_theo = tk.StringVar()
        self.u_coupure_theo = tk.StringVar()
        self.calib_txt = tk.StringVar(value="Aucune raie enregistrée")
        self.laser_nm = tk.StringVar(value="532")
        self.laser_verif = tk.BooleanVar()
        self.txt_reference = None
        self.commandes = []

        root.title("Spectromètre — acquisition")
        self.construire()
        self.demarrer_camera()
        dossier_lasers = (
            DOSSIER_LASERS_PROFS
            if DOSSIER_LASERS_PROFS.is_dir()
            else DOSSIER_SORTIE / "calibration_lasers"
        )
        if dossier_lasers.is_dir():
            root.after_idle(lambda: self.choisir_calib(dossier_lasers))
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
        ttk.Label(blanc, text="Coupure théorique (nm, facultative)").pack(anchor="w")
        ttk.Entry(blanc, textvariable=self.coupure_theo, width=12).pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(blanc, text="Incertitude ± (nm, facultative)").pack(anchor="w")
        ttk.Entry(blanc, textvariable=self.u_coupure_theo, width=12).pack(
            anchor="w", pady=(0, 6)
        )
        bouton(blanc, "Mesurer la transmission", self.sequence)

        ttk.Label(laser, text="Longueur d’onde du laser (nm)").pack(anchor="w")
        ttk.Entry(laser, textvariable=self.laser_nm, width=12).pack(
            anchor="w", pady=(0, 4)
        )
        bouton(laser, "Longueur d’onde depuis un TXT…", self.choisir_txt)
        ttk.Checkbutton(
            laser, text="Raie de vérification", variable=self.laser_verif
        ).pack(anchor="w", pady=4)
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

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(side="right", fill="both", expand=True)
        t1, t2, t3 = (ttk.Frame(self.nb) for _ in range(3))
        self.nb.add(t1, text="Aperçu")
        self.nb.add(t2, text="Transmission")
        self.nb.add(t3, text="Étalonnage")

        self.lbl = ttk.Label(t1)
        self.lbl.pack()

        self.fig = Figure(figsize=(7, 4.2), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=t2)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.res = ScrolledText(t2, height=8, wrap="word")
        self.res.pack(fill="x")

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
            self.txt_reference = (Path(chemin).name, self.laser_nm.get())
            self.log(f"Pic du TXT : {nm:.3f} nm. Vérifiez qu’il s’agit du même laser.")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Spectre de référence", str(exc), parent=self.root)

    def ajouter_laser(self):
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
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Capture laser", str(exc), parent=self.root)
            return

        self.occupation(True)
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

            dossier = (
                Path(DOSSIER_SORTIE) / "calibration_lasers"
                / datetime.now().strftime("laser_%Y%m%d_%H%M%S_%f")
            )
            px, intensite = profil_laser(self.images(str(dossier), n))
            laser = Laser(px, nm, intensite.size, verif, source)
            nouveaux = self.lasers + [laser]
            verifier_lasers(nouveaux)

            meta = {"nm": nm, "verification": verif, "source": source}
            (dossier / "laser.json").write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8"
            )
            self.lasers = nouveaux
            self.afficher_calib((laser, intensite))
            self.nb.select(2)
            self.log(f"Laser {nm:g} nm enregistré au pixel {px:.2f}.")
        except (OSError, RuntimeError, ValueError, tk.TclError) as exc:
            messagebox.showerror("Capture laser", str(exc), parent=self.root)
        finally:
            self.occupation(False)

    def choisir_calib(self):
        chemin = filedialog.askdirectory(parent=self.root, title="Dossier de captures laser RAW")
        if not chemin:
            return

        self.occupation(True)
        try:
            dossier = Path(chemin)
            if dossier.resolve() == DOSSIER_LASERS_PROFS.resolve():
                self.lasers, dernier = charger_lasers_profs()
                self.afficher_calib(dernier)
                if manuel:
                    self.nb.select(2)
                self.log(f"{len(self.lasers)} lasers chargés depuis {dossier}.")
                return
            nouveaux, dernier = [], None
            for d in [dossier] + sorted(p for p in dossier.iterdir() if p.is_dir()):
                meta = d / "laser.json"
                nom = re.search(r"(\d+(?:[.,]\d+)?)\s*nm", d.name, re.I)
                fichiers = sorted(d.glob("raw_*.npy")) or sorted((d / "images").glob("raw_*.npy"))
                if meta.is_file():
                    infos = json.loads(meta.read_text(encoding="utf-8"))
                    nm = float(infos["nm"])
                    verif, source = infos["verification"], infos["source"]
                elif nom and fichiers:
                    nm = float(nom.group(1).replace(",", "."))
                    verif, source = False, d.name
                else:
                    continue

                px, intensite = profil_laser(np.load(f) for f in fichiers)
                laser = Laser(px, nm, intensite.size, verif, source)
                nouveaux.append(laser)
                dernier = (laser, intensite)

            if not nouveaux:
                raise ValueError("Aucune capture laser reconnue dans ce dossier.")
            verifier_lasers(nouveaux)
            self.lasers = nouveaux
            self.afficher_calib(dernier)
            self.nb.select(2)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            messagebox.showerror("Dossier de calibration", str(exc), parent=self.root)
        finally:
            self.occupation(False)

    def sans_calib(self):
        self.lasers = []
        self.afficher_calib()

    def afficher_calib(self, mesure=None):
        n_verif = sum(p.verification for p in self.lasers)
        self.calib_txt.set(
            f"Raies : {len(self.lasers) - n_verif} ajustement, {n_verif} vérification"
        )
        self.fig_cal.clear()
        haut, bas = self.fig_cal.subplots(2, 1)
        lignes = []

        if mesure is not None:
            laser, intensite = mesure
            haut.plot(intensite)
            haut.axvline(
                laser.px, ls="--", color="tab:orange",
                label=f"{laser.nm:g} nm → {laser.px:.2f} px",
            )
            haut.legend(fontsize=8)
        haut.set(xlabel="Pixel binné", ylabel="Intensité RAW")

        if self.lasers:
            x, desc = calibrer(self.lasers[0].largeur, self.lasers)
            bas.plot(x, color="0.5", label=desc)
            for verif, couleur, nom in (
                (False, "tab:blue", "Ajustement"),
                (True, "tab:orange", "Vérification"),
            ):
                pts = [p for p in self.lasers if p.verification == verif]
                if pts:
                    bas.scatter([p.px for p in pts], [p.nm for p in pts],
                                color=couleur, label=nom)
                    lignes.extend(
                        f"{nom}  {p.nm:g} nm  |  {p.px:.2f} px  |  {p.source}"
                        for p in pts
                    )
            bas.legend(fontsize=8)
            ecarts = ecarts_verification(x, self.lasers)
            lignes.extend(resume_verification(ecarts, incertitude_calibration(ecarts)))

        bas.set(xlabel="Pixel binné", ylabel="Longueur d’onde (nm)")
        self.fig_cal.tight_layout()
        self.canvas_cal.draw_idle()
        self.res_cal.delete("1.0", "end")
        self.res_cal.insert("end", "\n".join(lignes))

    def demarrer_camera(self):
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
        self.cam.set_controls({"ExposureTime": int(ms * 1000)})
        for _ in range(4):
            self.cam.capture_metadata()
        self.log(f"Exposition : {ms:.1f} ms")
        return True

    def apercu(self):
        if not self.occupe:
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
        try:
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

            mesures = []
            for k in range(na):
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
                self.analyser(dossier, ref, mesures, n, theorie)
        except (OSError, RuntimeError, ValueError) as exc:
            messagebox.showerror("Séquence", str(exc), parent=self.root)
        finally:
            self.occupation(False)

    def analyser(self, dossier, ref, mesures, n_img, theorie=None):
        npx = ref.shape[1]
        lasers = self.lasers
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
        raies = [nm for nm, _ in ecarts]
        lignes = [f"Étalonnage : {desc}"] + avertissement + resume_verification(ecarts, u_cal)
        if u_cal is None:
            lignes.append("Barres : répétabilité seule (± une incertitude-type).")
        elif len(raies) == 1:
            lignes.append("Une seule raie de vérification. L’estimation de u_cal est indicative.")

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        i_ref, _ = transmission(ref, mesures[0][1])
        colonnes, entete = [x, i_ref], "lbd_nm,I_ref"

        for a, profs in mesures:
            _, T = transmission(ref, profs)
            centres, niveau = coupures(x[ordre], T[ordre])
            ligne, = ax.plot(x, T)
            lignes.append(f"θ = {a:g}°")
            txt = []
            incertitudes = (
                bootstrap(x, ordre, ref, profs) if any(c is not None for c in centres) else []
            )

            for nom, c, (u_stat, nb) in zip(
                ("montante", "descendante"), centres, incertitudes
            ):
                if c is None:
                    continue

                u = None
                if u_stat is None:
                    valeur = f"{c:.2f} nm"
                    detail = (
                        "répétabilité indisponible : au moins 2 images par série"
                        if nb is None else f"coupure instable : {nb}/{N_BOOT} tirages"
                    )
                else:
                    u = u_stat if u_cal is None else np.hypot(u_stat, u_cal)
                    valeur = f"({c:.2f} ± {u:.3g}) nm"
                    detail = f"u_rép = {u_stat:.3g} nm, {nb}/{N_BOOT} tirages"
                    if raies and not min(raies) <= c <= max(raies):
                        detail += ", hors de la plage des raies de vérification"

                txt.append(valeur)
                lignes.append(f"    {nom} : {valeur} ({detail})")
                ax.axvline(c, color=ligne.get_color(), ls="--", lw=0.7)
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
            f"Bootstrap : {N_BOOT} tirages, seuil de stabilité {TAUX_MIN:.0%}",
            "Barres horizontales : incertitude-type (k=1)",
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
            self.cam.stop()
            self.cam.close()
        finally:
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()