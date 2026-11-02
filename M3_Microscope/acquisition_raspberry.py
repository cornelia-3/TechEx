"""Acquisition du microscope — Raspberry Pi 4 + caméra HQ (capteur IMX477, pixels de 1,55 µm).

Au lancement, deux choix dans le terminal :

1. L'OPTION (une lettre)
   A  2028 x 1520, gris   capture pleine résolution + somme 2x2 du Bayer · RAM, pile TIFF écrite à la fin
   B  2028 x 1520, gris   comme A, mais pile TIFF écrite EN CONTINU pendant l'acquisition
   C  1014 x 760,  gris   mode capteur 2028 x 1520 + somme 2x2 · pile TIFF en continu (40X et 60X)
   D  4056 x 3040, Bayer  raw pleine résolution, aucun traitement · RAM, pile TIFF écrite à la fin
   E  4056 x 3040, Bayer  ancienne méthode : 30 images en RAM, puis un PNG 16 bits par image

2. APERÇU (direct seulement, rien n'est sauvegardé) ou ENREGISTREMENT.

Somme 2x2 du Bayer : chaque carré B G / G R du filtre coloré devient un seul pixel gris
(valeurs 0 à 4 x 4095 = 16 380). Plus de damier, 4x moins de données, environ 2x moins de bruit.

Le direct affiche exactement les images enregistrées (mêmes pixels). Touches dans la fenêtre :
    Espace ou Entrée : démarrer / arrêter l'enregistrement
    c : contraste automatique <-> échelle fixe (12 bits, ou 0-16 380 après la somme 2x2)
    z : zoom 1:1 au centre (pour la mise au point)
    q ou Échap : quitter (un enregistrement en cours est sauvegardé)
Dans le terminal : Entrée = démarrer / arrêter, q + Entrée = quitter.
Sans fenêtre possible (SSH), le script fonctionne quand même avec le terminal seulement.

Sorties (dossier courant), nom horodaté pour ne jamais rien écraser :
    <nom>_AAAA-MM-JJ_HHhMMmSS_<option>_<grossissement>X.tif   pile multipage non compressée (A à D)
    <nom>_..._E_..X/frame_0000.png ...                         un PNG par image (E)
    <nom>_....json   horodatages, exposition, gain, grossissement, fps réel, images perdues
<nom> est demandé au lancement en mode enregistrement (Entrée = « acq »).

Dépendances sur la Pi : picamera2, numpy, opencv (déjà utilisés), et tifffile pour A à D :
    sudo apt install python3-tifffile
"""
import json
import os
import queue
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2

try:
    import tifffile
except ImportError:
    tifffile = None

# ---------- Paramètres ----------
EXPOSURE_US = 90_000                     # temps d'exposition (µs)
GAIN = 1.0                               # gain analogique (monter le gain avant l'exposition)
N_MAX = {"A": 300, "D": 120, "E": 30}    # images par enregistrement (options en RAM; réduit si la RAM manque)
FRACTION_RAM = 0.75                      # part de la RAM libre qu'on s'autorise à remplir
LIVE_PENDANT_ENREGISTREMENT = True       # False = direct figé pendant l'enregistrement (si la Pi ne suit pas)
LARGEUR_FENETRE = 1000                   # largeur d'affichage du direct (px d'écran)
GROSSISSEMENT_DEFAUT = 20
DOSSIER_SORTIE = Path.cwd()

PLEINE = (4056, 3040)
OPTIONS = {
    "A": dict(capteur=PLEINE, somme2x2=True, ecriture="ram", format="tif",
              desc="2028 x 1520 gris, RAM puis pile TIFF (tous les objectifs)"),
    "B": dict(capteur=PLEINE, somme2x2=True, ecriture="continue", format="tif",
              desc="2028 x 1520 gris, pile TIFF écrite en continu (durée libre)"),
    "C": dict(capteur=(2028, 1520), somme2x2=True, ecriture="continue", format="tif",
              desc="1014 x 760 gris, pile TIFF en continu (40X et 60X)"),
    "D": dict(capteur=PLEINE, somme2x2=False, ecriture="ram", format="tif",
              desc="4056 x 3040 raw Bayer pleine résolution, RAM puis pile TIFF"),
    "E": dict(capteur=PLEINE, somme2x2=False, ecriture="ram", format="png",
              desc="ancienne méthode : 4056 x 3040 raw, 30 images, un PNG par image"),
}
PIXEL_UM = {"A": 3.1, "B": 3.1, "C": 6.2, "D": 1.55, "E": 1.55}  # pixel effectif au capteur (µm)
DUREE_MIN_US = {PLEINE: 100_000, (2028, 1520): 25_000}   # durée d'image minimale par mode capteur


#############################################################################################################
#                                       CHOIX DANS LE TERMINAL
#############################################################################################################

def choisir_option():
    print("Options :")
    for lettre, o in OPTIONS.items():
        print(f"  {lettre}  {o['desc']}")
    while True:
        rep = input("Option (A-E) : ").strip().upper()
        if rep in OPTIONS:
            return rep
        print("  Choix invalide.")


def choisir_mode():
    while True:
        rep = input("a = aperçu (rien n'est sauvegardé) · e = enregistrement : ").strip().lower()
        if rep in ("a", "aperçu", "apercu"):
            return "apercu"
        if rep in ("e", "enregistrement"):
            return "enregistrement"
        print("  Taper a ou e.")


def choisir_nom():
    """Nom de base des fichiers. La date et l'heure sont ajoutées : rien n'est jamais écrasé."""
    rep = input("Nom des fichiers [Entrée = acq] : ").strip()
    propre = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in rep).strip("._")
    if rep and propre != rep:
        print(f"  Nom ajusté : {propre}")
    return propre or "acq"


def choisir_grossissement():
    while True:
        rep = input(f"Grossissement de l'objectif [Entrée = {GROSSISSEMENT_DEFAUT}] : ").strip()
        rep = rep.upper().rstrip("X").replace(",", ".")
        if not rep:
            return GROSSISSEMENT_DEFAUT
        try:
            g = float(rep)
            if g > 0:
                return int(g) if g.is_integer() else g
        except ValueError:
            pass
        print("  Entrer un nombre positif.")


#############################################################################################################
#                                       IMAGES ET MÉMOIRE
#############################################################################################################

def somme_bayer(raw):
    """Somme des 4 pixels B, G, G, R de chaque carré 2x2 -> image grise de demi-taille (max 16 380)."""
    return raw[0::2, 0::2] + raw[0::2, 1::2] + raw[1::2, 0::2] + raw[1::2, 1::2]


def ram_disponible():
    """Octets de RAM disponibles (MemAvailable), ou None si illisible."""
    try:
        with open("/proc/meminfo") as f:
            for ligne in f:
                if ligne.startswith("MemAvailable:"):
                    return int(ligne.split()[1]) * 1024
    except OSError:
        pass
    return None


def bilan_temps(stamps_ns, durees_us):
    """(fps réel, nombre d'images perdues) à partir des horodatages du capteur."""
    if len(stamps_ns) < 2:
        return float("nan"), 0
    d = np.diff(np.asarray(stamps_ns, dtype=np.int64))
    attendu = np.median(durees_us) * 1e3 if durees_us else np.median(d)
    perdues = int(np.sum(np.maximum(np.round(d / attendu) - 1, 0)))
    return 1e9 / np.median(d), perdues


#############################################################################################################
#                                       ENREGISTREMENT
#############################################################################################################

class Enregistreur:
    """Un enregistrement : reçoit les images une à une, écrit les fichiers et le .json à la fin."""

    def __init__(self, lettre, opt, forme, dtype, n_max, grossissement, consignes, nom):
        self.lettre, self.opt, self.n_max = lettre, opt, n_max
        self.forme, self.dtype = forme, dtype
        self.grossissement, self.consignes = grossissement, consignes
        date = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S")
        self.base = DOSSIER_SORTIE / f"{nom}_{date}_{lettre}_{grossissement}X"
        self.images, self.stamps, self.expos, self.gains, self.durees = [], [], [], [], []
        self.n = 0
        if opt["ecriture"] == "continue":
            self.file = queue.Queue(maxsize=32)
            self.fil = threading.Thread(target=self._ecrivain, daemon=True)
            self.fil.start()

    def _ecrivain(self):
        with tifffile.TiffWriter(self.base.with_suffix(".tif"), bigtiff=True) as tif:
            while (img := self.file.get()) is not None:
                tif.write(img, contiguous=True)

    def ajouter(self, img, md):
        self.stamps.append(md.get("SensorTimestamp", time.monotonic_ns()))
        self.expos.append(md.get("ExposureTime"))
        self.gains.append(md.get("AnalogueGain"))
        if md.get("FrameDuration"):
            self.durees.append(md["FrameDuration"])
        if self.opt["ecriture"] == "continue":
            self.file.put(img)        # si la clé ne suit pas, ça bloque : les images perdues se voient
        else:                         # dans les horodatages
            self.images.append(img)
        self.n += 1

    def plein(self):
        return self.n_max is not None and self.n >= self.n_max

    def terminer(self):
        t0 = time.time()
        if self.opt["ecriture"] == "continue":
            self.file.put(None)
            self.fil.join()
            fichier = self.base.with_suffix(".tif")
        elif self.opt["format"] == "tif":
            fichier = self.base.with_suffix(".tif")
            volume = self.n * self.images[0].nbytes / 1e9 if self.images else 0
            print(f"\nÉcriture de {self.n} images ({volume:.1f} Go) dans {fichier.name}. "
                  f"Sur une clé USB, ça peut prendre plusieurs minutes...", flush=True)
            with tifffile.TiffWriter(fichier, bigtiff=True) as tif:
                for k, img in enumerate(self.images, 1):
                    tif.write(img, contiguous=True)
                    if k % max(1, self.n // 10) == 0 or k == self.n:
                        ecoule = time.time() - t0
                        print(f"  {k}/{self.n} images écrites ({ecoule:.0f} s, "
                              f"{k * img.nbytes / 1e6 / max(ecoule, 1e-3):.0f} Mo/s)", flush=True)
        else:                                            # E : ancienne méthode, un PNG par image
            fichier = self.base
            fichier.mkdir()

            def save(i):
                cv2.imwrite(str(fichier / f"frame_{i:04d}.png"), self.images[i],
                            [cv2.IMWRITE_PNG_COMPRESSION, 1])
                print(f"sauvegardé {i + 1}/{self.n}", flush=True)

            with ThreadPoolExecutor(max_workers=2) as ex:
                list(ex.map(save, range(self.n)))
            np.save(fichier / "timestamps_ns.npy", np.array(self.stamps))
            np.save(fichier / "exposure_us.npy", np.array(self.expos))
        self.images = []                                 # libère la RAM
        duree_ecriture = time.time() - t0

        fps, perdues = bilan_temps(self.stamps, self.durees)
        h, w = self.forme
        meta = {
            "option": self.lettre,
            "description": self.opt["desc"],
            "date": datetime.now().isoformat(timespec="seconds"),
            "grossissement": self.grossissement,
            "fichier_images": fichier.name,
            "largeur_px": w, "hauteur_px": h, "dtype": str(self.dtype),
            "contenu": ("somme 2x2 du Bayer BGGR (gris), valeurs 0-16380"
                        if self.opt["somme2x2"] else "raw Bayer BGGR 12 bits, valeurs 0-4095"),
            "pixel_effectif_capteur_um": PIXEL_UM[self.lettre],
            "pixel_plan_objet_um_nominal": PIXEL_UM[self.lettre] / float(self.grossissement),
            **self.consignes,
            "n_images": self.n,
            "fps_reel": round(float(fps), 4) if np.isfinite(fps) else None,
            "images_perdues": perdues,
            "timestamps_ns": [int(s) for s in self.stamps],
            "exposure_us": self.expos,
            "analogue_gain": self.gains,
        }
        with open(self.base.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)

        print(f"\n=== Enregistrement terminé : {self.n} images -> {fichier.name}")
        print(f"    fps réel = {fps:.2f}   images perdues = {perdues}   écriture finale = {duree_ecriture:.1f} s")
        if perdues:
            print("    ATTENTION : images perdues. Essayer LIVE_PENDANT_ENREGISTREMENT = False, "
                  "une exposition plus longue (fps plus bas) ou l'option C.")


#############################################################################################################
#                                       CAMÉRA (fil séparé)
#############################################################################################################

class Etat:
    def __init__(self):
        self.verrou = threading.Lock()
        self.image = None             # dernière image pour le direct
        self.n_capturees = 0
        self.n_rec = 0
        self.n_max_rec = None
        self.enregistre = False
        self.ecriture = False
        self.fps = float("nan")
        self.quitter = threading.Event()
        self.basculer = threading.Event()


def boucle_camera(picam2, etat, lettre, opt, mode, forme, grossissement, consignes, nom):
    rec = None
    derniers = []
    taille = int(np.prod(forme)) * 2
    try:
        while not etat.quitter.is_set():
            req = picam2.capture_request()
            try:
                raw = req.make_array("raw").view(np.uint16)[:consignes["raw_hauteur"], :consignes["raw_largeur"]]
                img = somme_bayer(raw) if opt["somme2x2"] else raw.copy()
                md = req.get_metadata()
            finally:
                req.release()

            derniers = (derniers + [md.get("SensorTimestamp", time.monotonic_ns())])[-20:]
            with etat.verrou:
                etat.n_capturees += 1
                if len(derniers) > 1:
                    etat.fps = 1e9 * (len(derniers) - 1) / (derniers[-1] - derniers[0])
                if rec is None or LIVE_PENDANT_ENREGISTREMENT:
                    etat.image = img

            if etat.basculer.is_set():
                etat.basculer.clear()
                if mode == "apercu":
                    print("\n(Mode aperçu : rien n'est enregistré. Relancer en mode e pour enregistrer.)")
                elif rec is None:
                    n_max = None
                    if opt["ecriture"] == "ram":
                        n_max = N_MAX[lettre]
                        libre = ram_disponible()
                        if libre is not None:
                            n_ram = int(FRACTION_RAM * libre / taille)
                            if n_ram < n_max:
                                print(f"\nRAM libre : {libre / 1e9:.1f} Go -> limité à {n_ram} images "
                                      f"(au lieu de {n_max})")
                                n_max = n_ram
                    rec = Enregistreur(lettre, opt, forme, img.dtype, n_max, grossissement, consignes, nom)
                    with etat.verrou:
                        etat.enregistre, etat.n_rec, etat.n_max_rec = True, 0, n_max
                    fin = f"{n_max} images max" if n_max else "Entrée pour arrêter"
                    print(f"\n● Enregistrement démarré ({fin})", flush=True)
                    continue          # l'image de déclenchement n'est pas enregistrée
                else:
                    rec = _arreter(rec, etat)
                    continue

            if rec is not None:
                rec.ajouter(img, md)
                with etat.verrou:
                    etat.n_rec = rec.n
                if rec.plein():
                    rec = _arreter(rec, etat)
    finally:
        if rec is not None:
            _arreter(rec, etat)
        etat.quitter.set()


def _arreter(rec, etat):
    with etat.verrou:
        etat.enregistre, etat.ecriture = False, True
    rec.terminer()
    with etat.verrou:
        etat.ecriture = False
    if not etat.quitter.is_set():
        print("\nPrêt. Entrée (ou Espace dans la fenêtre) = nouvel enregistrement · q = quitter", flush=True)
    return None


#############################################################################################################
#                                       AFFICHAGE ET CLAVIER
#############################################################################################################

def lire_terminal(etat):
    for ligne in sys.stdin:
        s = ligne.strip().lower()
        if s == "q":
            etat.quitter.set()
            return
        if s == "":
            etat.basculer.set()


def preparer_affichage(img, vmax, auto, zoom):
    h, w = img.shape
    if zoom:                                                  # 1 pixel image = 1 pixel écran, au centre
        hz = min(h, int(LARGEUR_FENETRE * 0.75))
        wz = min(w, LARGEUR_FENETRE)
        y0, x0 = (h - hz) // 2, (w - wz) // 2
        vue = img[y0:y0 + hz, x0:x0 + wz]
    elif w > LARGEUR_FENETRE:
        vue = cv2.resize(img, (LARGEUR_FENETRE, round(h * LARGEUR_FENETRE / w)), interpolation=cv2.INTER_AREA)
    else:
        vue = img
    if auto:
        bas, haut = np.percentile(vue[::2, ::2], (0.5, 99.8))
        haut = max(haut, bas + 1)
    else:
        bas, haut = 0, vmax
    vue8 = np.clip((vue.astype(np.float32) - bas) * (255.0 / (haut - bas)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(vue8, cv2.COLOR_GRAY2BGR), bas, haut


def texte(img, s, y, couleur=(255, 255, 255)):
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 1, cv2.LINE_AA)


def boucle_fenetre(etat, titre, vmax):
    auto, zoom = True, False
    while not etat.quitter.is_set():
        with etat.verrou:
            img, n, rec, ecr = etat.image, etat.n_rec, etat.enregistre, etat.ecriture
            fps, n_max = etat.fps, etat.n_max_rec
        if img is not None:
            vue, bas, haut = preparer_affichage(img, vmax, auto, zoom)
            echelle = f"auto {bas:.0f}-{haut:.0f}" if auto else f"fixe 0-{vmax}"
            texte(vue, f"{fps:.1f} fps   contraste {echelle}{'   zoom 1:1' if zoom else ''}", 22)
            if rec:
                total = f"/{n_max}" if n_max else ""
                fige = "  (direct fige)" if not LIVE_PENDANT_ENREGISTREMENT else ""
                texte(vue, f"REC  {n}{total}{fige}", 48, (0, 0, 255))
                cv2.circle(vue, (vue.shape[1] - 25, 25), 10, (0, 0, 255), -1)
            elif ecr:
                texte(vue, "Ecriture des fichiers...", 48, (0, 200, 255))
            cv2.imshow(titre, vue)
        touche = cv2.waitKey(30) & 0xFF
        if touche in (ord("q"), 27):
            etat.quitter.set()
        elif touche in (ord(" "), 13, 10):
            etat.basculer.set()
        elif touche == ord("c"):
            auto = not auto
        elif touche == ord("z"):
            zoom = not zoom
        if img is not None and cv2.getWindowProperty(titre, cv2.WND_PROP_VISIBLE) < 1:
            etat.quitter.set()                                # fenêtre fermée avec le X
    cv2.destroyAllWindows()


def boucle_texte(etat):
    while not etat.quitter.is_set():
        with etat.verrou:
            n, rec, ecr, fps, n_max = etat.n_rec, etat.enregistre, etat.ecriture, etat.fps, etat.n_max_rec
        if rec:
            print(f"\r● REC {n}{f'/{n_max}' if n_max else ''}   {fps:.1f} fps   ", end="", flush=True)
        elif not ecr:
            print(f"\r  direct (sans fenêtre)   {fps:.1f} fps   ", end="", flush=True)
        time.sleep(1)


def fenetre_possible(titre):
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        cv2.namedWindow(titre, cv2.WINDOW_AUTOSIZE)
        return True
    except cv2.error:
        return False


#############################################################################################################
#                                                 MAIN
#############################################################################################################

def main():
    lettre = choisir_option()
    opt = OPTIONS[lettre]
    mode = choisir_mode()
    grossissement = choisir_grossissement() if mode == "enregistrement" else None
    nom = choisir_nom() if mode == "enregistrement" else None

    if mode == "enregistrement" and opt["format"] == "tif" and tifffile is None:
        sys.exit("Le module tifffile manque. Sur la Pi : sudo apt install python3-tifffile "
                 "(ou pip install tifffile --break-system-packages), puis relancer.")

    frame_us = max(DUREE_MIN_US[opt["capteur"]], EXPOSURE_US + 2_000)
    picam2 = Picamera2()
    extra = {k: v for k, v in {"AeEnable": False, "AwbEnable": False}.items()
             if k in picam2.camera_controls}             # seulement si cette version de libcamera les connaît
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},   # flux obligatoire pour picamera2, non utilisé
        raw={"size": opt["capteur"], "format": "SBGGR12"},
        buffer_count=4,
        controls={"FrameDurationLimits": (frame_us, frame_us),
                  "ExposureTime": EXPOSURE_US, "AnalogueGain": GAIN, **extra},
    )
    picam2.configure(config)
    W, H = picam2.camera_configuration()["raw"]["size"]
    forme = (H // 2, W // 2) if opt["somme2x2"] else (H, W)
    vmax = 4 * 4095 if opt["somme2x2"] else 4095
    consignes = {"exposition_consigne_us": EXPOSURE_US, "gain_consigne": GAIN,
                 "duree_image_consigne_us": frame_us, "raw_largeur": W, "raw_hauteur": H}

    taille_mo = np.prod(forme) * 2 / 1e6
    print(f"\nOption {lettre} : {opt['desc']}")
    print(f"  capteur {W} x {H} -> image {forme[1]} x {forme[0]} ({taille_mo:.1f} Mo), "
          f"fps visé = {1e6 / frame_us:.1f}")
    if mode == "enregistrement":
        if opt["ecriture"] == "continue":
            libre = shutil.disk_usage(DOSSIER_SORTIE).free
            debit = taille_mo * 1e6 / frame_us
            print(f"  écriture continue : {debit:.0f} Mo/s requis, "
                  f"{libre / 1e9:.1f} Go libres (≈ {libre / 1e6 / debit / 60:.0f} min)")
        else:
            libre = ram_disponible()
            n_ram = int(FRACTION_RAM * libre / (taille_mo * 1e6)) if libre else None
            print(f"  RAM : {N_MAX[lettre]} images demandées"
                  + (f", {n_ram} possibles avec {libre / 1e9:.1f} Go libres" if n_ram is not None else ""))
        print(f"  sortie : {DOSSIER_SORTIE}")

    picam2.start()
    time.sleep(1)                                         # laisse les réglages se stabiliser

    etat = Etat()
    titre = f"Microscope - option {lettre} ({'enregistrement' if mode == 'enregistrement' else 'apercu'})"
    fenetre = fenetre_possible(titre)
    cam = threading.Thread(target=boucle_camera,
                           args=(picam2, etat, lettre, opt, mode, forme, grossissement, consignes, nom))
    cam.start()
    threading.Thread(target=lire_terminal, args=(etat,), daemon=True).start()

    if mode == "enregistrement":
        print("\nDirect lancé. Entrée (terminal) ou Espace (fenêtre) = démarrer / arrêter l'enregistrement · "
              "q = quitter")
    else:
        print("\nAperçu lancé (rien n'est sauvegardé). q = quitter")
    if fenetre:
        print("Fenêtre : c = contraste auto / fixe · z = zoom 1:1 · q ou Échap = quitter")
    else:
        print("Pas d'affichage graphique disponible : terminal seulement.")

    try:
        if fenetre:
            boucle_fenetre(etat, titre, vmax)
        else:
            boucle_texte(etat)
    except KeyboardInterrupt:
        pass
    finally:
        etat.quitter.set()
        cam.join()                                        # termine et sauvegarde un enregistrement en cours
        picam2.stop()
        print("\nTerminé.")


if __name__ == "__main__":
    main()
