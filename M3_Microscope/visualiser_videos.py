"""Visualise les vidéos / piles d'images du microscope avec la bonne profondeur de bits.

Formats lus :
  - .mp4 / .h264 : flux H.264 de la Pi (8 bits). Attention : les fichiers de videos_20261026
    sont des flux H.264 BRUTS renommés en .mp4 (aucun conteneur), d'où l'erreur 0xC00D36C4
    dans le lecteur Windows. OpenCV les lit quand même.
  - .tif / .tiff : pile multipage (ex. films de référence Moodle, futures acquisitions).
  - dossier de .png : une image par fichier (ancienne sortie de acquisition_raspberry.py).

Profondeur de bits : uint8 -> 8 bits; uint16 avec max <= 4095 -> 12 bits (raw du capteur HQ
rangé dans un conteneur 16 bits); sinon 16 bits. Une image 12 bits affichée sur l'échelle
16 bits (0-65535) n'occupe que les 6 % du bas de l'échelle : elle paraît noire.

Au lancement : on choisit UNE vidéo et un fps dans le terminal. La vidéo joue ensuite en
boucle (même image à trois échelles + histogramme de toute la vidéo) jusqu'à ce qu'on
ferme la fenêtre. Barre d'espace = pause / reprise.
Bonjour test111
"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

DOSSIER = Path(__file__).resolve().parent / "videos_20261026"
EXTENSIONS = (".mp4", ".h264", ".tif", ".tiff")
FPS_DEFAUT = 30        # proposé dans le terminal (le flux H.264 brut ne contient pas le fps)


#############################################################################################################
#                                       LECTURE
#############################################################################################################

def lire_pile(chemin):
    """Retourne un tableau (n_frames, hauteur, largeur) en niveaux de gris, dtype d'origine."""
    chemin = Path(chemin)
    if chemin.is_dir():
        fichiers = sorted(chemin.glob("*.png")) + sorted(chemin.glob("*.tif"))
        images = [cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in fichiers]
    elif chemin.suffix.lower() in (".tif", ".tiff"):
        ok, images = cv2.imreadmulti(str(chemin), flags=cv2.IMREAD_UNCHANGED)
        if not ok:
            raise RuntimeError(f"Lecture TIFF impossible : {chemin}")
    else:                                            # .mp4, .h264, .avi...
        cap = cv2.VideoCapture(str(chemin))
        if not cap.isOpened():
            raise FileNotFoundError(f"Impossible d'ouvrir : {chemin}")
        images = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            images.append(im)
        cap.release()

    if not images:
        raise RuntimeError(f"Aucune image lue dans {chemin}")
    gris = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im for im in images]
    return np.stack(gris)


def profondeur_bits(pile):
    """(bits du contenu, bits du conteneur)."""
    if pile.dtype == np.uint8:
        return 8, 8
    if pile.dtype == np.uint16:
        return (12 if pile.max() <= 4095 else 16), 16
    raise TypeError(f"dtype non prévu : {pile.dtype}")


#############################################################################################################
#                                       AFFICHAGE
#############################################################################################################

def resume(nom, pile):
    bits, conteneur = profondeur_bits(pile)
    n, h, w = pile.shape
    print(f"\n{nom}")
    print(f"  {n} frames, {w} x {h} px, contenu {bits} bits dans un conteneur {conteneur} bits")
    print(f"  intensité : min={pile.min()}  max={pile.max()}  moyenne={pile.mean():.1f}  "
          f"(échelle complète = {2**bits - 1})")
    print(f"  fraction de l'échelle {conteneur} bits occupée par le max : {pile.max() / (2**conteneur - 1):.1%}")


def afficher(nom, pile, fps):
    """Joue la pile en boucle à `fps` images/s jusqu'à la fermeture de la fenêtre."""
    bits, conteneur = profondeur_bits(pile)
    max_cont, max_bits = 2**conteneur - 1, 2**bits - 1
    bas, haut = np.percentile(pile, [0.5, 99.8])
    if haut <= bas:
        haut = bas + 1

    fig, ax = plt.subplots(1, 4, figsize=(18, 5), gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})
    fig.suptitle(f"{nom} — {fps:g} fps (espace = pause)")
    vues = [
        (0, max_cont, f"Échelle du conteneur : 0–{max_cont}"),
        (0, max_bits, f"Échelle {bits} bits : 0–{max_bits}"),
        (bas, haut, f"Contraste étiré : {bas:.0f}–{haut:.0f}"),
    ]
    ims = []
    for a, (vmin, vmax, titre) in zip(ax[:3], vues):
        ims.append(a.imshow(pile[0], cmap="gray", vmin=vmin, vmax=vmax, animated=True))
        a.set(title=titre, xlabel="x (px)", ylabel="y (px)")
    compteur = ax[0].text(0.02, 0.97, "", transform=ax[0].transAxes, color="yellow",
                          va="top", fontsize=9, animated=True)

    # Histogramme de toute la vidéo (fixe : le recalculer à chaque frame ralentirait la lecture)
    ax[3].hist(pile.ravel(), bins=256, range=(0, max_bits), color="0.3", log=True)
    ax[3].axvline(max_bits, color="r", ls="--", lw=1, label=f"max {bits} bits")
    if conteneur > bits:
        ax[3].text(0.98, 0.95, f"échelle {conteneur} bits\nva jusqu'à {max_cont}",
                   transform=ax[3].transAxes, ha="right", va="top", fontsize=8)
    ax[3].set(title="Histogramme, toute la vidéo (log)", xlabel="intensité", ylabel="nb de pixels")
    ax[3].legend(fontsize=8, loc="center right")
    plt.tight_layout()

    def maj(i):
        for im in ims:
            im.set_data(pile[i])
        compteur.set_text(f"frame {i}/{len(pile) - 1}   t = {i / fps:.2f} s")
        return (*ims, compteur)

    anim = FuncAnimation(fig, maj, frames=len(pile), interval=1000 / fps, blit=True, repeat=True)

    en_pause = [False]
    def touche(evt):
        if evt.key == " ":
            (anim.resume if en_pause[0] else anim.pause)()
            en_pause[0] = not en_pause[0]
    fig.canvas.mpl_connect("key_press_event", touche)
    fig._anim = anim                # garde une référence, sinon l'animation s'arrête
    return fig


#############################################################################################################
#                                       CHOIX DANS LE TERMINAL
#############################################################################################################

def choisir_fichier():
    chemins = sorted(p for p in DOSSIER.iterdir()
                     if p.suffix.lower() in EXTENSIONS or (p.is_dir() and any(p.glob("*.png"))))
    if not chemins:
        raise FileNotFoundError(f"Aucune vidéo dans {DOSSIER}")
    print(f"Vidéos dans {DOSSIER.name} :")
    for k, p in enumerate(chemins, 1):
        print(f"  {k}. {p.name}")
    while True:
        rep = input(f"Numéro de la vidéo (1-{len(chemins)}) : ").strip()
        if rep.isdigit() and 1 <= int(rep) <= len(chemins):
            return chemins[int(rep) - 1]
        print("  Choix invalide.")


def choisir_fps():
    while True:
        rep = input(f"fps de lecture [Entrée = {FPS_DEFAUT}] : ").strip().replace(",", ".")
        if not rep:
            return float(FPS_DEFAUT)
        try:
            fps = float(rep)
            if fps > 0:
                return fps
        except ValueError:
            pass
        print("  Entrer un nombre positif.")


#############################################################################################################
#                                                 MAIN
#############################################################################################################

if __name__ == "__main__":
    chemin = choisir_fichier()
    fps = choisir_fps()
    pile = lire_pile(chemin)
    resume(chemin.name, pile)
    print(f"  lecture en boucle à {fps:g} fps — fermer la fenêtre pour quitter")
    afficher(chemin.name, pile, fps)
    plt.show()
