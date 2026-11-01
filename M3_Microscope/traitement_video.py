import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DOSSIER_VIDEOS = Path(r"M3_Microscope\videos_20261026")
NOM_VIDEO = "20x.mp4"


#############################################################################################################
#                                       CHARGEMENT DES IMAGES
#############################################################################################################

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


def stats_video(frames, fps):
    n, h, w = frames.shape
    return {
        "n_frames": n,
        "largeur": w,
        "hauteur": h,
        "fps": fps,
        "duree_s": n / fps if fps > 0 else float("nan"),
    }


def afficher_frame(frames, indice, fps=None):
    """Affiche la frame `indice` (0 = première) avec une barre d'intensité."""
    n = len(frames)
    if not 0 <= indice < n:
        raise IndexError(f"Indice {indice} hors de [0, {n - 1}]")

    gris = frames[indice]
    titre = f"Frame {indice}/{n - 1}"
    if fps:
        titre += f"  (t = {indice / fps:.3f} s)"

    fig, ax = plt.subplots()
    im = ax.imshow(gris, cmap="gray", vmin=0, vmax=255)
    ax.set_title(titre)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    fig.colorbar(im, ax=ax, label="Intensité (8 bits)")
    plt.show()
    return gris


#############################################################################################################
#                                       DÉTECTION DES PARTICULES
#############################################################################################################





if __name__ == "__main__":
    chemin = DOSSIER_VIDEOS / NOM_VIDEO
    frames, fps = charger_video(chemin)
    s = stats_video(frames, fps)

    print(f"Vidéo            : {chemin}")
    print(f"Nombre de frames : {s['n_frames']}")
    print(f"Résolution       : {s['largeur']} x {s['hauteur']} pixels")
    print(f"Frame rate       : {s['fps']:.2f} fps")
    print(f"Durée            : {s['duree_s']:.2f} s")

    INDICE_FRAME = 120   # à modifier
    gris = afficher_frame(frames, INDICE_FRAME, s["fps"])
    print(f"Intensité : min={gris.min()}, max={gris.max()}, moyenne={gris.mean():.1f}")