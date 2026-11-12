"""MP4 des exemples : mosaïque (vraie vidéo + clones à plusieurs tailles, même contraste, même région)
et comparaison plein champ vraie / clone.
Usage : python3 rendu.py <ref> <dossier_exemples> <tailles> <x0,y0,l,h de la région>"""
import sys, json
from pathlib import Path
import numpy as np
import tifffile
from gen_exemples import VRAIES, en_8bits, etiquette, mp4

if __name__ == "__main__":
    k, dos = sys.argv[1], Path(sys.argv[2])
    tailles = [float(x) for x in sys.argv[3].split(",")]
    echelle = float(sys.argv[4])                                  # réduction de la mosaïque (plein champ)
    import cv2
    chemin_vraie = next(VRAIES.glob(f"inconnu2_{k}*_C_20X.tif"))
    vraie = tifffile.imread(str(chemin_vraie))
    fps = json.loads(chemin_vraie.with_suffix(".json").read_text())["fps_reel"]
    lo, hi = np.percentile(vraie[len(vraie) // 2], (0.5, 99.95))         # contraste de la VRAIE vidéo, pour tous
    sims = {a: tifffile.imread(str(dos / f"sim_{a:g}um_clone-{k}_C_20X.tif")) for a in tailles}
    n = min(len(vraie), *(len(s) for s in sims.values()))
    red = lambda im: cv2.resize(im, None, fx=echelle, fy=echelle, interpolation=cv2.INTER_AREA)
    noms = [("Vraie vidéo " + k, "billes de 0,475 µm (test du 9 nov.)", vraie)] + \
           [(f"Simulation : a = {a:g} µm".replace(".", ","), f"clone de {k}, seule la taille change", sims[a])
            for a in tailles]
    images = []
    for f in range(n):
        tuiles = [etiquette(red(en_8bits(v[f], lo, hi)), t, s) for t, s, v in noms]
        while len(tuiles) % 3:
            tuiles.append(np.zeros_like(tuiles[0]))
        lignes = [np.hstack(tuiles[i:i + 3] + []) for i in range(0, len(tuiles), 3)]
        sep = np.full((4, lignes[0].shape[1], 3), 40, np.uint8)
        im = np.vstack(sum([[L, sep] for L in lignes], [])[:-1])
        im = np.pad(im, ((0, im.shape[0] % 2), (0, im.shape[1] % 2), (0, 0)))
        images.append(im)
    mp4(images, dos / f"mosaique_{k}_tailles.mp4", fps)
    # plein champ : vraie | clone 0,475
    if 0.475 in sims:
        pc = []
        for f in range(n):
            g = etiquette(en_8bits(vraie[f], lo, hi), "Vraie vidéo " + k, "billes de 0,475 µm")
            d = etiquette(en_8bits(sims[0.475][f], lo, hi), "Simulation : a = 0,475 µm", f"clone de {k}")
            im = np.hstack([g, np.full((g.shape[0], 6, 3), 40, np.uint8), d])
            pc.append(np.pad(im, ((0, im.shape[0] % 2), (0, im.shape[1] % 2), (0, 0))))
        mp4(pc, dos / f"plein-champ_{k}_vraie-vs-clone.mp4", fps)
    print("ok", n, "images")
