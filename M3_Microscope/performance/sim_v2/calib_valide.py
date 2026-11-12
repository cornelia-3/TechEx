"""Étalonne chaque clone sur sa vraie vidéo (nombre de billes), génère les clones à un rayon donné,
puis les passe dans le même traitement automatique que les vraies vidéos (mesure_reel.py).

Usage : python3 calib_valide.py <a_um> <dossier_sortie> [videos=p2,p3,...] [graine=0] [p=2]
"""
import sys, json, pickle, contextlib, io
from pathlib import Path
import numpy as np
import simv2 as S
import traitement_video_tiff as M
import mesure_reel

ICI = Path(__file__).resolve().parent
HC = 11.0                                            # épaisseur de la chambre (µm), voir LISEZMOI
Z_F = dict(p2=2.1, p3=0.9, p4=2.4, p5=13.0, p6=17.0, p7=12.0)   # plan focal au-dessus du fond (µm)


def det_libres(frames, ref):
    """Nombre moyen de détections par image (prototype), comme dans la vraie vidéo."""
    kw, dmax, demi = M.parametres_pixels(ref["pe"])
    return np.mean([len(M.detecter_particules(f, **kw)) for f in frames])


def etalonner(ref, k, graine=123):
    """N_col tel que le nombre moyen de détections par image du clone à 0,475 µm égale celui de la vraie vidéo.
    Détections = (billes collées et taches fixes, mesurées avec N_col = 0) + c·N_col, c moyenné sur 6 tirages."""
    f_calib = ICI / "res" / f"calib_{k}.json"
    if f_calib.exists():
        return json.loads(f_calib.read_text())["N_col"]
    cible = float(np.mean(ref["n_det"]))
    fr, _ = S.simuler(S.A_REF, ref, np.random.default_rng(graine), Z_F[k], 0, Hc=HC, n_images=4)
    d0 = det_libres(fr, ref)
    N0, ms = 12, []
    for it in range(6):
        fr, _ = S.simuler(S.A_REF, ref, np.random.default_rng(graine + 1 + it), Z_F[k], N0, Hc=HC, n_images=12)
        ms.append(det_libres(fr, ref))
    c = (np.mean(ms) - d0) / N0
    N = int(max(1, round((cible - d0) / max(c, 1e-3))))
    print(f"  {k}: fixes {d0:.1f}, {c:.3f} détection par bille libre -> N_col = {N} (vraie : {cible:.1f} dét./image)", flush=True)
    f_calib.write_text(json.dumps(dict(N_col=N, z_f=Z_F[k], Hc=HC, d0=d0, c=c)))
    return N


if __name__ == "__main__":
    a = float(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    vids = sys.argv[3].split(",") if len(sys.argv) > 3 else ["p2", "p3", "p4", "p5", "p6", "p7"]
    graine = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    p = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
    for k in vids:
        ref = pickle.load(open(ICI / "res" / f"{k}.pkl", "rb"))
        N = etalonner(ref, k)
        rng = np.random.default_rng(1000 * graine + int(k[1]))
        fr, info = S.simuler(a, ref, rng, Z_F[k], N, p=p, Hc=HC)
        nom = out / f"sim_{a:g}um_clone-{k}_g{graine}_C_20X.tif"
        S.ecrire(fr, ref, info, nom)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            try:
                mesure_reel.main(nom, nom.with_suffix(".mes.pkl"))
            except Exception as e:
                print("échec analyse", e)
        print(f"a={a:g} {k}: " + buf.getvalue().strip().split("\n")[-1], flush=True)
