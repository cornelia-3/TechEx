"""Vidéos d'exemple : clones d'une vraie vidéo où seule la taille des billes change.
Même graine pour toutes les tailles : mêmes billes, mêmes positions de départ, mêmes tirages aléatoires;
seuls D(a), la sédimentation, la paroi, la brillance ∝ a^p et la taille propre de la sphère changent.

Usage : python3 gen_exemples.py <ref p4|p5…> <tailles séparées par des virgules> <dossier>"""
import sys, json, pickle, subprocess
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
import simv2 as S
from calib_valide import Z_F, HC

ICI = Path(__file__).resolve().parent
VRAIES = ICI.parents[1] / "techex_M3_videos" / "_avant_premier_trou"


def police(t):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(f, t)
        except OSError:
            pass
    return ImageFont.load_default()


def en_8bits(fr, lo, hi):
    return (np.clip((fr.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def etiquette(img8, texte, sous=""):
    im = Image.fromarray(img8).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 44 if sous else 26], fill=(0, 0, 0))
    d.text((8, 3), texte, fill=(255, 220, 120), font=police(19))
    if sous:
        d.text((8, 25), sous, fill=(200, 200, 200), font=police(14))
    return np.asarray(im)


def mp4(images, chemin, fps):
    h, w, _ = images[0].shape
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                          "-crf", "18", "-pix_fmt", "yuv420p", str(chemin)], stdin=subprocess.PIPE)
    for im in images:
        p.stdin.write(np.ascontiguousarray(im).tobytes())
    p.stdin.close(); p.wait()


if __name__ == "__main__":
    k = sys.argv[1]
    tailles = [float(x) for x in sys.argv[2].split(",")]
    out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    ref = pickle.load(open(ICI / "res" / f"{k}.pkl", "rb"))
    N = json.loads((ICI / "res" / f"calib_{k}.json").read_text())["N_col"]
    vraie = tifffile.imread(str(next(VRAIES.glob(f"inconnu2_{k}*_C_20X.tif"))))
    for a in tailles:
        f = out / f"sim_{a:g}um_clone-{k}_C_20X.tif"
        if f.exists():
            continue
        fr, info = S.simuler(a, ref, np.random.default_rng(2026), Z_F[k], N, Hc=HC)
        S.ecrire(fr, ref, info, f)
        print(f"{k} a={a:g} µm : D0={info['D0']:.3f} µm²/s, l_g={info['l_g']:.2f} µm, {info['n_colle']} collées",
              flush=True)
    print("vidéos simulées prêtes", flush=True)
