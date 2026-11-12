"""Simulation Monte-Carlo de vidéos du microscope, traitées par le code du prototype (traitement_video_tiff.py).

Modèle physique (une bille sphérique fluorescente dans l'eau, 22 °C) :
  - diffusion 3D libre, D = kT / (6πηa) (Stokes-Einstein), aucune paroi;
  - flou de mouvement : la position est intégrée pendant l'exposition (sous-pas de dt / N_SOUS);
  - image d'une bille = gaussienne d'écart type σ(z) = σ0·sqrt(1 + (z/z_R)²), σ0² = σ_diff² + a²/5
    (tache de diffraction + projection d'une sphère uniforme), photons totaux conservés hors foyer;
  - photons totaux ∝ a^p (p = 2 par défaut), étalonnés sur les billes du test réel (a = 0,475 µm, 20X);
  - bruit : grenaille (gain en comptes/photoélectron) + bruit de lecture + fond mesurés; saturation 16 380.
Usage : python3 sim_perf.py <config: 20X|40X> <a_um> <n_rep> [L] [dt_s] [texp_s] [N_images] [etiquette]
Sortie : sim/<config>_<a>_<etiquette>.pkl"""
import sys, pickle, time
from pathlib import Path
import numpy as np
ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import traitement_video_tiff as M

CONF = {  # px : pixel dans le plan objet (µm); NA; σ_diff ≈ 0.21 λ/NA (λ = 0.58 µm); z_R ≈ n λ / NA²
    "20X": dict(px=6.2 / 16.566, NA=0.40, lam=0.58, n=1.33, s_ab=0.45),
    "40X": dict(px=6.2 / 33.507, NA=0.65, lam=0.58, n=1.33, s_ab=0.45),
}
CAM = dict(fond=1143.0, lecture=7.4, gain=0.5, sat=16380)   # remplacé par les valeurs mesurées (etalon.json)
N_SOUS = 12


def simuler_video(a_um, cfg, rng, N=140, taille=360, n_part=7, dt=0.092, texp=0.090, etal=None):
    D = M.kB * M.T / (6 * np.pi * M.ETA * a_um * 1e-6) * 1e12            # µm²/s
    px = cfg["px"]
    s_diff = 0.21 * cfg["lam"] / cfg["NA"]
    s0 = np.sqrt(s_diff**2 + cfg.get("s_ab", 0)**2 + a_um**2 / 5) / px   # s_ab : aberrations mesurées                            # px
    zR = cfg["n"] * cfg["lam"] / cfg["NA"]**2                              # µm
    # photons totaux de la bille au foyer (étalonnés), plafonnés par la saturation plus bas
    Ntot = etal["N_ref"] * (a_um / etal["a_ref"])**etal.get("p", 2) * (cfg["NA"] / etal["NA_ref"])**2 * (texp / 0.090)   # photons ∝ exposition
    fond, lect, g = etal["fond"], etal["lecture"], etal["gain"]
    B = (fond - etal["noir"]) / g                                           # photons de fond par pixel
    n_exp = max(1, int(round(N_SOUS * texp / dt)))
    h = dt / N_SOUS
    pos = np.column_stack([rng.uniform(50, taille - 50, n_part), rng.uniform(50, taille - 50, n_part),
                           rng.uniform(-1.2 * zR, 1.2 * zR, n_part) / px])           # px (z aussi en px)
    pas = np.sqrt(2 * D * h) / px
    Y, X = np.mgrid[0:taille, 0:taille]
    frames = np.empty((N, taille, taille), np.uint16)
    vrai = np.zeros((N, n_part, 3))
    for f in range(N):
        img = np.full((taille, taille), B)
        acc = np.zeros((n_part, 3))
        for k in range(N_SOUS):
            pos += rng.normal(0, pas, pos.shape)
            if k < n_exp:
                acc += pos
                for j in range(n_part):
                    x, y, z = pos[j]
                    s = s0 * np.sqrt(1 + (z * px / zR)**2)
                    r = int(np.ceil(4 * s)) + 1
                    xi, yi = int(round(x)), int(round(y))
                    x0, x1, y0, y1 = max(0, xi - r), min(taille, xi + r + 1), max(0, yi - r), min(taille, yi + r + 1)
                    if x0 >= x1 or y0 >= y1:
                        continue
                    sx, sy = X[y0:y1, x0:x1] - x, Y[y0:y1, x0:x1] - y
                    img[y0:y1, x0:x1] += Ntot / n_exp / (2 * np.pi * s**2) * np.exp(-(sx**2 + sy**2) / (2 * s**2))
        vrai[f] = acc / n_exp
        signal = g * rng.poisson(img)                       # photoélectrons -> comptes
        im = etal["noir"] + signal + rng.normal(0, lect, img.shape)
        frames[f] = np.clip(np.round(im), 0, CAM["sat"]).astype(np.uint16)
    return frames, vrai, D


def traiter(frames, dt, texp, px_um, L):
    pe = px_um * 1e-6
    kw, dmax, demi = M.parametres_pixels(pe)
    try:
        s, _, n_fit = M.estimer_sigma_psf(frames, demi=demi, **kw)
    except RuntimeError:
        return None, None
    kw["distance_min"] = max(kw["distance_min"], int(np.ceil(2 * s)))
    dmax = max(dmax, 1.5 * s)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        det = M.localiser_toutes_frames(frames, s, **kw)
    pistes = M.lier_trajectoires(det, None, dist_max_px=dmax, trou_max=M.TROU_MAX)
    tab = M.pistes_en_tableaux(pistes, n_min=M.N_MIN)
    out = {}
    for l in (L if isinstance(L, (list, tuple)) else [L]):
        res = M.analyser_pistes(tab, dt, pixel_eff=pe, lag_max=l, t_exp=texp)
        out[l] = [dict(D=r["D"], a=r["a"], inc=r["inc"], inc_psf=r["inc_psf"], inc_stat=r["inc_stat"],
                       sigma_exp=r["sigma_exp"], n_pts=r["n_pts"], raison=r["raison_immobile"]) for r in res]
    return out, dict(sigma_psf=s, n_det=float(np.mean([len(d) for d in det])),
                     longueurs=[len(p["frames"]) for p in pistes])


if __name__ == "__main__":
    import json
    conf, a_um, n_rep = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    L = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [2, 3, 4, 5]
    dt = float(sys.argv[5]) if len(sys.argv) > 5 else 0.092
    texp = float(sys.argv[6]) if len(sys.argv) > 6 else 0.090
    N = int(sys.argv[7]) if len(sys.argv) > 7 else 140
    etiq = sys.argv[8] if len(sys.argv) > 8 else "base"
    etal = json.load(open(ICI / "etalon.json"))
    cfg = CONF[conf]
    (ICI / "sim").mkdir(exist_ok=True)
    reps = []
    t0 = time.time()
    for rep in range(n_rep):
        rng = np.random.default_rng(int(a_um * 1e5) + 1000 * rep + (7 if conf == "40X" else 0))
        frames, vrai, D = simuler_video(a_um, cfg, rng, N=N, dt=dt, texp=texp, etal=etal)
        out, info = traiter(frames, dt, texp, cfg["px"], L)
        reps.append(dict(out=out, info=info))
    pickle.dump(dict(conf=conf, a=a_um, D=D, L=L, dt=dt, texp=texp, N=N, reps=reps),
                open(ICI / "sim" / f"{conf}_{a_um:g}_{etiq}.pkl", "wb"))
    # résumé rapide
    for l in L:
        moy = []
        for r in reps:
            if r["out"] is None:
                continue
            g = [p for p in r["out"][l] if p["raison"] == "non"]
            if g:
                Dm = np.mean([p["D"] for p in g])
                moy.append(M.kB * M.T / (6 * np.pi * M.ETA * Dm) * 1e6 if Dm > 0 else np.nan)
        moy = np.array(moy)
        print(f"{conf} a={a_um:g} L={l}: {np.sum(np.isfinite(moy))}/{n_rep} vidéos, a_mes = {np.nanmean(moy):.4f} "
              f"(biais {100*(np.nanmean(moy)/a_um-1):+.1f} %, disp. {100*np.nanstd(moy)/a_um:.1f} %) "
              f"| {time.time()-t0:.0f} s", flush=True)
