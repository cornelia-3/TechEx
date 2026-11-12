"""Borne basse : effet de la taille de pixel (optique fixe, 20X NA 0,40) et de la limite de liaison.
Usage : python3 sim_pixel.py <a_um> <px_um> <n_rep>  -> sim_px/<a>_<px>.pkl
Modèle de caméra par pixel de sortie : fond en photons ∝ px² (même éclairement par µm²),
bruit de lecture ∝ px (somme de pixels bruts indépendants), photons de la bille inchangés.
Le champ (≈ 135 µm) est le même pour tous les pixels."""
import sys, json, pickle, time, contextlib, io
from pathlib import Path
import numpy as np
import traitement_video_tiff as M
import sim_perf as SP

a_um, px, n_rep = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
DMAX = (1.47, 2.5, 3.5)
etal0 = json.load(open("etalon.json"))
r = px / (6.2 / 16.566)
etal = dict(etal0)
etal["fond"] = etal0["noir"] + (etal0["fond"] - etal0["noir"]) * r**2
etal["lecture"] = etal0["lecture"] * r
cfg = dict(SP.CONF["20X"], px=px)
taille = int(round(135 / px))
Path("sim_px").mkdir(exist_ok=True)
out = {d: [] for d in DMAX}
info = []
t0 = time.time()
for rep in range(n_rep):
    rng = np.random.default_rng(int(a_um * 1e5) + 1000 * rep + int(px * 1000))
    fr, vrai, D = SP.simuler_video(a_um, cfg, rng, N=140, taille=taille, n_part=7, etal=etal)
    pe = px * 1e-6
    kw, _, demi = M.parametres_pixels(pe)
    try:
        s, _, _ = M.estimer_sigma_psf(fr, demi=demi, **kw)
    except RuntimeError:
        for d in DMAX:
            out[d].append(None)
        info.append(dict(sigma=np.nan, ndet=0)); continue
    kw["distance_min"] = max(kw["distance_min"], int(np.ceil(2 * s)))
    with contextlib.redirect_stdout(io.StringIO()):
        det = M.localiser_toutes_frames(fr, s, **kw)
    info.append(dict(sigma=s, ndet=float(np.mean([len(x) for x in det]))))
    for d in DMAX:
        dmax = max(d / px, 1.5 * s)
        pistes = M.lier_trajectoires(det, None, dist_max_px=dmax, trou_max=M.TROU_MAX)
        tab = M.pistes_en_tableaux(pistes, n_min=M.N_MIN)
        res = M.analyser_pistes(tab, 0.092, pixel_eff=pe, lag_max=2, t_exp=0.090)
        out[d].append([dict(D=q["D"], a=q["a"], inc_psf=q["inc_psf"], inc_stat=q["inc_stat"], raison=q["raison_immobile"])
                       for q in res])
pickle.dump(dict(a=a_um, px=px, D=D, dmax=DMAX, out=out, info=info), open(f"sim_px/{a_um:g}_{px:g}.pkl", "wb"))
for d in DMAX:
    moy, nb = [], []
    for rr in out[d]:
        if not rr:
            nb.append(0); continue
        g = [q for q in rr if q["raison"] == "non" and np.isfinite(q["D"])]
        nb.append(len(g))
        if g:
            Dm = np.mean([q["D"] for q in g])
            if Dm > 0:
                moy.append(M.kB * M.T / (6 * np.pi * M.ETA * Dm) * 1e6)
    print(f"a={a_um:g} px={px:g} dmax={d}: {len(moy)}/{n_rep} vidéos, biais {100*(np.mean(moy)/a_um-1) if moy else np.nan:+.1f} %, "
          f"disp {100*np.std(moy)/a_um if moy else np.nan:.1f} %, pistes/bille {np.mean(nb)/7:.2f}, "
          f"σ tache {np.nanmean([i['sigma'] for i in info]):.2f} px | {time.time()-t0:.0f} s", flush=True)
