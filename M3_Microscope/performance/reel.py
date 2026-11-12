"""Analyse automatique (code de base) d'une vidéo réelle, pour les graphiques de performance.
Usage : python3 reel.py <nom sans .tif> [fps si pas de json]
Sortie : res/<nom>.pkl  (info vidéo, pistes localisées, résultats par piste pour L = 2..5, SNR)"""
import sys, time, pickle
from pathlib import Path
import numpy as np
ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import traitement_video_tiff as M

nom = sys.argv[1]
fps = float(sys.argv[2]) if len(sys.argv) > 2 else None
(ICI / "res").mkdir(exist_ok=True)
t0 = time.time()
acq = M.charger_acquisition(ICI / "videos" / f"{nom}.tif", fps)
frames, dt, pe = acq["frames"], acq["dt"], acq["pixel_eff"]
kw, dmax, demi = M.parametres_pixels(pe)
fond_retire = False
try:
    s, s_mad, n_fit = M.estimer_sigma_psf(frames, demi=demi, **kw)
    if n_fit < 5:
        raise RuntimeError
except RuntimeError:
    M.retirer_fond_lent(frames, 2 * demi + 1)
    fond_retire = True
    s, s_mad, n_fit = M.estimer_sigma_psf(frames, demi=demi, **kw)
kw["distance_min"] = max(kw["distance_min"], int(np.ceil(2 * s)))
dmax = max(dmax, 1.5 * s)
t1 = time.time()
det = M.localiser_toutes_frames(frames, s, **kw)
t_loc = time.time() - t1
pistes = M.lier_trajectoires(det, acq["indices"], dist_max_px=dmax, trou_max=M.TROU_MAX)
tab = M.pistes_en_tableaux(pistes, n_min=M.N_MIN)
longueurs = [len(p["frames"]) for p in pistes]          # toutes les pistes, même courtes

# Signal et bruit à chaque position suivie
med = np.array([np.median(f[::4, ::4]) for f in frames], float)
bruit = np.array([1.4826 * np.median(np.abs(f[::4, ::4].astype(float) - m)) for f, m in zip(frames, med)])
sat = float(np.mean([np.mean(f >= 16380) for f in frames[::10]]))
snr = []
for p in tab:
    amps = []
    for k, (x, y) in enumerate(p["pos_px"]):
        if not np.isfinite(x):
            continue
        fi = p["f0"] + k
        i = int(np.searchsorted(acq["indices"], fi))
        if i >= len(frames) or acq["indices"][i] != fi:
            continue
        xi, yi = int(round(x)), int(round(y))
        sub = frames[i][max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2].astype(float)
        if sub.size:
            amps.append((sub.max() - med[i]) / max(bruit[i], 1))
    snr.append(np.median(amps) if amps else np.nan)

resL = {}
for L in (1, 2, 3, 4, 5):
    res = M.analyser_pistes(tab, dt, pixel_eff=pe, lag_max=L, t_exp=acq["t_exp"])
    resL[L] = [dict(piste=r["piste"], D=r["D"], a=r["a"], inc=r["inc"], inc_psf=r["inc_psf"], inc_stat=r["inc_stat"],
                    sigma_exp=r["sigma_exp"], n_pts=r["n_pts"], immobile=r["immobile"],
                    raison=r["raison_immobile"], sig_px=float(np.nanmedian(r["sig_px"])), snr=snr[i],
                    x0=float(np.nanmedian(r["pos_px"][:, 0])), y0=float(np.nanmedian(r["pos_px"][:, 1])))
               for i, r in enumerate(res)]

info = dict(nom=nom, n_images=len(frames), n_creneaux=int(acq["indices"][-1]) + 1, dt=dt, pixel_eff=pe,
            t_exp=acq["t_exp"], M=acq["grossissement_reel"], sigma_psf=s, sigma_psf_mad=s_mad, n_fit=n_fit,
            fond_retire=fond_retire, fond=float(np.median(med)), bruit=float(np.median(bruit)), saturation=sat,
            t_loc_par_image=t_loc / len(frames), h=frames.shape[1], w=frames.shape[2],
            n_det_moy=float(np.mean([len(d) for d in det])), longueurs=longueurs, duree=time.time() - t0,
            source=acq["source"])
pistes_light = [dict(f0=p["f0"], pos_px=p["pos_px"], inc_px=p["inc_px"]) for p in tab]
pickle.dump(dict(info=info, resL=resL, pistes=pistes_light, indices=acq["indices"]),
            open(ICI / "res" / f"{nom}.pkl", "wb"))
gard = [r for r in resL[3] if r["raison"] == "non"]
f = M.rayon_final([dict(r, a=r["a"]) for r in gard]) if gard else None
print(f"{nom}: {len(frames)} im, M={info['M']:.2f}, px={pe*1e9:.0f} nm, σ_PSF={s:.2f} px ({s*pe*1e9:.0f} nm), "
      f"{len(tab)} pistes ({len(gard)} mobiles), SNR méd {np.nanmedian(snr):.1f}, "
      f"a(L=3)={f['a']*1e6 if f else np.nan:.3f}±{f['inc']*1e6 if f else np.nan:.3f} µm χ²={f['chi2_red'] if f else np.nan:.1f}"
      f" | loc {t_loc/len(frames)*1e3:.0f} ms/im, total {time.time()-t0:.0f} s", flush=True)
