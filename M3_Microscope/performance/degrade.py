"""Dégradation numérique d'une vraie vidéo : profondeur de bits et regroupement de pixels (binning).
Usage : python3 degrade.py <nom>  -> deg_<nom>.pkl"""
import sys, pickle, contextlib, io, numpy as np
import traitement_video_tiff as M
nom = sys.argv[1]
acq = M.charger_acquisition(f"videos/{nom}.tif")
F0, dt, pe0 = acq["frames"], acq["dt"], acq["pixel_eff"]
d0 = pickle.load(open(f"res/{nom}.pkl", "rb"))
out = {}
def run(frames, pe, cle):
    kw, dmax, demi = M.parametres_pixels(pe)
    try:
        s, _, nfit = M.estimer_sigma_psf(frames, demi=demi, **kw)
    except Exception as e:
        out[cle] = dict(err=str(e)); print(cle, "échec", e, flush=True); return
    kw["distance_min"] = max(kw["distance_min"], int(np.ceil(2 * s))); dmax = max(dmax, 1.5 * s)
    with contextlib.redirect_stdout(io.StringIO()):
        det = M.localiser_toutes_frames(frames, s, **kw)
    tab = M.pistes_en_tableaux(M.lier_trajectoires(det, acq["indices"], dist_max_px=dmax, trou_max=M.TROU_MAX), n_min=M.N_MIN)
    r3 = {}
    for L in (2, 3):
        res = M.analyser_pistes(tab, dt, pixel_eff=pe, lag_max=L, t_exp=acq["t_exp"])
        g = [r for r in res if r["raison_immobile"] == "non"]
        f = M.rayon_final(g) if g else None
        r3[L] = dict(a=f["a"] if f else np.nan, inc=f["inc"] if f else np.nan, n=len(g), ntot=len(res),
                     sexp=float(np.nanmedian([r["sigma_exp"] for r in g])) if g else np.nan)
    out[cle] = dict(sigma_psf=s, ndet=float(np.mean([len(x) for x in det])), r=r3, pe=pe)
    print(cle, f"σ={s:.2f}px ndet={out[cle]['ndet']:.1f} a(L2)={r3[2]['a']*1e6:.3f}±{r3[2]['inc']*1e6:.3f} ({r3[2]['n']}/{r3[2]['ntot']}) σexp={r3[2]['sexp']*1e9:.0f} nm", flush=True)
for b in (14, 12, 10, 9, 8, 7, 6):
    q = 2 ** (14 - b)
    run((F0 // q * q).astype(np.uint16), pe0, ("bits", b))
for k in (2, 3, 4):
    h, w = (F0.shape[1] // k) * k, (F0.shape[2] // k) * k
    Fb = F0[:, :h, :w].reshape(len(F0), h // k, k, w // k, k).sum(axis=(2, 4)) / k**2
    run(Fb.astype(np.uint16), pe0 * k, ("bin", k))
pickle.dump(out, open(f"deg_{nom}.pkl", "wb"))
