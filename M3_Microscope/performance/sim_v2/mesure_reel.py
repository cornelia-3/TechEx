"""Mesure, sur une VRAIE vidéo, tous les paramètres dont la simulation v2 a besoin.

Usage : python3 mesure_reel.py <video.tif> <sortie.pkl>

Tout passe par les fonctions du prototype (traitement_video_tiff.py) pour la détection, l'ajustement
gaussien et la liaison, avec les mêmes paramètres que l'analyse de l'équipe. Mesures :
  fond      : carte médiane temporelle (éclairage, motif fixe, pixels chauds), offset global par image g(t)
  bruit     : variance temporelle contre niveau moyen sur les pixels sans bille -> gain et bruit de lecture
  billes    : nombre par image, σ et photons de chaque détection, pistes (prototype), immobiles
  PSF       : pile de vignettes recentrées (PSF empirique), ellipticité
  intensité : photoblanchiment (tendance), papillotement (fluctuation au-delà du bruit de grenaille)
  dérive    : déplacement moyen des billes libres image par image, et mouvement des billes immobiles
"""
import sys, pickle, json, contextlib, io
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # M3_Microscope/traitement_video_tiff.py
import traitement_video_tiff as M

NOIR = 1024.0            # niveau noir de la somme 2x2 (4 × 256)
VIGN = 12                # demi-largeur des vignettes de PSF (px)


def gauss_ell(coords, A, x0, y0, sx, sy, th, b):
    X, Y = coords
    c, s = np.cos(th), np.sin(th)
    xr, yr = c * (X - x0) + s * (Y - y0), -s * (X - x0) + c * (Y - y0)
    return (A * np.exp(-xr**2 / (2 * sx**2) - yr**2 / (2 * sy**2)) + b).ravel()


def main(chemin, sortie):
    chemin = Path(chemin)
    meta = json.loads(chemin.with_suffix(".json").read_text(encoding="utf-8"))
    fr = tifffile.imread(str(chemin)).astype(np.float32)
    n, H, W = fr.shape
    t = (np.asarray(meta["timestamps_ns"], np.int64) - meta["timestamps_ns"][0]) / 1e9
    dt = float(np.median(np.diff(t)))
    texp = float(np.median(meta["exposure_us"])) * 1e-6
    pe = meta["pixel_effectif_capteur_um"] / meta.get("grossissement_etalonne", 16.566) * 1e-6
    kw, dmax, demi = M.parametres_pixels(pe)

    # ---------------- détection / localisation / liaison (prototype) ----------------
    fr16 = fr.astype(np.uint16)
    s_psf, s_mad, n_fit = M.estimer_sigma_psf(fr16, demi=demi, **kw)
    kw["distance_min"] = max(kw["distance_min"], int(np.ceil(2 * s_psf)))
    dmax = max(dmax, 1.5 * s_psf)
    with contextlib.redirect_stdout(io.StringIO()):
        det = M.localiser_toutes_frames(fr16, s_psf, **kw)
    pistes = M.lier_trajectoires(det, None, dist_max_px=dmax, trou_max=M.TROU_MAX)
    tab = M.pistes_en_tableaux(pistes, n_min=M.N_MIN)
    res = M.analyser_pistes(tab, dt, pixel_eff=pe, lag_max=3, t_exp=texp)

    # ---------------- fond : carte sans billes ----------------
    masque_billes = np.zeros((n, H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W]
    for f, d in enumerate(det):
        for x, y, *_ , s in d:
            r = int(4 * max(s, s_psf)) + 3
            y0, y1, x0, x1 = max(0, int(y) - r), min(H, int(y) + r + 1), max(0, int(x) - r), min(W, int(x) + r + 1)
            masque_billes[f, y0:y1, x0:x1] = True
    libre = ~masque_billes
    # offset global par image (fond seulement), puis carte médiane après retrait de l'offset
    g = np.array([np.median(fr[f][libre[f]]) for f in range(n)])
    g = g - np.median(g)
    corr = fr - g[:, None, None]
    tmp = np.where(libre, corr, np.nan)
    carte = np.nanmedian(tmp, axis=0)
    jamais = ~np.isfinite(carte)                     # pixels toujours couverts (billes collées) -> lissage
    carte_lisse = ndi.gaussian_filter(np.where(jamais, np.nanmedian(carte), carte), 20)
    carte = np.where(jamais, carte_lisse, carte)
    del tmp

    # ---------------- bruit : variance temporelle des différences d'images, contre le niveau ----------------
    pix_libre = libre.all(axis=0)
    dif = np.diff(corr, axis=0)                      # retire carte fixe et dérives lentes
    var_pix = 0.5 * np.var(dif[:, pix_libre], axis=0)
    moy_pix = corr[:, pix_libre].mean(axis=0) - NOIR
    # régression var = gain·(moyenne) + lecture² par classes de niveau
    bins = np.percentile(moy_pix, np.linspace(1, 99, 15))
    cls = np.digitize(moy_pix, bins)
    mx = np.array([moy_pix[cls == k].mean() for k in range(1, len(bins)) if np.sum(cls == k) > 50])
    vy = np.array([np.median(var_pix[cls == k]) for k in range(1, len(bins)) if np.sum(cls == k) > 50])
    gain, lect2 = np.polyfit(mx, vy, 1)
    # corrélation spatiale du bruit (pixels voisins)
    d0 = dif[:, :H // 2, :W // 2]
    rho_x = np.corrcoef(d0[:, :, :-1].ravel()[::7], d0[:, :, 1:].ravel()[::7])[0, 1]
    # autocorrélation spatiale 2D du bruit temporel (lags -4..4), sur les pixels sans bille
    lib2 = (libre[1:] & libre[:-1]).astype(np.float32)
    P = 0; Pm = 0
    for k in range(0, n - 1, 3):                     # une image sur 3 suffit
        P = P + np.abs(np.fft.rfft2(np.where(lib2[k] > 0, dif[k], 0.0)))**2
        Pm = Pm + np.abs(np.fft.rfft2(lib2[k]))**2
    ac = np.fft.irfft2(P, s=(H, W)) / np.maximum(np.fft.irfft2(Pm, s=(H, W)), 1)
    ac = np.fft.fftshift(ac)[H // 2 - 4:H // 2 + 5, W // 2 - 4:W // 2 + 5]
    ac = ac / ac[4, 4]
    del lib2, P, Pm
    # pixels chauds : niveau moyen > 6 MAD au-dessus de la carte lissée
    resid = carte - ndi.median_filter(carte, 5)
    mad = 1.4826 * np.median(np.abs(resid))
    chauds = int(np.sum(resid > 6 * mad))
    del dif

    # ---------------- détections : σ, photons, ellipticité, vignettes ----------------
    lignes = []                                      # f, x, y, sigma, A, N_comptes
    vign, vign_s = [], []
    ell = []
    for f, d in enumerate(det):
        img = corr[f] - carte                        # image sans fond
        for x, y, ex, ey, s in d:
            sub, X, Y = M.region_interet(img, x, y, int(np.ceil(3.5 * s)) + 1)
            if sub.shape[0] != sub.shape[1]:
                continue
            Ncnt = float(sub.sum())                  # comptes intégrés (fond retiré)
            A = float(img[int(round(y)), int(round(x))])
            lignes.append((f, x, y, s, A, Ncnt))
            if VIGN <= x < W - VIGN - 1 and VIGN <= y < H - VIGN - 1 and len(vign) < 4000:
                v = img[int(y) - VIGN - 1:int(y) + VIGN + 2, int(x) - VIGN - 1:int(x) + VIGN + 2]
                v = ndi.shift(v, (int(y) - y, int(x) - x), order=3)[1:-1, 1:-1]
                vign.append(v.astype(np.float32)); vign_s.append(s)
            if f % 10 == 0 and len(ell) < 600:
                sub2, X2, Y2 = M.region_interet(img, x, y, int(np.ceil(3 * s)) + 1)
                try:
                    p, _ = curve_fit(gauss_ell, (X2, Y2), sub2.ravel(),
                                     p0=[max(sub2.max(), 1), x, y, s, s, 0.0, 0.0], maxfev=4000)
                    sx, sy = abs(p[3]), abs(p[4])
                    ell.append((s, max(sx, sy), min(sx, sy), p[5] % np.pi))
                except Exception:
                    pass
    lignes = np.array(lignes)

    # ---------------- pistes : intensité dans le temps, immobiles, dérive ----------------
    pis = []
    for r, p in zip(res, tab):
        f0 = p["f0"]
        fr_idx = f0 + np.flatnonzero(np.isfinite(p["pos_px"][:, 0]))
        pos = p["pos_px"][np.isfinite(p["pos_px"][:, 0])]
        # photons de chaque position (plus proche ligne de détection)
        Ns, ss = [], []
        for f, (x, y) in zip(fr_idx, pos):
            L = lignes[lignes[:, 0] == f]
            k = np.argmin((L[:, 1] - x)**2 + (L[:, 2] - y)**2)
            Ns.append(L[k, 5]); ss.append(L[k, 3])
        pis.append(dict(f=fr_idx, pos=pos, N=np.array(Ns), s=np.array(ss), D=r["D"], a=r["a"],
                        immobile=r["immobile"], sigma_exp=r["sigma_exp"]))
    # dérive : déplacement moyen image à image des billes libres présentes aux deux images
    dep = [[] for _ in range(n - 1)]
    dep_imm = [[] for _ in range(n - 1)]
    for p in pis:
        ff, pp = p["f"], p["pos"]
        ok = np.diff(ff) == 1
        for i in np.flatnonzero(ok):
            (dep_imm if p["immobile"] else dep)[ff[i]].append(pp[i + 1] - pp[i])
    moy = lambda L: np.mean(L, axis=0) if len(L) else np.array([np.nan, np.nan])
    v_lib = np.array([moy(L) for L in dep]); n_lib = np.array([len(L) for L in dep])
    v_imm = np.array([moy(L) for L in dep_imm]); n_imm = np.array([len(L) for L in dep_imm])

    out = dict(nom=chemin.stem, n=n, H=H, W=W, dt=dt, texp=texp, pe=pe, t=t, s_psf=s_psf, s_mad=s_mad,
               kw=kw, dmax=dmax, n_det=np.array([len(d) for d in det]), carte=carte.astype(np.float32),
               g=g, gain=gain, lecture=np.sqrt(max(lect2, 0)), var_classes=(mx, vy), rho_x=rho_x, ac=ac,
               chauds=chauds, lignes=lignes, vign=np.array(vign), vign_s=np.array(vign_s), ell=np.array(ell),
               pistes=pis, v_lib=v_lib, n_lib=n_lib, v_imm=v_imm, n_imm=n_imm,
               rayons=[(r["a"], r["immobile"], r["n_pts"]) for r in res])
    pickle.dump(out, open(sortie, "wb"))
    a_lib = [r["a"] for r in res if not r["immobile"]]
    D_lib = [r["D"] for r in res if not r["immobile"]]
    print(f"{chemin.stem}: {n} images, σ_psf={s_psf:.2f} px, {np.mean(out['n_det']):.1f} dét./image, "
          f"{len(res)} pistes ({sum(r['immobile'] for r in res)} immobiles), gain={gain:.3f}, "
          f"lecture={out['lecture']:.1f}, a(D moyen)={M.kB*M.T/(6*np.pi*M.ETA*np.mean(D_lib))*1e6:.3f} µm")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
