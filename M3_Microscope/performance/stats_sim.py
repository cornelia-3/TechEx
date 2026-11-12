"""Métriques de performance tirées des simulations (sim/<conf>_<a>_<etiquette>.pkl)."""
import glob, pickle, numpy as np
import traitement_video_tiff as M

N_PART = 7


def a_de_D(D):
    return M.kB * M.T / (6 * np.pi * M.ETA * D) * 1e6 if D > 0 else np.nan


def metriques(fichier, L):
    d = pickle.load(open(fichier, "rb"))
    a = d["a"]
    moy, inc, n_ok, cover, ab, Dr = [], [], [], [], [], []
    for r in d["reps"]:
        if r["out"] is None or L not in r["out"]:
            n_ok.append(0); continue
        g = [p for p in r["out"][L] if p["raison"] == "non" and np.isfinite(p["D"])]
        n_ok.append(len(g))
        ab += [p["a"] * 1e6 for p in g]
        Dr += [p["D"] * 1e12 / d["D"] for p in g]
        if not g:
            continue
        f = M.rayon_final([dict(p) for p in g])
        if f is None or not np.isfinite(f["a"]):
            continue
        moy.append(f["a"] * 1e6); inc.append(f["inc"] * 1e6)
        cover.append(abs(f["a"] * 1e6 - a) <= f["inc"] * 1e6)
    moy, inc = np.array(moy), np.array(inc)
    R = len(moy)
    out = dict(a=a, conf=d["conf"], L=L, dt=d["dt"], texp=d["texp"], N=d["N"], R=R, n_rep=len(d["reps"]),
               frac=np.mean(n_ok) / N_PART, D=d["D"])
    if R >= 2:
        out.update(biais=moy.mean() / a - 1, s_biais=moy.std(ddof=1) / a / np.sqrt(R),
                   cv=moy.std(ddof=1) / a, s_cv=moy.std(ddof=1) / a / np.sqrt(2 * (R - 1)),
                   inc_rel=np.median(inc) / a, couverture=np.mean(cover),
                   cv_bille=np.std(np.array(ab) / a) if len(ab) > 2 else np.nan, n_billes=np.mean(n_ok),
                   cv_D=np.std(Dr), cv_pred=np.std(Dr) / np.sqrt(max(np.mean(n_ok), 1)))
    else:
        out.update(biais=np.nan, s_biais=np.nan, cv=np.nan, s_cv=np.nan, inc_rel=np.nan, couverture=np.nan,
                   cv_bille=np.nan, n_billes=np.mean(n_ok), cv_D=np.nan, cv_pred=np.nan)
    return out


def tableau(etiq="base", L=3):
    lignes = []
    for f in sorted(glob.glob(f"sim/*_{etiq}.pkl")):
        lignes.append(metriques(f, L))
    return sorted(lignes, key=lambda x: (x["conf"], x["a"]))


if __name__ == "__main__":
    import sys
    etiq = sys.argv[1] if len(sys.argv) > 1 else "base"
    for L in (1, 2, 3, 4, 5):
        print(f"--- L = {L}")
        for m in tableau(etiq, L):
            print(f"{m['conf']} a={m['a']:6.3f}  R={m['R']:2d}/{m['n_rep']}  biais {100*m['biais']:+6.1f} ± {100*m['s_biais']:4.1f} %  "
                  f"CV vidéo {100*m['cv']:5.1f} %  inc. affichée {100*m['inc_rel']:5.1f} %  couverture {m['couverture']:.2f}  "
                  f"billes/vidéo {m['n_billes']:.1f}  CV bille {100*m['cv_bille']:5.1f} %")
