"""Compare les vraies vidéos du test et leurs clones simulés à 0,475 µm (même traitement automatique).
Usage : python3 comparer.py <dossier_clones> <figure.png>"""
import sys, pickle, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # M3_Microscope/traitement_video_tiff.py
import traitement_video_tiff as M

VIDS = ["p2", "p3", "p4", "p5", "p6", "p7"]


def resume(r):
    lib = [q for q in r["pistes"] if not q["immobile"]]
    imm = [q for q in r["pistes"] if q["immobile"]]
    D = [q["D"] for q in lib if np.isfinite(q["D"])]
    a = M.kB * M.T / (6 * np.pi * M.ETA * np.mean(D)) * 1e6 if D and np.mean(D) > 0 else np.nan
    s_det = r["lignes"][:, 3] * r["pe"] * 1e6 if len(r["lignes"]) else np.array([np.nan])
    N_lib = [np.median(q["N"]) for q in lib]
    sexp = [q["sigma_exp"] * 1e6 for q in lib if np.isfinite(q["sigma_exp"])]
    return dict(det=np.mean(r["n_det"]), s_psf=r["s_psf"] * r["pe"] * 1e6, n_lib=len(lib), n_imm=len(imm),
                long=np.median([len(q["f"]) for q in lib]) if lib else np.nan, a=a,
                gain=r["gain"], lect=r["lecture"], bruit_t=np.sqrt(np.median(r["var_classes"][1])),
                s_det=s_det, N_lib=np.array(N_lib), sexp=np.median(sexp) if sexp else np.nan,
                a_pistes=np.array([q["a"] * 1e6 for q in lib]))


if __name__ == "__main__":
    dossier, fig_out = Path(sys.argv[1]), sys.argv[2]
    R, S = {}, {}
    for k in VIDS:
        R[k] = resume(pickle.load(open(f"res/{k}.pkl", "rb")))
        f = glob.glob(str(dossier / f"*clone-{k}_*.mes.pkl"))
        if f:
            S[k] = resume(pickle.load(open(f[0], "rb")))
    print(f"{'':4}{'dét./image':>14}{'σ tache (µm)':>16}{'pistes libres':>16}{'immobiles':>12}{'long. méd.':>14}"
          f"{'σ_exp (µm)':>14}{'bruit t (cpt)':>16}{'a auto (µm)':>14}")
    for k in VIDS:
        r, s = R[k], S.get(k)
        if s is None:
            continue
        f = lambda c, fmt: f"{fmt.format(r[c])} / {fmt.format(s[c])}"
        print(f"{k:4}{f('det','{:.1f}'):>14}{f('s_psf','{:.2f}'):>16}{f('n_lib','{:d}'):>16}{f('n_imm','{:d}'):>12}"
              f"{f('long','{:.0f}'):>14}{f('sexp','{:.2f}'):>14}{f('bruit_t','{:.1f}'):>16}{f('a','{:.3f}'):>14}")
    ks = [k for k in VIDS if k in S]
    ar, as_ = np.array([R[k]["a"] for k in ks]), np.array([S[k]["a"] for k in ks])
    print(f"moyenne des rayons auto : vraies {np.mean(ar):.3f} ± {np.std(ar, ddof=1):.3f} µm, "
          f"clones {np.mean(as_):.3f} ± {np.std(as_, ddof=1):.3f} µm (vrai rayon des clones : 0,475 µm)")

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    sd_r = np.concatenate([R[k]["s_det"] for k in ks]); sd_s = np.concatenate([S[k]["s_det"] for k in ks])
    b = np.linspace(0.4, 4.5, 30)
    ax[0].hist(sd_r, b, density=True, alpha=.5, label="vraies"); ax[0].hist(sd_s, b, density=True, alpha=.5, label="clones")
    ax[0].set_xlabel("σ de la tache ajustée (µm)"); ax[0].legend(); ax[0].set_title("Défocalisation")
    nr = np.concatenate([R[k]["N_lib"] for k in ks]); ns = np.concatenate([S[k]["N_lib"] for k in ks])
    b = np.logspace(3, 6.5, 25)
    ax[1].hist(nr, b, alpha=.5, label="vraies"); ax[1].hist(ns, b, alpha=.5, label="clones"); ax[1].set_xscale("log")
    ax[1].set_xlabel("brillance d'une bille libre (comptes)"); ax[1].set_title("Brillance")
    x = np.arange(len(ks))
    ax[2].bar(x - .2, [R[k]["det"] for k in ks], .4, label="vraies"); ax[2].bar(x + .2, [S[k]["det"] for k in ks], .4, label="clones")
    ax[2].set_xticks(x, ks); ax[2].set_ylabel("détections par image"); ax[2].set_title("Densité")
    ax[3].plot(x - .1, ar, "o", label="vraies"); ax[3].plot(x + .1, as_, "s", label="clones")
    ax[3].axhline(0.475, color="k", lw=.8, ls="--"); ax[3].set_xticks(x, ks); ax[3].set_ylabel("rayon auto, L = 3 (µm)")
    ax[3].set_title("Rayon (même traitement)"); ax[3].legend()
    plt.tight_layout(); plt.savefig(fig_out, dpi=110)
