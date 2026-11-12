"""Décimation avec et sans retrait de la dérive d'ensemble (moyenne des pas de toutes les billes mobiles)."""
import pickle, sys, numpy as np
import traitement_video_tiff as M

def derive(d, mobiles):
    n = d["info"]["n_creneaux"]
    S = np.zeros((n, 2)); C = np.zeros(n)
    for j in mobiles:
        p = d["pistes"][j]; q = p["pos_px"]
        st = np.diff(q, axis=0); ok = np.isfinite(st[:, 0])
        idx = p["f0"] + 1 + np.flatnonzero(ok)
        S[idx] += st[ok]; C[idx] += 1
    m = np.where(C[:, None] > 0, S / np.maximum(C, 1)[:, None], 0.0)
    corr = np.where(C > 1, C / (C - 1), 1.0)                 # la moyenne de N pas browniens retire 1/N de la variance
    return np.cumsum(m, axis=0), C

out = {}
for nom in sys.argv[1:]:
    d = pickle.load(open(f"res/{nom}.pkl", "rb")); i = d["info"]
    mob = [r["piste"] for r in d["resL"][3] if r["raison"] == "non"]
    dr, C = derive(d, mob)
    Nmoy = np.mean(C[C > 0])
    for mode in ("brut", "derive"):
        for k in (1, 2, 3, 4, 5, 6):
            for L in (2, 3):
                per = []
                for ph in range(k):
                    tab = []
                    for j, p in enumerate(d["pistes"]):
                        q = p["pos_px"] - (dr[p["f0"]:p["f0"] + len(p["pos_px"])] if mode == "derive" else 0)
                        s0 = (ph - p["f0"]) % k
                        pos, inc = q[s0::k], p["inc_px"][s0::k]
                        if np.isfinite(pos[:, 0]).sum() >= max(8, M.N_MIN // k):
                            tab.append(dict(f0=0, pos_px=pos, inc_px=inc, sig_px=np.full(len(pos), np.nan), j=j))
                    if not tab: continue
                    res = M.analyser_pistes(tab, k * i["dt"], pixel_eff=i["pixel_eff"], lag_max=L, t_exp=i["t_exp"])
                    g = [r for r, t in zip(res, tab) if t["j"] in mob]
                    if mode == "derive":   # la dérive d'ensemble retire ~1/N de la diffusion propre : on la rend
                        for r in g: r["D"] *= Nmoy / (Nmoy - 1) if Nmoy > 1.5 else 1
                    f = M.rayon_final(g) if g else None
                    if f and np.isfinite(f["a"]): per.append((f["a"], f["inc"]))
                if per:
                    out[(nom, mode, k, L)] = dict(a=np.mean([x[0] for x in per]), inc=np.mean([x[1] for x in per]),
                                                  dt=k * i["dt"], n=len(mob), drift_um=float(np.ptp(dr, axis=0).max() * i["pixel_eff"] * 1e6))
pickle.dump(out, open("decim2.pkl", "wb"))
for key, v in sorted(out.items()):
    if key[3] == 2: print(key[:3], f"{v['a']*1e6:.3f} ± {v['inc']*1e6:.3f}  dérive totale {v['drift_um']:.1f} µm")
