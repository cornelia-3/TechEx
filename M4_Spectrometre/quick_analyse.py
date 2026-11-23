#!/usr/bin/env python3
"""
PHS3910 M4 - Analyse des fichiers RAW (raw_XXX.npy) acquis par le script Pi.

Dossiers attendus (un sous-dossier par type, créés par l'acquisition) :
  reference/ (ou images/) : sans filtre      mesure/ : avec filtre
  dark/ (optionnel)       : même exposition, sans lumière
  laser/ (optionnel)      : acquisition du laser de calibration (images/)

"""
import glob, os
import numpy as np
import matplotlib.pyplot as plt

# ---- Chemins (à modifier) ----
DOSSIER = r"C:\Users\corne\OneDrive - polymtlus\Documents\GitHub\TechEx\M4_Spectrometre\data\20261120\20261109_180326"
DOSSIER_LASER = None                  # acquisition du laser de calibration (ou None)
LBD_LASER = 532.0                     # nm
# ---- Montage (mêmes valeurs que le notebook) ----
LAMBDA, ORDRE = 1 / 600e3, 1          # pas du réseau (m), ordre
F2, PIXEL = 30e-3, 2 * 1.55e-6        # focale de sortie (m), pixel après binning 2x2 (m)
LBD_C, SENS = 550e-9, 1              # lbd centrale (Littrow), sens de l'axe
# ---- Traitement ----
BLANC, NOIR = 4095, 256               # saturation RAW 12 bits, noir (si pas de dark)
DEMI_H, LISS = 50, 5                  # demi-bande (px binnés), lissage (px)
SEUIL_REF, FRAC = 0.10, 0.50          # T ignorée si I_réf < 10 % du max ; coupure à 50 %
U_CAL_NM = None  # Incertitude-type d’étalonnage en nm, si connue
N_BOOT = 2000
TAUX_MIN = 0.95  # Proportion minimale de tirages où la coupure doit être retrouvée


def lisser(y, n=LISS):
    return np.convolve(y, np.ones(n) / n, mode="same") if n > 1 else y


def fichiers_raw(d, *noms):
    for sous in noms:
        f = sorted(glob.glob(os.path.join(d, sous, "raw_*.npy")))
        if f:
            return f
    return sorted(glob.glob(os.path.join(d, "raw_*.npy")))


def binner(r):
    h, w = r.shape[0] // 2 * 2, r.shape[1] // 2 * 2
    r = r[:h, :w]
    return r[::2, ::2] + r[::2, 1::2] + r[1::2, ::2] + r[1::2, 1::2]


def noir_moyen(dossier_dark):
    f = fichiers_raw(dossier_dark, "dark", "")
    if not f:
        return NOIR
    return np.mean([np.load(x).astype(np.float32) for x in f], axis=0)


def profils(fichiers, noir, bande=None):
    """Profil (moyenne sur la bande de lignes) de chaque image -> (n, W), bande."""
    out = []
    for f in fichiers:
        raw = np.load(f).astype(np.float32)
        if bande is None:  # bande centrée sur la ligne la plus intense
            y0 = int(np.argmax(lisser(binner(np.clip(raw - NOIR, 0, None)).sum(axis=1), 15)))
            bande = (max(0, y0 - DEMI_H), y0 + DEMI_H)
        y1, y2 = bande
        sat = (raw[2 * y1:2 * y2] >= BLANC - 5).mean() * 100
        if sat > 0.1:
            print(f"ATTENTION {os.path.basename(f)} : {sat:.2f} % de photosites saturés")
        out.append(binner(np.clip(raw - noir, 0, None))[y1:y2].mean(axis=0))
    return np.array(out), bande


def centroide(y):
    i = int(np.argmax(y))
    g, d = i, i
    while g > 0 and y[g] > y[i] / 2: g -= 1
    while d < len(y) - 1 and y[d] > y[i] / 2: d += 1
    s = slice(g, d + 1)
    return np.sum(np.arange(len(y))[s] * y[s]) / np.sum(y[s])


def axe_lbd(n, px_ref, lbd_ref_nm):
    """Pixel -> longueur d'onde (nm) : modèle réseau en réflexion + lentille f2, calé sur 1 laser."""
    th_i = np.arcsin(ORDRE * LBD_C / (2 * LAMBDA))
    th_ref = np.arcsin(ORDRE * lbd_ref_nm * 1e-9 / LAMBDA - np.sin(th_i))
    th = np.arctan(np.tan(th_ref) + SENS * (np.arange(n) - px_ref) * PIXEL / F2)
    return LAMBDA * (np.sin(th) + np.sin(th_i)) / ORDRE * 1e9


def coupures(x, T):
    """Renvoie les coupures montante et descendante à FRAC * max(T)."""
    if not np.any(np.isfinite(T)) or np.nanmax(T) <= 0:
        return None, None

    ordre = np.argsort(x)
    x, T = x[ordre], T[ordre]
    s = T - FRAC * np.nanmax(T)
    g, d = s[:-1], s[1:]
    montantes = np.flatnonzero((g < 0) & (d >= 0))
    descendantes = np.flatnonzero((g >= 0) & (d < 0))

    def interpole(i):
        return x[i] - g[i] * (x[i + 1] - x[i]) / (d[i] - g[i])

    return (interpole(montantes[0]) if montantes.size else None,
            interpole(descendantes[-1]) if descendantes.size else None)


def transmission(ref_moy, mes_moy):
    i_ref, i_mes = lisser(ref_moy), lisser(mes_moy)
    valide = i_ref > SEUIL_REF * i_ref.max()
    T = np.full(i_ref.shape, np.nan)
    T[valide] = i_mes[valide] / i_ref[valide]
    return i_ref, i_mes, T


def incertitudes_bootstrap(x, ref, mes):
    """Renvoie (incertitude-type, nombre de coupures retrouvées) pour chaque bord."""
    n_ref, n_mes = len(ref), len(mes)
    if min(n_ref, n_mes) < 2:
        print("Bootstrap indisponible : il faut au moins 2 images par série.")
        return (None, None), (None, None)

    rng = np.random.default_rng(3910)
    valeurs = [[], []]
    for _ in range(N_BOOT):
        ref_b = ref[rng.integers(n_ref, size=n_ref)].mean(axis=0)
        mes_b = mes[rng.integers(n_mes, size=n_mes)].mean(axis=0)
        _, _, T_b = transmission(ref_b, mes_b)
        for serie, c in zip(valeurs, coupures(x, T_b)):
            if c is not None:
                serie.append(c)

    return tuple(
        (np.std(serie, ddof=1) if len(serie) >= TAUX_MIN * N_BOOT else None,
         len(serie))
        for serie in valeurs
    )


def main():
    noir = noir_moyen(DOSSIER)
    f_ref = fichiers_raw(DOSSIER, "reference", "images")
    f_mes = fichiers_raw(DOSSIER, "mesure")
    if not f_ref or not f_mes:
        raise SystemExit("Il faut des raw_*.npy dans reference/ (ou images/) ET mesure/.")
    ref, bande = profils(f_ref, noir)
    mes, _ = profils(f_mes, noir, bande)          # même bande que la référence
    n = ref.shape[1]
    print(f"{len(f_ref)} réf., {len(f_mes)} mesures, bande {bande}, "
          f"noir = {'dark' if isinstance(noir, np.ndarray) else NOIR}")

    # --- Calibration (laser) ---
    if DOSSIER_LASER:
        pl, _ = profils(fichiers_raw(DOSSIER_LASER, "images"), noir_moyen(DOSSIER_LASER))
        px_ref = centroide(lisser(pl.mean(axis=0)))
        x = axe_lbd(n, px_ref, LBD_LASER)
        print(f"Laser {LBD_LASER} nm -> pixel {px_ref:.2f}")
    else:
        x = axe_lbd(n, n / 2, LBD_C * 1e9)
        print("ATTENTION : pas de laser -> lbd = 550 nm supposée au centre du capteur.")
        # --- Transmission et coupures du spectre moyen ---
    i_ref, i_mes, T = transmission(ref.mean(axis=0), mes.mean(axis=0))
    centres = coupures(x, T)

    if U_CAL_NM is None:
        print("Étalonnage : incertitude inconnue. Les barres indiquent la répétabilité seule.")
    elif not np.isfinite(U_CAL_NM) or U_CAL_NM < 0:
        raise ValueError("U_CAL_NM doit être None ou une valeur finie positive ou nulle.")

    estimations = (
        incertitudes_bootstrap(x, ref, mes)
        if any(c is not None for c in centres)
        else ((None, None), (None, None))
    )

    res = []
    for nom, c, (u_stat, n_valides) in zip(
        ("Montante", "Descendante"), centres, estimations
    ):
        if c is None:
            print(f"{nom} : coupure non détectée.")
            continue

        if u_stat is None:
            u = None
            label = f"{nom} : {c:.3f} nm (incertitude indisponible)"
        else:
            u = u_stat if U_CAL_NM is None else np.hypot(u_stat, U_CAL_NM)
            type_u = "répétabilité" if U_CAL_NM is None else "rép. + étal."
            label = f"{nom} : ({c:.3f} ± {u:.3g}) nm ({type_u})"

        detail = "" if n_valides is None else f" | {n_valides}/{N_BOOT} tirages avec coupure"
        print(label + detail)
        res.append((c, u, label))

    # --- Sorties ---
    np.savetxt(
        os.path.join(DOSSIER, "transmission.csv"),
        np.column_stack([x, i_ref, i_mes, T]),
        delimiter=",", header="lbd_nm,I_ref,I_mes,T", comments="",
    )
    plt.figure(figsize=(9, 4.5))
    plt.plot(x, T, "r")
    if res:
        seuil = FRAC * np.nanmax(T)
        for c, u, label in res:
            plt.axvline(c, color="g", ls="--", lw=0.8)
            plt.errorbar(c, seuil, xerr=u, fmt="go", capsize=4, label=label)
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER, "transmission.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()