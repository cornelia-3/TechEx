"""Simulation v2 : « clone » d'une vraie vidéo du test (billes de 0,475 µm, 20X, option C) où SEULE la taille
des billes change.

Principe : tout ce qui vient de l'instrument et de l'échantillon est MESURÉ sur la vraie vidéo de référence
(mesure_reel.py) et recopié tel quel; seule la physique qui dépend du rayon a est recalculée.

Repris de la vraie vidéo (même distribution, souvent la même série exacte) :
  - fond : carte d'éclairage pixel par pixel (non-uniformité, motif fixe, pixels chauds) + offset global g(t)
  - bruit : loi variance = gain·niveau + lecture² mesurée, corrélation verticale entre pixels voisins (0,145)
  - PSF au foyer : moyenne des vraies taches recentrées (cœur + halo), défocalisation = disque géométrique
  - horodatages, nombre d'images, exposition, taille de pixel
  - dérive : mouvement commun des billes collées (vibration) + écoulement lent des billes libres, image par image
  - billes collées : mêmes positions, même défocalisation, même brillance relative
  - brillance des billes libres : tirée de la distribution mesurée (pistes libres des 6 vidéos)
  - papillotement (≈ 2 % d'une image à l'autre) et photoblanchiment mesurés
  - plan de mise au point et nombre de billes : étalonnés pour que le clone à 0,475 µm reproduise la vraie vidéo

Recalculé pour chaque rayon a (physique) :
  - D0 = kT/(6πηa), diffusion 3D, flou de mouvement pendant l'exposition (12 sous-pas par image)
  - sédimentation (longueur l_g = kT/(Δm g)) et freinage par les parois (Faxén ∥, Brenner ⊥)
  - brillance ∝ a^p (p = 2 par défaut), taille propre de la sphère ajoutée à la PSF pour a > 0,475 µm
"""
import json, pickle
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from scipy.interpolate import UnivariateSpline
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # M3_Microscope/traitement_video_tiff.py
import traitement_video_tiff as M

ICI = Path(__file__).resolve().parent
NOIR, SAT = 1024.0, 16380
A_REF = 0.475                     # µm, billes du test
NA, N_EAU = 0.40, 1.33
TAN_DEF = NA / np.sqrt(N_EAU**2 - NA**2)      # rayon du disque de défocalisation / |Δz|
RHO_PS, RHO_EAU, G = 1050.0, 997.8, 9.81
N_SOUS = 12
OS = 8                            # suréchantillonnage de la PSF (pas de 1/8 px)
C_VERT = 0.0735                   # noyau vertical [c, 1, c] -> corrélation 0,145 entre lignes voisines
FLICKER = 0.023                   # fluctuation relative de brillance d'une image à l'autre (billes collées)
BLANCH = -0.0052                  # 1/s, photoblanchiment médian mesuré (négligeable)


# ------------------------------------------------------------------ PSF empirique
def psf_foyer():
    p = ICI / "res" / "psf_foyer.npy"
    if p.exists():
        return np.load(p)
    V, S = [], []
    for k in ["p2", "p3", "p4", "p5", "p6", "p7"]:
        r = pickle.load(open(ICI / "res" / f"{k}.pkl", "rb"))
        V.append(r["vign"]); S.append(r["vign_s"])
    V, S = np.concatenate(V), np.concatenate(S)
    v = V[S < 1.8]
    v = v / v.sum((1, 2), keepdims=True)
    psf = np.clip(np.median(v, 0), 0, None)          # médiane : robuste aux voisines
    psf /= psf.sum()
    np.save(p, psf)
    return psf


class BiblioPSF:
    """PSF(a, R) = PSF au foyer ⊗ (projection de la sphère si a > A_REF) ⊗ disque de rayon R (défocalisation)."""
    def __init__(self, a_um, pe_um, R_max_px=30.0, pas=0.5):
        base = psf_foyer()
        b = ndi.zoom(base, OS, order=3)
        b = np.clip(b, 0, None); b /= b.sum()
        if a_um > A_REF * 1.05:                       # taille propre de la sphère (épaisseur projetée)
            ae = np.sqrt(a_um**2 - A_REF**2) / pe_um * OS
            h = int(np.ceil(ae)) + 1
            y, x = np.mgrid[-h:h + 1, -h:h + 1]
            sph = np.sqrt(np.clip(ae**2 - x**2 - y**2, 0, None))
            b = fftconvolve(b, sph / sph.sum(), mode="full")
        self.base = b / b.sum()
        self.R = np.arange(0, R_max_px + pas, pas)
        self.cache = {}

    def noyau_os(self, iR):
        if ("os", iR) not in self.cache:
            R = self.R[iR] * OS
            h = int(np.ceil(R)) + 1
            y, x = np.mgrid[-h:h + 1, -h:h + 1]
            disque = np.clip(R + 0.5 - np.hypot(x, y), 0, 1)
            k = fftconvolve(self.base, disque / disque.sum(), mode="full") if R > 0.5 else self.base
            k = np.clip(k, 0, None)
            # recadre à un multiple de OS centré
            c = k.shape[0] // 2
            hp = (k.shape[0] // 2) // OS + 1
            pad = hp * OS + OS
            kk = np.zeros((2 * pad + 1, 2 * pad + 1))
            kk[pad - c:pad - c + k.shape[0], pad - c:pad - c + k.shape[1]] = k
            self.cache[("os", iR)] = (kk / kk.sum(), pad)
        return self.cache[("os", iR)]

    def noyau(self, R_px, fx, fy):
        """Noyau en pixels pour une bille dont le centre est à (entier + fx, entier + fy), fx, fy ∈ [0, 1)."""
        iR = int(np.clip(round(R_px / (self.R[1] - self.R[0])), 0, len(self.R) - 1))
        ix, iy = int(fx * OS) % OS, int(fy * OS) % OS
        cle = (iR, ix, iy)
        if cle not in self.cache:
            kk, pad = self.noyau_os(iR)
            # décale de (ix, iy) sous-pixels puis somme par blocs OS×OS
            k = np.roll(np.roll(kk, iy, 0), ix, 1)
            n = 2 * ((pad - OS) // OS) + 1
            k = k[OS // 2:OS // 2 + n * OS, OS // 2:OS // 2 + n * OS].reshape(n, OS, n, OS).sum((1, 3))
            self.cache[cle] = ((k / k.sum()).astype(np.float32), n // 2)
        return self.cache[cle]


# ------------------------------------------------------------------ physique dépendant de a
def D0_um2(a_um):
    return M.kB * M.T / (6 * np.pi * M.ETA * a_um * 1e-6) * 1e12


def l_g_um(a_um):
    m = 4 / 3 * np.pi * (a_um * 1e-6)**3 * (RHO_PS - RHO_EAU)
    return M.kB * M.T / (m * G) * 1e6


def faxen_par(a, z, Hc):
    """D∥/D0 entre deux parois (superposition des deux corrections de Faxén)."""
    def f(r):
        r = np.clip(r, 0, 0.999)
        return 1 - 9 / 16 * r + r**3 / 8 - 45 / 256 * r**4 - r**5 / 16
    return 1 / (1 / f(a / z) + 1 / f(a / (Hc - z)) - 1)


def brenner_perp(a, z, Hc):
    def f(h):                                         # Bevan & Prieve (2000), h = espace bille-paroi
        h = np.maximum(h, 1e-3 * a)
        return (6 * h**2 + 2 * a * h) / (6 * h**2 + 9 * a * h + 2 * a**2)
    return 1 / (1 / f(z - a) + 1 / f(Hc - a - z) - 1)


# ------------------------------------------------------------------ dérive mesurée
def derives(ref, lissage_s=2.0):
    """Retourne (commun_px (n,2), ecoulement_px (n,2)) : vibration (billes collées) et écoulement lent (libres)."""
    n, dt = ref["n"], ref["dt"]
    vi = np.nan_to_num(ref["v_imm"]); vl = ref["v_lib"]
    commun = np.vstack([[0, 0], np.cumsum(vi, 0)])
    ok = np.isfinite(vl[:, 0])
    rel = np.where(ok[:, None], vl - vi, 0.0)
    cum = np.vstack([[0, 0], np.cumsum(rel, 0)])
    t = np.arange(n) * dt
    ecoul = np.zeros_like(cum)
    for j in range(2):
        nk = max(1, int(t[-1] / lissage_s))
        sp = UnivariateSpline(t, cum[:, j], k=3, s=len(t) * np.var(np.diff(cum[:, j])) * 1.0)
        ecoul[:, j] = sp(t)
    ecoul -= ecoul[0]
    return commun, ecoul


def brillances_libres():
    N = []
    for k in ["p2", "p3", "p4", "p5", "p6", "p7"]:
        r = pickle.load(open(ICI / "res" / f"{k}.pkl", "rb"))
        N += [np.median(p["N"]) for p in r["pistes"] if not p["immobile"] and len(p["f"]) >= 40]
    return np.array(N)


def courbe_sigma_R(lib, pe_um):
    """σ ajusté (gaussienne du prototype) en fonction du rayon de défocalisation R (px), sans bruit."""
    Rs = np.arange(0, 22, 0.5)
    out = []
    for R in Rs:
        k, h = lib.noyau(R, 0, 0)
        img = np.zeros((2 * h + 41, 2 * h + 41)); c = img.shape[0] // 2
        img[c - h:c + h + 1, c - h:c + h + 1] = k * 1e5
        r = M.fit_gauss_local(img + 100, c, c, 100, max(1.5, R / 2))
        out.append(r[4] if r is not None else np.nan)
    return Rs, np.array(out)


# ------------------------------------------------------------------ simulation
def simuler(a_um, ref, rng, z_f, N_col, p=2.0, Hc=20.0, n_images=None, mise_au_point="fixe"):
    n = n_images or ref["n"]
    H, W, dt, texp, pe_m = ref["H"], ref["W"], ref["dt"], ref["texp"], ref["pe"]
    pe = pe_m * 1e6
    lib = BiblioPSF(a_um, pe)
    D0 = D0_um2(a_um); lg = l_g_um(a_um)
    if mise_au_point == "billes":                     # option : mise au point refaite sur la couche de billes
        z_f = min(z_f, a_um + min(lg, Hc / 2))
    # --- billes libres : positions initiales (Boltzmann entre les parois), µm
    marge = 15.0
    Lx, Ly = W * pe + 2 * marge, H * pe + 2 * marge
    zmin, zmax = a_um * 1.001, Hc - a_um * 1.001
    u = rng.uniform(size=N_col)
    if lg < 50 * Hc:
        z = zmin - lg * np.log(1 - u * (1 - np.exp(-(zmax - zmin) / lg)))
    else:
        z = zmin + u * (zmax - zmin)
    pos = np.column_stack([rng.uniform(-marge, W * pe + marge, N_col), rng.uniform(-marge, H * pe + marge, N_col), z])
    Nlib = rng.choice(brillances_libres(), N_col) * (a_um / A_REF)**p
    # --- billes collées de la vraie vidéo (positions, défocalisation, brillance)
    Rs, sR = courbe_sigma_R(BiblioPSF(A_REF, pe), pe)
    ok = np.isfinite(sR)
    colle = []
    for q in ref["pistes"]:
        if q["immobile"] and np.median(q["N"]) > 1e4 and len(q["f"]) > 0.5 * ref["n"]:
            s = np.median(q["s"])
            R = float(np.interp(s, sR[ok], Rs[ok]))
            colle.append((*np.nanmean(q["pos"], 0), R, np.median(q["N"]) * (a_um / A_REF)**p))
    commun, ecoul = derives(ref)
    # --- dynamique (sous-pas)
    h = dt / N_SOUS
    n_exp = max(1, int(round(N_SOUS * texp / dt)))
    carte, g = ref["carte"].astype(np.float64), ref["g"]
    gain, lect2 = ref["gain"], ref["lecture"]**2
    kv = np.array([[C_VERT], [1.0], [C_VERT]]); fac = (kv**2).sum()
    frames = np.empty((n, H, W), np.uint16)
    vrai = np.full((n, N_col, 4), np.nan)              # x, y (px, moyenne pendant l'exposition), z (µm), R (px)
    Nb = Nlib.copy()
    for f in range(n):
        img = np.zeros((H, W))
        acc = np.zeros((N_col, 3))
        for k in range(N_SOUS):
            zz = pos[:, 2]
            Dp, Dz = D0 * faxen_par(a_um, zz, Hc), D0 * brenner_perp(a_um, zz, Hc)
            dDz = (D0 * brenner_perp(a_um, zz + 1e-3, Hc) - Dz) / 1e-3
            pos[:, 0] += rng.normal(0, 1, N_col) * np.sqrt(2 * Dp * h)
            pos[:, 1] += rng.normal(0, 1, N_col) * np.sqrt(2 * Dp * h)
            pos[:, 2] += (-Dz / lg + dDz) * h + rng.normal(0, 1, N_col) * np.sqrt(2 * Dz * h)
            pos[:, 2] = np.where(pos[:, 2] < zmin, 2 * zmin - pos[:, 2], pos[:, 2])
            pos[:, 2] = np.where(pos[:, 2] > zmax, 2 * zmax - pos[:, 2], pos[:, 2])
            pos[:, 2] = np.clip(pos[:, 2], zmin, zmax)
            pos[:, 0] = (pos[:, 0] + marge) % Lx - marge          # champ périodique (hors image)
            pos[:, 1] = (pos[:, 1] + marge) % Ly - marge
            if k < n_exp:
                fr_ = f + (k + 0.5) / N_SOUS                     # instant du sous-pas (en images)
                i0 = min(int(fr_), n - 1)
                dlib = commun[i0] + ecoul[i0]                    # px
                xs = pos[:, 0] / pe + dlib[0]; ys = pos[:, 1] / pe + dlib[1]
                Rp = np.abs(pos[:, 2] - z_f) * TAN_DEF / pe
                acc += np.column_stack([xs, ys, Rp])
                w = Nb / n_exp
                for j in range(N_col):
                    ajouter(img, lib, xs[j], ys[j], Rp[j], w[j])
                for (cx, cy, R, Nc) in colle:
                    ajouter(img, lib, cx + commun[i0][0] - commun[0][0], cy + commun[i0][1] - commun[0][1], R,
                            Nc / n_exp * np.exp(BLANCH * f * dt))
        vrai[f, :, 0:2] = acc[:, :2] / n_exp
        vrai[f, :, 2] = pos[:, 2]
        vrai[f, :, 3] = acc[:, 2] / n_exp
        # brillance : papillotement + photoblanchiment
        Nb = Nlib * np.exp(BLANCH * (f + 1) * dt) * np.exp(rng.normal(0, FLICKER, N_col))
        # image attendue -> bruit mesuré (variance ∝ niveau, corrélé verticalement) -> quantification
        E = carte + g[min(f, len(g) - 1)] + img
        var = np.maximum(gain * (E - NOIR) + lect2, 1.0) / fac
        bruit = ndi.convolve(rng.normal(0, 1, (H, W)) * np.sqrt(var), kv, mode="reflect")
        frames[f] = np.clip(np.round(E + bruit), 0, SAT).astype(np.uint16)
    return frames, dict(a=a_um, D0=D0, l_g=lg, z_f=z_f, N_col=N_col, p=p, Hc=Hc, vrai=vrai,
                        n_colle=len(colle), commun=commun, ecoul=ecoul)


def ajouter(img, lib, x, y, R, N):
    xi, yi = int(np.floor(x)), int(np.floor(y))
    k, hh = lib.noyau(R, x - xi, y - yi)
    H, W = img.shape
    y0, y1, x0, x1 = yi - hh, yi + hh + 1, xi - hh, xi + hh + 1
    if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
        return
    ky0, kx0 = max(0, -y0), max(0, -x0)
    ky1, kx1 = k.shape[0] - max(0, y1 - H), k.shape[1] - max(0, x1 - W)
    img[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] += N * k[ky0:ky1, kx0:kx1]


def ecrire(frames, ref, info, chemin):
    import tifffile
    chemin = Path(chemin)
    tifffile.imwrite(str(chemin), frames)
    t0 = 4_000_000_000_000
    meta = dict(option="C", description="SIMULATION v2 (clone de " + ref["nom"] + ")", grossissement=20,
                largeur_px=int(ref["W"]), hauteur_px=int(ref["H"]), dtype="uint16",
                contenu="somme 2x2 du Bayer BGGR (gris), valeurs 0-16380", pixel_effectif_capteur_um=6.2,
                grossissement_etalonne=round(6.2 / (ref["pe"] * 1e6), 4), exposition_consigne_us=90000,
                duree_image_consigne_us=92000, n_images=int(len(frames)), images_perdues=0,
                timestamps_ns=[int(t0 + x * 1e9) for x in ref["t"][:len(frames)]],
                exposure_us=[int(ref["texp"] * 1e6)] * len(frames), analogue_gain=[1.0] * len(frames),
                simulation=dict(a_um=info["a"], z_f_um=info["z_f"], N_col=info["N_col"], p=info["p"],
                                Hc_um=info["Hc"], l_g_um=info["l_g"], D0_um2_s=info["D0"], ref=ref["nom"]))
    chemin.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    pickle.dump(info, open(chemin.with_suffix(".verite.pkl"), "wb"))
