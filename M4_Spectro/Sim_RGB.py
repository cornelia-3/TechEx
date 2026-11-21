"""Simulation du signal perçu par la caméra (sans filtre) - spectromètre 4f + réseau blazé.
Unités internes : µm."""
import numpy as np
import matplotlib.pyplot as plt

# ---------- Paramètres de conception (valeurs de la section IV) ----------
Lam  = 1000/600          # pas du réseau [µm]
f1, f2 = 50e3, 30e3      # focales [µm]
a, b = 10.0, 3000.0      # fente : largeur, hauteur [µm]
D    = 2000.0            # diamètre du faisceau sur le réseau = N*Lam [µm]
p    = 1.55              # taille des pixels [µm]
npx_x, npx_y = 3600, 3040  # fenêtre utilisée du capteur HQ
lamB = 0.55              # longueur d'onde de blaze [µm]
dfmax = 100.0            # défocalisation max. du doublet achromatique [µm] (à 400/700 nm)
lam0 = 0.55
sinL = lam0/(2*Lam)      # Littrow à 550 nm
thL  = np.arcsin(sinL)

# ---------- Modèle ----------
def theta_d(lam):  return np.arcsin(lam/Lam - sinL)
def x_pos(lam):    return f2*np.tan(theta_d(lam) - thL)            # éq. (12) [µm]
def eta(lam):      return np.sinc(1 - lamB/lam)**2                 # efficacité ordre 1
def w_eff(lam):                                                    # largeur de l'image de la fente + chromatisme
    wf = a*f2/f1
    wc = dfmax*np.abs(lam-lam0)/0.15 * D/f2
    return np.sqrt(wf**2 + wc**2)
def dx_r(lam):     return lam*f2/(D*np.cos(thL))                   # demi-largeur de diffraction, éq. (20)

def wl2rgb(lam_um):
    """Longueur d'onde -> RGB (approximation de Bruton)."""
    w = lam_um*1e3
    R = np.zeros_like(w); G = np.zeros_like(w); B = np.zeros_like(w)
    m = (w>=380)&(w<440); R[m] = -(w[m]-440)/60; B[m] = 1
    m = (w>=440)&(w<490); G[m] = (w[m]-440)/50;  B[m] = 1
    m = (w>=490)&(w<510); G[m] = 1;              B[m] = -(w[m]-510)/20
    m = (w>=510)&(w<580); R[m] = (w[m]-510)/70;  G[m] = 1
    m = (w>=580)&(w<645); R[m] = 1;              G[m] = -(w[m]-645)/65
    m = (w>=645)&(w<=780); R[m] = 1
    f = np.ones_like(w)
    m = w<420;  f[m] = 0.3+0.7*(w[m]-380)/40
    m = w>700;  f[m] = 0.3+0.7*(780-w[m])/80
    return np.stack([R, G, B], axis=-1)*f[:, None]

x_px = (np.arange(npx_x) - npx_x/2 + 0.5)*p                        # coordonnées des pixels [µm]
y_px = (np.arange(npx_y) - npx_y/2 + 0.5)*p

def row_signal(lams, weights):
    """Intensité (3 canaux) sur une ligne de pixels : somme incohérente (éq. 17)."""
    out = np.zeros((npx_x, 3))
    offs = np.linspace(-0.5, 0.5, 9)                               # intégration sur l'image de la fente
    for lam, wgt in zip(lams, weights):
        u0 = x_pos(lam)
        prof = np.zeros(npx_x)
        for o in offs:
            prof += np.sinc((x_px - u0 - o*w_eff(lam))/dx_r(lam))**2
        out += (wgt*prof/len(offs))[:, None]*wl2rgb(np.array([lam]))[0]
    return out

def to_image(row, gamma=1/1.8):
    row = row/row.max()
    yprof = 0.5*(np.tanh((b*f2/f1/2 - np.abs(y_px))/30) + 1)        # image de la fente : 1,8 mm de haut
    img = (yprof[:, None, None]*row[None, :, :]).astype(np.float32)
    return np.clip(img, 0, 1)**gamma

# ---------- 1) Source blanche (S(λ)=1) ----------
lams = np.arange(0.400, 0.7001, 0.0001)
row_w = row_signal(lams, eta(lams))
img_w = to_image(row_w)

# ---------- 2) Lasers de calibration (exemple : 405, 532, 650 nm) ----------
las = np.array([0.405, 0.532, 0.650])
row_l = row_signal(las, np.ones(3))
img_l = to_image(row_l)

# ---------- Figure ----------
ext = [x_px[0]/1e3, x_px[-1]/1e3, y_px[-1]/1e3, y_px[0]/1e3]
lg = np.linspace(0.38, 0.72, 2000); xg = x_pos(lg)/1e3; lg_nm = lg*1e3
fig, axs = plt.subplots(2, 1, figsize=(9, 6.6), facecolor="white")
for ax, img, ttl in zip(axs, [img_w, img_l],
        ["(a) Source à large bande, sans filtre",
         "(b) Lasers de calibration (405, 532 et 650 nm, exemple)"]):
    ax.imshow(img, extent=ext, aspect="equal", interpolation="nearest")
    ax.set_title(ttl, fontsize=10, loc="left")
    ax.set_ylabel("$y_2$ (mm)"); ax.set_xlabel("$x_2$ (mm)")
    sec = ax.secondary_xaxis("top", functions=(lambda x: np.interp(x, xg, lg_nm),
                                               lambda l: np.interp(l, lg_nm, xg)))
    sec.set_xlabel("Longueur d'onde (nm)")
    sec.set_xticks(np.arange(400, 701, 50))
plt.tight_layout()
plt.savefig("figure_signal_camera.png", dpi=250)   # enregistré à côté du script
plt.show()  

# ---------- Valeurs pour l'analyse ----------
print("x(400), x(550), x(700) [mm] :", x_pos(np.array([0.4, 0.55, 0.7]))/1e3)
print("Étendue spectre [mm] :", (x_pos(0.7)-x_pos(0.4))/1e3, " -> pixels :", (x_pos(0.7)-x_pos(0.4))/p)
for l in las:                                                       # FWHM mesurée des raies
    k = np.argmin(np.abs(x_px - x_pos(l)))
    seg = row_l[:, :].sum(axis=1)
    lo, hi = max(0, k-40), k+40
    s = seg[lo:hi]; s = s/s.max(); idx = np.where(s >= 0.5)[0]
    fw_um = (idx[-1]-idx[0]+1)*p
    disp = f2/(Lam*np.cos(theta_d(l)))*1e-3                        # µm/nm
    print(f"{l*1e3:.0f} nm : FWHM ≈ {fw_um:.1f} µm = {fw_um/p:.1f} px ≈ {fw_um/disp:.2f} nm")
E = eta(np.array([0.4, 0.55, 0.7])); print("eta :", E)
print("Largeur image fente [µm] :", a*f2/f1, " hauteur [mm] :", b*f2/f1/1e3)