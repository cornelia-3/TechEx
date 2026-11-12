"""Modèles analytiques : paroi (Faxén) + sédimentation, température, temps de sortie de la profondeur de champ."""
import numpy as np
kB, T, g = 1.380649e-23, 295.15, 9.81
eta = 0.955e-3; drho = 1050 - 997.8          # polystyrène - eau (kg/m³)
def faxen(r):                                 # r = a/h ; D∥/D0 (Schäffer 2007)
    return 1 - 9/16*r + r**3/8 - 45/256*r**4 - r**5/16
def lg(a):                                    # longueur de sédimentation (m)
    return kB*T/(4/3*np.pi*a**3*drho*g)
def biais_paroi(a, zf=None, dof=2e-6, H=100e-6):
    """Rapport a_apparent/a pour les billes vues dans le plan focal zf (centre, hauteur depuis le fond)."""
    h = np.linspace(a*1.02, H - a, 20000)
    p = np.exp(-(h - a)/lg(a))                # Boltzmann
    if zf is None:                            # on focalise là où sont les billes : maximum de densité vue
        zf = a + min(lg(a), H/2)
    w = p*np.exp(-(h - zf)**2/(2*dof**2))     # billes dans la profondeur de champ
    return np.sum(w)/np.sum(w*faxen(a/h)), zf
if __name__ == "__main__":
    for a in (0.05, 0.1, 0.2, 0.475, 1.0, 2.5, 5.0):
        A = a*1e-6
        r_fond, zf = biais_paroi(A, zf=A + 0.0)            # plan focal au contact
        r_lg, zf2 = biais_paroi(A)                          # plan focal à a + l_g
        r10, _ = biais_paroi(A, zf=10e-6)
        print(f"a={a:5.3f} µm  l_g={lg(A)*1e6:9.2f} µm  v_sed={2*drho*g*A**2/(9*eta)*1e6:.4f} µm/s  biais(zf=a+l_g, {zf2*1e6:.1f} µm)={100*(r_lg-1):+.1f} %  biais(zf=10 µm)={100*(r10-1):+.1f} %")
    # température : d ln(T/η)/dT avec Vogel
    A_, B_, C_ = 2.414e-5, 247.8, 140.0
    etaT = lambda TT: A_*10**(B_/(TT - C_))
    for dT in (-2, -1, 1, 2, 3):
        r = (T/etaT(T))/((T+dT)/etaT(T+dT))
        print(f"T réelle = 22{dT:+d} °C -> a_calculé/a = {r:.4f} ({100*(r-1):+.1f} %)")
    print("η(22 °C) Vogel =", etaT(T))
