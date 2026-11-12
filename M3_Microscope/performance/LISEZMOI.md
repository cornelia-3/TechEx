# Tests de performance — brochure du mandat 3

Scripts qui produisent les graphiques de performance (brouillons G1 à G8).
À placer à côté de `traitement_video_tiff.py`, qu'ils importent.

| Script | Rôle |
| --- | --- |
| `reel.py <nom> [fps]` | Analyse automatique d'une vraie vidéo (code de base), L = 1 à 5, rapport signal/bruit, sortie `res/<nom>.pkl` |
| `sim_perf.py <20X|40X> <a_um> <n_vidéos> [L] [dt] [t_exp] [N] [étiquette]` | Vidéos simulées (flou, défocalisation, bruit étalonné) traitées par le code du prototype → `sim/` |
| `stats_sim.py [étiquette]` | Biais, dispersion, couverture de l'incertitude, pistes par bille |
| `decim2.py <noms…>` | Décimation temporelle des vraies pistes, avec et sans retrait de la dérive d'ensemble |
| `degrade.py <nom>` | Réduction de bits et regroupement de pixels sur une vraie vidéo |
| `modeles.py` | Paroi (Faxén) + sédimentation, sensibilité à la température |
| `figures.py` | Toutes les figures dans `fig/` |
| `etalon.json` | Bruit de caméra et luminosité étalonnés sur les vidéos du test réel |

Les vidéos sont lues dans `videos/` (liens vers techex_M3_videos, renommés `…_C_20X.tif`).
