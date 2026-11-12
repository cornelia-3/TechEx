# Simulation v2 : clones des vraies vidéos où seule la taille des billes change

Chaque vidéo simulée est le « clone » d'une des 6 vraies vidéos du test (inconnu2 p2 à p7 : billes de 0,475 µm, 20X, option C, avant la première image perdue). Tout ce qui vient de l'instrument et de l'échantillon est mesuré sur la vraie vidéo et recopié. Seule la physique qui dépend du rayon a est recalculée.

## Ce qui est repris de la vraie vidéo (mesuré par `mesure_reel.py`)

| Paramètre | Valeur mesurée (6 vidéos) | Ancienne simulation (`sim_perf.py`) |
| --- | --- | --- |
| Fond | carte pixel par pixel de chaque vidéo : 1250 à 1330 comptes, non-uniformité spatiale (MAD) de 25 à 36 comptes, pixels chauds | fond uniforme de 1290 |
| Offset global g(t) | série réelle, écart type de 1,7 à 25 comptes | absent |
| Bruit temporel | 4,8 à 5,5 comptes au fond ; variance = 0,090·(niveau − 1024) + 1,7² | 31 comptes (lecture 28,8) : c'était la non-uniformité du fond, pas du bruit |
| Corrélation du bruit | 0,145 entre lignes voisines, 0,01 entre colonnes | aucune |
| Seuil de détection du prototype | fond + 125 comptes (fixé par la non-uniformité) | fond + 22 comptes |
| PSF au foyer | moyenne de 785 vraies taches recentrées : σ ajusté = 0,63 µm, avec un halo | gaussienne σ0 = √(0,30² + 0,45² + a²/5) µm |
| Défocalisation | disque géométrique de rayon 0,315·\|Δz\| (NA 0,40 dans l'eau) ⊗ PSF au foyer | gaussienne élargie, z_R = 4,8 µm |
| Brillance des billes libres | tirée des 35 pistes libres mesurées : 4·10³ à 5·10⁵ comptes, médiane ≈ 10⁵ | une seule valeur pour toutes |
| Papillotement / photoblanchiment | 2,3 % d'une image à l'autre / −0,5 %/s | absents |
| Billes collées | mêmes positions, même défocalisation, même brillance relative (1 à 6 par vidéo) | absentes |
| Vibration | mouvement commun mesuré sur les billes collées, image par image | absente |
| Dérive (écoulement) | mouvement moyen des billes libres, lissé sur 2 s : jusqu'à 9 µm (p4) | absente |
| Horodatages, exposition, nombre d'images | ceux de la vraie vidéo (88 à 184 images, Δt = 92 ms, 90 ms) | 140 images régulières |
| Image | plein champ, 1014 × 760 | 360 × 360 |

## Étalonné pour que le clone à 0,475 µm reproduise la vraie vidéo (`calib_valide.py`)

- Plan de mise au point au-dessus du fond : 2,1 / 0,9 / 2,4 / 13 / 17 / 12 µm (p2 à p7), déduit de la défocalisation des billes collées.
- Épaisseur de la chambre : 11 µm, déduite de p5 et p6 (billes libres entre 7 et 11 µm du foyer, billes collées à 18 µm). **Hypothèse à confirmer.**
- Nombre de billes dans la chambre : 5 à 12 par champ, ajusté pour retrouver le nombre de détections par image.

## Recalculé pour chaque rayon a (`simv2.py`)

- D0 = kT/(6πηa), avec les constantes du prototype (22 °C, η = 0,95 mPa·s).
- Diffusion 3D avec flou de mouvement (12 sous-pas par image).
- Sédimentation : l_g = kT/(Δm·g).
- Freinage par les deux parois : Faxén pour le déplacement parallèle, Brenner pour le déplacement perpendiculaire.
- Brillance ∝ a^p. On prend p = 2 ; les données ne le contraignent pas.
- Taille propre de la sphère ajoutée à la PSF pour a > 0,475 µm.

Une même graine donne les mêmes billes, aux mêmes positions de départ, avec les mêmes tirages, à toutes les tailles.

## Validation : clones à 0,475 µm contre vraies vidéos, même traitement automatique (L = 3)

| | vraies | clones |
| --- | --- | --- |
| Détections par image | 6,9 à 10,2 | 7,0 à 10,0 |
| σ de la tache (µm) | 0,69 à 2,43 | 0,72 à 2,62 |
| σ_exp (µm) | 0,09 à 0,23 | 0,06 à 0,22 |
| Bruit temporel (comptes) | 4,8 à 5,5 | 4,8 à 5,5 |
| Rayon automatique (µm) | 0,670 ± 0,195 | 0,628 ± 0,093 |

Voir `res/validation_0475.png`. Les clones reproduisent aussi le biais du traitement automatique sur les vraies vidéos (+30 à +40 %). Il vient de la vibration et de la dérive, pas du code.

## Limites connues

- **Épaisseur de la chambre.** Avec 11 µm, la paroi seule biaise déjà le rayon de +17 à +29 % à 0,475 µm dans les clones (vérité sans dérive). C'est plus que le +5 % mesuré avec le protocole complet. Soit la chambre est plus épaisse, soit le tri manuel retire les billes freinées. À trancher avant d'utiliser les clones pour G1.
- **Exposant de brillance p.** Il change directement a_min (seuil de détection).
- **Brillance libre.** Elle est tirée des billes détectées, donc un peu biaisée vers les brillantes.
- **Objets fixes et billes collées.** Ils sont lissés dans la carte de fond (taches floues autour).
- **PSF à 0,475 µm.** Elle est mesurée sur des billes de 0,475 µm. Pour a < 0,475 µm, la tache est donc un peu trop large (≈ 6 %).

## Utilisation (depuis ce dossier, `traitement_video_tiff.py` deux niveaux plus haut)

```
python mesure_reel.py <vraie.tif> res/p4.pkl          # déjà fait : res/*.pkl
python calib_valide.py 0.475 sim/a0.475               # étalonne, génère et analyse les 6 clones
python comparer.py sim/a0.475 res/validation.png
python gen_exemples.py p4 0.1,0.15,0.3,0.475,1,2.5 sim/exemples_p4
python rendu.py p4 sim/exemples_p4 0.475,0.1,0.15,1,2.5 0.6
```

Les clones sont écrits en `.tif` + `.json` au même format que `acquisition_raspberry.py`, donc lisibles tels quels par `traitement_video_tiff.py` et `traitement_video_tiff_extra.py`. Le fichier `.verite.pkl` contient les vraies trajectoires.
