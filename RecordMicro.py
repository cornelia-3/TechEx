# Vous devez installer les librairies suivantes :
# PIP :  pip install sounddevice scipy numpy matplotlib
# Anaconda : conda install -c conda-forge python-sounddevice scipy numpy matplotlib
#

import queue
import time
import threading
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

# --------------------------------------------------------------------------
# Pour avoir un joli affichage
# --------------------------------------------------------------------------

verrou_affichage = threading.Lock()

def print_sync(*args, **kwargs):
    """Print thread-safe : évite que les lignes de threads différents se mélangent."""
    with verrou_affichage:
        print(*args, **kwargs, flush=True)

# ---------------------------------------------------------------------------
# Paramètres globaux
# ---------------------------------------------------------------------------

FREQ_S = 44100          # Fréquence d'échantillonnage
DUREE_CARTO = 3         # Durée (s) d'enregistrement pour la cartographie
DUREE_IMPACT = 0.3        # Durée (s) d'enregistrement pour un impact ponctuel
SEUIL_INTENSITE = 0.05  # à ajuster selon la sensibilité de ton piézo
TAILLE_FENETRE = int(0.3 * FREQ_S)  # durée du tampon roulant, remplace DUREE_IMPACT
BLOCKSIZE = 512                     # taille des blocs du callback (~12 ms à 44100 Hz)
PERIODE_REFRACTAIRE = 0.4  # secondes à ignorer après un déclenchement (échos/résonance)

# ---------------------------------------------------------------------------
# Enregistrement de base
# ---------------------------------------------------------------------------

def enregistrer(duree=DUREE_CARTO, freq_s=FREQ_S):
    """
    Enregistre un signal mono depuis le micro/piézo par défaut.
    Retourne un vecteur 1D (et non un tableau colonne (N,1)).
    """
    signal = sd.rec(int(duree * freq_s), samplerate=freq_s, channels=1)
    sd.wait()
    return signal.flatten()

def rogner_signal(signal, freq_s, seuil=SEUIL_INTENSITE, marge_avant=0.01, marge_apres=0.15):
    """Isole la portion utile du signal autour de l'impact."""
    indices = np.where(np.abs(signal) >= seuil)[0]
    if len(indices) == 0:
        return signal  # aucun impact détecté
    debut = max(0, indices[0] - int(marge_avant * freq_s))
    fin = min(len(signal), indices[0] + int(marge_apres * freq_s))
    return signal[debut:fin]

def afficher_signal(signal, freq_s=FREQ_S, titre=""):
    """Affiche un signal temporel pour vérification visuelle."""
    t = np.arange(len(signal)) / freq_s
    plt.figure()
    plt.plot(t, signal)
    plt.xlabel("Temps [s]")
    plt.ylabel("Amplitude")
    plt.title(titre)
    plt.show()

def afficher_correlations(correlations):
    """Affiche les courbes de corrélation par retournement temporel."""
    plt.figure()
    for pos, correlation in correlations.items():
        plt.plot(correlation, label=pos)
    plt.legend()
    plt.title("Corrélations par retournement temporel")
    plt.xlabel("Décalage [échantillons]")
    plt.ylabel("Amplitude corrélée")
    plt.show()

# ---------------------------------------------------------------------------
# Étape 1 : Cartographie de la boîte
# ---------------------------------------------------------------------------

def cartographier_boite(positions, duree=DUREE_CARTO, freq_s=FREQ_S, afficher=True, seuil=SEUIL_INTENSITE):
    """
    Enregistre un signal de référence pour chaque position déclarée.

    positions : liste de noms de positions, ex. ["gauche", "droite"]
    Retourne un dict {nom_position: signal (np.ndarray)}
    """
    cartographie = {}
    for pos in positions:
        input(f"-> Position '{pos}' : appuyez sur Entrée puis cognez "
              f"une seule fois pendant les {duree} s d'enregistrement...")
        print(f"Enregistrement de la position '{pos}'...")
        signal = enregistrer(duree, freq_s)
        signal = rogner_signal(signal, freq_s, seuil)
        cartographie[pos] = signal
        if afficher:
            afficher_signal(signal, freq_s, titre=f"Référence — {pos}")
        print(f"Position '{pos}' enregistrée.\n")
    return cartographie


def sauvegarder_cartographie(cartographie, chemin="cartographie.npz"):
    """Sauvegarde la cartographie sur disque pour réutilisation ultérieure."""
    np.savez(chemin, **cartographie)
    print(f"Cartographie sauvegardée dans '{chemin}'.")


def charger_cartographie(chemin="cartographie.npz"):
    """Recharge une cartographie sauvegardée précédemment."""
    data = np.load(chemin)
    return {k: data[k] for k in data.files}

def meilleure_correlation_par_categorie(signal, cartographie_individuelle, categories):
    """
    Corrèle le signal avec chaque référence individuelle et retient, pour
    chaque catégorie définie dans `categories`, le meilleur score et la
    courbe de corrélation associée.

    categories : dict {nom_categorie: liste_des_cles_de_cartographie}
                 ex. {"gauche": ["gauche_1", "gauche_2", ...], "droite": [...]}
                 ou, sans regroupement : {"gauche": ["gauche"], "droite": ["droite"]}
    """
    meilleur_pic = {}
    meilleure_correlation = {}
    for categorie, cles in categories.items():
        for cle in cles:
            if cle not in cartographie_individuelle:
                print_sync(f"Attention : clé '{cle}' absente de la cartographie, ignorée.")
                continue
            reference = cartographie_individuelle[cle]
            correlation = correler_retournement_temporel(signal, reference)
            pic = np.max(np.abs(correlation))
            if categorie not in meilleur_pic or pic > meilleur_pic[categorie]:
                meilleur_pic[categorie] = pic
                meilleure_correlation[categorie] = correlation

    meilleure_categorie = max(meilleur_pic, key=meilleur_pic.get)
    return meilleure_categorie, meilleur_pic, meilleure_correlation

# ---------------------------------------------------------------------------
# Étape 2-3 : Corrélation par retournement temporel
# ---------------------------------------------------------------------------
#
# Principe : le retournement temporel d'un signal de référence h(t) correspond
# à h(-t) (échantillons inversés dans le temps). Corréler le signal mesuré s(t)
# avec h(-t), c'est-à-dire calculer s(t) * h(-t) (convolution), équivaut à
# calculer la corrélation croisée entre s(t) et h(t). Le pic de ce produit
# indique l'instant où s(t) « ressemble » le plus à h(t) : c'est le principe
# du filtre adapté (matched filter), qui est l'implémentation numérique du
# retournement temporel à un seul capteur.
#
# On normalise chaque signal par son énergie avant de corréler, pour que la
# comparaison entre positions ne soit pas biaisée par des différences
# d'amplitude ou de durée d'enregistrement.

def normaliser_energie(signal):
    """Normalise un signal pour qu'il ait une énergie unitaire (norme L2 = 1)."""
    norme = np.linalg.norm(signal)
    if norme == 0:
        return signal
    return signal / norme


def correler_retournement_temporel(signal, reference):
    s = normaliser_energie(signal)
    ref = normaliser_energie(reference)
    reference_inversee = ref[::-1]
    correlation = fftconvolve(s, reference_inversee, mode="full")
    return correlation


def determiner_meilleure_correlation(signal, cartographie, afficher=False):
    """
    Compare le signal mesuré à chaque référence de la cartographie et
    retourne la position dont le pic de corrélation est le plus élevé.

    Retourne (meilleure_position, dict des pics par position, dict des
    corrélations complètes par position).
    """
    pics = {}
    correlations = {}
    for pos, reference in cartographie.items():
        correlation = correler_retournement_temporel(signal, reference)
        correlations[pos] = correlation
        pics[pos] = np.max(np.abs(correlation))

    meilleure_position = max(pics, key=pics.get)

    if afficher:
        plt.figure()
        for pos, correlation in correlations.items():
            plt.plot(correlation, label=pos)
        plt.legend()
        plt.title("Corrélations par retournement temporel")
        plt.xlabel("Décalage [échantillons]")
        plt.ylabel("Amplitude corrélée")
        plt.show()

    return meilleure_position, pics, correlations


def impulsion_detectee(signal, seuil=SEUIL_INTENSITE):
    """Retourne True si l'amplitude max du signal dépasse le seuil (impact réel)."""
    amplitude_max = np.max(np.abs(signal))
    return amplitude_max >= seuil

# ---------------------------------------------------------------------------
# Étape 4 : Jouer la note correspondante
# ---------------------------------------------------------------------------

def jouer_note(position, notes, bloquant=True):
    """
    Joue le fichier audio associé à la position détectée.

    notes : dict {nom_position: chemin_fichier_wav}
    """
    if position not in notes:
        print(f"Aucune note associée à la position '{position}'.")
        return
    freq_s_fichier, data = wavfile.read(notes[position])
    sd.play(data, freq_s_fichier)
    if bloquant:
        sd.wait()


# ---------------------------------------------------------------------------
# Boucle complète : lecture -> corrélation -> note ou STREAM
# ---------------------------------------------------------------------------
 

def demarrer_stream_detection(cartographie, notes, categories=None, freq_s=FREQ_S,
                               seuil=SEUIL_INTENSITE, afficher_correlation=False,
                               periode_refractaire=PERIODE_REFRACTAIRE):
    if categories is None:
        categories = {pos: [pos] for pos in cartographie}  # 1 catégorie = 1 position, par défaut

    tampon = np.zeros(TAILLE_FENETRE, dtype=np.float32)
    file_evenements = queue.Queue()
    file_plots = queue.Queue()
    en_traitement = threading.Event()
    arret = threading.Event()
    dernier_declenchement = -periode_refractaire

    def callback(indata, frames, time_info, status):
        nonlocal tampon, dernier_declenchement
        if status:
            print_sync(status)
        bloc = indata[:, 0]
        tampon = np.concatenate([tampon, bloc])[-TAILLE_FENETRE:]

        maintenant = time.time()
        assez_de_temps_ecoule = (maintenant - dernier_declenchement) >= periode_refractaire

        if not en_traitement.is_set() and assez_de_temps_ecoule and np.max(np.abs(bloc)) >= seuil:
            en_traitement.set()
            dernier_declenchement = maintenant
            file_evenements.put(tampon.copy())

    def traiter_evenements():
        while not arret.is_set():
            try:
                signal = file_evenements.get(timeout=0.1)
            except queue.Empty:
                continue

            signal_rogne = rogner_signal(signal, freq_s, seuil)
            position, pics, correlations = meilleure_correlation_par_categorie(
                signal_rogne, cartographie, categories
            )

            valeurs_triees = sorted(pics.values(), reverse=True)
            marge = (valeurs_triees[0] - valeurs_triees[1]) / valeurs_triees[0] if valeurs_triees[0] > 0 else 0
            print_sync(f"{position} — marge de confiance : {marge:.2%}")

            jouer_note(position, notes, bloquant=False)
            en_traitement.clear()

            if afficher_correlation:
                file_plots.put(correlations)

    def attendre_arret():
        input("Détection en continu (flux)... appuyez sur Entrée pour arrêter.\n")
        arret.set()

    threading.Thread(target=attendre_arret, daemon=True).start()
    thread_traitement = threading.Thread(target=traiter_evenements, daemon=True)
    thread_traitement.start()

    with sd.InputStream(samplerate=freq_s, channels=1, blocksize=BLOCKSIZE, callback=callback):
        while not arret.is_set():
            try:
                correlations = file_plots.get(timeout=0.05)
                afficher_correlations(correlations)
            except queue.Empty:
                pass
            sd.sleep(50)

    print_sync("\nArrêt de la détection.")


# ---------------------------------------------------------------------------
# Exemple d'utilisation
# ---------------------------------------------------------------------------
# 
if __name__ == "__main__":

    # --- Cas simple : une position = une catégorie ---
    positions = ["gauche", "droite"]
    categories = {
        "gauche": ["gauche"],
        "droite": ["droite"],
    }

    # --- Cas avec plusieurs prises par côté (regroupées) ---
    # positions = ["gauche_1", "gauche_2", "gauche_3", "gauche_4", "gauche_5", "gauche_6",
    #              "droite_1", "droite_2", "droite_3", "droite_4", "droite_5", "droite_6"]
    # categories = {
    #     "gauche": [f"gauche_{i}" for i in range(1, 7)],
    #     "droite": [f"droite_{i}" for i in range(1, 7)],
    # }

    notes = {
        "gauche": "a3.wav",
        "droite": "c5.wav",
    }

    cartographie = cartographier_boite(positions, duree=DUREE_CARTO)
    sauvegarder_cartographie(cartographie)
    # cartographie = charger_cartographie()

    demarrer_stream_detection(cartographie, notes, categories=categories, afficher_correlation=True)