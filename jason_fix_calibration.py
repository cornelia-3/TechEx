# fix_roles.py — double-clic ou : python fix_roles.py
import json
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

root = tk.Tk()
root.withdraw()
chemin = filedialog.askdirectory(title="Dossier de calibration (contient 532, 555, ...)")
if not chemin:
    raise SystemExit("Aucun dossier choisi.")

racine = Path(chemin)
lasers = []
for d in sorted(p for p in racine.iterdir() if p.is_dir()):
    meta = d / "laser.json"
    if meta.is_file():
        infos = json.loads(meta.read_text(encoding="utf-8"))
        lasers.append((d, meta, infos))
        print(f"{d.name:>6} nm : {'VERIF' if infos['verification'] else 'ajustement'}")

if not lasers:
    raise SystemExit("Aucun laser.json trouvé dans ce dossier.")

reponse = input(
    "\nNoms des dossiers à mettre en VÉRIFICATION, séparés par des espaces\n"
    "(laisser vide = tous en ajustement) : "
)
verif = set(reponse.split())

for d, meta, infos in lasers:
    infos["verification"] = d.name in verif
    meta.write_text(json.dumps(infos, ensure_ascii=False), encoding="utf-8")
    print(f"{d.name:>6} nm -> {'VERIF' if infos['verification'] else 'ajustement'}")