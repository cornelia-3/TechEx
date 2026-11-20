from pathlib import Path
import shutil

data = Path(r"C:\Users\corne\OneDrive - polymtlus\Documents\GitHub\TechEx\M4_Spectrometre\data")
resume = data / "resume"
resume.mkdir(exist_ok=True)

for dossier in sorted(data.iterdir()):
    src = dossier / "spectre.png"
    if dossier.is_dir() and src.exists():
        shutil.copy(src, resume / f"{dossier.name}_spectre.png")
        print("copié :", dossier.name)