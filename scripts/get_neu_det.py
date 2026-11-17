"""Recupere NEU-DET (detection : images + annotations Pascal VOC) pour le chapitre Detection du cours.

Usage (depuis la racine du projet) :

    python scripts/get_neu_det.py

Produit data/raw/neu_det/IMAGES/*.jpg et data/raw/neu_det/ANNOTATIONS/*.xml.
Source officielle sans lien direct : on utilise une copie GitHub figee a un
commit, en clone partiel (seuls les dossiers utiles sont telecharges).
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import subprocess  # subprocess : lance un autre programme (ici git) depuis Python
from pathlib import Path  # Path : represente un chemin de fichier

REPO_URL = "https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection.git"  # depot GitHub source
COMMIT = "351bee21b56bbf97a1e1d18262bceb202eb7ef41"  # version precise du depot : donnees figees
SUBDIRS = ["IMAGES", "ANNOTATIONS"]  # seuls dossiers a telecharger : images et fichiers d'annotations XML


# telecharge les images et les annotations dans target ; ne renvoie rien
def download(target: Path) -> None:
    if all((target / sub).is_dir() for sub in SUBDIRS):  # si les deux dossiers sont deja la
        print(f"Deja telecharge : {target}")  # on le signale
        return  # et on s'arrete
    target.mkdir(parents=True, exist_ok=True)  # cree le dossier de destination
    # petite fonction : run("init") lance "git init" dans target ; check=True arrete tout si git echoue
    # ("noqa: E731" dit a ruff d'accepter une lambda rangee dans une variable)
    run = lambda *args: subprocess.run(["git", *args], cwd=target, check=True)  # noqa: E731
    if not (target / ".git").is_dir():  # si le depot git local n'existe pas encore (1er lancement)
        run("init", "-q")  # cree un depot git vide
        run("remote", "add", "origin", REPO_URL)  # lui indique l'adresse du depot distant
    run("sparse-checkout", "set", "--no-cone", *[f"/{sub}/" for sub in SUBDIRS])  # ne recuperer que ces dossiers (clone partiel)
    run("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)  # telecharge ce seul commit, sans l'historique
    run("checkout", "-q", "FETCH_HEAD")  # ecrit les fichiers de ce commit sur le disque
    print(f"{target} : {sum(1 for _ in (target / 'IMAGES').glob('*.jpg'))} images")  # affiche le nombre d'images obtenues


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    download(Path("data/raw/neu_det"))  # telecharge dans data/raw/neu_det
