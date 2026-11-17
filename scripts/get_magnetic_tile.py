"""Recupere le jeu Magnetic Tile Defect (chapitre Segmentation du cours).

Usage (depuis la racine du projet) :

    python scripts/get_magnetic_tile.py
    python scripts/get_magnetic_tile.py --source chemin/vers/depot-deja-clone

Produit data/raw/magnetic_tile/ :

    MT_Blowhole/Imgs/   115 images + 115 masques   (trou)
    MT_Break/Imgs/       85 + 85                    (cassure)
    MT_Crack/Imgs/       57 + 57                    (fissure)
    MT_Fray/Imgs/        32 + 32                    (effilochage)
    MT_Uneven/Imgs/     103 + 103                   (surface irreguliere)
    MT_Free/Imgs/       952 + 952                   (sans defaut ; masque vide)

Chaque image `x.jpg` (niveaux de gris, tailles variables autour de 250 x 300) a
son masque `x.png` : blanc (255) sur le defaut, noir ailleurs. Source : Huang
et al., "Surface defect saliency of magnetic tile" (2018), depot GitHub de
l'auteur, fige a un commit (environ 110 Mo).
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (--source, --raw-dir)
import shutil  # shutil : copie de fichiers
import subprocess  # subprocess : lance un autre programme (ici git) depuis Python
from pathlib import Path  # Path : represente un chemin de fichier

REPO_URL = "https://github.com/abin24/Magnetic-tile-defect-datasets..git"  # depot source (son nom finit par un point, d'ou "..git")
COMMIT = "cc4eb530b57b9cb7225bf76023873043ffd50655"  # version precise du depot : donnees figees
CLASSES = ("MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Uneven", "MT_Free")  # les 6 dossiers de classes a recuperer


# telecharge les 6 dossiers de classes ; renvoie le dossier telecharge
def download(target: Path) -> Path:
    """Clone partiel (les 6 dossiers de classes) au commit fige ; renvoie le dossier."""
    if all((target / name / "Imgs").is_dir() for name in CLASSES):  # si tout est deja la
        print(f"Deja telecharge : {target}")  # on le signale
        return target  # et on ne retelecharge pas
    target.mkdir(parents=True, exist_ok=True)  # cree le dossier de telechargement
    # petite fonction : run("init") lance "git init" dans target ; check=True arrete tout si git echoue
    # ("noqa: E731" dit a ruff d'accepter une lambda rangee dans une variable)
    run = lambda *args: subprocess.run(["git", *args], cwd=target, check=True)  # noqa: E731
    run("init", "-q")  # cree un depot git vide
    run("remote", "add", "origin", REPO_URL)  # lui indique l'adresse du depot distant
    run("sparse-checkout", "set", "--no-cone", *[f"/{name}/" for name in CLASSES])  # ne recuperer que ces dossiers (clone partiel)
    run("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)  # telecharge ce seul commit, sans l'historique
    run("checkout", "-q", "FETCH_HEAD")  # ecrit les fichiers de ce commit sur le disque
    return target  # dossier qui contient maintenant les 6 classes


def main() -> None:  # point d'entree : telecharge si besoin puis copie images et masques ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--source", type=Path, default=None, help="depot deja clone")  # option : eviter le telechargement
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))  # option : dossier racine des donnees brutes
    args = parser.parse_args()  # lit la ligne de commande

    source = args.source or download(args.raw_dir / "_download_magnetic_tile")  # dossier fourni, sinon on telecharge
    dest = args.raw_dir / "magnetic_tile"  # dossier final utilise par le cours
    for name in CLASSES:  # pour chaque classe
        out = dest / name / "Imgs"  # dossier de destination de la classe
        out.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
        files = sorted((source / name / "Imgs").iterdir())  # tous les fichiers du dossier source, dans l'ordre
        for path in files:  # pour chaque fichier
            if path.suffix.lower() in {".jpg", ".png"}:  # seulement les images (.jpg) et les masques (.png)
                shutil.copy2(path, out / path.name)  # copie le fichier
        n_images = len(list(out.glob("*.jpg")))  # nombre d'images copiees
        print(f"  {name:12s} {n_images:4d} images, {len(list(out.glob('*.png')))} masques")  # bilan de la classe, colonnes alignees
    print(f"{dest} pret")  # fin


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
