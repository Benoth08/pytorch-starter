"""Recupere la categorie `bottle` de MVTec AD (chapitre Anomalies du cours).

Usage (depuis la racine du projet) :

    python scripts/get_mvtec_bottle.py
    python scripts/get_mvtec_bottle.py --source chemin/vers/bottle   # deja telecharge (site officiel)

Produit data/raw/mvtec_bottle/ avec l'arborescence officielle de MVTec AD :

    train/good/*.png                  209 images normales (les seules utilisees pour apprendre)
    test/good/*.png                    20 images normales de test
    test/<type_de_defaut>/*.png        63 images defectueuses (broken_large, broken_small, contamination)
    ground_truth/<type>/*_mask.png     le masque pixel de chaque defaut (pour l'AUROC pixel)

Le site officiel (mvtec.com) exige un formulaire : on utilise une copie GitHub de
la categorie, figee a un commit, en clone partiel (environ 300 Mo). Licence
MVTec AD : CC BY-NC-SA 4.0 (usage non commercial). Toute autre categorie
(carpet, screw, ...) a la meme arborescence : le code du cours ne change pas.
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (--source, --raw-dir)
import shutil  # shutil : copie de fichiers
import subprocess  # subprocess : lance un autre programme (ici git) depuis Python
from pathlib import Path  # Path : represente un chemin de fichier

REPO_URL = "https://github.com/Manjeet908/C_25.git"  # depot GitHub qui contient une copie de la categorie bottle
COMMIT = "850495af6de6551c59208de26f815212ff1e6824"  # version precise du depot : donnees figees
REMOTE_DIR = "dataset/bottle"  # seul dossier du depot a telecharger


# telecharge seulement le dossier bottle du depot ; renvoie son chemin
def download(target: Path) -> Path:
    """Clone partiel du depot au commit fige ; renvoie le dossier bottle."""
    if (target / REMOTE_DIR).is_dir():  # si le dossier est deja la
        print(f"Deja telecharge : {target / REMOTE_DIR}")  # on le signale
        return target / REMOTE_DIR  # et on ne retelecharge pas
    target.mkdir(parents=True, exist_ok=True)  # cree le dossier de telechargement
    # petite fonction : run("init") lance "git init" dans target ; check=True arrete tout si git echoue
    # ("noqa: E731" dit a ruff d'accepter une lambda rangee dans une variable)
    run = lambda *args: subprocess.run(["git", *args], cwd=target, check=True)  # noqa: E731
    run("init", "-q")  # cree un depot git vide
    run("remote", "add", "origin", REPO_URL)  # lui indique l'adresse du depot distant
    run("sparse-checkout", "set", "--no-cone", f"/{REMOTE_DIR}/")  # ne recuperer que ce dossier (clone partiel)
    run("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)  # telecharge ce seul commit, sans l'historique
    run("checkout", "-q", "FETCH_HEAD")  # ecrit les fichiers de ce commit sur le disque
    return target / REMOTE_DIR  # chemin du dossier bottle


# recopie l'arborescence MVTec en gardant les memes sous-dossiers ; renvoie le nombre de fichiers copies
def copy_tree(source: Path, dest: Path) -> int:
    """Copie train/, test/ et ground_truth/ ; renvoie le nombre d'images copiees."""
    count = 0  # compteur de fichiers copies
    for subset in ("train", "test", "ground_truth"):  # pour chacun des trois dossiers officiels
        for image in sorted((source / subset).rglob("*.png")):  # pour chaque PNG, y compris dans les sous-dossiers (rglob)
            out = dest / image.relative_to(source)  # meme chemin relatif, sous dest (ex. test/good/000.png)
            out.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier de destination si besoin
            shutil.copy2(image, out)  # copie le fichier
            count += 1  # un fichier de plus
    return count  # nombre total de fichiers copies


def main() -> None:  # point d'entree : telecharge si besoin, copie, affiche un bilan ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--source", type=Path, default=None, help="dossier bottle deja telecharge")  # option : eviter le telechargement
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))  # option : dossier racine des donnees brutes
    args = parser.parse_args()  # lit la ligne de commande

    source = args.source or download(args.raw_dir / "_download_mvtec")  # dossier fourni, sinon on telecharge
    dest = args.raw_dir / "mvtec_bottle"  # dossier final utilise par le cours
    count = copy_tree(source, dest)  # copie tout et compte
    print(f"{dest} : {count} fichiers PNG (images + masques)")  # bilan global
    for subset in ("train/good", "test/good", "test/broken_large", "test/broken_small", "test/contamination"):  # pour chaque sous-dossier
        print(f"  {subset:22s} {len(list((dest / subset).glob('*.png')))} images")  # nombre d'images, colonnes alignees


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
