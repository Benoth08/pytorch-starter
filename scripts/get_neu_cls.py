"""Recupere NEU-CLS et le range selon les trois organisations vues dans le cours.

Usage (depuis la racine du projet) :

    python scripts/get_neu_cls.py            # telecharge puis prepare
    python scripts/get_neu_cls.py --source chemin/vers/NEU-CLS-deja-telecharge

Produit dans data/raw :

    neu_cls_split/   cas A : train/ val/ test/ puis un sous-dossier par classe
    neu_cls_flat/    cas B : un sous-dossier par classe, sans decoupage
    neu_cls_csv/     cas C : images/ a plat + labels.csv (filename, label)

Le jeu officiel (Northeastern University) n'a pas de lien direct : on utilise
la copie GitHub d'un reconditionnement Kaggle, figee a un commit precis, en
clone partiel (seul le dossier d'images est telecharge, environ 130 Mo).
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (--source, --raw-dir)
import csv  # csv : ecrit le fichier labels.csv
import shutil  # shutil : copie de fichiers
import subprocess  # subprocess : lance un autre programme (ici git) depuis Python
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

REPO_URL = "https://github.com/aviralchharia/Surface-Defect-Detection-in-Hot-Rolled-Steel-Strips.git"  # depot GitHub source
COMMIT = "2a1e01db3b04a6eb9502b73e30fc88a6bb242575"  # version precise du depot : les donnees ne changeront jamais sous nos pieds
REMOTE_DIR = "NEU Metal Surface Defects Data"  # seul dossier du depot a telecharger
# Noms d'origine (Kaggle) -> noms utilises dans le cours (minuscules, sans espace).
CLASS_RENAME = {  # dictionnaire nom d'origine -> nom du cours
    "Crazing": "crazing",  # craquelures
    "Inclusion": "inclusion",  # inclusions (corps etrangers)
    "Patches": "patches",  # taches
    "Pitted": "pitted",  # piqures
    "Rolled": "rolled",  # calamine incrustee au laminage
    "Scratches": "scratches",  # rayures
}
SUBSET_RENAME = {"train": "train", "valid": "val", "test": "test"}  # sous-dossiers d'origine -> noms du cours


# telecharge seulement le dossier d'images du depot ; renvoie le chemin de ce dossier
def download(target: Path) -> Path:
    """Clone partiel du depot au commit fige ; renvoie le dossier des images."""
    if (target / REMOTE_DIR).is_dir():  # si le dossier est deja la (script deja lance)
        print(f"Deja telecharge : {target / REMOTE_DIR}")  # on le signale
        return target / REMOTE_DIR  # et on ne retelecharge pas
    target.mkdir(parents=True, exist_ok=True)  # cree le dossier de telechargement
    # petite fonction : run("init") lance "git init" dans target ; check=True arrete tout si git echoue
    # ("noqa: E731" dit a ruff d'accepter une lambda rangee dans une variable)
    run = lambda *args: subprocess.run(["git", *args], cwd=target, check=True)  # noqa: E731
    run("init", "-q")  # cree un depot git vide (-q = silencieux)
    run("remote", "add", "origin", REPO_URL)  # lui indique l'adresse du depot distant
    run("sparse-checkout", "set", "--no-cone", f"/{REMOTE_DIR}/")  # ne recuperer que ce dossier (clone partiel)
    run("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)  # telecharge ce seul commit, sans l'historique
    run("checkout", "-q", "FETCH_HEAD")  # ecrit les fichiers de ce commit sur le disque
    return target / REMOTE_DIR  # chemin du dossier d'images


# cas A : recopie en gardant le decoupage train / val / test fourni ; ne renvoie rien
def layout_a(source: Path, dest: Path) -> None:
    """Cas A : on garde le decoupage fourni (train / val / test)."""
    for subset_src, subset_dst in SUBSET_RENAME.items():  # pour chaque sous-ensemble (nom d'origine, nom du cours)
        for class_src, class_dst in CLASS_RENAME.items():  # pour chaque classe (nom d'origine, nom du cours)
            out = dest / subset_dst / class_dst  # dossier de destination, ex. neu_cls_split/train/crazing
            out.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
            for image in sorted((source / subset_src / class_src).glob("*.bmp")):  # pour chaque image .bmp de cette classe
                shutil.copy2(image, out / image.name)  # copie l'image (copy2 garde aussi la date du fichier)


# cas B : regroupe toutes les images d'une classe dans un seul dossier ; ne renvoie rien
def layout_b(split_root: Path, dest: Path) -> None:
    """Cas B : toutes les images d'une classe dans un seul dossier."""
    for subset in SUBSET_RENAME.values():  # pour chaque sous-ensemble du cas A (train, val, test)
        for class_dir in sorted((split_root / subset).iterdir()):  # pour chaque dossier de classe
            out = dest / class_dir.name  # dossier de la classe, sans sous-ensemble
            out.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
            for image in sorted(class_dir.glob("*.bmp")):  # pour chaque image
                shutil.copy2(image, out / image.name)  # copie l'image


# cas C : un dossier d'images a plat + un CSV qui donne la classe ; ne renvoie rien
def layout_c(flat_root: Path, dest: Path) -> None:
    """Cas C : un dossier d'images a plat et un CSV (filename, label)."""
    images_dir = dest / "images"  # dossier unique des images
    images_dir.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
    rows = []  # lignes du futur CSV
    for class_dir in sorted(flat_root.iterdir()):  # pour chaque dossier de classe du cas B
        for image in sorted(class_dir.glob("*.bmp")):  # pour chaque image
            shutil.copy2(image, images_dir / image.name)  # copie l'image dans le dossier unique
            rows.append({"filename": image.name, "label": class_dir.name})  # note son nom et sa classe
    # ouvre labels.csv en ecriture ; newline="" evite des lignes vides sous Windows ; with referme le fichier a la fin
    with open(dest / "labels.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "label"])  # ecrivain CSV qui prend des dictionnaires
        writer.writeheader()  # 1re ligne : les noms de colonnes
        writer.writerows(rows)  # une ligne par image


def main() -> None:  # point d'entree : telecharge si besoin puis produit les trois rangements ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--source", type=Path, default=None, help="dossier NEU-CLS deja telecharge")  # option : eviter le telechargement
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))  # option : dossier racine des donnees brutes
    args = parser.parse_args()  # lit la ligne de commande (--raw-dir devient args.raw_dir)

    source = args.source or download(args.raw_dir / "_download_neu_cls")  # dossier fourni, sinon on telecharge
    split_root, flat_root, csv_root = (  # les trois dossiers de destination
        args.raw_dir / "neu_cls_split",  # cas A
        args.raw_dir / "neu_cls_flat",  # cas B
        args.raw_dir / "neu_cls_csv",  # cas C
    )
    layout_a(source, split_root)  # produit le cas A depuis la source
    layout_b(split_root, flat_root)  # produit le cas B depuis le cas A
    layout_c(flat_root, csv_root)  # produit le cas C depuis le cas B
    for root in (split_root, flat_root, csv_root):  # pour chaque rangement produit
        print(f"{root} : {sum(1 for _ in root.rglob('*.bmp'))} images")  # affiche le nombre d'images (controle rapide)


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
