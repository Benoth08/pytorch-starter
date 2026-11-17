"""Script 1 sur 5 (etapes 3 et 4 du cours) : catalogue des images et decoupage train / val / test, enregistre en CSV.

    python scripts/make_split.py --config configs/default.yaml

A lancer une seule fois par jeu de donnees : tous les autres scripts relisent
le fichier split.csv produit ici. Le relancer avec la meme graine redonne le
meme decoupage.
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (ici --config)
import logging  # logging : messages horodates du programme
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import pandas as pd  # pandas : tableaux de donnees (ici pour un tableau croise classes x sous-ensembles)

from vision_starter.config import load_config  # lit le fichier YAML de reglages
from vision_starter.data import build_catalog, check_split, make_split  # catalogue des images, controle et decoupage
from vision_starter.utils import setup_logging  # configure l'affichage des messages

logger = logging.getLogger(__name__)  # journal propre a ce script


def main() -> None:  # point d'entree du script : catalogue, decoupage, ecriture du CSV ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # option obligatoire : chemin du fichier YAML
    args = parser.parse_args()  # lit la ligne de commande ; args.config contient le chemin
    config = load_config(args.config)  # charge et valide les reglages
    setup_logging()  # messages affiches dans la console

    catalog = build_catalog(config)  # tableau (path, label) de toutes les images
    if "split" not in catalog.columns:  # cas A : le decoupage fourni est deja la ; sinon on le fabrique
        catalog = make_split(catalog, config.val_fraction, config.test_fraction, config.seed)  # decoupage stratifie au hasard
    check_split(catalog)  # refuse les doublons et les sous-ensembles vides

    split_file = Path(config.split_file)  # chemin du CSV a ecrire
    split_file.parent.mkdir(parents=True, exist_ok=True)  # cree son dossier s'il n'existe pas
    catalog.to_csv(split_file, index=False)  # ecrit le CSV (index=False : pas de colonne de numeros de ligne)
    logger.info("%d images, %d classes -> %s", len(catalog), catalog["label"].nunique(), split_file)  # bilan : nombres et fichier
    logger.info("\n%s", pd.crosstab(catalog["label"], catalog["split"]))  # tableau : nombre d'images par classe et par sous-ensemble


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement (pas quand on l'importe)
    main()  # on execute le script
