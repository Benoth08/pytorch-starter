"""Detection, script 1 sur 5 (etapes 3, 4 et 7 du cours) : catalogue des images annotees, decoupage, conversion au format YOLO.

    python scripts/prepare_yolo.py --config configs/yolo.yaml

A lancer une seule fois par jeu de donnees. Produit split.csv (relu par les
autres scripts) et l'arborescence yolo_dir/images/<split>, labels/<split> et
data.yaml qu'Ultralytics attend. Les scripts suivants ne lisent que data.yaml.
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (ici --config)
import logging  # logging : messages horodates du programme
from pathlib import Path  # Path : represente un chemin de fichier

import pandas as pd  # pandas : tableaux de donnees (ici pour un tableau croise classes x sous-ensembles)

from vision_starter.config import DetectionConfig, load_config  # classe de reglages de la detection et lecture du YAML
from vision_starter.data import check_split, make_split  # decoupage stratifie et son controle (memes outils qu'en classification)

# catalogue VOC, liste des classes des boites, ecriture au format YOLO (voir src/vision_starter/detection.py)
from vision_starter.detection import catalog_voc, class_names_of_boxes, export_yolo_dataset
from vision_starter.utils import setup_logging  # configuration des messages

logger = logging.getLogger(__name__)  # journal propre a ce script


def main() -> None:  # point d'entree : catalogue, decoupage, conversion YOLO ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # option obligatoire : chemin du fichier YAML
    args = parser.parse_args()  # lit la ligne de commande
    config: DetectionConfig = load_config(args.config, DetectionConfig)  # charge les reglages de detection
    config.output_path.mkdir(parents=True, exist_ok=True)  # cree le dossier des resultats
    setup_logging(config.output_path / "prepare.log")  # messages a l'ecran et dans prepare.log

    # Catalogue : une ligne par image, avec son fichier d'annotations. Meme decoupage stratifie qu'en classification.
    catalog = catalog_voc(Path(config.data_dir))  # tableau (image, annotation, classe majoritaire, nombre de boites...)
    class_names = class_names_of_boxes(catalog)  # toutes les classes presentes dans les boites, triees
    # bilan : nombre d'images, nombre total de boites, classes
    logger.info("%d images | %d boites | classes : %s", len(catalog), int(catalog["n_boxes"].sum()), class_names)
    catalog = make_split(catalog, config.val_fraction, config.test_fraction, config.seed)  # decoupage stratifie par classe majoritaire
    check_split(catalog)  # refuse doublons et sous-ensembles vides
    split_file = Path(config.split_file)  # chemin du CSV de decoupage
    split_file.parent.mkdir(parents=True, exist_ok=True)  # cree son dossier si besoin
    catalog.to_csv(split_file, index=False)  # fige le decoupage
    logger.info("\n%s", pd.crosstab(catalog["label"], catalog["split"]))  # tableau : nombre d'images par classe et sous-ensemble

    # Format YOLO : images/<split>/x.jpg <-> labels/<split>/x.txt ("classe cx cy w h" normalises) + data.yaml.
    data_yaml = export_yolo_dataset(catalog, class_names, Path(config.yolo_dir))  # copie et convertit tout ; renvoie le data.yaml
    example = sorted((Path(config.yolo_dir) / "labels" / "train").glob("*.txt"))[0]  # premier fichier d'etiquettes, pour montrer
    logger.info("data.yaml -> %s\n%s", data_yaml, data_yaml.read_text(encoding="utf-8").strip())  # affiche le contenu du data.yaml
    # affiche l'exemple d'etiquettes sur une ligne (les retours a la ligne deviennent " | ")
    logger.info("exemple d'etiquette %s : %s", example.name, example.read_text(encoding="utf-8").strip().replace("\n", " | "))


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
