"""Detection, script 3 sur 5 (etape 11 du cours) : mAP sur le test, une seule fois, avec le meilleur modele.

    python scripts/evaluate_yolo.py --config configs/yolo.yaml

conf=0.001 pour l'evaluation : toute la courbe precision-rappel est parcourue
(le seuil de production, conf_predict, ne sert qu'a decider). Produit metrics.json
avec le mAP global et l'AP par classe.
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (ici --config)
import json  # json : ecrit les metriques dans un fichier texte lisible par tout langage
import logging  # logging : messages horodates du programme
from pathlib import Path  # Path : represente un chemin de fichier

import yaml  # PyYAML : relit data.yaml pour connaitre les noms des classes
from ultralytics import YOLO  # Ultralytics : bibliotheque YOLO (detection d'objets)

from vision_starter.config import DetectionConfig, load_config  # classe de reglages de la detection et lecture du YAML
from vision_starter.utils import setup_logging  # configuration des messages

logger = logging.getLogger(__name__)  # journal propre a ce script


def main() -> None:  # point d'entree : mesure le mAP sur le test et l'ecrit ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # option obligatoire : chemin du fichier YAML
    args = parser.parse_args()  # lit la ligne de commande
    config: DetectionConfig = load_config(args.config, DetectionConfig)  # charge les reglages
    output = config.output_path  # dossier du modele et des resultats
    setup_logging(output / "evaluate.log")  # messages a l'ecran et dans evaluate.log

    data_yaml = Path(config.yolo_dir) / "data.yaml"  # carte du jeu de donnees
    # noms des classes dans l'ordre des numeros (la cle "names" est un dictionnaire numero -> nom)
    class_names = list(yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"].values())
    model = YOLO(output / "runs" / "train" / "weights" / "best.pt")  # charge le meilleur modele entraine
    metrics = model.val(  # evaluation par Ultralytics ; renvoie un objet de metriques
        data=str(data_yaml),  # ou trouver les images et les etiquettes
        split="test",  # sous-ensemble de test (jamais vu pendant l'entrainement)
        imgsz=config.image_size,  # taille des images
        batch=config.batch_size,  # images par lot
        conf=config.conf_eval,  # seuil tres bas : on garde presque toutes les boites pour tracer toute la courbe
        iou=config.iou_nms,  # seuil de recouvrement de la NMS (suppression des boites en double)
        device="cpu",  # calcul sur processeur
        project=str((output / "runs").resolve()),  # dossier des runs, en chemin absolu
        name="test",  # sous-dossier : runs/test
        exist_ok=True,  # reutilise ce dossier s'il existe deja
        plots=False,  # pas de figures automatiques
        verbose=False,  # sortie console plus courte
    )
    # AP par classe (AP = aire sous la courbe precision-rappel) au seuil d'IoU 0.5 et en moyenne de 0.5 a 0.95
    per_class = {name: {"ap50": float(metrics.box.ap50[i]), "ap50_95": float(metrics.box.ap[i])} for i, name in enumerate(class_names)}
    # mAP = moyenne des AP de toutes les classes
    results = {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map), "per_class": per_class}
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")  # ecrit les resultats en JSON
    logger.info("test : mAP@0.5 = %.3f | mAP@0.5:0.95 = %.3f", results["map50"], results["map50_95"])  # bilan global
    for name, ap in per_class.items():  # pour chaque classe
        logger.info("  %-16s AP@0.5 = %.3f | AP@0.5:0.95 = %.3f", name, ap["ap50"], ap["ap50_95"])  # son AP, colonnes alignees


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
