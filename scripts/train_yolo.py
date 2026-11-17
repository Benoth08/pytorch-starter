"""Detection, script 2 sur 5 (etapes 5 a 10 du cours) : entrainement YOLO11 (la boucle est dans Ultralytics).

    python scripts/train_yolo.py --config configs/yolo.yaml

Produit dans output_dir : config_used.yaml, train.log, runs/train/weights/best.pt
(meilleur modele selon le mAP de validation, choisi par Ultralytics) et
train_summary.json. Ultralytics fait ici les etapes 5 a 10 : Dataset, loader,
augmentations, boucle, validation a chaque epoque, arret anticipe.
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (ici --config)
import json  # json : ecrit le resume dans un fichier texte lisible par tout langage
import logging  # logging : messages horodates du programme
from pathlib import Path  # Path : represente un chemin de fichier

from ultralytics import YOLO  # Ultralytics : bibliotheque YOLO (detection d'objets) avec sa propre boucle d'entrainement

from vision_starter.config import DetectionConfig, load_config, save_config  # reglages de detection : classe, lecture, copie
from vision_starter.utils import seed_everything, setup_logging  # graine du hasard et messages

logger = logging.getLogger(__name__)  # journal propre a ce script


def main() -> None:  # point d'entree : entraine YOLO et ecrit un resume ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # option obligatoire : chemin du fichier YAML
    args = parser.parse_args()  # lit la ligne de commande
    config: DetectionConfig = load_config(args.config, DetectionConfig)  # charge les reglages
    output = config.output_path  # dossier des resultats
    output.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
    setup_logging(output / "train.log")  # messages a l'ecran et dans train.log
    save_config(config, output / "config_used.yaml")  # copie des reglages utilises
    seed_everything(config.seed)  # fixe le hasard

    data_yaml = Path(config.yolo_dir) / "data.yaml"  # carte du jeu de donnees produite par prepare_yolo.py
    if not data_yaml.exists():  # si elle n'existe pas encore
        raise FileNotFoundError(f"{data_yaml} absent : lancer scripts/prepare_yolo.py d'abord")  # message qui dit quoi faire

    model = YOLO(config.weights)  # pre-entraine COCO ; la tete est remplacee pour nos classes au premier train
    results = model.train(  # lance l'entrainement complet ; renvoie les metriques de validation finales
        data=str(data_yaml),  # ou trouver les images et les etiquettes
        epochs=config.epochs,  # nombre maximal d'epoques
        patience=config.patience,  # arret anticipe apres tant d'epoques sans progres
        imgsz=config.image_size,  # taille des images (cote du carre)
        batch=config.batch_size,  # images par lot
        device="cpu",  # calcul sur processeur (mettre 0 pour le premier GPU)
        seed=config.seed,  # graine du hasard
        workers=config.num_workers,  # processus de chargement en parallele
        project=str((output / "runs").resolve()),  # chemin absolu, sinon Ultralytics prefixe runs/detect/
        name="train",  # sous-dossier du run : runs/train
        exist_ok=True,  # reutilise ce dossier au lieu de creer train2, train3...
        close_mosaic=config.close_mosaic,  # coupe l'augmentation mosaic sur les dernieres epoques
        warmup_epochs=config.warmup_epochs,  # montee progressive du pas d'apprentissage au debut
        plots=False,  # pas de figures automatiques
        verbose=False,  # sortie console plus courte
    )
    best_weights = Path(results.save_dir) / "weights" / "best.pt"  # meilleur modele, ecrit par Ultralytics
    if best_weights.is_relative_to(Path.cwd()):  # si ce chemin est sous le dossier courant
        best_weights = best_weights.relative_to(Path.cwd())  # on l'ecrit en relatif (plus lisible, portable)
    # resume : chemin des poids, mAP@0.5 et mAP@0.5:0.95 de validation (mAP = qualite moyenne des detections, 1 = parfait)
    summary = {"best_weights": str(best_weights), "val_map50": float(results.box.map50), "val_map50_95": float(results.box.map)}
    (output / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")  # ecrit le resume en JSON
    # bilan final
    logger.info("validation : mAP@0.5 = %.3f | mAP@0.5:0.95 = %.3f -> %s", summary["val_map50"], summary["val_map50_95"], best_weights)


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
