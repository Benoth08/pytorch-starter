"""Detection, script 4 sur 5 (etape 13 du cours) : prediction sur une image ou un dossier, boites dessinees et CSV.

    python scripts/predict_yolo.py --config configs/yolo.yaml --source chemin/image.jpg
    python scripts/predict_yolo.py --config configs/yolo.yaml --source chemin/dossier --csv predictions.csv

conf_predict (0.25) est le seuil de production : une boite en dessous n'est pas
rendue. Les images annotees vont dans output_dir/predictions/.
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (--config, --source, --csv)
import logging  # logging : messages horodates du programme
import time  # time : mesure du temps de prediction
from pathlib import Path  # Path : represente un chemin de fichier

import cv2  # OpenCV : lecture et ecriture d'images (en ordre de couleurs BGR)
import pandas as pd  # pandas : tableau des detections, ecrit en CSV
from ultralytics import YOLO  # Ultralytics : bibliotheque YOLO (detection d'objets)

from vision_starter.config import DetectionConfig, load_config  # classe de reglages de la detection et lecture du YAML
from vision_starter.data import EXTENSIONS  # extensions de fichiers reconnues comme images
from vision_starter.detection import draw_boxes  # dessine les boites sur une image
from vision_starter.utils import setup_logging  # configuration des messages

logger = logging.getLogger(__name__)  # journal propre a ce script


# liste les images a traiter ; renvoie une liste de chemins
def list_images(source: Path) -> list[Path]:
    if source.is_file():  # si la source est un seul fichier
        return [source]  # liste d'une seule image
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in EXTENSIONS)  # sinon toutes les images du dossier et sous-dossiers


def main() -> None:  # point d'entree : detecte les objets de chaque image, dessine et enregistre ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # option obligatoire : chemin du fichier YAML
    parser.add_argument("--source", required=True, type=Path, help="image ou dossier")  # option obligatoire : quoi analyser
    parser.add_argument("--csv", type=Path, default=None, help="fichier de sortie (facultatif)")  # option : ou ecrire les detections
    args = parser.parse_args()  # lit la ligne de commande
    config: DetectionConfig = load_config(args.config, DetectionConfig)  # charge les reglages
    output = config.output_path  # dossier du modele et des resultats
    setup_logging()  # messages dans la console
    model = YOLO(output / "runs" / "train" / "weights" / "best.pt")  # charge le meilleur modele entraine
    drawn_dir = output / "predictions"  # dossier des images annotees
    drawn_dir.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas

    rows, start = [], time.time()  # lignes du futur CSV (une par boite) ; heure de debut
    for path in list_images(args.source):  # pour chaque image a traiter
        result = model.predict(  # detection sur cette image ; [0] a la fin = resultat de la seule image passee
            str(path),  # chemin de l'image
            imgsz=config.image_size,  # taille de travail (comme a l'entrainement)
            conf=config.conf_predict,  # seuil de production : les boites moins sures sont ecartees
            iou=config.iou_nms,  # seuil de recouvrement de la NMS (suppression des doublons)
            device="cpu",  # calcul sur processeur
            verbose=False,  # pas d'affichage par image
        )[0]  # predict renvoie une liste (une entree par image) : on prend la seule
        boxes = result.boxes.xyxy.numpy()  # coins des boites (xmin, ymin, xmax, ymax) en pixels de l'image d'origine
        classes, scores = result.boxes.cls.numpy().astype(int), result.boxes.conf.numpy()  # numero de classe et confiance de chaque boite
        for box, cls, score in zip(boxes, classes, scores):  # pour chaque boite detectee
            corners = dict(zip(("xmin", "ymin", "xmax", "ymax"), box.round(1).tolist()))  # coins arrondis, ranges dans un dictionnaire
            # ligne du CSV : image, nom de la classe, confiance arrondie, puis les 4 coins (** deplie le dictionnaire)
            rows.append({"path": path.as_posix(), "label": result.names[cls], "score": round(float(score), 4), **corners})
        named = [(f"{result.names[c]} {s:.2f}", *b) for b, c, s in zip(boxes, classes, scores)]  # (texte "classe score", coins...)
        cv2.imwrite(str(drawn_dir / path.name), draw_boxes(cv2.imread(str(path)), named))  # lit l'image, dessine, enregistre
        found = ", ".join(f"{result.names[c]} {s:.2f}" for c, s in zip(classes, scores))  # liste lisible des detections
        logger.info("%-24s -> %d detection(s) : %s", path.name, len(boxes), found)  # affiche le resultat de l'image
    elapsed = time.time() - start  # duree totale
    logger.info("%d boites en %.2f s -> %s", len(rows), elapsed, drawn_dir)  # bilan : nombre de boites, duree, dossier
    if args.csv:  # si un fichier de sortie a ete demande
        pd.DataFrame(rows).to_csv(args.csv, index=False)  # ecrit toutes les detections en CSV
        logger.info("-> %s", args.csv)  # indique ou


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
