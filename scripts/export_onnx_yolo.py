"""Detection, script 5 sur 5 (etape 12 du cours) : export ONNX sans NMS, post-traitement maison verifie contre Ultralytics.

    python scripts/export_onnx_yolo.py --config configs/yolo.yaml

Le fichier ONNX sort les boites brutes (lot, 4 + C, ancres). Le programme qui
l'utilise doit reproduire le letterbox en entree et le filtrage + NMS en sortie :
ces deux fonctions sont dans vision_starter.detection, et le script echoue si
leurs boites ne recouvrent pas celles d'Ultralytics (IoU moyen < 0,98).
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (--config, --n-images)
import json  # json : ecrit la "carte" du modele dans un fichier texte lisible par tout langage
import logging  # logging : messages horodates du programme
import shutil  # shutil : deplacement de fichiers
from pathlib import Path  # Path : represente un chemin de fichier

import cv2  # OpenCV : lecture des images (en ordre de couleurs BGR)
import numpy as np  # numpy : moyenne des IoU

# ONNX Runtime : execute un modele ONNX (format standard de modele, utilisable sans PyTorch ni Ultralytics)
import onnxruntime as ort
import pandas as pd  # pandas : relit le decoupage split.csv
import yaml  # PyYAML : relit data.yaml pour les noms des classes
from ultralytics import YOLO  # Ultralytics : bibliotheque YOLO (export ONNX et predictions de reference)

from vision_starter.config import DetectionConfig, load_config  # classe de reglages de la detection et lecture du YAML

# IoU de deux boites, post-traitement (filtrage + NMS) et pretraitement (letterbox) ecrits a la main
from vision_starter.detection import box_iou, postprocess, preprocess
from vision_starter.utils import setup_logging  # configuration des messages

logger = logging.getLogger(__name__)  # journal propre a ce script


def main() -> None:  # point d'entree : export ONNX, verification contre Ultralytics, carte du modele ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # option obligatoire : chemin du fichier YAML
    # option : nombre d'images de validation a comparer (16 par defaut)
    parser.add_argument("--n-images", type=int, default=16, help="images de validation utilisees pour verifier")
    args = parser.parse_args()  # lit la ligne de commande (--n-images devient args.n_images)
    config: DetectionConfig = load_config(args.config, DetectionConfig)  # charge les reglages
    output = config.output_path  # dossier du modele et des resultats
    setup_logging(output / "export_onnx.log")  # messages a l'ecran et dans export_onnx.log
    size = config.image_size  # taille d'entree du modele

    best_weights = output / "runs" / "train" / "weights" / "best.pt"  # meilleur modele entraine
    model = YOLO(best_weights)  # le charge
    # export ONNX par Ultralytics : taille de lot libre (dynamic), SANS NMS dans le fichier (on la fera nous-memes) ; renvoie le chemin
    exported = Path(model.export(format="onnx", imgsz=size, dynamic=True, nms=False, simplify=False, device="cpu"))
    onnx_path = output / "model.onnx"  # emplacement final du fichier ONNX
    shutil.move(str(exported), onnx_path)  # Ultralytics ecrit a cote des poids ; on range avec les autres sorties
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # charge le fichier ONNX, execution sur CPU
    input_name = session.get_inputs()[0].name  # nom de l'entree choisi par Ultralytics (lu plutot que devine)

    # Verification : memes images de validation, boites ONNX + post-traitement contre boites Ultralytics.
    paths = pd.read_csv(config.split_file).query("split == 'val'")["path"].head(args.n_images).tolist()  # n premieres images de val
    ious, n_onnx, n_ref = [], 0, 0  # IoU de chaque boite ONNX avec sa meilleure boite de reference ; compteurs de boites
    for path in paths:  # pour chaque image de verification
        tensor, ratio, pad = preprocess(cv2.imread(path), size)  # letterbox + mise en forme (1, 3, size, size), comme Ultralytics
        raw = session.run(None, {input_name: tensor})[0]  # sortie brute du modele ONNX : (1, 4 + classes, ancres)
        onnx_boxes = postprocess(raw, ratio, pad, size, conf=config.conf_predict, iou=config.iou_nms)  # filtrage + NMS maison
        # boites de reference calculees par Ultralytics avec les memes seuils
        result = model.predict(path, imgsz=size, conf=config.conf_predict, iou=config.iou_nms, device="cpu", verbose=False)[0]
        reference = result.boxes.xyxy.numpy()  # coins des boites de reference
        n_onnx, n_ref = n_onnx + len(onnx_boxes), n_ref + len(reference)  # cumule les nombres de boites des deux cotes
        if len(reference):  # si Ultralytics a trouve au moins une boite
            # pour chaque boite ONNX : son meilleur recouvrement (IoU) avec une boite de reference ; box[:4] = les 4 coins
            ious.extend(max(box_iou(box[:4], ref) for ref in reference) for box in onnx_boxes)
    mean_iou = float(np.mean(ious)) if ious else float("nan")  # IoU moyen (nan s'il n'y a aucune boite a comparer)
    logger.info("sortie ONNX brute : %s = (lot, 4 + classes, ancres)", raw.shape)  # forme de la sortie brute (derniere image)
    # bilan : images verifiees, boites des deux cotes, IoU moyen
    logger.info("%d images | %d boites ONNX | %d boites Ultralytics | IoU moyen %.4f", len(paths), n_onnx, n_ref, mean_iou)
    # Meme nombre de boites et IoU moyen > 0,98 : les boites d'un modele peu entraine (scores pres du seuil) bougent un peu plus.
    if n_onnx != n_ref or not (mean_iou > 0.98):  # "not (x > 0.98)" echoue aussi si mean_iou vaut nan
        raise RuntimeError("le post-traitement ONNX ne reproduit pas Ultralytics")  # on echoue : pas fiable en production

    # noms des classes dans l'ordre des numeros, relus dans data.yaml
    class_names = list(yaml.safe_load((Path(config.yolo_dir) / "data.yaml").read_text(encoding="utf-8"))["names"].values())
    card = {  # carte du modele : ce qu'un programme sans Ultralytics doit savoir
        "input_name": input_name,  # nom de l'entree
        "output_name": session.get_outputs()[0].name,  # nom de la sortie
        "layout": "NCHW",  # ordre des dimensions : lot, canaux, hauteur, largeur
        "image_size": size,  # cote du carre d'entree
        "channels": "RGB",  # ordre des couleurs attendu
        "preprocess": "letterbox to image_size (keep ratio, pad 114), BGR->RGB, divide by 255",  # pretraitement a reproduire
        # forme et contenu de la sortie brute
        "output_layout": "(batch, 4 + num_classes, anchors) : cx cy w h in letterboxed pixels, then one score per class",
        # post-traitement a reproduire : seuil de score, NMS par classe, retour aux coordonnees de l'image d'origine
        "postprocess": f"score = max class score, keep > {config.conf_predict}, NMS per class iou {config.iou_nms}, undo letterbox",
        "class_names": class_names,  # numero -> nom de classe
    }
    (output / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")  # ecrit la carte
    # bilan final : fichier, taille en Mo, emplacement de la carte
    logger.info("export OK : %s (%.1f Mo) | carte -> %s", onnx_path, onnx_path.stat().st_size / 1e6, output / "model_card.json")


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
