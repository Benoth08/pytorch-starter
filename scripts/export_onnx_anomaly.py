"""Anomalies, script 3 sur 3 (etape 12 du cours) : export ONNX du scoreur complet (auto-encodeur + lissage + score), verifie.

    python scripts/export_onnx_anomaly.py --config configs/anomaly_ae.yaml

Produit dans output_dir : anomaly_scorer.onnx (entree "input" (N, 3, S, S) en
[0, 1] ; sorties "reconstruction", "anomaly_map", "score") et model_card.json
(pretraitement et seuils a appliquer cote production).
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (ici --config)
import json  # json : ecrit la "carte" du modele dans un fichier texte lisible par tout langage
import logging  # logging : messages horodates du programme
from pathlib import Path  # Path : represente un chemin de fichier

import numpy as np  # numpy : comparaison des sorties PyTorch et ONNX

# ONNX Runtime : execute un modele ONNX (format standard de modele, utilisable sans PyTorch, depuis C#, C++...)
import onnxruntime as ort
import pandas as pd  # pandas : relit le decoupage split.csv
import torch  # PyTorch : export ONNX et calcul de reference

# DataLoader, transformations et rechargement du scoreur (voir src/vision_starter/anomaly.py)
from vision_starter.anomaly import build_anomaly_loader, build_anomaly_transforms, load_scorer
from vision_starter.config import AnomalyConfig, load_config  # classe de reglages et lecture du YAML
from vision_starter.utils import setup_logging  # configuration des messages

logger = logging.getLogger(__name__)  # journal propre a ce script


def main() -> None:  # point d'entree : export, verification, carte du modele ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # option obligatoire : chemin du fichier YAML
    args = parser.parse_args()  # lit la ligne de commande
    config: AnomalyConfig = load_config(args.config, AnomalyConfig)  # charge les reglages
    output = config.output_path  # dossier du modele et des resultats
    setup_logging()  # messages dans la console

    scorer, saved = load_scorer(output / "model.pt", torch.device("cpu"))  # recharge le scoreur sur CPU (l'export se fait sur CPU)
    size = saved["image_size"]  # taille des images attendue
    onnx_path = output / "anomaly_scorer.onnx"  # fichier ONNX a produire
    torch.onnx.export(  # convertit le scoreur PyTorch en fichier ONNX
        scorer,  # le modele a exporter (auto-encodeur + lissage + score)
        (torch.zeros(2, 3, size, size),),  # exemple d'entree : un lot de 2 images noires (ni 0 ni 1, cas particuliers)
        str(onnx_path),  # fichier de sortie
        input_names=["input"],  # nom de l'entree
        output_names=["reconstruction", "anomaly_map", "score"],  # noms des trois sorties, dans l'ordre de forward
        opset_version=config.onnx_opset,  # version du catalogue d'operations ONNX, fixee dans le YAML (cle onnx_opset)
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=256)}},  # taille du lot libre (1 a 256) pour l'argument x
        dynamo=True,  # nouvel exporteur de PyTorch (base sur torch.export)
        external_data=False,  # tous les poids dans un seul fichier .onnx
    )

    # Verification sur de vraies images de validation, avec un lot d'une autre taille que l'export.
    _, eval_transform = build_anomaly_transforms(size)  # transformation d'evaluation (_ = celle d'entrainement, inutile)
    catalog = pd.read_csv(output / "split.csv")  # decoupage fige a l'entrainement
    # DataLoader sur les 8 premieres images de validation
    loader = build_anomaly_loader(catalog[catalog["split"] == "val"].head(8), eval_transform, config, train=False)
    images, _, _ = next(iter(loader))  # premier lot (masques et etiquettes inutiles)
    with torch.no_grad():  # pas de gradient : on ne fait que predire
        _, torch_map, torch_score = scorer(images)  # cartes et scores de reference calcules par PyTorch
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # charge le fichier ONNX, execution sur CPU
    _, onnx_map, onnx_score = session.run(None, {"input": images.numpy()})  # memes images passees au modele ONNX
    # plus grands ecarts entre PyTorch et ONNX, sur les scores puis sur les cartes
    gap_score, gap_map = float(np.abs(onnx_score - torch_score.numpy()).max()), float(np.abs(onnx_map - torch_map.numpy()).max())
    if gap_score > 1e-4:  # si les scores different trop
        raise RuntimeError(f"ecart PyTorch / ONNX trop grand sur les scores : {gap_score:.2e}")  # on echoue : export pas fiable
    (output / "model_card.json").write_text(  # ecrit la carte du modele : ce que la production doit savoir
        json.dumps(  # dictionnaire -> texte JSON
            {  # contenu de la carte
                "onnx_file": onnx_path.name,  # nom du fichier ONNX
                "input": {  # description de l'entree
                    "name": "input",  # nom
                    "shape": ["batch", 3, size, size],  # forme : lot, 3 canaux, hauteur, largeur
                    "preprocessing": "RGB, resize bilineaire, /255, pas de normalisation",  # pretraitement a reproduire
                },
                # formes des trois sorties
                "outputs": {"reconstruction": ["batch", 3, size, size], "anomaly_map": ["batch", 1, size, size], "score": ["batch"]},
                "image_threshold": saved["image_threshold"],  # seuil de decision par image
                "pixel_threshold": saved["pixel_threshold"],  # seuil de decision par pixel
                "max_abs_gap_score": gap_score,  # ecart mesure sur les scores (preuve de la verification)
                "max_abs_gap_map": gap_map,  # ecart mesure sur les cartes
            },
            indent=2,  # mise en page lisible
        )
    )
    # bilan : fichier produit, sa taille en Mo et les ecarts mesures
    logger.info("%s (%.1f Mo) | ecart max score %.2e, carte %.2e", onnx_path, onnx_path.stat().st_size / 1e6, gap_score, gap_map)


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
