"""Script 4 sur 5 (etape 12 du cours) : export ONNX du modele sauvegarde, puis verification avec onnxruntime.

    python scripts/export_onnx.py --config configs/default.yaml

Le script echoue si les sorties PyTorch et ONNX different sur de vraies images
de validation : un export qui passe sans verification ne vaut rien.
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (ici --config)
import json  # json : ecrit la "carte" du modele dans un fichier texte lisible par tout langage
import logging  # logging : messages horodates du programme
import platform  # platform : version de Python, notee dans la carte du modele
from datetime import UTC, datetime  # date et heure de l'export, en temps universel
from pathlib import Path  # Path : represente un chemin de fichier

import numpy as np  # numpy : comparaison des sorties PyTorch et ONNX
import onnx  # onnx : bibliotheque du format ONNX (ici pour noter sa version)

# ONNX Runtime : execute un modele ONNX (format standard de modele, lisible depuis C#, C++, Java... sans PyTorch)
import onnxruntime as ort
import torch  # PyTorch : chargement du modele et export ONNX

from vision_starter.config import load_config  # lit le fichier YAML de reglages
from vision_starter.data import build_loader, build_transforms, load_split  # DataLoader, transformations, decoupage
from vision_starter.model import rebuild_model  # recree le modele enregistre, avec ses poids
from vision_starter.utils import setup_logging  # configuration des messages

logger = logging.getLogger(__name__)  # journal propre a ce script


def main() -> None:  # point d'entree : export ONNX, verification, ecriture de la carte du modele ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # option obligatoire : chemin du fichier YAML
    args = parser.parse_args()  # lit la ligne de commande
    config = load_config(args.config)  # charge et valide les reglages
    output = config.output_path  # dossier du modele et des resultats
    setup_logging(output / "export_onnx.log")  # messages a l'ecran et dans export_onnx.log
    device = torch.device("cpu")  # l'export se fait toujours sur CPU

    saved = torch.load(output / "model.pt", map_location=device, weights_only=True)  # relit le modele sauvegarde par train.py
    class_names = saved["class_names"]  # noms des classes dans l'ordre de l'entrainement
    # architecture notee dans model.pt + poids appris, en mode evaluation (dropout coupe, BatchNorm figee : indispensable ici)
    model = rebuild_model(saved, config)

    example = torch.zeros(2, 3, saved["image_size"], saved["image_size"])  # lot de 2 : ni 0 ni 1, cas particuliers
    onnx_path = output / "model.onnx"  # fichier ONNX a produire
    torch.onnx.export(  # convertit le modele PyTorch en fichier ONNX
        model,  # le modele a exporter
        (example,),  # exemple d'entree (tuple) : sert a suivre les calculs du modele
        str(onnx_path),  # fichier de sortie
        input_names=["input"],  # nom de l'entree dans le fichier ONNX
        output_names=["logits"],  # nom de la sortie (scores bruts par classe)
        opset_version=config.onnx_opset,  # version du catalogue d'operations ONNX, fixee dans le YAML (cle onnx_opset)
        # la 1re dimension (taille du lot) de l'argument x de forward reste libre : de 1 a 1024 images par appel
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=1024)}},
        dynamo=True,  # utilise le nouvel exporteur de PyTorch (base sur torch.export)
        external_data=False,  # tous les poids dans un seul fichier .onnx (pas de fichier annexe)
    )

    # Verification sur de vraies images de validation, avec un lot d'une autre taille que l'export.
    _, eval_transform = build_transforms(saved["image_size"], saved["mean"], saved["std"])  # pretraitement d'evaluation
    class_to_idx = {name: i for i, name in enumerate(class_names)}  # nom de classe -> numero
    # DataLoader sur les 16 premieres images de validation (head(16))
    loader = build_loader(load_split(Path(config.split_file), "val").head(16), class_to_idx, eval_transform, config, train=False)
    images, _ = next(iter(loader))  # prend le premier lot (les etiquettes ne servent pas)
    with torch.no_grad():  # pas de gradient : on ne fait que predire
        torch_logits = model(images).numpy()  # sortie de reference calculee par PyTorch
    # session ONNX Runtime : charge le fichier et l'execute sur le processeur
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {"input": images.numpy()})[0]  # sortie ONNX pour les memes images (None = toutes les sorties)
    max_diff = float(np.abs(torch_logits - onnx_logits).max())  # plus grand ecart entre les deux sorties
    if not np.allclose(torch_logits, onnx_logits, atol=1e-4, rtol=1e-3):  # si les sorties ne sont pas egales aux arrondis pres
        raise RuntimeError(f"export ONNX incorrect : ecart max {max_diff:.2e}")  # on echoue : ce fichier ONNX n'est pas fiable

    # La carte du modele : tout ce qu'un programme (Python, C#, C++) doit savoir pour l'utiliser.
    metrics_file = output / "test_metrics.json"  # chiffres du test, ecrits par evaluate.py (a lancer avant l'export)
    card = {  # dictionnaire ecrit en JSON a cote du .onnx
        "model_name": saved["model_name"],  # architecture du reseau
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),  # date de l'export, en temps universel
        "input_name": "input",  # nom de l'entree
        "output_name": "logits",  # nom de la sortie
        "layout": "NCHW",  # ordre des dimensions : lot, canaux, hauteur, largeur
        "image_size": saved["image_size"],  # taille des images attendue
        "resize": "stretch to image_size x image_size, bilinear, antialias",  # redimensionnement a reproduire (sans garder le rapport)
        "channels": "RGB",  # ordre des couleurs attendu
        "scale": "divide by 255 then (x - mean) / std per channel",  # pretraitement a reproduire
        "mean": saved["mean"],  # moyenne par canal
        "std": saved["std"],  # ecart-type par canal
        "class_names": class_names,  # numero de sortie -> nom de classe
        "opset": config.onnx_opset,  # version du catalogue d'operations du fichier .onnx
        "max_diff_pytorch_onnx": max_diff,  # plus grand ecart mesure a la verification
        "metrics": json.loads(metrics_file.read_text()) if metrics_file.is_file() else {},  # chiffres du test ; vide si pas evalue
        "versions": {  # versions des bibliotheques : la premiere question quand un resultat differe
            "python": platform.python_version(),  # version de Python
            "torch": torch.__version__,  # version de PyTorch
            "onnx": onnx.__version__,  # version d'onnx
            "onnxruntime": ort.__version__,  # version du moteur d'execution
        },
    }
    if not card["metrics"]:  # carte sans chiffres : le modele n'a pas ete mesure sur le test
        # avertissement dans le journal : l'ordre attendu est evaluate.py, puis export_onnx.py
        logger.warning("test_metrics.json absent : lancer scripts/evaluate.py avant l'export pour que la carte porte les chiffres")
    (output / "model_card.json").write_text(json.dumps(card, indent=2))  # ecrit la carte
    # bilan : fichier produit, ecart max mesure et taille du lot verifie
    logger.info("export OK : %s | ecart max PyTorch / ONNX %.2e | lot verifie %d", onnx_path, max_diff, len(images))


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
