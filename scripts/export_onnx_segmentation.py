"""Segmentation, script 3 sur 3 (etape 12 du cours) : export ONNX du U-Net, verifie sur de vraies images.

    python scripts/export_onnx_segmentation.py --config configs/segmentation.yaml

Produit dans output_dir : unet.onnx (entree "input" (N, 3, S, S) normalisee
ImageNet, sortie "logits" (N, 1, S, S)) et model_card.json (pretraitement,
sigmoide et seuil a appliquer cote production).
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (ici --config)
import json  # json : ecrit la "carte" du modele dans un fichier texte lisible par tout langage
import logging  # logging : messages horodates du programme
from pathlib import Path  # Path : represente un chemin de fichier

import numpy as np  # numpy : comparaison des sorties PyTorch et ONNX

# ONNX Runtime : execute un modele ONNX (format standard de modele, utilisable sans PyTorch, depuis C#, C++...)
import onnxruntime as ort
import pandas as pd  # pandas : relit le decoupage
import torch  # PyTorch : export ONNX et calcul de reference

from vision_starter.config import SegmentationConfig, load_config  # classe de reglages et lecture du YAML
from vision_starter.data import IMAGENET_MEAN, IMAGENET_STD  # normalisation ImageNet, a reporter dans la carte du modele

# DataLoader, transformations et rechargement du U-Net (voir src/vision_starter/segmentation.py)
from vision_starter.segmentation import build_segmentation_loader, build_segmentation_transforms, load_unet
from vision_starter.utils import setup_logging  # configuration des messages

logger = logging.getLogger(__name__)  # journal propre a ce script


def main() -> None:  # point d'entree : export, verification, carte du modele ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # option obligatoire : chemin du fichier YAML
    args = parser.parse_args()  # lit la ligne de commande
    config: SegmentationConfig = load_config(args.config, SegmentationConfig)  # charge les reglages
    output = config.output_path  # dossier du modele et des resultats
    setup_logging()  # messages dans la console

    model, saved = load_unet(output / "model.pt", torch.device("cpu"))  # recharge le U-Net sur CPU (l'export se fait sur CPU)
    size = saved["image_size"]  # taille des images attendue
    onnx_path = output / "unet.onnx"  # fichier ONNX a produire
    torch.onnx.export(  # convertit le U-Net PyTorch en fichier ONNX
        model,  # le modele a exporter
        (torch.zeros(2, 3, size, size),),  # exemple d'entree : un lot de 2 images (ni 0 ni 1, cas particuliers)
        str(onnx_path),  # fichier de sortie
        input_names=["input"],  # nom de l'entree
        output_names=["logits"],  # nom de la sortie : un logit par pixel
        opset_version=config.onnx_opset,  # version du catalogue d'operations ONNX, fixee dans le YAML (cle onnx_opset)
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=256)}},  # taille du lot libre (1 a 256) pour l'argument x
        dynamo=True,  # nouvel exporteur de PyTorch (base sur torch.export)
        external_data=False,  # tous les poids dans un seul fichier .onnx
    )

    _, eval_transform = build_segmentation_transforms(size)  # transformation d'evaluation (_ = celle d'entrainement, inutile)
    catalog = pd.read_csv(config.split_file)  # decoupage fige a l'entrainement
    # premier lot construit a partir des 3 premieres images de validation (le masque, _, ne sert pas)
    images, _ = next(iter(build_segmentation_loader(catalog[catalog["split"] == "val"].head(3), eval_transform, config, train=False)))
    with torch.no_grad():  # pas de gradient : on ne fait que predire
        torch_logits = model(images).numpy()  # sortie de reference calculee par PyTorch
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # charge le fichier ONNX, execution sur CPU
    onnx_logits = session.run(None, {"input": images.numpy()})[0]  # sortie ONNX pour les memes images
    # Ecart RELATIF : les logits valent plusieurs unites, et deux moteurs n'accumulent pas les convolutions dans le meme
    # ordre ; un ecart absolu de 1e-3 sur des valeurs de 10 est du bruit numerique, pas une erreur d'export.
    gap = float(np.abs(onnx_logits - torch_logits).max())  # plus grand ecart absolu entre les deux sorties
    scale = float(np.abs(torch_logits).max())  # plus grande valeur absolue des logits (ordre de grandeur)
    if gap / max(scale, 1.0) > 1e-3:  # si l'ecart depasse 0,1 % de cet ordre de grandeur
        # on echoue avec un message qui donne l'ecart et l'ordre de grandeur
        raise RuntimeError(f"ecart PyTorch / ONNX trop grand sur les logits : {gap:.2e} pour des valeurs allant jusqu'a {scale:.1f}")
    (output / "model_card.json").write_text(  # ecrit la carte du modele : ce que la production doit savoir
        json.dumps(  # dictionnaire -> texte JSON
            {  # contenu de la carte
                "onnx_file": onnx_path.name,  # nom du fichier ONNX
                "input": {  # description de l'entree
                    "name": "input",  # nom
                    "shape": ["batch", 3, size, size],  # forme : lot, 3 canaux, hauteur, largeur
                    "preprocessing": "RGB, resize bilineaire, /255, normalisation ImageNet",  # pretraitement a reproduire
                    "mean": IMAGENET_MEAN,  # moyenne par canal
                    "std": IMAGENET_STD,  # ecart-type par canal
                },
                # sortie : nom, forme, et post-traitement (sigmoide puis comparaison au seuil retenu)
                "output": {"name": "logits", "shape": ["batch", 1, size, size], "postprocessing": f"sigmoid > {saved['threshold']}"},
                "classes": ["fond", "defaut"],  # signification de 0 et 1
                "max_abs_gap_logits": gap,  # ecart mesure (preuve de la verification)
                "max_abs_logit": scale,  # ordre de grandeur des logits
            },
            indent=2,  # mise en page lisible
        )
    )
    logger.info(  # bilan : fichier, taille en Mo, ecart mesure et ordre de grandeur
        "%s (%.1f Mo) | ecart max PyTorch / ONNX %.2e sur des logits jusqu'a %.1f",  # modele du message
        onnx_path,  # fichier produit
        onnx_path.stat().st_size / 1e6,  # taille en Mo (octets / 1 million)
        gap,  # ecart max mesure
        scale,  # plus grand logit
    )


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
