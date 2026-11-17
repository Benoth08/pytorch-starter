"""Segmentation, script 2 sur 3 (etape 11 du cours) : evaluation sur le test au seuil retenu, IoU global et par classe, figure.

    python scripts/evaluate_segmentation.py --config configs/segmentation.yaml

Produit dans output_dir : test_metrics.json et predictions.png (image, verite,
prediction pour chaque classe).
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (ici --config)
import json  # json : ecrit les metriques dans un fichier texte lisible par tout langage
import logging  # logging : messages horodates du programme
from pathlib import Path  # Path : represente un chemin de fichier

import matplotlib  # matplotlib : bibliotheque de graphiques
import numpy as np  # numpy : tableaux de nombres (masques pour la figure)
import pandas as pd  # pandas : tableaux de donnees (catalogue de test, tableau par classe)
import torch  # PyTorch : tenseurs des logits et des masques
from PIL import Image  # Pillow : ouvre les images pour la figure

matplotlib.use("Agg")  # moteur de dessin sans fenetre : les figures vont dans des fichiers
# pyplot : l'outil de trace de matplotlib, importe apres le choix du moteur ;
# "noqa: E402" dit a ruff d'accepter ces imports places apres du code (c'est voulu)
import matplotlib.pyplot as plt  # noqa: E402

from vision_starter.config import SegmentationConfig, load_config  # noqa: E402  (classe de reglages et lecture du YAML)

# briques de la segmentation (voir src/vision_starter/segmentation.py)
from vision_starter.segmentation import (
    build_segmentation_loader,  # fabrique un DataLoader (images + masques par lots)
    build_segmentation_transforms,  # transformations entrainement / evaluation
    collect_logits,  # rassemble les sorties du modele pour tout un loader
    confusion_pixels,  # compte pixels communs / predits / vrais
    iou_dice,  # calcule IoU et Dice a partir de ces comptes
    load_unet,  # recharge le U-Net entraine
)  # noqa: E402
from vision_starter.utils import get_device, setup_logging  # noqa: E402  (choix CPU/GPU, messages)

logger = logging.getLogger(__name__)  # journal propre a ce script


# metriques separees par classe ; renvoie {classe: {metrique: valeur}}
def per_class_metrics(test_df: pd.DataFrame, logits: torch.Tensor, masks: torch.Tensor, threshold: float, min_area: int) -> dict[str, dict]:
    """IoU et Dice par type de defaut ; pour les images sans defaut, le taux de fausses detections (plus de min_area pixels)."""
    result = {}  # dictionnaire des resultats, rempli classe par classe
    for name in sorted(test_df["label"].unique()):  # pour chaque classe presente dans le test, par ordre alphabetique
        sel = torch.tensor((test_df["label"] == name).to_numpy())  # masque vrai / faux : les images de cette classe
        inter, pred, truth = confusion_pixels(logits[sel], masks[sel], threshold)  # comptes de pixels pour ces images
        if truth == 0:  # si aucun pixel de defaut (classe "free") : l'IoU n'a pas de sens
            # part des images ou le modele "voit" plus de min_area pixels de defaut alors qu'il n'y en a pas (fausse detection)
            flagged = ((torch.sigmoid(logits[sel]) > threshold).flatten(1).sum(dim=1) > min_area).float().mean().item()
            # nombre d'images, taux de fausses detections, pixels predits a tort
            result[name] = {"images": int(sel.sum()), "false_detections": round(flagged, 3), "predicted_pixels": pred}
        else:  # sinon (classe avec defauts)
            iou, dice = iou_dice(inter, pred, truth)  # recouvrement prediction / verite
            result[name] = {"images": int(sel.sum()), "iou": round(iou, 3), "dice": round(dice, 3)}  # nombre d'images, IoU, Dice
    return result  # toutes les classes


# figure : pour chaque classe, l'image, la vraie zone et la zone predite ; ne renvoie rien
def plot_predictions(test_df: pd.DataFrame, logits: torch.Tensor, masks: torch.Tensor, threshold: float, size: int, path: Path) -> None:
    names = sorted(test_df["label"].unique())  # classes du test, par ordre alphabetique
    fig, axes = plt.subplots(3, len(names), figsize=(2.5 * len(names), 7.6))  # grille : 3 lignes, une colonne par classe
    for col, name in enumerate(names):  # pour chaque classe (col = numero de colonne)
        i = int(np.flatnonzero((test_df["label"] == name).to_numpy())[0])  # indice de la 1re image de cette classe
        image = Image.open(test_df.iloc[i]["path"]).convert("L").resize((size, size))  # image en gris, a la taille du modele
        pred = torch.sigmoid(logits[i, 0]).numpy() > threshold  # zone predite : probabilite > seuil (tableau vrai / faux)
        truth = masks[i, 0].numpy() > 0.5  # vraie zone de defaut
        inter, union = (pred & truth).sum(), (pred | truth).sum()  # pixels communs (ET) et pixels de l'union (OU)
        axes[0, col].imshow(image, cmap="gray")  # ligne 1 : l'image seule
        axes[0, col].set_title(name, fontsize=9)  # nom de la classe en titre
        axes[1, col].imshow(image, cmap="gray")  # ligne 2 : l'image...
        # ... avec la verite en couleur par-dessus (masked_where cache les pixels hors defaut, alpha = transparence)
        axes[1, col].imshow(np.ma.masked_where(~truth, truth), cmap="autumn", alpha=0.6)
        axes[2, col].imshow(image, cmap="gray")  # ligne 3 : l'image...
        axes[2, col].imshow(np.ma.masked_where(~pred, pred), cmap="winter", alpha=0.6)  # ... avec la prediction par-dessus
        # titre : IoU de cette image, ou message si ni verite ni prediction (union vide)
        axes[2, col].set_title(f"IoU {inter / union:.2f}" if union else "rien a detecter, rien detecte", fontsize=8)
    for ax, name in zip(axes[:, 0], ("image", "verite", "prediction")):  # pour la 1re colonne de chaque ligne
        ax.set_ylabel(name)  # nom de la ligne sur le cote gauche
    for ax in axes.ravel():  # pour chaque case de la grille
        ax.set_xticks([])  # pas de graduations horizontales
        ax.set_yticks([])  # pas de graduations verticales (on garde le nom de ligne)
    fig.tight_layout()  # evite les chevauchements
    fig.savefig(path, dpi=110)  # enregistre l'image
    plt.close(fig)  # libere la memoire de la figure


def main() -> None:  # point d'entree : predit le test, calcule les metriques, ecrit les resultats ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # option obligatoire : chemin du fichier YAML
    args = parser.parse_args()  # lit la ligne de commande
    config: SegmentationConfig = load_config(args.config, SegmentationConfig)  # charge les reglages
    output = config.output_path  # dossier du modele et des resultats
    setup_logging(output / "evaluate.log")  # messages a l'ecran et dans evaluate.log
    device = get_device()  # GPU si disponible, sinon CPU

    model, saved = load_unet(output / "model.pt", device)  # recharge le U-Net et les reglages sauvegardes (taille, seuil)
    _, eval_transform = build_segmentation_transforms(saved["image_size"])  # transformation d'evaluation
    catalog = pd.read_csv(config.split_file)  # decoupage fige a l'entrainement
    test_df = catalog[catalog["split"] == "test"].reset_index(drop=True)  # lignes du test, renumerotees 0..n-1
    # logits et masques de tout le test (le DataLoader est cree directement dans l'appel)
    logits, masks = collect_logits(model, build_segmentation_loader(test_df, eval_transform, config, train=False), device)

    iou, dice = iou_dice(*confusion_pixels(logits, masks, saved["threshold"]))  # IoU et Dice globaux au seuil retenu
    metrics = {  # metriques a enregistrer
        "iou": iou,  # IoU global
        "dice": dice,  # Dice global
        "threshold": saved["threshold"],  # seuil utilise
        "n_images": len(test_df),  # nombre d'images de test
        "per_class": per_class_metrics(test_df, logits, masks, saved["threshold"], config.min_defect_area),  # detail par classe
    }
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2))  # ecrit les metriques en JSON
    # dessine la figure image / verite / prediction
    plot_predictions(test_df, logits, masks, saved["threshold"], saved["image_size"], output / "predictions.png")
    logger.info(  # bilan ; chaque %... est remplace par la valeur correspondante ci-dessous
        "test (%d images) : IoU %.3f | Dice %.3f (seuil %.2f)\n%s",  # modele du message
        len(test_df),  # nombre d'images
        iou,  # IoU global
        dice,  # Dice global
        saved["threshold"],  # seuil
        pd.DataFrame(metrics["per_class"]).T.to_string(na_rep=""),  # tableau par classe (.T = une ligne par classe ; cases vides si absent)
    )


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
