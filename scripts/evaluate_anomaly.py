"""Anomalies, script 2 sur 3 (etape 11 du cours) : evaluation sur le test (normales + defauts), une seule fois.

    python scripts/evaluate_anomaly.py --config configs/anomaly_ae.yaml

Produit dans output_dir : test_metrics.json (AUROC image et pixel, rappel et
fausses alarmes au seuil calibre, par type de defaut), scores_test.csv et
anomaly_maps.png (image, carte, contour du masque, pour chaque type).
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (ici --config)
import json  # json : ecrit les metriques dans un fichier texte lisible par tout langage
import logging  # logging : messages horodates du programme
from pathlib import Path  # Path : represente un chemin de fichier

import matplotlib  # matplotlib : bibliotheque de graphiques
import numpy as np  # numpy : tableaux de nombres (scores, cartes, masques)
import pandas as pd  # pandas : tableaux de donnees (catalogue de test, scores)
from PIL import Image  # Pillow : ouvre les images pour la figure

matplotlib.use("Agg")  # moteur de dessin sans fenetre : les figures vont dans des fichiers
# pyplot : l'outil de trace de matplotlib, importe apres le choix du moteur ;
# "noqa: E402" dit a ruff d'accepter ces imports places apres du code (c'est voulu)
import matplotlib.pyplot as plt  # noqa: E402

# metriques, DataLoader, transformations, rechargement du scoreur, scores d'un loader (voir src/vision_starter/anomaly.py)
from vision_starter.anomaly import anomaly_metrics, build_anomaly_loader, build_anomaly_transforms, load_scorer, score_loader  # noqa: E402
from vision_starter.config import AnomalyConfig, load_config  # noqa: E402  (classe de reglages et lecture du YAML)
from vision_starter.utils import get_device, setup_logging  # noqa: E402  (choix CPU/GPU, messages)

logger = logging.getLogger(__name__)  # journal propre a ce script


# figure : pour chaque type de defaut, une image (en haut) et sa carte d'anomalie (en bas) ; ne renvoie rien
def plot_maps(
    test_df: pd.DataFrame,  # catalogue du test (une ligne par image)
    scores: np.ndarray,  # score de chaque image
    maps: np.ndarray,  # carte d'anomalie de chaque image
    masks: np.ndarray,  # vrai masque de chaque image
    threshold: float,  # seuil de decision image
    pixel_threshold: float,  # seuil pixel (sert a regler l'echelle des couleurs)
    size: int,  # taille des images affichees
    path: Path,  # fichier image a ecrire
) -> None:  # ne renvoie rien
    types = list(dict.fromkeys(test_df["defect_type"]))  # dans l'ordre d'apparition, sans doublon
    fig, axes = plt.subplots(2, len(types), figsize=(3.3 * len(types), 6.4))  # grille : 2 lignes, une colonne par type
    for col, defect_type in enumerate(types):  # pour chaque type (col = numero de colonne)
        i = int(np.flatnonzero((test_df["defect_type"] == defect_type).to_numpy())[0])  # indice de la 1re image de ce type
        axes[0, col].imshow(Image.open(test_df.iloc[i]["path"]).convert("RGB").resize((size, size)))  # en haut : l'image
        # titre : type, score, et position par rapport au seuil
        axes[0, col].set_title(f"{defect_type} | {scores[i]:.4f} {'> seuil' if scores[i] > threshold else '<= seuil'}", fontsize=8)
        # en bas : la carte en fausses couleurs ; echelle fixe de 0 a 2 x seuil pixel (max evite une echelle nulle)
        axes[1, col].imshow(maps[i], cmap="inferno", vmin=0, vmax=max(2 * pixel_threshold, 1e-6))
        if masks[i].any():  # si l'image a un vrai defaut
            axes[1, col].contour(masks[i], levels=[0.5], colors="cyan", linewidths=0.8)  # dessine le contour du vrai masque en cyan
    for ax in axes.ravel():  # pour chaque case de la grille
        ax.axis("off")  # cache les axes gradues (inutiles sur des images)
    fig.tight_layout()  # evite les chevauchements
    fig.savefig(path, dpi=110)  # enregistre l'image
    plt.close(fig)  # libere la memoire de la figure


def main() -> None:  # point d'entree : score le test, calcule les metriques, ecrit les resultats ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # option obligatoire : chemin du fichier YAML
    args = parser.parse_args()  # lit la ligne de commande
    config: AnomalyConfig = load_config(args.config, AnomalyConfig)  # charge les reglages
    output = config.output_path  # dossier du modele et des resultats
    setup_logging(output / "evaluate.log")  # messages a l'ecran et dans evaluate.log
    device = get_device()  # GPU si disponible, sinon CPU

    scorer, saved = load_scorer(output / "model.pt", device)  # recharge le scoreur et les reglages sauvegardes (seuils...)
    _, eval_transform = build_anomaly_transforms(saved["image_size"])  # transformation d'evaluation (_ = celle d'entrainement)
    catalog = pd.read_csv(output / "split.csv")  # decoupage fige a l'entrainement
    test_df = catalog[catalog["split"] == "test"].reset_index(drop=True)  # lignes du test, renumerotees 0..n-1
    test_loader = build_anomaly_loader(test_df, eval_transform, config, train=False)  # lots du test, ordre fixe
    scores, maps, masks, labels = score_loader(scorer, test_loader, device)  # scores, cartes, masques et etiquettes du test

    metrics = anomaly_metrics(scores, maps, masks, labels, saved["image_threshold"])  # AUROC, rappel, fausses alarmes, F1
    metrics["image_threshold"] = saved["image_threshold"]  # garde aussi le seuil utilise
    flagged = (scores > saved["image_threshold"]).astype(int)  # 1 = image signalee comme anomalie
    # part d'images signalees pour chaque type de defaut (pour "good", c'est le taux de fausses alarmes)
    metrics["flagged_by_type"] = test_df.assign(flagged=flagged).groupby("defect_type")["flagged"].mean().round(3).to_dict()
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2))  # ecrit les metriques en JSON
    test_df.assign(score=scores, flagged=flagged).to_csv(output / "scores_test.csv", index=False)  # detail par image en CSV
    plot_maps(  # dessine la figure des cartes d'anomalie
        test_df,  # catalogue du test
        scores,  # scores des images
        maps,  # cartes d'anomalie
        masks,  # vrais masques
        saved["image_threshold"],  # seuil image calibre
        saved["pixel_threshold"],  # seuil pixel calibre
        saved["image_size"],  # taille des images
        output / "anomaly_maps.png",  # fichier de sortie
    )
    logger.info(  # bilan ; chaque %... est remplace par la valeur correspondante ci-dessous
        "test (%d images) : AUROC image %.3f | AUROC pixel %.3f | rappel %.3f | fausses alarmes %.3f | F1 %.3f",  # modele du message
        metrics["n_images"],  # nombre d'images
        metrics["auroc_image"],  # AUROC image
        metrics.get("auroc_pixel", float("nan")),  # AUROC pixel (nan si absente : aucun masque de defaut)
        metrics["recall"],  # rappel
        metrics["false_alarm_rate"],  # taux de fausses alarmes
        metrics["f1"],  # F1
    )
    logger.info("part signalee par type : %s", metrics["flagged_by_type"])  # detail par type


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
