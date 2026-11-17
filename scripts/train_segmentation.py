"""Segmentation, script 1 sur 3 (etapes 3 a 10 du cours) : catalogue, decoupage, entrainement du U-Net, choix du seuil sur la validation.

    python scripts/train_segmentation.py --config configs/segmentation.yaml

Produit dans output_dir : config_used.yaml, train.log, split.csv, history.csv,
curves.png, best.pt et model.pt (poids + taille d'image + seuil retenu).
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (ici --config)
import json  # json : ecrit les metriques dans un fichier texte lisible par tout langage
import logging  # logging : messages horodates du programme
from pathlib import Path  # Path : represente un chemin de fichier

import matplotlib  # matplotlib : bibliotheque de graphiques
import pandas as pd  # pandas : tableaux de donnees (catalogue, historique)
import torch  # PyTorch : sauvegarde et rechargement des poids

matplotlib.use("Agg")  # moteur de dessin sans fenetre : les figures vont dans des fichiers
# pyplot : l'outil de trace de matplotlib, importe apres le choix du moteur ;
# "noqa: E402" dit a ruff d'accepter ces imports places apres du code (c'est voulu)
import matplotlib.pyplot as plt  # noqa: E402

from vision_starter.config import SegmentationConfig, load_config, save_config  # noqa: E402  (reglages : classe, lecture, copie)
from vision_starter.data import check_group_leak, check_split  # noqa: E402  (controles du decoupage : fuites, doublons, vides)

# briques de la segmentation (voir src/vision_starter/segmentation.py)
from vision_starter.segmentation import (  # noqa: E402
    ResNetUNet,  # le U-Net (encodeur ResNet18 + decodeur)
    build_segmentation_loader,  # fabrique un DataLoader (images + masques par lots)
    build_segmentation_transforms,  # transformations entrainement / evaluation
    catalog_magnetic_tile,  # catalogue des images Magnetic Tile et de leurs masques
    choose_threshold,  # choisit le seuil de probabilite sur la validation
    collect_logits,  # rassemble les sorties du modele pour tout un loader
    fit_segmentation,  # boucle d'entrainement complete
    split_segmentation,  # decoupage train / val / test
)

# outils communs : compter les poids, choisir CPU/GPU, fixer le hasard, configurer les messages
from vision_starter.utils import count_parameters, get_device, seed_everything, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)  # journal propre a ce script


# trace la perte (a gauche) et l'IoU / le Dice de validation (a droite) et enregistre l'image ; ne renvoie rien
def plot_history(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))  # une figure avec 2 graphiques cote a cote
    axes[0].plot(history["epoch"], history["train_loss"], label="train")  # perte d'entrainement
    axes[0].plot(history["epoch"], history["val_loss"], label="val")  # perte de validation
    axes[0].set_title("perte BCE + Dice")  # titre du graphique de gauche
    axes[1].plot(history["epoch"], history["val_iou"], label="IoU val")  # IoU de validation (recouvrement prediction / verite)
    axes[1].plot(history["epoch"], history["val_dice"], label="Dice val")  # Dice de validation
    axes[1].set_title("qualite des masques (val)")  # titre du graphique de droite
    for ax in axes:  # pour chacun des deux graphiques
        ax.set_xlabel("epoque")  # legende de l'axe horizontal
        ax.legend()  # affiche la legende
        ax.grid(alpha=0.3)  # quadrillage leger
    fig.tight_layout()  # evite les chevauchements
    fig.savefig(path, dpi=110)  # enregistre l'image
    plt.close(fig)  # libere la memoire de la figure


def main() -> None:  # point d'entree : donnees, entrainement, choix du seuil, sauvegardes ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # option obligatoire : chemin du fichier YAML
    args = parser.parse_args()  # lit la ligne de commande
    config: SegmentationConfig = load_config(args.config, SegmentationConfig)  # charge les reglages de segmentation
    output = config.output_path  # dossier des resultats
    output.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
    setup_logging(output / "train.log")  # messages a l'ecran et dans train.log
    save_config(config, output / "config_used.yaml")  # copie des reglages utilises
    seed_everything(config.seed)  # fixe le hasard : resultats reproductibles
    device = get_device()  # GPU si disponible, sinon CPU

    # Donnees : catalogue (image + masque), decoupage stratifie fige en CSV.
    # catalogue des images (avec n_free images sans defaut au plus) puis decoupage train / val / test
    catalog = split_segmentation(catalog_magnetic_tile(Path(config.data_dir), config.n_free, config.seed), config)
    check_split(catalog)  # refuse doublons et sous-ensembles vides
    check_group_leak(catalog, "group")  # refuse un groupe de quasi-doublons a cheval sur deux sous-ensembles
    group_sizes = catalog["group"].value_counts()  # nombre d'images par groupe
    logger.info(  # bilan des quasi-doublons : ils restent ensemble, du meme cote du decoupage
        "%d images | %d groupes | %d groupes de quasi-doublons (%d images)",  # modele du message
        len(catalog),  # nombre d'images
        len(group_sizes),  # nombre de groupes (une image seule forme un groupe)
        int((group_sizes > 1).sum()),  # groupes de 2 images ou plus
        int(group_sizes[group_sizes > 1].sum()),  # images concernees
    )
    split_file = Path(config.split_file)  # chemin du CSV de decoupage
    split_file.parent.mkdir(parents=True, exist_ok=True)  # cree son dossier si besoin
    catalog.to_csv(split_file, index=False)  # fige le decoupage (relu par l'evaluation et l'export)
    logger.info("\n%s", pd.crosstab(catalog["label"], catalog["split"]))  # tableau : nombre d'images par classe et sous-ensemble
    train_transform, eval_transform = build_segmentation_transforms(config.image_size)  # avec retournements / sans
    # lots d'entrainement (melanges, avec retournements)
    train_loader = build_segmentation_loader(catalog[catalog["split"] == "train"], train_transform, config, train=True)
    val_loader = build_segmentation_loader(catalog[catalog["split"] == "val"], eval_transform, config, train=False)  # validation

    # Modele et boucle.
    model = ResNetUNet(pretrained=config.pretrained).to(device)  # cree le U-Net et l'envoie sur le device
    logger.info("U-Net : %d parametres", count_parameters(model))  # taille du modele
    history = fit_segmentation(model, train_loader, val_loader, config, device, output / "best.pt")  # entrainement
    history.to_csv(output / "history.csv", index=False)  # historique en CSV
    plot_history(history, output / "curves.png")  # courbes en image
    best = torch.load(output / "best.pt", map_location=device, weights_only=True)  # relit le meilleur point de l'entrainement
    model.load_state_dict(best["model_state"])  # remet ses poids dans le modele

    # Le seuil par pixel se choisit sur la validation, jamais sur le test.
    val_logits, val_masks = collect_logits(model, val_loader, device)  # sorties du modele et vrais masques de la validation
    threshold, sweep = choose_threshold(val_logits, val_masks, config.thresholds)  # meilleur seuil et IoU de chaque candidat
    # affiche l'IoU obtenue avec chaque seuil (arrondie) et le seuil retenu
    logger.info("IoU de validation par seuil : %s -> seuil retenu %.2f", {t: round(v, 3) for t, v in sweep.items()}, threshold)
    torch.save(  # ecrit model.pt : tout ce qu'il faut pour predire plus tard
        # poids, taille d'image, seuil retenu, epoque du meilleur modele
        {"model_state": model.state_dict(), "image_size": config.image_size, "threshold": threshold, "best_epoch": best["epoch"]},
        output / "model.pt",  # fichier de destination
    )
    (output / "val_metrics.json").write_text(  # ecrit les resultats de validation en JSON
        # meilleure epoque, sa perte, le seuil retenu et l'IoU de chaque seuil essaye
        json.dumps({"best_epoch": best["epoch"], "val_loss": best["val_loss"], "threshold": threshold, "iou_by_threshold": sweep}, indent=2)
    )
    logger.info("meilleure epoque %d (val perte %.4f) -> %s", best["epoch"], best["val_loss"], output / "model.pt")  # bilan final


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
