"""Anomalies, script 1 sur 3 (etapes 3 a 10 du cours) : catalogue, decoupage, entrainement de l'auto-encodeur, calibration des seuils.

    python scripts/train_anomaly.py --config configs/anomaly_ae.yaml

Produit dans output_dir : config_used.yaml, train.log, split.csv, history.csv,
curves.png, best.pt, model.pt (poids + reglages pour scorer) et calibration.json
(seuils image et pixel, fixes sur des images normales de validation).
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (ici --config)
import json  # json : ecrit les seuils dans un fichier texte lisible par tout langage
import logging  # logging : messages horodates du programme
from pathlib import Path  # Path : represente un chemin de fichier

import matplotlib  # matplotlib : bibliotheque de graphiques
import pandas as pd  # pandas : tableaux de donnees (catalogue, historique)
import torch  # PyTorch : sauvegarde et rechargement des poids

matplotlib.use("Agg")  # moteur de dessin sans fenetre : les figures vont dans des fichiers
# pyplot : l'outil de trace de matplotlib, importe apres le choix du moteur ;
# "noqa: E402" dit a ruff d'accepter ces imports places apres du code (c'est voulu)
import matplotlib.pyplot as plt  # noqa: E402

# briques de la detection d'anomalies (voir src/vision_starter/anomaly.py)
from vision_starter.anomaly import (  # noqa: E402
    AnomalyScorer,  # auto-encodeur + lissage -> carte et score d'anomalie
    ConvAutoencoder,  # l'auto-encodeur (compresse puis reconstruit l'image)
    build_anomaly_loader,  # fabrique un DataLoader (images par lots)
    build_anomaly_transforms,  # transformations entrainement / evaluation
    calibrate_thresholds,  # seuils de decision a partir des normales
    catalog_mvtec,  # catalogue des images MVTec AD
    fit_autoencoder,  # boucle d'entrainement complete
    score_loader,  # scores et cartes pour tout un loader
    split_normals,  # met de cote des normales pour la validation
)
from vision_starter.config import AnomalyConfig, load_config, save_config  # noqa: E402  (reglages : classe, lecture, copie)

# outils communs : compter les poids, choisir CPU/GPU, fixer le hasard, configurer les messages
from vision_starter.utils import count_parameters, get_device, seed_everything, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)  # journal propre a ce script


# trace la perte (MSE) d'entrainement et de validation par epoque et l'enregistre en image ; ne renvoie rien
def plot_history(history: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.2))  # une figure avec un seul graphique
    ax.plot(history["epoch"], history["train_loss"], label="train")  # courbe d'entrainement
    ax.plot(history["epoch"], history["val_loss"], label="val (normales)")  # courbe de validation
    ax.set_xlabel("epoque")  # legende de l'axe horizontal
    ax.set_ylabel("MSE de reconstruction")  # legende de l'axe vertical
    ax.set_yscale("log")  # echelle logarithmique : on voit mieux les petites baisses en fin d'entrainement
    ax.legend()  # affiche la legende
    ax.grid(alpha=0.3)  # quadrillage leger
    fig.tight_layout()  # evite les chevauchements
    fig.savefig(path, dpi=110)  # enregistre l'image
    plt.close(fig)  # libere la memoire de la figure


def main() -> None:  # point d'entree : donnees, entrainement, calibration, sauvegardes ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # option obligatoire : chemin du fichier YAML
    args = parser.parse_args()  # lit la ligne de commande
    config: AnomalyConfig = load_config(args.config, AnomalyConfig)  # charge les reglages avec la classe AnomalyConfig
    output = config.output_path  # dossier des resultats
    output.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
    setup_logging(output / "train.log")  # messages a l'ecran et dans train.log
    save_config(config, output / "config_used.yaml")  # copie des reglages utilises
    seed_everything(config.seed)  # fixe le hasard : resultats reproductibles
    device = get_device()  # GPU si disponible, sinon CPU

    # Donnees : seules les normales servent a apprendre et a calibrer ; le test n'est pas lu ici.
    catalog = split_normals(catalog_mvtec(Path(config.data_dir)), config.val_fraction, config.seed)  # catalogue + validation
    catalog.to_csv(output / "split.csv", index=False)  # fige le decoupage (relu par l'evaluation et l'export)
    logger.info("\n%s", pd.crosstab(catalog["label"], catalog["split"]))  # tableau : nombre d'images par etiquette et sous-ensemble
    train_transform, eval_transform = build_anomaly_transforms(config.image_size)  # avec retournements / sans
    # lots d'entrainement : normales du train, melangees, avec retournements
    train_loader = build_anomaly_loader(catalog[catalog["split"] == "train"], train_transform, config, train=True)
    val_loader = build_anomaly_loader(catalog[catalog["split"] == "val"], eval_transform, config, train=False)  # validation

    # Modele et boucle.
    autoencoder = ConvAutoencoder(config.latent_channels).to(device)  # cree l'auto-encodeur et l'envoie sur le device
    # affiche la taille du modele et la largeur du goulot
    logger.info("auto-encodeur : %d parametres | goulot %d canaux", count_parameters(autoencoder), config.latent_channels)
    history = fit_autoencoder(autoencoder, train_loader, val_loader, config, device, output / "best.pt")  # entrainement
    history.to_csv(output / "history.csv", index=False)  # historique en CSV
    plot_history(history, output / "curves.png")  # courbes en image
    best = torch.load(output / "best.pt", map_location=device, weights_only=True)  # relit le meilleur point de l'entrainement
    autoencoder.load_state_dict(best["model_state"])  # remet ses poids dans le modele

    # Calibration : scores des normales de validation -> seuils. Aucun defaut n'a ete vu.
    scorer = AnomalyScorer(autoencoder, config.gaussian_sigma).to(device)  # ajoute le lissage et le calcul du score
    val_scores, val_maps, _, _ = score_loader(scorer, val_loader, device)  # scores et cartes des normales de validation
    # seuils = quantiles de ces scores (ex. 99 % des normales passent sous le seuil image)
    calibration = calibrate_thresholds(val_scores, val_maps, config.image_quantile, config.pixel_quantile)
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2))  # ecrit les seuils en JSON
    torch.save(  # ecrit model.pt : tout ce qu'il faut pour scorer plus tard
        {  # dictionnaire sauvegarde
            "model_state": autoencoder.state_dict(),  # les poids
            "latent_channels": config.latent_channels,  # largeur du goulot (pour recreer l'architecture)
            "image_size": config.image_size,  # taille d'entree attendue
            "gaussian_sigma": config.gaussian_sigma,  # force du lissage de la carte
            "best_epoch": best["epoch"],  # epoque du meilleur modele
            **calibration,  # ajoute les cles du dictionnaire des seuils (image_threshold, pixel_threshold, n_val)
        },
        output / "model.pt",  # fichier de destination
    )
    logger.info(  # bilan final ; chaque %... est remplace par la valeur correspondante ci-dessous
        "meilleure epoque %d (val MSE %.5f) | seuil image %.5f | seuil pixel %.5f -> %s",  # modele du message
        best["epoch"],  # meilleure epoque
        best["val_loss"],  # sa MSE de validation
        calibration["image_threshold"],  # seuil image
        calibration["pixel_threshold"],  # seuil pixel
        output / "model.pt",  # fichier produit
    )


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
