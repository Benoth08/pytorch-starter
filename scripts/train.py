"""Script 2 sur 5 (etapes 5 a 10 du cours) : entrainement. Assemble les briques : config -> donnees -> modele -> boucle.

    python scripts/train.py --config configs/default.yaml

Produit dans output_dir : config_used.yaml, train.log, history.csv, best.pt,
model.pt (poids + tout ce qu'il faut pour predire), curves.png.
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (ici --config)
import json  # json : ecrit les metriques dans un fichier texte lisible par tout langage
import logging  # logging : messages horodates du programme
from pathlib import Path  # Path : represente un chemin de fichier

import matplotlib  # matplotlib : bibliotheque de graphiques
import torch  # PyTorch : tenseurs, sauvegarde des poids, optimiseurs
import torch.nn as nn  # nn : briques de reseaux de neurones (ici la fonction de perte)

matplotlib.use("Agg")  # moteur de dessin sans fenetre : les figures vont dans des fichiers (marche aussi sur un serveur)
# pyplot : l'outil de trace de matplotlib ; importe APRES le choix du moteur.
# "noqa: E402" dit a ruff d'accepter ces imports places apres du code (c'est voulu ici)
import matplotlib.pyplot as plt  # noqa: E402

from vision_starter.config import load_config, save_config  # noqa: E402  (lecture et copie des reglages YAML)

# fonctions de donnees du projet (voir src/vision_starter/data.py)
from vision_starter.data import (  # noqa: E402
    IMAGENET_MEAN,  # moyenne des pixels d'ImageNet par canal
    IMAGENET_STD,  # ecart-type des pixels d'ImageNet par canal
    build_loader,  # fabrique un DataLoader (images par lots)
    build_transforms,  # fabrique les transformations entrainement / evaluation
    class_names_of,  # liste triee des classes
    class_weights,  # poids des classes pour la perte
    compute_mean_std,  # moyenne et ecart-type calcules sur notre train
    load_split,  # relit un sous-ensemble du decoupage
)
from vision_starter.engine import fit  # noqa: E402  (la boucle d'entrainement complete)
from vision_starter.model import build_model, build_optimizer  # noqa: E402  (fabrique du modele, fabrique de l'optimiseur)

# outils communs : compter les poids, choisir CPU/GPU, fixer le hasard, configurer les messages
from vision_starter.utils import count_parameters, get_device, seed_everything, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)  # journal propre a ce script


# trace les courbes de perte et d'accuracy (train et validation) et les enregistre en image ; ne renvoie rien
def plot_history(history, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))  # une figure avec 2 graphiques cote a cote (taille en pouces)
    for ax, metric in zip(axes, ("loss", "acc")):  # graphique de gauche : perte ; de droite : accuracy
        ax.plot(history["epoch"], history[f"train_{metric}"], label="train")  # courbe d'entrainement
        ax.plot(history["epoch"], history[f"val_{metric}"], label="val")  # courbe de validation
        ax.set_title(metric)  # titre du graphique
        ax.set_xlabel("epoque")  # legende de l'axe horizontal
        ax.legend()  # affiche la legende train / val
        ax.grid(alpha=0.3)  # quadrillage leger
    fig.tight_layout()  # evite que titres et axes se chevauchent
    fig.savefig(path, dpi=110)  # enregistre l'image (dpi = finesse)
    plt.close(fig)  # libere la memoire de la figure


def main() -> None:  # point d'entree : donnees, modele, entrainement, sauvegardes ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # option obligatoire : chemin du fichier YAML
    args = parser.parse_args()  # lit la ligne de commande
    config = load_config(args.config)  # charge et valide les reglages
    output = config.output_path  # dossier des resultats
    output.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
    setup_logging(output / "train.log")  # messages a l'ecran ET dans train.log
    save_config(config, output / "config_used.yaml")  # garde une copie des reglages utilises (tracabilite)
    seed_everything(config.seed)  # fixe le hasard : relancer donne le meme resultat
    device = get_device()  # GPU si disponible, sinon CPU
    logger.info("device : %s", device)  # affiche le device choisi

    # Donnees (etapes 5 a 7) : le script ne lit que train et val, jamais test.
    split_file = Path(config.split_file)  # CSV du decoupage produit par make_split.py
    class_names = class_names_of(split_file)  # liste triee des classes
    class_to_idx = {name: i for i, name in enumerate(class_names)}  # nom de classe -> numero (0, 1, 2...)
    train_df, val_df = load_split(split_file, "train"), load_split(split_file, "val")  # lignes du train et de la validation
    if config.normalization == "imagenet":  # si on normalise avec les statistiques d'ImageNet (modele pre-entraine)
        mean, std = IMAGENET_MEAN, IMAGENET_STD  # valeurs fixes connues
    else:  # sinon : statistiques de nos propres images
        mean, std = compute_mean_std(train_df, class_to_idx, config.image_size)  # calculees sur le train seulement (pas de fuite)
    train_transform, eval_transform = build_transforms(config.image_size, mean, std, config.augment)  # avec augmentations / sans
    train_loader = build_loader(train_df, class_to_idx, train_transform, config, train=True)  # lots melanges pour apprendre
    val_loader = build_loader(val_df, class_to_idx, eval_transform, config, train=False)  # lots fixes pour mesurer
    # affiche les classes, le nombre d'images et la moyenne arrondie a 3 decimales
    logger.info("classes %s | train %d | val %d | mean %s", class_names, len(train_df), len(val_df), [round(m, 3) for m in mean])

    # Modele, perte, optimiseur, scheduler (etapes 8 et 9).
    model = build_model(config, len(class_names)).to(device)  # cree le modele choisi dans le YAML et l'envoie sur le device
    logger.info("modele %s | parametres entrainables %d", config.model_name, count_parameters(model))  # taille du modele
    loss_weights = class_weights(train_df, class_names).to(device) if config.class_weighting else None  # poids par classe, ou aucun
    # perte de classification (entropie croisee) : grande quand la bonne classe recoit un petit score
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    # optimiseur (algorithme qui corrige les poids a partir des gradients) choisi dans le YAML : cle optimizer, adamw par defaut
    optimizer = build_optimizer(model, config)
    # scheduler : multiplie le pas par scheduler_factor quand la perte de validation stagne plus de scheduler_patience epoques
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.scheduler_factor, patience=config.scheduler_patience)

    # Boucle complete (etape 10), puis sauvegarde de tout ce qu'il faut pour predire (etape 12).
    history = fit(  # entraine et renvoie l'historique ; le meilleur modele est ecrit dans best.pt
        model,  # le reseau
        train_loader,  # lots d'entrainement
        val_loader,  # lots de validation
        criterion,  # la perte
        optimizer,  # l'optimiseur
        scheduler,  # le reglage automatique du pas
        config.epochs,  # nombre maximal d'epoques
        config.patience,  # patience de l'arret anticipe
        device,  # CPU ou GPU
        output / "best.pt",  # fichier du meilleur modele
    )
    history.to_csv(output / "history.csv", index=False)  # historique en CSV
    plot_history(history, output / "curves.png")  # courbes en image
    best = torch.load(output / "best.pt", map_location=device, weights_only=True)  # relit le meilleur point (pas forcement le dernier)
    model.load_state_dict(best["model_state"])  # remet ses poids dans le modele
    torch.save(  # ecrit model.pt : tout ce qu'il faut pour predire plus tard sans relire le YAML
        {  # dictionnaire sauvegarde
            "model_state": model.state_dict(),  # les poids
            "model_name": config.model_name,  # l'architecture a recreer
            "class_names": class_names,  # numero -> nom de classe
            "image_size": config.image_size,  # taille d'entree attendue
            "mean": mean,  # moyenne de normalisation
            "std": std,  # ecart-type de normalisation
            "best_epoch": best["epoch"],  # epoque du meilleur modele
        },
        output / "model.pt",  # fichier de destination
    )
    # ecrit la meilleure epoque et sa perte de validation dans un JSON (indent=2 : lisible)
    (output / "val_metrics.json").write_text(json.dumps({"best_epoch": best["epoch"], "val_loss": best["val_loss"]}, indent=2))
    logger.info("meilleure epoque %d (val perte %.4f) -> %s", best["epoch"], best["val_loss"], output / "model.pt")  # bilan final


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
