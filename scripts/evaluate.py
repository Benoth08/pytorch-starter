"""Script 3 sur 5 (etape 11 du cours) : evaluation finale sur le test, une seule fois, avec le modele sauvegarde.

    python scripts/evaluate.py --config configs/default.yaml

Produit test_metrics.json, classification_report.txt et confusion_matrix.png.
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (ici --config)
import json  # json : ecrit les metriques dans un fichier texte lisible par tout langage
import logging  # logging : messages horodates du programme
from pathlib import Path  # Path : represente un chemin de fichier

import matplotlib  # matplotlib : bibliotheque de graphiques
import numpy as np  # numpy : tableaux de nombres (la matrice de confusion en est un)
import torch  # PyTorch : chargement du modele sauvegarde
import torch.nn as nn  # nn : briques de reseaux de neurones (ici la fonction de perte)

# classification_report : precision / rappel / F1 par classe ; confusion_matrix : tableau verite x prediction ;
# f1_score : F1 = compromis entre precision et rappel
from sklearn.metrics import classification_report, confusion_matrix, f1_score

matplotlib.use("Agg")  # moteur de dessin sans fenetre : les figures vont dans des fichiers
# pyplot : l'outil de trace de matplotlib, importe apres le choix du moteur ;
# "noqa: E402" dit a ruff d'accepter ces imports places apres du code (c'est voulu)
import matplotlib.pyplot as plt  # noqa: E402

from vision_starter.config import load_config  # noqa: E402  (lecture des reglages YAML)
from vision_starter.data import build_loader, build_transforms, load_split  # noqa: E402  (DataLoader, transformations, decoupage)
from vision_starter.engine import evaluate  # noqa: E402  (passe d'evaluation sans apprentissage)
from vision_starter.model import rebuild_model  # noqa: E402  (recree le modele enregistre, avec ses poids)
from vision_starter.utils import get_device, setup_logging  # noqa: E402  (choix CPU/GPU, messages)

logger = logging.getLogger(__name__)  # journal propre a ce script


# dessine la matrice de confusion (lignes = verite, colonnes = prediction) et l'enregistre ; ne renvoie rien
def plot_confusion(matrix: np.ndarray, class_names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))  # une figure avec un seul graphique
    ax.imshow(matrix, cmap="Blues")  # affiche le tableau comme une image : plus c'est fonce, plus il y a d'images
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")  # noms des classes en abscisse, inclines
    ax.set_yticks(range(len(class_names)), class_names)  # noms des classes en ordonnee
    ax.set_xlabel("prediction")  # axe horizontal : classe predite
    ax.set_ylabel("verite")  # axe vertical : vraie classe
    for i in range(len(class_names)):  # pour chaque ligne (vraie classe)
        for j in range(len(class_names)):  # pour chaque colonne (classe predite)
            # ecrit le nombre au centre de la case, en blanc sur les cases foncees, en noir sinon (lisibilite)
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()  # evite les chevauchements
    fig.savefig(path, dpi=110)  # enregistre l'image
    plt.close(fig)  # libere la memoire de la figure


def main() -> None:  # point d'entree : recharge le modele, l'evalue sur le test, ecrit les resultats ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # option obligatoire : chemin du fichier YAML
    args = parser.parse_args()  # lit la ligne de commande
    config = load_config(args.config)  # charge et valide les reglages
    output = config.output_path  # dossier ou se trouve model.pt et ou ecrire les resultats
    setup_logging(output / "evaluate.log")  # messages a l'ecran et dans evaluate.log
    device = get_device()  # GPU si disponible, sinon CPU

    saved = torch.load(output / "model.pt", map_location=device, weights_only=True)  # relit le modele sauvegarde par train.py
    class_names = saved["class_names"]  # classes dans l'ordre de l'entrainement
    class_to_idx = {name: i for i, name in enumerate(class_names)}  # nom de classe -> numero
    model = rebuild_model(saved, config).to(device)  # meme architecture (sans retelecharger de poids) + les poids appris

    # transformation d'evaluation avec la taille et la normalisation de l'entrainement (_ = celle d'entrainement, inutile ici)
    _, eval_transform = build_transforms(saved["image_size"], saved["mean"], saved["std"])
    # lots du sous-ensemble de test, jamais vu jusqu'ici
    test_loader = build_loader(load_split(Path(config.split_file), "test"), class_to_idx, eval_transform, config, train=False)
    # perte, accuracy, classes predites et vraies classes sur tout le test
    test_loss, test_acc, preds, targets = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    metrics = {  # metriques a enregistrer
        "test_loss": test_loss,  # perte moyenne sur le test
        "accuracy": test_acc,  # part d'images bien classees
        "f1_macro": float(f1_score(targets, preds, average="macro")),  # F1 moyen des classes (chaque classe compte pareil)
        "n_images": int(len(targets)),  # nombre d'images de test
    }
    rng = np.random.default_rng(0)  # generateur de hasard reproductible : le meme intervalle a chaque lancement
    draws = [rng.integers(0, len(targets), len(targets)) for _ in range(1000)]  # 1000 retirages du test, avec remise (bootstrap)
    f1_draws = [f1_score(targets[i], preds[i], average="macro") for i in draws]  # la metrique principale sur chaque retirage
    metrics["f1_macro_ci95"] = [round(float(v), 3) for v in np.percentile(f1_draws, [2.5, 97.5])]  # la fourchette a 95 %, a cote du chiffre
    metrics["errors"] = int((preds != targets).sum())  # le nombre d'erreurs : "8 sur 270" parle plus que 0.970
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2))  # ecrit les metriques en JSON lisible
    report = classification_report(targets, preds, target_names=class_names, digits=3)  # rapport texte par classe
    (output / "classification_report.txt").write_text(report)  # l'enregistre
    plot_confusion(confusion_matrix(targets, preds), class_names, output / "confusion_matrix.png")  # calcule et dessine la matrice
    # bilan dans les messages : accuracy, F1 macro, sa fourchette a 95 %, nombre d'erreurs et d'images
    logger.info(
        "test : accuracy %.3f | F1 macro %.3f [%.3f ; %.3f] | %d erreurs sur %d images",  # modele du message
        test_acc,  # part d'images bien classees
        metrics["f1_macro"],  # F1 moyen des classes
        *metrics["f1_macro_ci95"],  # bornes basse et haute de l'intervalle de confiance
        metrics["errors"],  # nombre d'images mal classees
        len(targets),  # nombre d'images de test
    )
    logger.info("rapport par classe\n%s", report)  # precision, rappel et F1 de chaque classe


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
