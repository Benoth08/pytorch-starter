"""Script 5 sur 5 (etape 13 du cours) : prediction sur une image ou un dossier, avec le modele sauvegarde.

    python scripts/predict.py --config configs/default.yaml --source chemin/image.bmp
    python scripts/predict.py --config configs/default.yaml --source chemin/dossier --csv predictions.csv

Le pretraitement vient du fichier model.pt (taille, mean, std) : il est donc
exactement celui de l'entrainement, sans rien recopier a la main.
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (--config, --source, --csv)
import logging  # logging : messages horodates du programme
import time  # time : mesure du temps de prediction
from pathlib import Path  # Path : represente un chemin de fichier

import pandas as pd  # pandas : tableau des predictions, ecrit en CSV
import torch  # PyTorch : chargement du modele, calcul des probabilites
from PIL import Image  # Pillow : ouvre les fichiers image

from vision_starter.config import load_config  # lit le fichier YAML de reglages
from vision_starter.data import EXTENSIONS, build_transforms  # extensions d'images reconnues ; transformations
from vision_starter.model import rebuild_model  # recree le modele enregistre, avec ses poids
from vision_starter.utils import get_device, setup_logging  # choix CPU/GPU ; configuration des messages

logger = logging.getLogger(__name__)  # journal propre a ce script


# recharge le modele sauvegarde ; renvoie (modele en mode evaluation, transformation, noms des classes)
def load_saved_model(model_file: Path, config, device: torch.device):
    """Modele en mode eval + transformation d'evaluation + noms de classes."""
    saved = torch.load(model_file, map_location=device, weights_only=True)  # lit model.pt (weights_only : donnees seulement, plus sur)
    # architecture notee dans model.pt, sans retelecharger de poids, puis les poids appris ; mode evaluation, sur le device
    model = rebuild_model(saved, config).to(device)
    # transformation d'evaluation avec la taille et la normalisation de l'entrainement (_ = celle d'entrainement, inutile)
    _, transform = build_transforms(saved["image_size"], saved["mean"], saved["std"])
    return model, transform, saved["class_names"]  # tout ce qu'il faut pour predire


@torch.no_grad()  # pas de gradient : on ne fait que predire (plus rapide, moins de memoire)
# predit la classe d'une image ; renvoie (nom de la classe, probabilite de cette classe)
def predict_image(path: Path, model, transform, class_names: list[str], device: torch.device) -> tuple[str, float]:
    """Une image -> (classe, probabilite)."""
    # ouvre l'image en RGB, la transforme en tenseur normalise, unsqueeze(0) ajoute la dimension lot -> (1, 3, H, W)
    batch = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    probabilities = torch.softmax(model(batch), dim=1)[0]  # softmax : logits -> probabilites qui somment a 1 ; [0] = 1re image du lot
    index = int(probabilities.argmax())  # numero de la classe la plus probable
    return class_names[index], float(probabilities[index])  # son nom et sa probabilite


# liste les images a traiter ; renvoie une liste de chemins
def list_images(source: Path) -> list[Path]:
    if source.is_file():  # si la source est un seul fichier
        return [source]  # liste d'une seule image
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in EXTENSIONS)  # sinon toutes les images du dossier et sous-dossiers


def main() -> None:  # point d'entree : charge le modele et predit chaque image ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # option obligatoire : chemin du fichier YAML
    parser.add_argument("--source", required=True, type=Path, help="image ou dossier")  # option obligatoire : quoi predire
    parser.add_argument("--csv", type=Path, default=None, help="fichier de sortie (facultatif)")  # option : ou ecrire les resultats
    args = parser.parse_args()  # lit la ligne de commande
    config = load_config(args.config)  # charge et valide les reglages
    setup_logging()  # messages dans la console
    device = get_device()  # GPU si disponible, sinon CPU
    model, transform, class_names = load_saved_model(config.output_path / "model.pt", config, device)  # modele pret a predire

    rows, start = [], time.time()  # lignes du futur CSV ; heure de debut pour mesurer la vitesse
    for path in list_images(args.source):  # pour chaque image a traiter
        label, probability = predict_image(path, model, transform, class_names, device)  # classe predite et sa probabilite
        rows.append({"path": path.as_posix(), "label": label, "probability": round(probability, 4)})  # memorise le resultat
        logger.info("%-30s -> %-10s (%.2f)", path.name, label, probability)  # affiche nom, classe et probabilite (colonnes alignees)
    # bilan : nombre d'images, duree totale et duree moyenne par image en millisecondes (max(..., 1) evite de diviser par 0)
    logger.info("%d images en %.2f s (%.0f ms / image)", len(rows), time.time() - start, 1000 * (time.time() - start) / max(len(rows), 1))
    if args.csv:  # si un fichier de sortie a ete demande
        pd.DataFrame(rows).to_csv(args.csv, index=False)  # ecrit toutes les predictions en CSV
        logger.info("-> %s", args.csv)  # indique ou


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
