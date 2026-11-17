"""PatchCore avec anomalib : memoire des normales, test, export ONNX, en un seul script (chapitre Anomalies du cours).

    python scripts/train_patchcore.py --config configs/patchcore.yaml

Pas de boucle d'entrainement a ecrire : la bibliotheque fournit datamodule,
modele, seuils, metriques et export. Le script ne fait que les assembler
d'apres le YAML, et ranger les resultats comme les autres scripts du projet.
Necessite l'extra "anomaly" (uv sync --extra anomaly).

Produit dans output_dir : config_used.yaml, train.log, test_metrics.json (metriques
et seuils bruts), export/weights/onnx/model.onnx + model.onnx.data, et les images
d'anomalib (image, carte, masque predit) sous Patchcore/.
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import argparse  # argparse : lit les options passees en ligne de commande (ici --config)
import json  # json : ecrit les metriques dans un fichier texte lisible par tout langage
import logging  # logging : messages horodates du programme
import time  # time : mesure la duree de construction de la memoire
from pathlib import Path  # Path : represente un chemin de fichier

# ONNX Runtime : execute un modele ONNX (format standard de modele) ; sert ici a relire et verifier l'export
import onnxruntime as ort
import torch  # PyTorch : ici seulement pour savoir si un GPU est disponible
from torchvision import models  # architectures connues (ResNet...) avec leurs poids pre-entraines

from vision_starter.config import PatchcoreConfig, load_config, save_config  # reglages PatchCore : classe, lecture, copie
from vision_starter.utils import quiet_libraries, seed_everything, setup_logging  # messages et graine du hasard

logger = logging.getLogger(__name__)  # journal propre a ce script


# choisit le reseau pre-entraine qui extrait les caracteristiques ; renvoie un nom (timm) ou un modele (torchvision)
def build_backbone(config: PatchcoreConfig):
    """Un nom timm (poids Hugging Face) ou le modele torchvision du meme nom (poids download.pytorch.org)."""
    if config.backbone_source == "timm":  # si on passe par la bibliotheque timm
        return config.backbone  # anomalib accepte directement le nom et telecharge lui-meme les poids
    if config.backbone_source == "torchvision":  # si on prend le modele de torchvision
        # getattr(models, "resnet18") = la fonction models.resnet18 ; weights="DEFAULT" = poids ImageNet
        return getattr(models, config.backbone)(weights="DEFAULT")
    raise ValueError(f"backbone_source inconnu : {config.backbone_source}")  # aucune valeur reconnue : faute dans le YAML


def main() -> None:  # point d'entree : memoire des normales, test, export ONNX ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # lecteur d'options ; description = 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # option obligatoire : chemin du fichier YAML
    args = parser.parse_args()  # lit la ligne de commande
    config: PatchcoreConfig = load_config(args.config, PatchcoreConfig)  # charge les reglages
    output = config.output_path  # dossier des resultats
    output.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
    setup_logging(output / "train.log")  # messages a l'ecran et dans train.log
    save_config(config, output / "config_used.yaml")  # copie des reglages utilises
    seed_everything(config.seed)  # fixe le hasard

    from anomalib.data import MVTecAD  # imports ici : le reste du projet marche sans anomalib
    from anomalib.deploy import ExportType  # liste des formats d'export (ONNX, OpenVINO...)
    from anomalib.engine import Engine  # le "moteur" qui pilote apprentissage, test et export
    from anomalib.models import Patchcore  # le modele PatchCore

    quiet_libraries()  # apres l'import : Lightning remet son journal a INFO en s'important

    datamodule = MVTecAD(  # "datamodule" : prepare tout seul les DataLoader train et test au format MVTec AD
        root=config.data_root,  # dossier qui contient la categorie
        category=config.category,  # nom du sous-dossier (ex. mvtec_bottle)
        train_batch_size=config.batch_size,  # images par lot a l'entrainement
        eval_batch_size=config.batch_size,  # images par lot au test
        num_workers=config.num_workers,  # processus de chargement en parallele
        seed=config.seed,  # graine du hasard
    )
    model = Patchcore(  # PatchCore : garde en memoire des vecteurs de morceaux d'images normales
        backbone=build_backbone(config),  # reseau pre-entraine qui transforme chaque morceau d'image en vecteur
        layers=config.layers,  # couches du reseau dont on prend les vecteurs
        coreset_sampling_ratio=config.coreset_sampling_ratio,  # part des vecteurs gardes en memoire (coreset = echantillon representatif)
        num_neighbors=config.num_neighbors,  # nombre de voisins consultes pour noter un morceau
    )
    engine = Engine(  # moteur anomalib (base sur Lightning, qui ecrit la boucle a notre place)
        default_root_dir=output,  # ou ecrire ses fichiers
        accelerator="cpu" if not torch.cuda.is_available() else "gpu",  # GPU si disponible, sinon CPU
        devices=1,  # un seul processeur / une seule carte
        logger=False,  # pas de journal d'experience Lightning (on a le notre)
        enable_progress_bar=False,  # pas de barre de progression (journal plus propre)
        enable_model_summary=False,  # pas de resume du modele au demarrage
    )

    seed_everything(config.seed)  # le coreset part d'un vecteur tire au hasard : graine refixee juste avant fit
    start = time.time()  # heure de debut
    engine.fit(model, datamodule=datamodule)  # "entrainement" : une seule passe sur les normales pour remplir la memoire
    fit_seconds = time.time() - start  # duree en secondes
    # taille de la memoire (nombre de vecteurs, longueur de chaque vecteur) et duree
    logger.info("memoire : %s vecteurs en %.0fs", tuple(model.model.memory_bank.shape), fit_seconds)

    results = engine.test(model, datamodule=datamodule)[0]  # evaluation sur le test ; [0] = resultats du 1er (seul) jeu
    metrics = {name: float(value) for name, value in results.items()}  # metriques converties en nombres Python simples
    metrics.update(  # ajoute d'autres informations
        {  # dictionnaire a fusionner
            "image_threshold_raw": float(model.post_processor.image_threshold),  # seuil image choisi par anomalib (score brut)
            "pixel_threshold_raw": float(model.post_processor.pixel_threshold),  # seuil pixel (score brut)
            "fit_seconds": round(fit_seconds),  # duree de construction de la memoire
        }
    )
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2))  # ecrit les metriques en JSON
    logger.info(  # bilan du test ; chaque %... est remplace par la valeur correspondante ci-dessous
        "test : image AUROC %.3f | pixel AUROC %.3f | image F1 %.3f",  # modele du message
        metrics["image_AUROC"],  # AUROC image (qualite du classement des scores, 1 = parfait)
        metrics["pixel_AUROC"],  # AUROC pixel
        metrics["image_F1Score"],  # F1 image au seuil choisi
    )

    onnx_path = engine.export(  # export ONNX du modele complet (reseau + memoire) ; renvoie le chemin du fichier
        model,  # le modele a exporter
        export_type=ExportType.ONNX,  # format ONNX
        export_root=output / "export",  # dossier de destination
        input_size=(config.image_size, config.image_size),  # taille d'entree figee dans le fichier
        onnx_kwargs={"opset_version": 18},  # l'opset produit par torch 2.15 ; le defaut d'anomalib (14) tente une conversion qui echoue
    )
    data_path = onnx_path.with_name(onnx_path.name + ".data")  # le graphe d'un cote, poids et memoire de l'autre
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # relit le fichier : verifie qu'il se charge
    logger.info(  # bilan de l'export ; chaque %... est remplace par la valeur correspondante ci-dessous
        "export ONNX : %s + %s (%.1f Mo) | entree %s | sorties %s | seuils bruts dans test_metrics.json",  # modele du message
        onnx_path.name,  # nom du fichier du graphe
        data_path.name,  # nom du fichier des poids
        (onnx_path.stat().st_size + data_path.stat().st_size) / 1e6,  # taille totale en Mo
        session.get_inputs()[0].shape,  # forme de l'entree attendue
        [o.name for o in session.get_outputs()],  # noms des sorties
    )


if __name__ == "__main__":  # vrai seulement quand on lance ce fichier directement
    main()  # on execute le script
