"""Configuration du projet : un fichier YAML, lu une seule fois, valide a l'entree.

Le YAML remplace la cellule CONFIG du notebook. Une dataclass fige la liste des
cles attendues : une faute de frappe dans le YAML provoque une erreur claire au
demarrage, pas un comportement silencieux au milieu de l'entrainement.
"""

from __future__ import annotations  # permet d'ecrire les types modernes (ex. str | None) dans les annotations

# dataclass : cree une classe "fiche de reglages" a partir d'une simple liste de champs types ;
# asdict la convertit en dictionnaire, field regle la valeur par defaut d'un champ, fields liste les champs
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import yaml  # PyYAML : lit et ecrit les fichiers .yaml (format texte lisible "cle: valeur")


@dataclass  # transforme la classe : chaque ligne "nom: type = valeur" devient un reglage avec sa valeur par defaut
class Config:  # tous les reglages d'une experience de classification d'images
    """Tous les reglages d'une experience."""

    data_dir: str = "data/raw/neu_cls_flat"  # dossier des images d'entree
    data_layout: str = "class_folders"  # class_folders | split_folders | csv ; facon dont les images sont rangees
    labels_csv: str | None = None  # cas csv seulement ; fichier qui donne la classe de chaque image
    output_dir: str = "outputs/default"  # dossier ou ecrire modele, courbes et resultats
    split_file: str = "outputs/split.csv"  # fichier qui fige la repartition train / validation / test

    image_size: int = 128  # cote (en pixels) des images carrees donnees au reseau
    batch_size: int = 32  # images traitees ensemble a chaque pas : plus grand = plus rapide mais plus de memoire
    num_workers: int = 2  # processus qui preparent les images en parallele pendant que le modele calcule
    val_fraction: float = 0.15  # 15 % des images pour la validation (choisir l'epoque et les reglages)
    test_fraction: float = 0.15  # 15 % pour le test final, jamais vues pendant l'entrainement
    seed: int = 42  # graine du hasard : memes tirages a chaque lancement, resultats reproductibles

    model_name: str = "simple_cnn"  # simple_cnn | small_resnet | resnet18 ; architecture du reseau
    pretrained: bool = True  # partir de poids deja appris sur ImageNet (resnet18 seulement)
    freeze_backbone: bool = True  # geler le "backbone" (partie qui extrait les caracteristiques) : seule la tete apprend
    unfreeze_layers: list[str] = field(default_factory=list)  # blocs degeles malgre freeze_backbone, ex. [layer4]
    normalization: str = "dataset"  # dataset (calculee sur le train) | imagenet ; moyenne / ecart-type pour centrer les pixels

    epochs: int = 15  # nombre maximal d'epoques (une epoque = un passage complet sur les images d'entrainement)
    learning_rate: float = 1e-3  # taille des pas de correction des poids pour les couches neuves
    backbone_learning_rate: float = 1e-4  # pas plus petit pour les couches pre-entrainees degelees (ne pas tout casser)
    weight_decay: float = 1e-4  # penalite sur les grands poids : limite le sur-apprentissage
    patience: int = 4  # arret si la validation ne s'ameliore plus pendant 4 epoques (early stopping)
    class_weighting: bool = True  # donner plus de poids aux classes rares dans la perte
    optimizer: str = "adamw"  # adamw | sgd ; l'algorithme qui corrige les poids a partir des gradients
    momentum: float = 0.9  # sgd seulement : part de la correction precedente reprise a chaque pas (l'elan)
    scheduler_factor: float = 0.5  # quand la validation stagne, le pas d'apprentissage est multiplie par ce nombre
    scheduler_patience: int = 2  # epoques sans progres tolerees avant de reduire le pas

    # augmentations appliquees au train, dans l'ordre : hflip | vflip | rotate | brightness (voir data.AUGMENTATIONS) ; [] = aucune
    augment: list[str] = field(default_factory=lambda: ["hflip", "vflip"])
    onnx_opset: int = 18  # version du catalogue d'operations ONNX a l'export : une version que le moteur de production connait

    extra: dict = field(default_factory=dict)  # reglages libres, non valides

    @property  # se lit comme un attribut (config.output_path) mais est calcule a la demande
    def output_path(self) -> Path:  # renvoie output_dir sous forme de Path, plus pratique pour composer des chemins
        return Path(self.output_dir)  # convertit le texte en objet chemin


@dataclass  # meme principe : une fiche de reglages, ici pour l'auto-encodeur
class AnomalyConfig:  # reglages de la detection d'anomalies par auto-encodeur
    """Detection d'anomalies par auto-encodeur (chapitre Anomalies du cours) : on n'apprend que sur des images normales."""

    data_dir: str = "data/raw/mvtec_bottle"  # arborescence MVTec AD : train/good, test/<type>, ground_truth/<type>
    output_dir: str = "outputs/anomaly_ae"  # dossier des resultats

    image_size: int = 256  # cote des images carrees en entree
    batch_size: int = 16  # images traitees ensemble a chaque pas
    num_workers: int = 2  # processus de chargement des images en parallele
    val_fraction: float = 0.15  # part des normales d'entrainement gardees pour calibrer le seuil
    seed: int = 42  # graine du hasard pour la reproductibilite

    latent_channels: int = 32  # largeur du goulot (nombre de canaux de la representation compressee)
    epochs: int = 30  # nombre maximal d'epoques
    learning_rate: float = 1e-3  # taille des pas de l'optimiseur
    weight_decay: float = 1e-5  # petite penalite sur les grands poids
    patience: int = 8  # epoques sans progres avant l'arret anticipe

    gaussian_sigma: float = 4.0  # lissage de la carte d'erreur (pixels) : enleve le bruit pixel a pixel
    image_quantile: float = 0.99  # seuil image = ce quantile des scores des normales de validation
    pixel_quantile: float = 0.999  # meme idee pour le seuil pixel a pixel (plus strict : il y a beaucoup de pixels)
    onnx_opset: int = 18  # version du catalogue d'operations ONNX a l'export

    extra: dict = field(default_factory=dict)  # reglages libres, non valides

    @property  # attribut calcule a la demande
    def output_path(self) -> Path:  # renvoie le dossier de sortie sous forme de Path
        return Path(self.output_dir)  # convertit le texte en objet chemin


@dataclass  # fiche de reglages pour PatchCore
class PatchcoreConfig:  # reglages de PatchCore (detection d'anomalies sans entrainement, bibliotheque anomalib)
    """PatchCore avec anomalib (chapitre Anomalies du cours) : pas d'entrainement, une memoire de vecteurs normaux."""

    data_root: str = "data/raw"  # contient <category>/ au format MVTec AD
    category: str = "mvtec_bottle"  # nom du sous-dossier du jeu de donnees a utiliser
    output_dir: str = "outputs/patchcore"  # dossier des resultats

    image_size: int = 256  # cote des images carrees en entree
    batch_size: int = 16  # images traitees ensemble
    num_workers: int = 0  # 0 = chargement dans le processus principal (plus simple, evite des soucis sous Windows)
    seed: int = 42  # graine du hasard

    backbone: str = "resnet18"  # resnet18 | wide_resnet50_2 (le reglage du papier) | tout nom timm
    backbone_source: str = "torchvision"  # torchvision (poids download.pytorch.org) | timm (poids Hugging Face)
    # couches du reseau dont on recupere les caracteristiques : niveau intermediaire (ni trop fin, ni trop abstrait)
    layers: list[str] = field(default_factory=lambda: ["layer2", "layer3"])
    coreset_sampling_ratio: float = 0.01  # 0.1 dans le papier ; 1 % suffit presque et tient sur CPU
    num_neighbors: int = 9  # nombre de voisins regardes dans la memoire pour noter un morceau d'image

    extra: dict = field(default_factory=dict)  # reglages libres, non valides

    @property  # attribut calcule a la demande
    def output_path(self) -> Path:  # renvoie le dossier de sortie sous forme de Path
        return Path(self.output_dir)  # convertit le texte en objet chemin


@dataclass  # fiche de reglages pour la segmentation
class SegmentationConfig:  # reglages de la segmentation (classer chaque pixel : defaut ou fond)
    """Segmentation binaire defaut / fond avec un U-Net (chapitre Segmentation du cours)."""

    data_dir: str = "data/raw/magnetic_tile"  # MT_<classe>/Imgs/x.jpg + x.png
    output_dir: str = "outputs/segmentation"  # dossier des resultats
    split_file: str = "outputs/segmentation/split.csv"  # repartition train / validation / test figee

    image_size: int = 256  # cote des images carrees en entree
    batch_size: int = 8  # images traitees ensemble (petit : un U-Net consomme beaucoup de memoire)
    num_workers: int = 2  # processus de chargement en parallele
    n_free: int = 150  # images sans defaut conservees (sur 952) : evite que le fond ecrase tout
    val_fraction: float = 0.15  # part pour la validation
    test_fraction: float = 0.15  # part pour le test final
    seed: int = 42  # graine du hasard
    duplicate_max_hamming: int = 6  # quasi-doublons : nombre maximal de bits differents entre deux empreintes (sur 64)
    duplicate_min_correlation: float = 0.85  # quasi-doublons : correlation minimale des details pour reunir deux images

    pretrained: bool = True  # encodeur initialise avec des poids appris sur ImageNet
    epochs: int = 12  # nombre maximal d'epoques
    learning_rate: float = 1e-3  # decodeur (neuf)
    encoder_learning_rate: float = 1e-4  # encodeur ResNet18 pre-entraine
    weight_decay: float = 1e-4  # penalite sur les grands poids
    patience: int = 4  # epoques sans progres avant l'arret anticipe
    # seuils de probabilite candidats (pixel = defaut si proba > seuil), choisis sur la validation
    thresholds: list[float] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7])  # candidats, choisis sur la validation
    min_defect_area: int = 20  # decision par image : "defaut" si plus de ce nombre de pixels sont predits en defaut
    onnx_opset: int = 18  # version du catalogue d'operations ONNX a l'export

    extra: dict = field(default_factory=dict)  # reglages libres, non valides

    @property  # attribut calcule a la demande
    def output_path(self) -> Path:  # renvoie le dossier de sortie sous forme de Path
        return Path(self.output_dir)  # convertit le texte en objet chemin


@dataclass  # fiche de reglages pour la detection d'objets
class DetectionConfig:  # reglages de la detection (trouver et encadrer les defauts) avec YOLO
    """Detection avec YOLO / Ultralytics (chapitre Detection du cours) : la librairie fait la boucle, nous les donnees et l'export."""

    data_dir: str = "data/raw/neu_det"  # IMAGES/*.jpg + ANNOTATIONS/*.xml (Pascal VOC)
    yolo_dir: str = "data/processed/neu_det_yolo"  # le meme jeu, range comme Ultralytics l'attend
    output_dir: str = "outputs/yolo"  # dossier des resultats
    split_file: str = "outputs/yolo/split.csv"  # repartition train / validation / test figee
    weights: str = "outputs/yolo/pretrained/yolo11n.pt"  # poids COCO ; Ultralytics les telecharge s'ils manquent

    image_size: int = 320  # cote des images carrees donnees a YOLO
    batch_size: int = 16  # images traitees ensemble
    num_workers: int = 2  # processus de chargement en parallele
    val_fraction: float = 0.15  # part pour la validation
    test_fraction: float = 0.15  # part pour le test final
    seed: int = 42  # graine du hasard

    epochs: int = 60  # nombre maximal d'epoques
    patience: int = 20  # epoques sans progres avant l'arret anticipe
    close_mosaic: int = 10  # dernieres epoques sans mosaic (augmentation qui colle 4 images en une)
    warmup_epochs: int = 3  # premieres epoques avec un pas d'apprentissage qui monte doucement
    conf_eval: float = 0.001  # evaluation : toute la courbe precision-rappel (on garde presque toutes les boites)
    conf_predict: float = 0.25  # production : seuil de decision (boites moins sures ignorees)
    iou_nms: float = 0.7  # NMS : supprime une boite qui chevauche une meilleure a plus de 70 % (IoU = recouvrement)

    extra: dict = field(default_factory=dict)  # reglages libres, non valides

    @property  # attribut calcule a la demande
    def output_path(self) -> Path:  # renvoie le dossier de sortie sous forme de Path
        return Path(self.output_dir)  # convertit le texte en objet chemin


# lit un fichier YAML et renvoie l'objet de reglages rempli (Config ou une autre classe de reglages)
def load_config(path: str | Path, config_class: type = Config):
    """Lit le YAML et refuse toute cle inconnue. config_class : Config (classification) ou une des autres taches."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}  # dictionnaire lu ; {} si le fichier est vide
    known = {f.name for f in fields(config_class)}  # ensemble des noms de reglages prevus par la classe
    unknown = set(raw) - known  # cles du YAML qui ne correspondent a aucun reglage (faute de frappe probable)
    if unknown:  # si au moins une cle est inconnue
        raise ValueError(f"cles inconnues dans {path} : {sorted(unknown)}")  # on arrete tout de suite avec un message clair
    return config_class(**raw)  # cree l'objet : les cles du YAML remplacent les valeurs par defaut


# enregistre les reglages dans un fichier YAML ; ne renvoie rien
def save_config(config, path: str | Path) -> None:
    """Copie la configuration a cote des resultats, pour savoir ce qui a ete lance."""
    # asdict : objet -> dictionnaire ; sort_keys=False garde l'ordre des champs de la classe
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
