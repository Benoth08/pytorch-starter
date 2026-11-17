"""Tout ce qui touche aux donnees : catalogue, decoupage, Dataset, transformations, DataLoader.

C'est la reunion des etapes 3 a 7 du notebook (du catalogue aux lots). Les scripts ne manipulent
jamais de chemins d'images directement : ils passent par le catalogue.
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import logging  # logging : messages horodates du programme
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (ici les vignettes des images)
import pandas as pd  # pandas : tableaux de donnees (DataFrame = tableau avec colonnes nommees, comme une feuille Excel)
import torch  # PyTorch : tenseurs (tableaux de nombres sur CPU ou GPU) et reseaux de neurones
from PIL import Image  # Pillow : ouvre les fichiers image (jpg, png, bmp...)
from scipy.ndimage import gaussian_filter  # scipy : flou gaussien (pour separer l'eclairage des details d'une image)
from sklearn.model_selection import train_test_split  # scikit-learn : coupe un tableau en deux parties au hasard
from torch.utils.data import DataLoader, Dataset  # Dataset : donne un exemple par indice ; DataLoader : les regroupe en lots
from torchvision.transforms import v2  # transformations d'images de torchvision (redimensionner, retourner, normaliser...)

from vision_starter.config import Config  # la fiche de reglages du projet
from vision_starter.utils import seed_worker  # graine propre a chaque processus de chargement

logger = logging.getLogger(__name__)  # journal propre a ce module (son nom apparait dans les messages)

EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}  # extensions de fichiers reconnues comme images
# moyenne et ecart-type des pixels d'ImageNet par canal (rouge, vert, bleu) : a utiliser avec un modele pre-entraine sur ImageNet
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# --- Catalogue : trois rangements d'origine, un seul tableau (path, label) ---


# lit un dossier range "une classe par sous-dossier" et renvoie le tableau (path, label)
def catalog_from_class_folders(root: Path) -> pd.DataFrame:
    """Cas B : root/<classe>/image.ext."""
    rows = [  # liste de lignes du futur tableau, une par image
        {"path": path.as_posix(), "label": class_dir.name}  # chemin ecrit avec des / (lisible partout) et classe (= nom du dossier)
        for class_dir in sorted(root.iterdir())  # pour chaque element de root, dans l'ordre alphabetique
        if class_dir.is_dir()  # on ne garde que les dossiers (un dossier = une classe)
        for path in sorted(class_dir.iterdir())  # pour chaque fichier de ce dossier de classe
        if path.suffix.lower() in EXTENSIONS  # seulement les fichiers images (extension reconnue, en minuscules)
    ]
    if not rows:  # si on n'a trouve aucune image
        raise FileNotFoundError(f"aucune image trouvee sous {root}")  # erreur claire : mauvais chemin ou dossier vide
    return pd.DataFrame(rows)  # transforme la liste de dictionnaires en tableau a deux colonnes


# lit un dossier deja coupe en train / val / test et renvoie le tableau (path, label, split)
def catalog_from_split_folders(root: Path) -> pd.DataFrame:
    """Cas A : root/{train,val,test}/<classe>/image.ext ; le decoupage fourni est conserve."""
    frames = []  # un tableau par sous-ensemble, assembles a la fin
    for subset in ("train", "val", "test"):  # pour chaque sous-ensemble fourni
        frame = catalog_from_class_folders(root / subset)  # lit root/<subset>/<classe>/...
        frame["split"] = subset  # note dans une colonne a quel sous-ensemble appartient chaque image
        frames.append(frame)  # garde ce tableau pour l'assemblage
    catalog = pd.concat(frames, ignore_index=True)  # empile les trois tableaux ; ignore_index renumerote les lignes 0..n-1
    # Piege du cas A : une classe absente d'un dossier decale les indices de classes.
    per_split = catalog.groupby("split")["label"].apply(set)  # pour chaque sous-ensemble, l'ensemble des classes presentes
    if any(classes != per_split["train"] for classes in per_split):  # si un sous-ensemble n'a pas les memes classes que train
        raise ValueError(f"les classes different entre sous-ensembles : {per_split.to_dict()}")  # on refuse avec le detail
    return catalog  # renvoie le catalogue complet


# lit un dossier d'images plat et un CSV d'etiquettes ; renvoie le tableau (path, label)
def catalog_from_csv(images_dir: Path, csv_path: Path) -> pd.DataFrame:
    """Cas C : un dossier d'images et un CSV avec les colonnes filename et label."""
    table = pd.read_csv(csv_path)  # lit le CSV dans un tableau
    # chemin de chaque fichier ecrit avec des / + etiquette convertie en texte (une classe "3" reste un nom, pas un nombre)
    return pd.DataFrame({"path": [(images_dir / f).as_posix() for f in table["filename"]], "label": table["label"].astype(str)})


# choisit la bonne fonction de lecture selon config.data_layout ; renvoie le catalogue
def build_catalog(config: Config) -> pd.DataFrame:
    """Choisit le lecteur de catalogue d'apres la configuration."""
    root = Path(config.data_dir)  # dossier des donnees sous forme de chemin
    if config.data_layout == "class_folders":  # rangement : un sous-dossier par classe
        return catalog_from_class_folders(root)  # catalogue sans decoupage
    if config.data_layout == "split_folders":  # rangement : train/val/test deja separes
        return catalog_from_split_folders(root)  # catalogue avec le decoupage fourni
    if config.data_layout == "csv":  # rangement : un dossier plat + un CSV d'etiquettes
        return catalog_from_csv(root, Path(config.labels_csv))  # catalogue lu depuis le CSV
    raise ValueError(f"data_layout inconnu : {config.data_layout}")  # aucune valeur reconnue : faute de frappe dans le YAML


# --- Decoupage ---


# repartit les images en train / val / test au hasard ; renvoie une copie du catalogue avec une colonne split
def make_split(catalog: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int) -> pd.DataFrame:
    """Ajoute une colonne split (train / val / test), stratifiee par label."""
    # 1re coupe : on met de cote le test ; stratify = chaque classe garde la meme proportion dans les deux parties
    train_val, test = train_test_split(catalog, test_size=test_fraction, stratify=catalog["label"], random_state=seed)
    # 2e coupe du reste : la fraction est recalculee pour que val fasse bien val_fraction du total (et pas du reste)
    train, val = train_test_split(train_val, test_size=val_fraction / (1 - test_fraction), stratify=train_val["label"], random_state=seed)
    catalog = catalog.copy()  # copie : on ne modifie pas le tableau recu par l'appelant
    catalog.loc[train.index, "split"] = "train"  # marque les lignes tirees pour l'entrainement
    catalog.loc[val.index, "split"] = "val"  # marque les lignes de validation
    catalog.loc[test.index, "split"] = "test"  # marque les lignes de test
    return catalog  # renvoie le catalogue complete


# variante du decoupage par groupes (une meme piece ne doit pas etre a la fois en train et en test) ; renvoie le catalogue
def make_grouped_split(catalog: pd.DataFrame, group_column: str, val_fraction: float, test_fraction: float, seed: int) -> pd.DataFrame:
    """Variante : toutes les images d'un meme groupe (piece, patient, video) vont dans le meme sous-ensemble."""
    from sklearn.model_selection import GroupShuffleSplit  # coupe au hasard en gardant chaque groupe entier d'un seul cote

    catalog = catalog.copy()  # copie : on ne modifie pas le tableau recu
    first = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)  # 1re coupe : reste / test
    rest_idx, test_idx = next(first.split(catalog, groups=catalog[group_column]))  # positions des lignes de chaque cote
    rest = catalog.iloc[rest_idx]  # lignes hors test (iloc = selection par position)
    second = GroupShuffleSplit(n_splits=1, test_size=val_fraction / (1 - test_fraction), random_state=seed)  # 2e coupe : train / val
    train_idx, val_idx = next(second.split(rest, groups=rest[group_column]))  # positions dans "rest" de chaque cote
    catalog.loc[rest.index[train_idx], "split"] = "train"  # convertit les positions en etiquettes de lignes puis marque train
    catalog.loc[rest.index[val_idx], "split"] = "val"  # marque val
    catalog.loc[catalog.index[test_idx], "split"] = "test"  # marque test
    return catalog  # renvoie le catalogue complete


# decoupage par groupes ET equilibre par classe ; renvoie une copie du catalogue avec une colonne split
def make_stratified_grouped_split(
    catalog: pd.DataFrame,  # le catalogue, avec une colonne label et une colonne de groupe
    group_column: str,  # nom de la colonne qui dit a quel groupe appartient chaque image
    val_fraction: float,  # part visee pour la validation
    test_fraction: float,  # part visee pour le test
    seed: int,  # graine du hasard
) -> pd.DataFrame:  # renvoie le catalogue avec sa colonne split
    """Aucun groupe n'est coupe, et chaque classe garde a peu pres la meme part partout. Les fractions sont approchees."""
    from sklearn.model_selection import StratifiedGroupKFold  # coupe en n plis sans couper de groupe, en equilibrant les classes

    catalog = catalog.copy()  # copie : on ne modifie pas le tableau recu
    # n plis de taille 1 / n : on garde le premier comme test ; 15 % demandes -> 7 plis -> environ 14 %
    folds = StratifiedGroupKFold(n_splits=round(1 / test_fraction), shuffle=True, random_state=seed)
    rest_idx, test_idx = next(folds.split(catalog, catalog["label"], groups=catalog[group_column]))  # positions : reste / test
    rest = catalog.iloc[rest_idx]  # lignes hors test
    # meme methode sur le reste, pour la validation : le nombre de plis redonne val_fraction du total
    folds = StratifiedGroupKFold(n_splits=round((1 - test_fraction) / val_fraction), shuffle=True, random_state=seed)
    train_idx, val_idx = next(folds.split(rest, rest["label"], groups=rest[group_column]))  # positions dans "rest"
    catalog.loc[rest.index[train_idx], "split"] = "train"  # marque les lignes d'entrainement
    catalog.loc[rest.index[val_idx], "split"] = "val"  # marque les lignes de validation
    catalog.loc[catalog.index[test_idx], "split"] = "test"  # marque les lignes de test
    return catalog  # renvoie le catalogue complete


# --- Quasi-doublons : les reperer, pour qu'ils restent du meme cote du decoupage ---


# calcule l'empreinte visuelle d'une image ; renvoie un entier de 64 bits
def dhash(image: Image.Image) -> int:
    """Empreinte perceptive : deux images presque identiques ont des empreintes egales a quelques bits pres."""
    small = np.asarray(image.convert("L").resize((9, 8), Image.BOX), dtype=np.int16)  # vignette grise de 9 colonnes x 8 lignes
    bits = (small[:, 1:] > small[:, :-1]).flatten()  # 1 si un pixel est plus clair que son voisin de gauche : 8 x 8 = 64 bits
    return int(np.packbits(bits).view(">u8")[0])  # les 64 bits ranges dans un seul entier


# resume les details fins d'une image ; renvoie un vecteur de 64 x 64 nombres, de longueur 1
def detail_signature(image: Image.Image, size: int = 64) -> np.ndarray:
    """Vignette sans son eclairage, centree et normee : le produit de deux signatures est une correlation (1 = memes details)."""
    small = np.asarray(image.convert("L").resize((size, size), Image.BOX), dtype=np.float32)  # vignette grise 64 x 64
    detail = small - gaussian_filter(small, 3)  # retire la version floue (eclairage, degrades) : il reste les details fins
    detail -= detail.mean()  # centre : moyenne a 0
    return (detail / max(float(np.linalg.norm(detail)), 1e-6)).ravel()  # longueur 1 (1e-6 evite de diviser par 0), mis a plat


# regroupe les images presque identiques ; renvoie un numero de groupe par image (deux images du meme groupe ont le meme numero)
def near_duplicate_groups(paths: list[str], max_hamming: int = 6, min_correlation: float = 0.85) -> list[int]:
    """Deux temps : empreintes a max_hamming bits pres, puis details correles. Les paires confirmees sont reunies de proche en proche.

    Toutes les paires sont comparees : convient jusqu'a quelques milliers d'images. Mieux vaut grouper trop que pas assez :
    un groupe en trop ne coute rien, un quasi-doublon a cheval sur train et test fausse le score.
    """
    hashes, signatures = [], []  # empreinte et signature de chaque image
    for path in paths:  # pour chaque image
        with Image.open(path) as image:  # ouvre le fichier (referme a la fin du bloc)
            hashes.append(dhash(image))  # son empreinte de 64 bits
            signatures.append(detail_signature(image))  # sa signature de details
    parent = list(range(len(paths)))  # chaque image pointe vers une image "parent" ; au depart elle-meme : un groupe par image

    def root(i: int) -> int:  # renvoie l'image qui represente le groupe de l'image i
        while parent[i] != i:  # tant que i n'est pas son propre parent, on remonte
            parent[i] = parent[parent[i]]  # raccourci : i pointe vers son grand-parent (les remontees suivantes sont plus courtes)
            i = parent[i]  # on monte d'un cran
        return i  # i est le representant du groupe

    for i in range(len(paths)):  # chaque image...
        for j in range(i + 1, len(paths)):  # ...comparee a toutes les suivantes
            close = (hashes[i] ^ hashes[j]).bit_count() <= max_hamming  # ^ marque les bits differents, bit_count les compte
            if close and float(signatures[i] @ signatures[j]) >= min_correlation:  # empreintes proches ET details correles
                parent[root(i)] = root(j)  # reunit les deux groupes
    return [root(i) for i in range(len(paths))]  # le representant de chaque image sert de numero de groupe


# verifie que le decoupage est sain ; ne renvoie rien mais leve une erreur en cas de probleme
def check_split(catalog: pd.DataFrame) -> None:
    """Refuse un decoupage avec des chemins en double ou un sous-ensemble vide."""
    if not catalog["path"].is_unique:  # si un meme chemin apparait plusieurs fois (risque de fuite entre train et test)
        raise ValueError("des images apparaissent deux fois dans le catalogue")  # on refuse
    missing = {"train", "val", "test"} - set(catalog["split"])  # sous-ensembles attendus mais absents
    if missing:  # si au moins un sous-ensemble est vide
        raise ValueError(f"sous-ensembles vides : {sorted(missing)}")  # on refuse avec la liste


# verifie qu'aucun groupe n'est a cheval sur deux sous-ensembles ; ne renvoie rien mais leve une erreur sinon
def check_group_leak(catalog: pd.DataFrame, group_column: str) -> None:
    """Refuse un decoupage ou des images d'un meme groupe (piece, quasi-doublons) sont dans deux sous-ensembles : c'est une fuite."""
    per_group = catalog.groupby(group_column)["split"].nunique()  # nombre de sous-ensembles differents touches par chaque groupe
    if (per_group > 1).any():  # si au moins un groupe est coupe
        raise ValueError(f"{int((per_group > 1).sum())} groupes a cheval sur plusieurs sous-ensembles")  # on refuse, avec le nombre


# relit le fichier de decoupage et renvoie seulement les lignes d'un sous-ensemble (train, val ou test)
def load_split(split_file: Path, subset: str) -> pd.DataFrame:
    """Relit le decoupage fige et renvoie un seul sous-ensemble (on choisit explicitement)."""
    catalog = pd.read_csv(split_file)  # lit tout le decoupage
    return catalog[catalog["split"] == subset].reset_index(drop=True)  # filtre le sous-ensemble et renumerote les lignes 0..n-1


# renvoie la liste triee des noms de classes : l'ordre fixe le numero de chaque classe
def class_names_of(split_file: Path) -> list[str]:
    return sorted(pd.read_csv(split_file)["label"].unique())  # classes distinctes, dans l'ordre alphabetique


# --- Dataset et transformations ---


class ImageDataset(Dataset):  # jeu de donnees PyTorch : dataset[i] renvoie (image, numero de classe)
    """Lit une image et son etiquette a partir du catalogue."""

    # constructeur : memorise les chemins, les numeros de classe et la transformation a appliquer
    def __init__(self, frame: pd.DataFrame, class_to_idx: dict[str, int], transform=None) -> None:
        self.paths = frame["path"].tolist()  # liste Python des chemins (plus rapide a indexer qu'une colonne pandas)
        self.targets = [class_to_idx[label] for label in frame["label"]]  # nom de classe -> numero (0, 1, 2...) attendu par la perte
        self.transform = transform  # transformation appliquee a chaque image (ou None)

    def __len__(self) -> int:  # nombre d'exemples ; utilise par len(dataset) et par le DataLoader
        return len(self.paths)  # une image par chemin

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:  # renvoie l'exemple numero index : (image, classe)
        image = Image.open(self.paths[index]).convert("RGB")  # ouvre l'image et force 3 canaux (meme une image en gris)
        if self.transform is not None:  # si une transformation est prevue
            image = self.transform(image)  # la PIL devient un tenseur redimensionne, augmente et normalise
        return image, self.targets[index]  # l'image et son numero de classe


# calcule la moyenne et l'ecart-type des pixels par canal ; renvoie deux listes de 3 nombres
def compute_mean_std(frame: pd.DataFrame, class_to_idx: dict[str, int], image_size: int) -> tuple[list[float], list[float]]:
    """Moyenne et ecart-type par canal, sur le train seulement, images en [0, 1]."""
    # transformation minimale : PIL -> tenseur, taille fixe, valeurs entieres 0-255 converties en reels 0-1 (scale=True)
    to_tensor = v2.Compose([v2.ToImage(), v2.Resize((image_size, image_size), antialias=True), v2.ToDtype(torch.float32, scale=True)])
    dataset = ImageDataset(frame, class_to_idx, to_tensor)  # dataset qui renvoie ces tenseurs
    total, total_sq, n_pixels = torch.zeros(3), torch.zeros(3), 0  # sommes par canal, sommes des carres, compteur de pixels
    for i in range(len(dataset)):  # pour chaque image du train
        image, _ = dataset[i]  # tenseur de forme (3, hauteur, largeur) ; l'etiquette ne sert pas ici
        total += image.sum(dim=(1, 2))  # ajoute la somme des pixels de chaque canal (on somme hauteur et largeur)
        total_sq += (image**2).sum(dim=(1, 2))  # ajoute la somme des carres (pour la variance)
        n_pixels += image.shape[1] * image.shape[2]  # ajoute le nombre de pixels de l'image
    mean = total / n_pixels  # moyenne par canal
    std = (total_sq / n_pixels - mean**2).sqrt()  # ecart-type = racine(moyenne des carres - carre de la moyenne)
    return mean.tolist(), std.tolist()  # tenseurs convertis en listes Python (faciles a ecrire en YAML / JSON)


# augmentations disponibles : nom ecrit dans le YAML (cle augment) -> fonction qui fabrique la transformation
AUGMENTATIONS = {
    "hflip": lambda: v2.RandomHorizontalFlip(p=0.5),  # miroir gauche-droite une fois sur deux (variete gratuite)
    "vflip": lambda: v2.RandomVerticalFlip(p=0.5),  # miroir haut-bas une fois sur deux (un defaut reste un defaut)
    "rotate": lambda: v2.RandomRotation(degrees=10),  # petite rotation au hasard, entre -10 et +10 degres
    "brightness": lambda: v2.ColorJitter(brightness=0.2, contrast=0.2),  # eclairage et contraste modifies de 20 % au plus
}


# construit les deux chaines de transformations ; renvoie (entrainement, evaluation)
def build_transforms(
    image_size: int,  # cote des images carrees donnees au reseau
    mean: list[float],  # moyenne de normalisation par canal
    std: list[float],  # ecart-type de normalisation par canal
    augment: list[str] | None = None,  # noms des augmentations du train (cle augment du YAML) ; None = les deux retournements
) -> tuple[v2.Compose, v2.Compose]:  # renvoie les deux chaines de transformations
    """(transformation d'entrainement avec augmentations, transformation d'evaluation sans)."""
    size = (image_size, image_size)  # taille cible (hauteur, largeur)
    names = ["hflip", "vflip"] if augment is None else augment  # sans precision : les augmentations du cours
    unknown = [name for name in names if name not in AUGMENTATIONS]  # noms qui ne correspondent a aucune augmentation
    if unknown:  # faute de frappe dans le YAML
        raise ValueError(f"augmentations inconnues : {unknown} (choix : {sorted(AUGMENTATIONS)})")  # erreur avec les choix possibles
    train_transform = v2.Compose(  # Compose enchaine les etapes dans l'ordre
        [  # liste des etapes, appliquees de haut en bas a chaque image
            v2.ToImage(),  # image PIL -> tenseur d'entiers (canaux, hauteur, largeur)
            v2.Resize(size, antialias=True),  # taille fixe ; antialias evite les effets d'escalier
            *[AUGMENTATIONS[name]() for name in names],  # les augmentations demandees, dans l'ordre du YAML (* = une etape chacune)
            v2.ToDtype(torch.float32, scale=True),  # entiers 0-255 -> reels 0-1
            v2.Normalize(mean=mean, std=std),  # normalisation : (pixel - moyenne) / ecart-type, valeurs centrees autour de 0
        ]
    )
    eval_transform = v2.Compose(  # evaluation : memes etapes mais sans hasard, pour des resultats stables
        # tenseur, taille fixe, reels 0-1, normalisation (pas de retournement)
        [v2.ToImage(), v2.Resize(size, antialias=True), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=mean, std=std)]
    )
    return train_transform, eval_transform  # les deux chaines


# cree le DataLoader qui fournit les images par lots ; renvoie ce DataLoader
def build_loader(frame: pd.DataFrame, class_to_idx: dict[str, int], transform, config: Config, train: bool) -> DataLoader:
    """Un DataLoader ; shuffle et drop_last seulement pour le train."""
    dataset = ImageDataset(frame, class_to_idx, transform)  # dataset des images de ce sous-ensemble
    generator = torch.Generator().manual_seed(config.seed)  # ordre de melange reproductible
    return DataLoader(  # objet a parcourir avec for : chaque tour donne un lot (images, etiquettes)
        dataset,  # ou lire les exemples
        batch_size=config.batch_size,  # nombre d'images par lot
        shuffle=train,  # melange a chaque epoque pour le train seulement (l'ordre ne doit pas influencer l'apprentissage)
        drop_last=train,  # train : jette le dernier lot incomplet (un lot minuscule perturbe la BatchNorm)
        num_workers=config.num_workers,  # processus de chargement en parallele
        generator=generator,  # generateur aleatoire du melange (graine fixe)
        worker_init_fn=seed_worker,  # graine propre a chaque processus de chargement (numpy et random)
        persistent_workers=config.num_workers > 0,  # garde les processus vivants entre epoques (evite de les relancer)
    )


# calcule un poids par classe pour la perte ; renvoie un tenseur d'un nombre par classe
def class_weights(frame: pd.DataFrame, class_names: list[str]) -> torch.Tensor:
    """Poids inverses aux effectifs, pour la perte (classes rares comptees plus fort)."""
    counts = frame["label"].value_counts().reindex(class_names).to_numpy()  # nombre d'images par classe, dans l'ordre des classes
    # poids = total / (nb de classes x effectif) : vaut 1 si les classes sont equilibrees, plus de 1 pour une classe rare
    return torch.tensor(len(frame) / (len(class_names) * counts), dtype=torch.float32)
