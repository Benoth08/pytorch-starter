# %% [markdown]
# # Segmentation : une classe par pixel, brique par brique
#
# Jeu Magnetic Tile Defect : images de tuiles magnetiques en niveaux de gris, cinq types de defauts,
# un masque pixel par image. Tache : defaut / pas defaut pour CHAQUE pixel (segmentation binaire).
# Par rapport a la classification, quatre briques changent : le Dataset (image + masque) et les
# transformations (identiques pour les deux) aux etapes 5 a 7, le modele (encodeur + decodeur, U-Net) a l'etape 8,
# la perte et les metriques (BCE + Dice, IoU) aux etapes 9 et 11. Tout le reste est le meme code,
# dans le meme fil en 13 etapes.

# %% imports
from __future__ import annotations  # permet d'ecrire les types modernes (int | None) sans souci de version de Python

import json  # ecriture de fichiers JSON (metriques, fiche du modele)
import random  # hasard de Python, fixe plus bas
import time  # mesure du temps ecoule
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import matplotlib.pyplot as plt  # matplotlib : affichage d'images et de graphiques
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import onnxruntime as ort  # onnxruntime : execute un modele exporte au format ONNX (format standard, sans PyTorch)
import pandas as pd  # pandas : tableaux de donnees (DataFrame), ici le catalogue des images
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU) et apprentissage profond
import torch.nn as nn  # nn : briques de reseau (couches, fonctions de perte)
import torch.nn.functional as F  # F : operations sous forme de fonctions (ici l'agrandissement d'images)
from PIL import Image  # PIL (Pillow) : ouvrir les fichiers images
from scipy.ndimage import gaussian_filter  # scipy : flou gaussien (pour separer l'eclairage des details d'une image)
from sklearn.model_selection import StratifiedGroupKFold  # coupe en plis sans couper de groupe, en equilibrant les classes
from torch.utils.data import DataLoader, Dataset  # Dataset : une image a la fois ; DataLoader : les regroupe en lots
from torchvision import models, tv_tensors  # models : reseaux pre-entraines ; tv_tensors : types speciaux (ici Mask)
from torchvision.transforms import v2  # transformations d'images (redimensionner, retourner, normaliser...)

SEG_CONFIG = {  # tous les reglages de la segmentation au meme endroit
    "data_dir": "data/raw/magnetic_tile",  # MT_<classe>/Imgs/x.jpg + x.png (scripts/get_magnetic_tile.py)
    "output_dir": "outputs/segmentation",  # dossier des resultats
    "image_size": 256,  # cote des images donnees au reseau (256 x 256)
    "batch_size": 8,  # 8 images traitees ensemble a chaque pas (lot / batch)
    "num_workers": 0,  # 0 : chargement des images dans le processus principal (simple, marche partout)
    "n_free": 150,  # images sans defaut gardees (sur 952) : le modele doit aussi apprendre a ne RIEN detecter
    "val_fraction": 0.15,  # 15 % des images pour la validation
    "test_fraction": 0.15,  # 15 % des images pour le test
    "min_defect_area": 20,  # decision par image : "defaut" si plus de 20 pixels sont predits en defaut
    "epochs": 12,  # nombre d'epoques (une epoque = un passage sur toutes les images d'entrainement)
    "learning_rate": 1e-3,  # decodeur (neuf)
    "encoder_learning_rate": 1e-4,  # encodeur ResNet18 pre-entraine (on l'ajuste doucement)
    "weight_decay": 1e-4,  # pousse les poids vers 0 : limite l'apprentissage par coeur
    "seed": 42,  # graine du hasard
}
OUTPUT_DIR = Path(SEG_CONFIG["output_dir"])  # chemin du dossier de sortie
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # cree le dossier s'il n'existe pas
SIZE = SEG_CONFIG["image_size"]  # raccourci pour la taille d'image
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # carte graphique (GPU) si elle existe, sinon CPU
# moyenne et ecart-type par canal des images ImageNet : l'encodeur pre-entraine attend des images normalisees ainsi
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def seed_everything(seed: int) -> None:  # fixe toutes les sources de hasard ; ne renvoie rien
    random.seed(seed)  # hasard de Python
    np.random.seed(seed)  # hasard de numpy
    torch.manual_seed(seed)  # hasard de PyTorch sur CPU
    torch.cuda.manual_seed_all(seed)  # hasard de PyTorch sur GPU


seed_everything(SEG_CONFIG["seed"])  # resultats reproductibles

# %% catalog_seg
# Etape 3 - Cataloguer : le catalogue d'un jeu de segmentation = (image, masque, classe de l'image).
# Ici le masque a le meme nom que l'image, en .png ; dans d'autres jeux il est dans un dossier masks/ parallele.
DATA_DIR = Path(SEG_CONFIG["data_dir"])  # dossier des donnees brutes
rows = []  # une ligne par image pour le catalogue
for class_dir in sorted(DATA_DIR.glob("MT_*")):  # pour chaque dossier de classe (MT_Blowhole, MT_Free...)
    label = class_dir.name.removeprefix("MT_").lower()  # blowhole, break, crack, fray, uneven, free
    for image_path in sorted((class_dir / "Imgs").glob("*.jpg")):  # pour chaque image .jpg de cette classe
        mask_path = image_path.with_suffix(".png")  # le masque : meme nom, extension .png
        assert mask_path.is_file(), f"masque manquant : {mask_path}"  # arrete tout si le masque n'existe pas
        rows.append({"path": image_path.as_posix(), "mask_path": mask_path.as_posix(), "label": label})  # chemins avec des /
catalog = pd.DataFrame(rows)  # le catalogue : un tableau pandas
# Trop d'images "free" noieraient les defauts (952 contre 392) : on en garde un echantillon fixe.
free = catalog[catalog["label"] == "free"].sample(n=SEG_CONFIG["n_free"], random_state=SEG_CONFIG["seed"])  # 150 "free" au hasard
# remet ensemble les images avec defaut et l'echantillon "free", triees par chemin, renumerotees a partir de 0
catalog = pd.concat([catalog[catalog["label"] != "free"], free]).sort_values("path").reset_index(drop=True)


def read_mask(path: str, size: int | None = None) -> np.ndarray:  # lit un masque ; renvoie un tableau de 0 et de 1
    """Masque 0/1. Seuil a 127 : les bords du PNG sont anti-crenelees (valeurs entre 0 et 255). NEAREST pour rester binaire."""
    mask = Image.open(path).convert("L")  # ouvre le PNG en niveaux de gris ("L" : un seul canal)
    if size is not None:  # si une taille est demandee
        mask = mask.resize((size, size), Image.NEAREST)  # redimensionne en plus proche voisin : pas de valeurs intermediaires
    return (np.array(mask) > 127).astype(np.uint8)  # 1 = defaut (pixel clair), 0 = fond


catalog["defect_fraction"] = [read_mask(p).mean() for p in catalog["mask_path"]]  # part des pixels en defaut dans chaque image
sizes = pd.Series([Image.open(p).size for p in catalog["path"]])  # taille (largeur, hauteur) de chaque image
# par classe : nombre d'images et part moyenne de pixels en defaut
print(catalog.groupby("label").agg(images=("path", "count"), pixels_defaut=("defect_fraction", "mean")).round(4))
# les images n'ont pas toutes la meme taille ; mode "L" = niveaux de gris
print("tailles d'images distinctes :", sizes.nunique(), "| exemple :", sizes.iloc[0], "| mode :", Image.open(catalog["path"][0]).mode)

# %% show_masks
# Toujours regarder les masques : un masque decale, inverse (fond blanc) ou mal seuille ruine l'entrainement.
class_names = sorted(catalog["label"].unique())  # liste triee des classes d'images
fig, axes = plt.subplots(2, len(class_names), figsize=(15, 5.2))  # grille : 2 lignes, une colonne par classe
for col, name in enumerate(class_names):  # pour chaque classe (col : numero de colonne)
    row = catalog[catalog["label"] == name].iloc[0]  # premiere image de cette classe
    image = Image.open(row["path"]).convert("L")  # l'image en niveaux de gris
    axes[0, col].imshow(image, cmap="gray")  # ligne 1 : l'image seule
    axes[0, col].set_title(f"{name} {image.size}", fontsize=9)  # titre : classe et taille
    axes[1, col].imshow(image, cmap="gray")  # ligne 2 : l'image...
    mask = read_mask(row["mask_path"])  # ...et son masque
    # masked_where cache les pixels de fond : seul le defaut est colore, par-dessus l'image (alpha = transparence)
    axes[1, col].imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.6, vmin=0, vmax=1)
    axes[1, col].set_title(f"{mask.mean():.1%} de pixels", fontsize=9)  # part de pixels en defaut
for ax in axes.ravel():  # pour chaque case de la grille
    ax.axis("off")  # cache les axes
plt.tight_layout()  # ajuste les marges
plt.show()  # affiche


# %% split_seg
# Etape 4 - Decouper : par groupes de quasi-doublons, et equilibre par la classe de l'image (free compris).
# Ce jeu contient des images presque identiques : la meme tuile, recadree ou eclairee autrement. Si l'une part en train et
# l'autre en test, le modele est mesure sur des images qu'il a deja vues, et le score est trop beau. On les repere donc
# d'abord, puis on decoupe sans jamais separer deux images d'un meme groupe (decoupage groupe, etape 4 du cours).
def dhash(image: Image.Image) -> int:  # empreinte visuelle d'une image ; renvoie un entier de 64 bits
    """Deux images presque identiques ont des empreintes egales a quelques bits pres."""
    small = np.asarray(image.convert("L").resize((9, 8), Image.BOX), dtype=np.int16)  # vignette grise de 9 colonnes x 8 lignes
    bits = (small[:, 1:] > small[:, :-1]).flatten()  # 1 si un pixel est plus clair que son voisin de gauche : 8 x 8 = 64 bits
    return int(np.packbits(bits).view(">u8")[0])  # les 64 bits ranges dans un seul entier


def detail_signature(image: Image.Image, size: int = 64) -> np.ndarray:  # details fins d'une image ; renvoie 64 x 64 nombres
    """Vignette sans son eclairage, centree et normee : le produit de deux signatures est une correlation (1 = memes details)."""
    small = np.asarray(image.convert("L").resize((size, size), Image.BOX), dtype=np.float32)  # vignette grise 64 x 64
    detail = small - gaussian_filter(small, 3)  # retire la version floue (eclairage, degrades) : il reste les details fins
    detail -= detail.mean()  # centre : moyenne a 0
    return (detail / max(float(np.linalg.norm(detail)), 1e-6)).ravel()  # longueur 1 (1e-6 evite de diviser par 0), mis a plat


# regroupe les images presque identiques ; renvoie un numero de groupe par image (meme numero = meme groupe)
def near_duplicate_groups(paths: list[str], max_hamming: int = 6, min_correlation: float = 0.85) -> list[int]:
    """Deux temps : empreintes a max_hamming bits pres, puis details correles. Mieux vaut grouper trop que pas assez."""
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


catalog["group"] = near_duplicate_groups(catalog["path"].tolist())  # numero de groupe de chaque image
group_sizes = catalog["group"].value_counts()  # nombre d'images par groupe
# bilan : une image seule forme un groupe ; les groupes de 2 images ou plus sont les quasi-doublons
print(f"{len(catalog)} images | {len(group_sizes)} groupes | {(group_sizes > 1).sum()} groupes de quasi-doublons", end=" ")
print(f"({group_sizes[group_sizes > 1].sum()} images, le plus grand en compte {group_sizes.max()})")  # fin de la meme ligne

# decoupe 1 : le test. 7 plis de taille 1 / 7 (15 % demandes), on garde le premier : aucun groupe coupe, classes equilibrees
folds = StratifiedGroupKFold(n_splits=round(1 / SEG_CONFIG["test_fraction"]), shuffle=True, random_state=SEG_CONFIG["seed"])
rest_idx, test_idx = next(folds.split(catalog, catalog["label"], groups=catalog["group"]))  # positions des lignes : reste / test
rest = catalog.iloc[rest_idx]  # ce qui reste apres le test
# decoupe 2 : la validation, par la meme methode sur le reste (0.85 / 0.15 -> 6 plis : environ 15 % du total)
n_val_folds = round((1 - SEG_CONFIG["test_fraction"]) / SEG_CONFIG["val_fraction"])  # nombre de plis pour la validation
folds = StratifiedGroupKFold(n_splits=n_val_folds, shuffle=True, random_state=SEG_CONFIG["seed"])  # decoupeur reproductible
train_idx, val_idx = next(folds.split(rest, rest["label"], groups=rest["group"]))  # positions dans "rest" : entrainement / validation
# ajoute la colonne split au catalogue : "train", "val" ou "test"
catalog.loc[rest.index[train_idx], "split"], catalog.loc[rest.index[val_idx], "split"] = "train", "val"
catalog.loc[catalog.index[test_idx], "split"] = "test"  # les lignes du premier pli
# les trois parties, sous forme de tableaux (une ligne par image)
train_df, val_df, test_df = (catalog[catalog["split"] == name] for name in ("train", "val", "test"))
assert catalog["path"].is_unique  # verifie qu'aucune image n'apparait deux fois
assert catalog.groupby("group")["split"].nunique().max() == 1  # aucun groupe a cheval sur deux parties : pas de fuite
catalog.to_csv(OUTPUT_DIR / "split.csv", index=False)  # sauvegarde le decoupage
print(pd.crosstab(catalog["label"], catalog["split"]))  # nombre d'images par classe et par partie


# %% dataset_seg
# Etapes 5 a 7 (pretraiter, augmenter, charger par lots) : le Dataset renvoie (image (3, S, S) normalisee, masque (1, S, S) en 0/1).
# tv_tensors.Mask : v2 applique le MEME retournement a l'image et au masque, redimensionne le masque en plus proche
# voisin, et ne lui applique NI la mise a l'echelle [0, 1] NI la normalisation (reservees aux images).
class SegmentationDataset(Dataset):  # jeu de donnees : renvoie une image et son masque, transformes ensemble
    def __init__(self, frame: pd.DataFrame, transform) -> None:  # recoit le tableau des images et la transformation
        self.paths = frame["path"].tolist()  # chemins des images
        self.mask_paths = frame["mask_path"].tolist()  # chemins des masques, dans le meme ordre
        self.transform = transform  # transformation a appliquer

    def __len__(self) -> int:  # nombre d'images du jeu
        return len(self.paths)  # une image par chemin

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:  # renvoie (image, masque) numero index
        image = Image.open(self.paths[index]).convert("RGB")  # gris -> 3 canaux identiques : l'encodeur ImageNet attend du RGB
        mask = tv_tensors.Mask(read_mask(self.mask_paths[index])[None])  # (1, H, W) ; [None] ajoute la dimension canal
        image, mask = self.transform(image, mask)  # memes transformations geometriques pour les deux
        return image, mask.float()  # masque en nombres a virgule : la perte BCE l'exige


train_transform = v2.Compose(  # transformations de l'entrainement, dans l'ordre
    [  # liste des etapes
        v2.ToImage(),  # image PIL -> tenseur image (canaux, hauteur, largeur)
        v2.Resize((SIZE, SIZE), antialias=True),  # toutes les images en 256 x 256
        v2.RandomHorizontalFlip(),  # augmentation : retourne gauche-droite une fois sur deux (masque compris)
        v2.RandomVerticalFlip(),  # augmentation : retourne haut-bas une fois sur deux
        v2.ToDtype(torch.float32, scale=True),  # valeurs 0-255 -> 0-1 (pas pour le masque)
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # normalisation ImageNet : (valeur - moyenne) / ecart-type
    ]
)
eval_transform = v2.Compose(  # transformations de validation / test : sans augmentation aleatoire
    [  # liste des etapes
        v2.ToImage(),  # image PIL -> tenseur image
        v2.Resize((SIZE, SIZE), antialias=True),  # 256 x 256
        v2.ToDtype(torch.float32, scale=True),  # valeurs 0-1
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # normalisation ImageNet
    ]
)


def make_loader(frame: pd.DataFrame, transform, train: bool) -> DataLoader:  # cree un DataLoader (fournit les images par lots)
    return DataLoader(  # renvoie le DataLoader
        SegmentationDataset(frame, transform),  # le jeu de donnees
        batch_size=SEG_CONFIG["batch_size"],  # taille des lots
        shuffle=train,  # melange seulement pour l'entrainement
        drop_last=train,  # a l'entrainement, jette le dernier lot incomplet
        num_workers=SEG_CONFIG["num_workers"],  # processus de chargement
    )


train_loader, val_loader, test_loader = (  # les trois DataLoaders
    make_loader(train_df, train_transform, True),  # entrainement : augmentation et melange
    make_loader(val_df, eval_transform, False),  # validation
    make_loader(test_df, eval_transform, False),  # test
)
images, masks = next(iter(train_loader))  # un premier lot, pour verifier
# formes (8, 3, 256, 256) et (8, 1, 256, 256) ; les masques ne contiennent que 0 et 1
print("images", tuple(images.shape), images.dtype, "| masques", tuple(masks.shape), masks.dtype, "valeurs", torch.unique(masks).tolist())

# %% show_batch_seg
fig, axes = plt.subplots(1, 8, figsize=(16, 2.4))  # 8 cases cote a cote (une par image du lot)
for ax, image, mask in zip(axes, images, masks):  # pour chaque image du lot et son masque
    # annule la normalisation (x * ecart-type + moyenne) pour afficher des couleurs normales ; clamp : garde entre 0 et 1
    shown = (image * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)).clamp(0, 1)
    ax.imshow(shown.permute(1, 2, 0).numpy())  # (C, H, W) -> (H, W, C) pour matplotlib
    if mask.any():  # si l'image contient un defaut
        ax.contour(mask[0].numpy(), levels=[0.5], colors="red", linewidths=1)  # dessine son contour en rouge
    ax.axis("off")  # cache les axes
plt.tight_layout()  # ajuste les marges
plt.show()  # affiche


# %% model_unet
# Etape 8 - Modele : U-Net = un encodeur qui reduit la resolution (ici ResNet18 pre-entraine, comme pour le transfert) et un
# decodeur qui la remonte, avec des "ponts" (skip connections) : a chaque etage, le decodeur recoit aussi les
# cartes de l'encodeur de meme resolution, pour retrouver les contours fins perdus par la reduction.
class UpBlock(nn.Module):  # un etage du decodeur : double la resolution et affine avec deux convolutions
    """x2 en resolution, concatenation avec le pont, deux convolutions 3x3."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:  # canaux en entree, du pont, en sortie
        super().__init__()  # initialise la partie nn.Module
        self.conv = nn.Sequential(  # les couches de l'etage, dans l'ordre
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),  # convolution 3x3 sur entree + pont
            nn.BatchNorm2d(out_channels),  # BatchNorm : remet les valeurs a une echelle stable
            nn.ReLU(inplace=True),  # ReLU (activation) : garde le positif, met 0 sinon
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),  # seconde convolution 3x3
            nn.BatchNorm2d(out_channels),  # BatchNorm
            nn.ReLU(inplace=True),  # ReLU
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:  # x : sortie de l'etage d'avant ; skip : le pont
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # agrandit x2 (interpolation bilineaire)
        if skip is not None:  # s'il y a un pont a cet etage
            x = torch.cat([x, skip], dim=1)  # colle les cartes de l'encodeur a la suite (dimension des canaux)
        return self.conv(x)  # passe dans les deux convolutions


class ResNetUNet(nn.Module):  # le U-Net complet : encodeur ResNet18 + decodeur ; renvoie un logit par pixel
    """Image (N, 3, S, S) -> logits (N, 1, S, S) : un score par pixel, avant sigmoide."""

    def __init__(self, pretrained: bool = True) -> None:  # pretrained : partir des poids ImageNet ou non
        super().__init__()  # initialise la partie nn.Module
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)  # ResNet18, poids ImageNet ou au hasard
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 canaux, S/2
        self.pool = resnet.maxpool  # reduction x2 du ResNet
        self.enc1, self.enc2, self.enc3, self.enc4 = (  # les 4 grands blocs du ResNet, reutilises tels quels
            resnet.layer1,  # bloc 1
            resnet.layer2,  # bloc 2
            resnet.layer3,  # bloc 3
            resnet.layer4,  # bloc 4
        )  # 64 S/4, 128 S/8, 256 S/16, 512 S/32
        self.up4 = UpBlock(512, 256, 256)  # S/16
        self.up3 = UpBlock(256, 128, 128)  # S/8
        self.up2 = UpBlock(128, 64, 64)  # S/4
        self.up1 = UpBlock(64, 64, 32)  # S/2
        self.up0 = UpBlock(32, 0, 16)  # S : plus de pont a pleine resolution (l'encodeur commence a S/2)
        self.head = nn.Conv2d(16, 1, 1)  # convolution 1x1 : 16 canaux -> 1 score (logit) par pixel

    def encoder_parameters(self):  # renvoie la liste des poids de l'encodeur (noms qui commencent par "enc")
        return [p for name, p in self.named_parameters() if name.startswith("enc")]  # pour leur donner un lr a part

    def decoder_parameters(self):  # renvoie la liste des autres poids (decodeur et tete)
        return [p for name, p in self.named_parameters() if not name.startswith("enc")]  # tout ce qui n'est pas l'encodeur

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # passage avant : image -> logits par pixel
        e0 = self.enc0(x)  # (N, 64, S/2, S/2)
        e1 = self.enc1(self.pool(e0))  # (N, 64, S/4, S/4)
        e2 = self.enc2(e1)  # (N, 128, S/8, S/8)
        e3 = self.enc3(e2)  # (N, 256, S/16, S/16)
        e4 = self.enc4(e3)  # (N, 512, S/32, S/32)
        d = self.up4(e4, e3)  # remonte a S/16, avec le pont e3
        d = self.up3(d, e2)  # remonte a S/8, avec le pont e2
        d = self.up2(d, e1)  # remonte a S/4, avec le pont e1
        d = self.up1(d, e0)  # remonte a S/2, avec le pont e0
        return self.head(self.up0(d, None))  # remonte a pleine taille S, puis un logit par pixel


model = ResNetUNet().to(DEVICE)  # cree le modele et le place sur CPU ou GPU
with torch.no_grad():  # sans gradient : simple verification
    out = model(torch.zeros(2, 3, SIZE, SIZE, device=DEVICE))  # 2 images factices
print("entree (2, 3, S, S) -> logits", tuple(out.shape))  # attendu : (2, 1, 256, 256)
# nombre total de poids et nombre de poids de l'encodeur ({:,} : separateur de milliers)
print(f"parametres : {sum(p.numel() for p in model.parameters()):,} dont encodeur {sum(p.numel() for p in model.encoder_parameters()):,}")


# %% loss_metrics
# Etapes 9 et 11 (perte, metrique). BCE seule echoue quand le defaut fait 0,2 % des pixels (tout predire "fond" donne deja 99,8 %).
# Dice = 2 |A inter B| / (|A| + |B|) ne regarde que les pixels du defaut : on additionne les deux.
# Metrique : IoU = |A inter B| / |A union B|, cumulee sur tout le jeu (pas la moyenne des IoU par image).
def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:  # renvoie 1 - Dice, moyenne du lot
    probs = torch.sigmoid(logits)  # sigmoide : logit -> probabilite de defaut entre 0 et 1, pour chaque pixel
    intersection = (probs * targets).sum(dim=(1, 2, 3))  # recouvrement prediction / verite, par image
    total = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))  # taille predite + taille vraie, par image
    return (1 - (2 * intersection + smooth) / (total + smooth)).mean()  # smooth evite 0/0 sur les images sans defaut


class DiceBCELoss(nn.Module):  # perte combinee : BCE (pixel par pixel) + Dice (recouvrement)
    def __init__(self) -> None:  # constructeur
        super().__init__()  # initialise la partie nn.Module
        self.bce = nn.BCEWithLogitsLoss()  # BCE : entropie croisee binaire, sigmoide incluse

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # renvoie la perte totale
        return self.bce(logits, targets) + dice_loss(logits, targets)  # somme des deux pertes


def confusion_pixels(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> tuple[int, int, int]:  # compte des pixels
    """(intersection, predits, vrais) en nombre de pixels, pour cumuler sur un jeu entier."""
    preds = (torch.sigmoid(logits) > threshold).float()  # decision par pixel : 1 si la probabilite depasse le seuil
    return int((preds * targets).sum()), int(preds.sum()), int(targets.sum())  # pixels communs, predits, vrais


def iou_dice(intersection: int, predicted: int, truth: int) -> tuple[float, float]:  # renvoie (IoU, Dice) a partir des comptes
    union = predicted + truth - intersection  # pixels dans l'un ou l'autre
    return intersection / max(union, 1), 2 * intersection / max(predicted + truth, 1)  # max(..., 1) evite une division par 0


criterion = DiceBCELoss()  # la perte utilisee pour l'entrainement
perfect = (torch.rand(2, 1, 8, 8) > 0.5).float()  # un faux masque 8 x 8 au hasard
print(  # perte d'une prediction parfaite (logits +10 / -10 : tres surs) : proche de 0 ; IoU = 1
    f"perte sur une prediction parfaite : {criterion(perfect * 20 - 10, perfect).item():.4f} | IoU : {iou_dice(*confusion_pixels(perfect * 20 - 10, perfect))[0]:.2f}"
)
print(f"perte en predisant 'fond partout' : {criterion(torch.full_like(perfect, -10), perfect).item():.4f}")  # perte elevee grace au Dice


# %% train_seg
# Etape 10 - Entrainer : les memes cinq lignes ; seules la perte et la metrique cumulee changent.
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer) -> float:  # entraine une epoque ; renvoie la perte moyenne
    model.train()  # mode entrainement (BatchNorm apprend ses statistiques)
    total, seen = 0.0, 0  # cumul de la perte, nombre d'images vues
    for images, masks in loader:  # pour chaque lot d'images et de masques
        images, masks = images.to(DEVICE), masks.to(DEVICE)  # sur le meme appareil que le modele
        optimizer.zero_grad()  # remet les gradients a zero
        loss = criterion(model(images), masks)  # passage avant et perte
        loss.backward()  # gradients (dans quel sens bouger chaque poids)
        optimizer.step()  # optimiseur : met a jour les poids
        total += loss.item() * len(images)  # ajoute la perte du lot (ponderee par sa taille)
        seen += len(images)  # compte les images
    return total / seen  # perte moyenne de l'epoque


@torch.no_grad()  # pas de gradient dans cette fonction : plus rapide, moins de memoire
def evaluate(model: nn.Module, loader: DataLoader, criterion, threshold: float = 0.5) -> dict[str, float]:  # renvoie perte, IoU, Dice
    model.eval()  # mode evaluation
    total, seen, inter, pred, truth = 0.0, 0, 0, 0, 0  # cumuls : perte, images, pixels communs, predits, vrais
    for images, masks in loader:  # pour chaque lot
        images, masks = images.to(DEVICE), masks.to(DEVICE)  # sur l'appareil du modele
        logits = model(images)  # logits par pixel
        total += criterion(logits, masks).item() * len(images)  # ajoute la perte du lot
        seen += len(images)  # compte les images
        i, p, t = confusion_pixels(logits, masks, threshold)  # comptes de pixels du lot
        inter, pred, truth = inter + i, pred + p, truth + t  # cumule sur tout le jeu
    iou, dice = iou_dice(inter, pred, truth)  # IoU et Dice globaux
    return {"loss": total / seen, "iou": iou, "dice": dice}  # resultats


optimizer = torch.optim.AdamW(  # optimiseur AdamW avec deux groupes de poids
    [  # un taux d'apprentissage par groupe
        {"params": model.decoder_parameters(), "lr": SEG_CONFIG["learning_rate"]},  # decodeur neuf : apprend vite
        {"params": model.encoder_parameters(), "lr": SEG_CONFIG["encoder_learning_rate"]},  # encodeur pre-entraine : doucement
    ],
    weight_decay=SEG_CONFIG["weight_decay"],  # weight decay commun
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)  # divise lr par 2 si val stagne 2 epoques
history, best_val = [], float("inf")  # historique des epoques ; meilleure perte de validation (infini au depart)
for epoch in range(1, SEG_CONFIG["epochs"] + 1):  # pour chaque epoque, numerotees a partir de 1
    start = time.time()  # heure de debut de l'epoque
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)  # apprend sur le train
    val = evaluate(model, val_loader, criterion)  # mesure sur la validation
    scheduler.step(val["loss"])  # le scheduler regarde la perte de validation
    # garde les chiffres de l'epoque pour les courbes
    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"], "val_iou": val["iou"], "val_dice": val["dice"]})
    improved = val["loss"] < best_val  # True si la validation s'ameliore
    if improved:  # si nouveau record
        best_val = val["loss"]  # retient la nouvelle meilleure perte
        torch.save(model.state_dict(), OUTPUT_DIR / "unet_best.pt")  # sauvegarde les poids du meilleur modele
    print(  # une ligne de suivi par epoque
        f"epoque {epoch:2d} | train perte {train_loss:.4f} | val perte {val['loss']:.4f} IoU {val['iou']:.3f} Dice {val['dice']:.3f} | {time.time() - start:.0f}s{' | meilleur' if improved else ''}"
    )
history = pd.DataFrame(history)  # l'historique en tableau pandas
model.load_state_dict(torch.load(OUTPUT_DIR / "unet_best.pt", weights_only=True))  # recharge les poids du meilleur modele

# %% curves_seg
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))  # deux graphiques cote a cote
axes[0].plot(history["epoch"], history["train_loss"], label="train")  # perte d'entrainement par epoque
axes[0].plot(history["epoch"], history["val_loss"], label="val")  # perte de validation par epoque
axes[0].set_title("perte BCE + Dice")  # titre du graphique de gauche
axes[1].plot(history["epoch"], history["val_iou"], label="IoU val")  # IoU de validation
axes[1].plot(history["epoch"], history["val_dice"], label="Dice val")  # Dice de validation
axes[1].set_title("qualite des masques (val)")  # titre du graphique de droite
for ax in axes:  # pour chaque graphique
    ax.set_xlabel("epoque")  # nom de l'axe horizontal
    ax.legend()  # legende (noms des courbes)
    ax.grid(alpha=0.3)  # quadrillage leger
plt.tight_layout()  # ajuste les marges
plt.show()  # affiche


# %% threshold_seg
# Etape 11 - Evaluer (seuil de decision) : le seuil sur la probabilite par pixel se choisit sur la VALIDATION, jamais sur le test.
@torch.no_grad()  # pas de gradient
def collect_logits(model: nn.Module, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:  # renvoie (logits, masques) du jeu
    model.eval()  # mode evaluation
    logits, masks = zip(*[(model(images.to(DEVICE)).cpu(), m) for images, m in loader])  # (logits, masques) de chaque lot, separes en deux
    return torch.cat(logits), torch.cat(masks)  # colle tous les lots


val_logits, val_masks = collect_logits(model, val_loader)  # logits et masques de la validation
sweep = {t: iou_dice(*confusion_pixels(val_logits, val_masks, t))[0] for t in (0.3, 0.4, 0.5, 0.6, 0.7)}  # IoU pour chaque seuil essaye
best_threshold = max(sweep, key=sweep.get)  # le seuil qui donne la meilleure IoU
print("IoU de validation par seuil :", {t: round(v, 3) for t, v in sweep.items()}, "-> seuil retenu", best_threshold)  # resultat

# %% evaluate_seg
# Etape 11 - Evaluer : une seule fois sur le test, au seuil retenu. IoU global, puis par type de defaut
# (sur les images de ce type), et le taux de fausses detections sur les images sans defaut.
test_logits, test_masks = collect_logits(model, test_loader)  # logits et masques du test
test_iou, test_dice = iou_dice(*confusion_pixels(test_logits, test_masks, best_threshold))  # IoU et Dice globaux sur le test
print(f"test : IoU {test_iou:.3f} | Dice {test_dice:.3f} (seuil {best_threshold})")  # affiche
per_class = {}  # resultats par classe
for name in class_names:  # pour chaque classe
    sel = torch.tensor((test_df["label"] == name).to_numpy())  # selection (True / False) des images de test de cette classe
    inter, pred, truth = confusion_pixels(test_logits[sel], test_masks[sel], best_threshold)  # comptes de pixels sur ces images
    if name == "free":  # images sans defaut : on mesure les fausses alertes
        # part des images "free" ou plus de min_defect_area pixels sont detectes comme defaut
        detected = (torch.sigmoid(test_logits[sel]) > best_threshold).flatten(1).sum(dim=1)  # pixels predits en defaut, par image
        flagged = (detected > SEG_CONFIG["min_defect_area"]).float().mean().item()  # part des images au-dessus de la surface minimale
        per_class[name] = {"images": int(sel.sum()), "fausses detections": round(flagged, 3), "pixels predits": pred}  # resultats "free"
    else:  # classes avec defaut
        per_class[name] = {  # resultats de la classe
            "images": int(sel.sum()),  # nombre d'images de test de cette classe
            "IoU": round(iou_dice(inter, pred, truth)[0], 3),  # IoU sur ces images
            "Dice": round(iou_dice(inter, pred, truth)[1], 3),  # Dice sur ces images
        }
print(pd.DataFrame(per_class).T.to_string(na_rep=""))  # tableau : une ligne par classe (cases vides si sans objet)
json.dump(  # sauvegarde les metriques de test dans un fichier JSON
    {"iou": test_iou, "dice": test_dice, "threshold": best_threshold, "per_class": per_class},  # contenu
    open(OUTPUT_DIR / "test_metrics.json", "w"),  # fichier de destination, ouvert en ecriture
    indent=2,  # JSON indente, lisible
)

# %% show_pred_seg
# Image | verite | prediction, pour chaque classe. Les erreurs sont plus instructives que le score.
fig, axes = plt.subplots(3, len(class_names), figsize=(15, 7.6))  # grille : 3 lignes, une colonne par classe
for col, name in enumerate(class_names):  # pour chaque classe
    i = int(np.flatnonzero((test_df["label"] == name).to_numpy())[0])  # position de la premiere image de test de cette classe
    image = Image.open(test_df.iloc[i]["path"]).convert("L").resize((SIZE, SIZE))  # l'image en gris, a la taille du modele
    prob = torch.sigmoid(test_logits[i, 0]).numpy()  # probabilite de defaut de chaque pixel
    pred = prob > best_threshold  # masque predit (True = defaut)
    truth = test_masks[i, 0].numpy() > 0.5  # vrai masque
    inter, union = (pred & truth).sum(), (pred | truth).sum()  # pixels communs, pixels dans l'un ou l'autre
    axes[0, col].imshow(image, cmap="gray")  # ligne 1 : l'image
    axes[0, col].set_title(name, fontsize=9)  # titre : la classe
    axes[1, col].imshow(image, cmap="gray")  # ligne 2 : l'image...
    axes[1, col].imshow(np.ma.masked_where(~truth, truth), cmap="autumn", alpha=0.6)  # ...avec la verite par-dessus
    axes[2, col].imshow(image, cmap="gray")  # ligne 3 : l'image...
    axes[2, col].imshow(np.ma.masked_where(~pred, pred), cmap="winter", alpha=0.6)  # ...avec la prediction par-dessus
    axes[2, col].set_title(f"IoU {inter / union:.2f}" if union else "rien a detecter, rien detecte", fontsize=8)  # IoU de cette image
for ax, name in zip(axes[:, 0], ("image", "verite", "prediction")):  # premiere colonne de chaque ligne
    ax.set_ylabel(name)  # nom de la ligne a gauche
for ax in axes.ravel():  # pour chaque case
    ax.set_xticks([])  # pas de graduations horizontales
    ax.set_yticks([])  # pas de graduations verticales (on garde le nom de ligne)
plt.tight_layout()  # ajuste les marges
plt.show()  # affiche

# %% onnx_seg
# Etape 12 - Sauvegarder et exporter : export, verification sur de vraies images, latence. La sortie est le logit par pixel :
# la sigmoide et le seuil sont dans model_card.json, a appliquer cote production (ou a ajouter au graphe).
model.eval().cpu()  # mode evaluation, sur CPU pour l'export
onnx_path = OUTPUT_DIR / "unet.onnx"  # fichier ONNX a creer (ONNX : format standard, executable sans PyTorch)
torch.onnx.export(  # exporte le modele au format ONNX
    model,  # le modele a exporter
    (torch.zeros(2, 3, SIZE, SIZE),),  # exemple d'entree (2 images) : sert a tracer les calculs
    str(onnx_path),  # chemin du fichier
    input_names=["input"],  # nom donne a l'entree
    output_names=["logits"],  # nom donne a la sortie
    dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=256)}},  # la dimension 0 (lot) peut varier de 1 a 256
    dynamo=True,  # utilise le nouvel exporteur de PyTorch
    external_data=False,  # tout dans un seul fichier (poids compris)
)
session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # charge le modele ONNX pour l'executer sur CPU
images, _ = next(iter(test_loader))  # un lot de test
with torch.no_grad():  # sans gradient
    torch_logits = model(images[:3]).numpy()  # logits PyTorch des 3 premieres images
onnx_logits = session.run(None, {"input": images[:3].numpy()})[0]  # logits ONNX des memes images
print(f"ecart max PyTorch / ONNX sur les logits : {np.abs(onnx_logits - torch_logits).max():.2e}")  # doit etre tres petit
assert np.allclose(onnx_logits, torch_logits, atol=1e-3)  # arrete tout si l'export ne donne pas les memes resultats
json.dump(  # ecrit la fiche du modele : ce qu'il faut pour l'utiliser ailleurs
    {  # contenu de la fiche
        "input": "input",  # nom de l'entree
        "shape": ["batch", 3, SIZE, SIZE],  # forme de l'entree
        "preprocessing": "RGB, resize bilineaire, /255, normalisation ImageNet",  # pretraitement a refaire avant
        "output": "logits (batch, 1, S, S)",  # sortie
        "postprocessing": f"sigmoid > {best_threshold}",  # post-traitement : sigmoide puis seuil
        "classes": ["fond", "defaut"],  # signification de 0 et 1
    },
    open(OUTPUT_DIR / "model_card.json", "w"),  # fichier de destination
    indent=2,  # JSON indente
)
one = images[:1].numpy()  # une seule image (lot de 1) pour mesurer la vitesse
for _ in range(3):  # 3 executions de chauffe (les premieres sont plus lentes)
    session.run(None, {"input": one})  # execution non mesuree
start = time.time()  # heure de depart
for _ in range(10):  # 10 executions mesurees
    session.run(None, {"input": one})  # execution ONNX
onnx_ms = (time.time() - start) / 10 * 1000  # temps moyen en millisecondes
start = time.time()  # nouveau depart pour PyTorch
with torch.no_grad():  # sans gradient
    for _ in range(10):  # 10 executions
        model(images[:1])  # execution PyTorch
torch_ms = (time.time() - start) / 10 * 1000  # temps moyen en millisecondes
print(f"latence lot de 1 sur CPU : PyTorch {torch_ms:.0f} ms | onnxruntime {onnx_ms:.0f} ms")  # comparaison des vitesses
