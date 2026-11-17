# %% [markdown]
# # Non supervise : detection d'anomalies, exploration sans etiquettes, Grad-CAM
#
# Trois situations ou l'on n'a pas (ou pas encore) d'etiquettes de defauts :
# 1. detection d'anomalies : apprendre ce qui est normal, signaler ce qui s'en ecarte
#    (auto-encodeur, puis caracteristiques pre-entrainees : PatchCore simplifie) ;
# 2. explorer un lot d'images sans etiquettes : embeddings + clustering, pour pre-etiqueter ;
# 3. Grad-CAM : ou regarde un classifieur deja entraine (controle avant mise en production).
# Meme fil en 13 etapes que la classification : reglages (1) -> catalogue (3) -> decoupage (4) -> Dataset (5 a 7)
# -> modele (8) -> perte et boucle (9, 10) -> evaluation (11) -> export (12).

# %% imports
from __future__ import annotations  # permet d'ecrire les types modernes (tuple[int, ...]) sans souci de version de Python

import json  # ecriture de fichiers JSON (seuils, fiche du modele)
import math  # fonctions mathematiques de base (arrondi superieur...)
import random  # hasard de Python, fixe plus bas
import time  # mesure du temps ecoule
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import matplotlib.pyplot as plt  # matplotlib : affichage d'images et de graphiques
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import onnxruntime as ort  # onnxruntime : execute un modele exporte au format ONNX (format standard, sans PyTorch)
import pandas as pd  # pandas : tableaux de donnees (DataFrame), ici le catalogue des images
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU) et apprentissage profond
import torch.nn as nn  # nn : briques de reseau (couches, fonctions de perte)
import torch.nn.functional as F  # F : operations sous forme de fonctions (perte MSE, convolution, redimensionnement)
from PIL import Image  # PIL (Pillow) : ouvrir les fichiers images
from sklearn.cluster import KMeans  # k-means : regroupe des points en k groupes de points proches
from sklearn.decomposition import PCA  # PCA : resume des vecteurs longs en 2 nombres, pour les dessiner
from sklearn.metrics import f1_score, roc_auc_score  # F1 (precision + rappel) et AUROC (qualite du classement des scores)
from sklearn.model_selection import train_test_split  # coupe un tableau en deux parties
from torch.utils.data import DataLoader, Dataset  # Dataset : une image a la fois ; DataLoader : les regroupe en lots
from torchvision import models, tv_tensors  # models : reseaux pre-entraines ; tv_tensors : types speciaux (ici Mask)
from torchvision.transforms import v2  # transformations d'images (redimensionner, retourner, normaliser...)

ANOMALY_CONFIG = {  # tous les reglages de la detection d'anomalies au meme endroit
    "data_dir": "data/raw/mvtec_bottle",  # arborescence officielle MVTec AD (scripts/get_mvtec_bottle.py)
    "output_dir": "outputs/anomaly",  # dossier des resultats
    "image_size": 256,  # cote des images donnees au reseau (256 x 256)
    "batch_size": 16,  # 16 images traitees ensemble a chaque pas (lot / batch)
    "num_workers": 0,  # 0 : chargement des images dans le processus principal (simple, marche partout)
    "val_fraction": 0.15,  # part des images normales d'entrainement gardees pour la validation
    "epochs": 30,  # nombre d'epoques (une epoque = un passage sur toutes les images d'entrainement)
    "learning_rate": 1e-3,  # taux d'apprentissage : taille du pas de chaque mise a jour des poids
    "weight_decay": 1e-5,  # pousse les poids vers 0 : limite l'apprentissage par coeur
    "latent_channels": 32,  # largeur du goulot : LE reglage de l'auto-encodeur
    "gaussian_sigma": 4.0,  # lissage de la carte d'erreur, en pixels
    "image_quantile": 0.99,  # seuil image = quantile des scores des images normales de validation
    "seed": 42,  # graine du hasard
}
OUTPUT_DIR = Path(ANOMALY_CONFIG["output_dir"])  # chemin du dossier de sortie
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # cree le dossier s'il n'existe pas
SIZE = ANOMALY_CONFIG["image_size"]  # raccourci pour la taille d'image
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # carte graphique (GPU) si elle existe, sinon CPU


def seed_everything(seed: int) -> None:  # fixe toutes les sources de hasard ; ne renvoie rien
    random.seed(seed)  # hasard de Python
    np.random.seed(seed)  # hasard de numpy
    torch.manual_seed(seed)  # hasard de PyTorch sur CPU
    torch.cuda.manual_seed_all(seed)  # hasard de PyTorch sur GPU


seed_everything(ANOMALY_CONFIG["seed"])  # resultats reproductibles

# %% catalog_anomaly
# Etape 3 - Cataloguer : le catalogue d'un jeu d'anomalies. Arborescence MVTec AD :
#   train/good/*.png             images normales (les seules pour apprendre)
#   test/good/*.png              normales de test
#   test/<type>/*.png            defectueuses, avec ground_truth/<type>/<nom>_mask.png
DATA_DIR = Path(ANOMALY_CONFIG["data_dir"])  # dossier des donnees brutes
rows = []  # une ligne par image pour le catalogue
for subset in ("train", "test"):  # pour chacune des deux parties officielles
    for class_dir in sorted((DATA_DIR / subset).iterdir()):  # pour chaque sous-dossier (good, broken_large...)
        for image_path in sorted(class_dir.glob("*.png")):  # pour chaque image .png
            defect_type = class_dir.name  # type de l'image = nom du sous-dossier
            # chemin du masque pour une image defectueuse ; None pour une image normale (pas de masque)
            mask_path = DATA_DIR / "ground_truth" / defect_type / f"{image_path.stem}_mask.png" if defect_type != "good" else None
            rows.append(  # ajoute la ligne de cette image
                {  # les informations gardees
                    "path": image_path.as_posix(),  # chemin de l'image, ecrit avec des /
                    "label": "good" if defect_type == "good" else "defect",  # la cible binaire
                    "defect_type": defect_type,  # type precis de defaut (ou "good")
                    "mask_path": str(mask_path) if mask_path else "",  # chemin du masque, texte vide si aucun
                    "split": subset,  # partie officielle : train ou test
                }
            )
catalog = pd.DataFrame(rows)  # le catalogue : un tableau pandas
assert all(Path(p).is_file() for p in catalog["mask_path"] if p), "masque manquant"  # chaque masque annonce doit exister
print(pd.crosstab(catalog["defect_type"], catalog["split"]))  # nombre d'images par type et par partie


# %% show_anomaly
# Une image normale et chaque type de defaut, avec le contour du masque : on verifie que le masque tombe sur le defaut.
def read_image(path: str, size: int) -> np.ndarray:  # lit une image couleur et renvoie un tableau (size, size, 3)
    return np.array(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR))  # ouvre, RGB, redimensionne


def read_mask(path: str, size: int) -> np.ndarray:  # lit un masque et renvoie un tableau (size, size) de 0 et de 1
    """Masque binaire 0/1 ; NEAREST : un masque redimensionne en bilineaire aurait des valeurs intermediaires."""
    if not path:  # pas de masque (image normale)
        return np.zeros((size, size), dtype=np.uint8)  # masque vide : que des 0
    return (np.array(Image.open(path).convert("L").resize((size, size), Image.NEAREST)) > 127).astype(np.uint8)  # 1 = defaut


examples = [catalog[catalog["defect_type"] == t].iloc[0] for t in ("good", "broken_large", "broken_small", "contamination")]  # 1 par type
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))  # 4 cases cote a cote
for ax, row in zip(axes, examples):  # pour chaque case et chaque exemple
    ax.imshow(read_image(row["path"], SIZE))  # affiche l'image
    mask = read_mask(row["mask_path"], SIZE)  # son masque
    if mask.any():  # s'il y a un defaut
        ax.contour(mask, levels=[0.5], colors="red", linewidths=1.2)  # dessine le contour du defaut en rouge
    ax.set_title(f"{row['defect_type']} ({mask.mean():.1%} de pixels)", fontsize=9)  # type et part de pixels en defaut
    ax.axis("off")  # cache les axes
plt.tight_layout()  # ajuste les marges
plt.show()  # affiche

# %% split_anomaly
# Etape 4 - Decouper : la validation ne contient QUE des images normales (prises dans train/good) ;
# le test officiel (normales + defauts) est conserve tel quel. Aucun defaut n'est vu avant l'evaluation finale.
train_good = catalog[catalog["split"] == "train"]  # les images normales d'entrainement
# 15 % d'entre elles pour la validation, le reste pour l'entrainement
train_df, val_df = train_test_split(train_good, test_size=ANOMALY_CONFIG["val_fraction"], random_state=ANOMALY_CONFIG["seed"])
catalog.loc[val_df.index, "split"] = "val"  # note "val" dans le catalogue pour ces images
test_df = catalog[catalog["split"] == "test"]  # le test officiel
assert (catalog.loc[catalog["split"] != "test", "label"] == "good").all(), "un defaut s'est glisse dans train ou val"  # controle
catalog.to_csv(OUTPUT_DIR / "split.csv", index=False)  # sauvegarde le decoupage
print(pd.crosstab(catalog["label"], catalog["split"]))  # nombre d'images normales / defauts par partie


# %% dataset_anomaly
# Etapes 5 a 7 (pretraiter, augmenter, charger par lots) : le Dataset renvoie (image en [0, 1], masque 0/1, etiquette 0/1).
# Pas de normalisation mean/std : la sortie sigmoide du decodeur doit pouvoir reproduire l'entree.
# tv_tensors.Mask : les transformations v2 appliquent le MEME retournement a l'image et au masque,
# et redimensionnent le masque en plus proche voisin (il reste binaire).
class AnomalyDataset(Dataset):  # jeu de donnees : renvoie image, masque et etiquette
    """Image RGB en [0, 1], masque (1, S, S) en 0/1, etiquette (0 = normal, 1 = defaut)."""

    def __init__(self, frame: pd.DataFrame, transform) -> None:  # recoit le tableau des images et la transformation
        self.paths = frame["path"].tolist()  # chemins des images
        self.mask_paths = frame["mask_path"].fillna("").tolist()  # chemins des masques (cases vides remplacees par "")
        self.targets = (frame["label"] == "defect").astype(int).tolist()  # etiquettes : 1 = defaut, 0 = normal
        self.transform = transform  # transformation a appliquer

    def __len__(self) -> int:  # nombre d'images du jeu
        return len(self.paths)  # une image par chemin

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:  # renvoie (image, masque, etiquette) numero index
        image = Image.open(self.paths[index]).convert("RGB")  # ouvre l'image en couleur
        if self.mask_paths[index]:  # si l'image a un masque
            mask = (np.array(Image.open(self.mask_paths[index]).convert("L")) > 127).astype(np.uint8)  # masque 0/1
        else:  # image normale
            mask = np.zeros((image.height, image.width), dtype=np.uint8)  # masque vide de la taille de l'image
        image, mask = self.transform(image, tv_tensors.Mask(mask[None]))  # memes transformations pour les deux ; [None] : ajoute le canal
        return image, mask.float(), self.targets[index]  # image, masque en nombres a virgule, etiquette


train_transform = v2.Compose(  # transformations de l'entrainement, dans l'ordre
    [  # liste des etapes
        v2.ToImage(),  # image PIL -> tenseur image (canaux, hauteur, largeur)
        v2.Resize((SIZE, SIZE), antialias=True),  # 256 x 256
        v2.RandomHorizontalFlip(),  # augmentation : retourne gauche-droite une fois sur deux
        v2.RandomVerticalFlip(),  # augmentation : retourne haut-bas une fois sur deux
        v2.ToDtype(torch.float32, scale=True),  # valeurs 0-255 -> 0-1
    ]
)
# validation / test : memes etapes sans augmentation aleatoire
eval_transform = v2.Compose([v2.ToImage(), v2.Resize((SIZE, SIZE), antialias=True), v2.ToDtype(torch.float32, scale=True)])


def make_loader(frame: pd.DataFrame, transform, shuffle: bool) -> DataLoader:  # cree un DataLoader (fournit les images par lots)
    return DataLoader(  # renvoie le DataLoader
        AnomalyDataset(frame, transform),  # le jeu de donnees
        batch_size=ANOMALY_CONFIG["batch_size"],  # taille des lots
        shuffle=shuffle,  # melanger ou non l'ordre des images
        num_workers=ANOMALY_CONFIG["num_workers"],  # processus de chargement
    )


train_loader = make_loader(train_df, train_transform, shuffle=True)  # entrainement : augmentation et melange
val_loader = make_loader(val_df, eval_transform, shuffle=False)  # validation (que des normales)
test_loader = make_loader(test_df, eval_transform, shuffle=False)  # test (normales + defauts), ordre du catalogue
images, masks, targets = next(iter(test_loader))  # un premier lot de test, pour verifier
print(  # formes, type et plage des valeurs, puis quelques etiquettes
    "images",  # texte affiche
    tuple(images.shape),  # (16, 3, 256, 256)
    images.dtype,  # float32
    f"[{images.min():.2f}, {images.max():.2f}] | masques",  # valeurs entre 0 et 1
    tuple(masks.shape),  # (16, 1, 256, 256)
    "| etiquettes",  # texte affiche
    targets[:8].tolist(),  # les 8 premieres etiquettes
)


# %% model_ae
# Etape 8 - Modele : l'auto-encodeur convolutif. Encodeur (resolution / 2 par bloc) -> goulot -> decodeur symetrique.
# Le goulot est le reglage central : trop large, le reseau recopie aussi les defauts ; trop etroit, il floute tout.
def down_block(in_channels: int, out_channels: int) -> nn.Sequential:  # renvoie un bloc qui divise la taille par 2
    """Conv 4x4 pas 2 : divise la resolution par 2."""
    return nn.Sequential(  # les couches du bloc, dans l'ordre
        nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),  # convolution 4x4 qui saute d'un pixel sur deux
        nn.BatchNorm2d(out_channels),  # BatchNorm : remet les valeurs a une echelle stable
        nn.LeakyReLU(0.2, inplace=True),  # activation : garde le positif, multiplie le negatif par 0.2
    )


def up_block(in_channels: int, out_channels: int) -> nn.Sequential:  # renvoie un bloc qui multiplie la taille par 2
    """Convolution transposee 4x4 pas 2 : multiplie la resolution par 2."""
    return nn.Sequential(  # les couches du bloc
        nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),  # "convolution a l'envers" : agrandit x2
        nn.BatchNorm2d(out_channels),  # BatchNorm
        nn.ReLU(inplace=True),  # ReLU : garde le positif, met 0 sinon
    )


class ConvAutoencoder(nn.Module):  # auto-encodeur : compresse l'image puis la reconstruit
    """Image (N, 3, S, S) en [0, 1] -> reconstruction de la meme forme, en [0, 1] (sigmoide)."""

    def __init__(self, latent_channels: int = 32, widths: tuple[int, ...] = (32, 64, 128, 256)) -> None:  # goulot et canaux des blocs
        super().__init__()  # initialise la partie nn.Module
        encoder, previous = [], 3  # couches de l'encodeur ; l'image a 3 canaux
        for width in widths:  # pour chaque bloc de l'encodeur
            encoder.append(down_block(previous, width))  # divise la taille par 2, passe a width canaux
            previous = width  # la sortie devient l'entree du bloc suivant
        encoder.append(nn.Conv2d(previous, latent_channels, 1))  # le goulot : une conv 1x1 qui compresse les canaux
        self.encoder = nn.Sequential(*encoder)  # l'encodeur complet
        decoder = [nn.Conv2d(latent_channels, widths[-1], 1), nn.ReLU(inplace=True)]  # decodeur : rouvre les canaux du goulot
        for width_in, width_out in zip(widths[::-1], widths[::-1][1:]):  # canaux dans l'ordre inverse : 256->128, 128->64, 64->32
            decoder.append(up_block(width_in, width_out))  # agrandit x2
        decoder += [nn.ConvTranspose2d(widths[0], 3, 4, stride=2, padding=1), nn.Sigmoid()]  # dernier x2, 3 canaux, sigmoide -> [0, 1]
        self.decoder = nn.Sequential(*decoder)  # le decodeur complet

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # passage avant : image -> reconstruction
        return self.decoder(self.encoder(x))  # compresse puis reconstruit


autoencoder = ConvAutoencoder(ANOMALY_CONFIG["latent_channels"]).to(DEVICE)  # cree le modele, le place sur CPU ou GPU
dummy = torch.zeros(2, 3, SIZE, SIZE, device=DEVICE)  # 2 images factices pour verifier les formes
latent = autoencoder.encoder(dummy)  # leur representation compressee (le goulot)
print("entree", tuple(dummy.shape), "-> goulot", tuple(latent.shape), "-> sortie", tuple(autoencoder(dummy).shape))  # formes
# combien de fois moins de nombres dans le goulot que dans l'image ; nombre de poids du modele
print(f"compression x{dummy[0].numel() / latent[0].numel():.0f} | parametres : {sum(p.numel() for p in autoencoder.parameters()):,}")


# %% train_ae
# Etapes 9 et 10 (perte, entrainement) : les cinq lignes, avec la perte MSE entre l'entree et la reconstruction (pas d'etiquette : c'est
# ce qui rend l'apprentissage "non supervise"). La validation mesure la meme perte sur des images normales jamais vues.
def run_epoch(model: nn.Module, loader: DataLoader, optimizer=None) -> float:  # une epoque d'entrainement ou d'evaluation
    """Une passe ; avec optimiseur = entrainement, sans = evaluation. Renvoie la MSE moyenne."""
    model.train(optimizer is not None)  # mode entrainement si optimiseur, sinon mode evaluation
    total, seen = 0.0, 0  # cumul de la perte, nombre d'images vues
    with torch.set_grad_enabled(optimizer is not None):  # gradients seulement si on entraine
        for images, _, _ in loader:  # pour chaque lot (masques et etiquettes ignores : non supervise)
            images = images.to(DEVICE)  # sur l'appareil du modele
            loss = F.mse_loss(model(images), images)  # MSE : ecart au carre moyen entre reconstruction et image d'origine
            if optimizer is not None:  # si on entraine
                optimizer.zero_grad()  # gradients a zero
                loss.backward()  # gradients (dans quel sens bouger chaque poids)
                optimizer.step()  # optimiseur : met a jour les poids
            total += loss.item() * len(images)  # ajoute la perte du lot
            seen += len(images)  # compte les images
    return total / seen  # MSE moyenne


# optimiseur AdamW (met a jour les poids), avec le taux d'apprentissage et le weight decay de la config
optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=ANOMALY_CONFIG["learning_rate"], weight_decay=ANOMALY_CONFIG["weight_decay"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)  # divise lr par 2 si val stagne 4 epoques
history, best_val = [], float("inf")  # historique ; meilleure perte de validation (infini au depart)
start = time.time()  # heure de depart
for epoch in range(1, ANOMALY_CONFIG["epochs"] + 1):  # pour chaque epoque, numerotees a partir de 1
    train_loss = run_epoch(autoencoder, train_loader, optimizer)  # apprend sur le train
    val_loss = run_epoch(autoencoder, val_loader)  # mesure sur la validation
    scheduler.step(val_loss)  # le scheduler regarde la perte de validation
    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})  # garde les chiffres pour les courbes
    if val_loss < best_val:  # si la validation s'ameliore
        best_val = val_loss  # nouveau record
        torch.save(autoencoder.state_dict(), OUTPUT_DIR / "ae_best.pt")  # sauvegarde les poids du meilleur modele
    if epoch % 5 == 0 or epoch == 1:  # affiche seulement la 1re epoque et toutes les 5 epoques
        print(f"epoque {epoch:2d} | train MSE {train_loss:.5f} | val MSE {val_loss:.5f} | {time.time() - start:.0f}s")  # suivi
history = pd.DataFrame(history)  # l'historique en tableau pandas
autoencoder.load_state_dict(torch.load(OUTPUT_DIR / "ae_best.pt", weights_only=True))  # recharge les meilleurs poids
print(f"meilleure val MSE {best_val:.5f} a l'epoque {int(history['val_loss'].idxmin()) + 1}")  # idxmin : ligne du minimum (0 = epoque 1)

# %% curves_ae
fig, ax = plt.subplots(figsize=(6, 3.2))  # un graphique
ax.plot(history["epoch"], history["train_loss"], label="train")  # perte d'entrainement par epoque
ax.plot(history["epoch"], history["val_loss"], label="val (normales)")  # perte de validation par epoque
ax.set_xlabel("epoque")  # nom de l'axe horizontal
ax.set_ylabel("MSE de reconstruction")  # nom de l'axe vertical
ax.set_yscale("log")  # axe vertical logarithmique : on voit mieux les petites valeurs
ax.legend()  # legende
ax.grid(alpha=0.3)  # quadrillage leger
plt.tight_layout()  # ajuste les marges
plt.show()  # affiche


# %% show_reconstruction
# Entree | reconstruction | erreur : sur une image normale l'erreur est faible partout ;
# sur un defaut, le reseau reconstruit une bouteille normale et l'erreur se concentre sur le defaut.
@torch.no_grad()  # pas de gradient : on ne fait que predire
def reconstruct(model: nn.Module, image: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:  # renvoie (reconstruction, carte d'erreur)
    model.eval()  # mode evaluation
    recon = model(image[None].to(DEVICE))[0].cpu()  # ajoute la dimension lot, reconstruit, reprend la seule image
    error = ((image - recon) ** 2).mean(dim=0)  # moyenne sur les 3 canaux -> (S, S)
    return recon.permute(1, 2, 0).numpy(), error.numpy()  # reconstruction en (H, W, C) pour l'affichage, et carte d'erreur


fig, axes = plt.subplots(3, 4, figsize=(13, 9.5))  # grille 3 lignes x 4 colonnes
for col, row in enumerate(examples):  # pour chaque exemple (normal, puis chaque type de defaut)
    image, mask, _ = AnomalyDataset(pd.DataFrame([row]), eval_transform)[0]  # charge l'image et son masque via le Dataset
    recon, error = reconstruct(autoencoder, image)  # reconstruction et erreur
    axes[0, col].imshow(image.permute(1, 2, 0).numpy())  # ligne 1 : l'entree
    axes[0, col].set_title(row["defect_type"], fontsize=9)  # titre : le type
    axes[1, col].imshow(recon)  # ligne 2 : la reconstruction
    axes[2, col].imshow(error, cmap="inferno", vmin=0, vmax=0.05)  # ligne 3 : l'erreur (meme echelle pour toutes)
    if mask.any():  # s'il y a un vrai defaut
        axes[2, col].contour(mask[0].numpy(), levels=[0.5], colors="cyan", linewidths=0.8)  # son contour en cyan
for ax, name in zip(axes[:, 0], ("entree", "reconstruction", "erreur (contour = verite)")):  # premiere colonne de chaque ligne
    ax.set_ylabel(name)  # nom de la ligne a gauche
for ax in axes.ravel():  # pour chaque case
    ax.set_xticks([])  # pas de graduations horizontales
    ax.set_yticks([])  # pas de graduations verticales
plt.tight_layout()  # ajuste les marges
plt.show()  # affiche


# %% scoring_anomaly
# Le score d'anomalie, a UN seul endroit : erreur par pixel -> lissage gaussien (un defaut occupe une zone, le bruit
# pixel a pixel est attenue) -> score image = maximum de la carte (un seul defaut localise suffit a rejeter la piece).
# C'est un nn.Module : il sera exporte en ONNX avec l'auto-encodeur, la production n'a rien a reimplementer.
def gaussian_kernel(sigma: float) -> torch.Tensor:  # renvoie un noyau de flou gaussien 2D (poids qui somment a 1)
    radius = max(1, math.ceil(3 * sigma))  # rayon du noyau : 3 sigma couvre presque toute la cloche
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)  # positions -rayon ... +rayon
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))  # courbe en cloche (gaussienne) sur ces positions
    kernel_1d /= kernel_1d.sum()  # normalise : la somme vaut 1 (le flou ne change pas le niveau moyen)
    return torch.outer(kernel_1d, kernel_1d)[None, None]  # (1, 1, k, k)


class AnomalyScorer(nn.Module):  # le scoreur complet : auto-encodeur + carte d'erreur lissee + score image
    """Image -> (reconstruction, carte d'anomalie (N, 1, S, S), score image (N,))."""

    def __init__(self, autoencoder: nn.Module, sigma: float) -> None:  # recoit l'auto-encodeur entraine et le sigma du flou
        super().__init__()  # initialise la partie nn.Module
        self.autoencoder = autoencoder  # garde l'auto-encodeur
        self.register_buffer("kernel", gaussian_kernel(sigma))  # suit le modele (.to, state_dict) sans etre entraine

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # renvoie reconstruction, carte, score
        recon = self.autoencoder(x)  # reconstruction
        error_map = ((x - recon) ** 2).mean(dim=1, keepdim=True)  # erreur au carre par pixel, moyenne des 3 canaux : (N, 1, S, S)
        radius = self.kernel.shape[-1] // 2  # rayon du noyau
        error_map = F.conv2d(F.pad(error_map, (radius,) * 4, mode="reflect"), self.kernel)  # flou gaussien (bords en miroir, meme taille)
        return recon, error_map, error_map.flatten(1).amax(dim=1)  # score image = valeur maximale de la carte


@torch.no_grad()  # pas de gradient
def score_loader(scorer: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # note tout un loader
    """Renvoie scores (N,), cartes (N, S, S), masques (N, S, S), etiquettes (N,) pour tout un loader."""
    scorer.eval()  # mode evaluation
    scores, maps, all_masks, labels = [], [], [], []  # listes a remplir lot par lot
    for images, masks, targets in loader:  # pour chaque lot
        _, error_map, score = scorer(images.to(DEVICE))  # carte et score (la reconstruction ne sert pas ici)
        scores.append(score.cpu())  # scores image
        maps.append(error_map[:, 0].cpu())  # cartes, sans la dimension canal
        all_masks.append(masks[:, 0])  # vrais masques
        labels.append(targets)  # vraies etiquettes
    return torch.cat(scores).numpy(), torch.cat(maps).numpy(), torch.cat(all_masks).numpy(), torch.cat(labels).numpy()  # tout colle


scorer = AnomalyScorer(autoencoder, ANOMALY_CONFIG["gaussian_sigma"]).to(DEVICE)  # cree le scoreur
val_scores, val_maps, _, _ = score_loader(scorer, val_loader)  # scores et cartes des normales de validation
# Calibration SANS defaut : le seuil est un quantile des scores d'images normales = un taux de fausses alarmes choisi.
threshold = float(np.quantile(val_scores, ANOMALY_CONFIG["image_quantile"]))  # 99 % des normales de val sont sous ce seuil
pixel_threshold = float(np.quantile(val_maps, 0.999))  # seuil pixel : 99,9 % des pixels normaux sont dessous
json.dump(  # sauvegarde les seuils dans un fichier JSON
    {"image_threshold": threshold, "pixel_threshold": pixel_threshold, "n_val": len(val_scores)},  # contenu
    open(OUTPUT_DIR / "calibration.json", "w"),  # fichier de destination, ouvert en ecriture
    indent=2,  # JSON indente, lisible
)
# resume des scores des normales de validation
print(f"scores de validation (normales) : min {val_scores.min():.5f} | mediane {np.median(val_scores):.5f} | max {val_scores.max():.5f}")
print(f"seuil image (quantile {ANOMALY_CONFIG['image_quantile']}) = {threshold:.5f} | seuil pixel = {pixel_threshold:.5f}")  # seuils

# %% evaluate_anomaly
# Etape 11 - Evaluer. AUROC : independant du seuil, probabilite qu'un defaut ait un score plus eleve qu'une normale.
# Au seuil calibre : rappel (defauts attrapes), fausses alarmes (normales rejetees), F1, par type de defaut.
test_scores, test_maps, test_masks, test_labels = score_loader(scorer, test_loader)  # tout le test
auroc_image = roc_auc_score(test_labels, test_scores)  # AUROC par image
auroc_pixel = roc_auc_score(test_masks.ravel(), test_maps.ravel())  # AUROC par pixel (tous les pixels mis a plat)
decisions = (test_scores > threshold).astype(int)  # decision : 1 = defaut si le score depasse le seuil
recall = decisions[test_labels == 1].mean()  # rappel : part des defauts signales
false_alarms = decisions[test_labels == 0].mean()  # fausses alarmes : part des normales signalees
print(f"AUROC image {auroc_image:.3f} | AUROC pixel {auroc_pixel:.3f}")  # affiche les AUROC
# resultats au seuil calibre : rappel, fausses alarmes et F1
print(f"au seuil {threshold:.5f} : rappel {recall:.3f} | fausses alarmes {false_alarms:.3f} | F1 {f1_score(test_labels, decisions):.3f}")
# part des images signalees pour chaque type (pour "good" : le taux de fausses alarmes)
per_type = test_df.assign(score=test_scores, flagged=decisions).groupby("defect_type")["flagged"].mean()
print("part des images signalees par type :", per_type.round(3).to_dict())  # affiche
# resultats de l'auto-encodeur, gardes pour la comparaison avec PatchCore
ae_results = {"AUROC image": auroc_image, "AUROC pixel": auroc_pixel, "rappel au seuil": recall, "fausses alarmes": false_alarms}

fig, ax = plt.subplots(figsize=(6.5, 3.2))  # un graphique
bins = np.linspace(min(test_scores.min(), val_scores.min()), test_scores.max(), 30)  # 30 intervalles communs aux deux histogrammes
ax.hist(test_scores[test_labels == 0], bins=bins, alpha=0.6, label="test normales")  # histogramme des scores des normales
ax.hist(test_scores[test_labels == 1], bins=bins, alpha=0.6, label="test defauts")  # histogramme des scores des defauts
ax.axvline(threshold, color="k", linestyle="--", label="seuil (val)")  # ligne verticale : le seuil
ax.set_xlabel("score d'anomalie (max de la carte lissee)")  # nom de l'axe horizontal
ax.legend()  # legende
plt.tight_layout()  # ajuste les marges
plt.show()  # affiche

# %% show_maps
# Les cartes sur le test : ou l'auto-encodeur voit le defaut, et ou il se trompe.
picks = [  # position (0, 1, 2...) dans test_df de la 2e image de chaque type
    test_df.index.get_loc(test_df[test_df["defect_type"] == t].index[1])  # index[1] : 2e image du type t ; get_loc : sa position
    for t in ("good", "broken_large", "broken_small", "contamination")  # pour chaque type ; position = indice dans test_scores
]
fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))  # grille 2 x 4
for col, i in enumerate(picks):  # pour chaque image choisie
    axes[0, col].imshow(read_image(test_df.iloc[i]["path"], SIZE))  # ligne 1 : l'image
    axes[0, col].set_title(  # titre : type, score et position par rapport au seuil
        # texte du titre : type, score, et "> seuil" ou "<= seuil"
        f"{test_df.iloc[i]['defect_type']} | score {test_scores[i]:.4f} {'> seuil' if test_scores[i] > threshold else '<= seuil'}",
        fontsize=8,  # petite police
    )
    axes[1, col].imshow(test_maps[i], cmap="inferno", vmin=0, vmax=max(pixel_threshold * 2, 1e-6))  # ligne 2 : la carte d'anomalie
    if test_masks[i].any():  # s'il y a un vrai defaut
        axes[1, col].contour(test_masks[i], levels=[0.5], colors="cyan", linewidths=0.8)  # son contour en cyan
for ax in axes.ravel():  # pour chaque case
    ax.axis("off")  # cache les axes
plt.tight_layout()  # ajuste les marges
plt.show()  # affiche


# %% patchcore
# La methode qui marche en industrie : PAS d'entrainement. Un reseau pre-entraine (ResNet18 ImageNet) decrit chaque
# zone de l'image par un vecteur ; on memorise les vecteurs des images normales ; un vecteur de test loin de tous les
# vecteurs memorises = anomalie. C'est PatchCore (Roth et al., 2022), ici simplifie (sous-echantillonnage aleatoire
# de la memoire au lieu du coreset). anomalib en contient la version complete.
class PatchFeatures(nn.Module):  # extracteur : renvoie une grille de vecteurs, un par zone de l'image
    """Caracteristiques locales : layer2 et layer3 d'un ResNet18 gele, concatenees a la resolution de layer2."""

    def __init__(self) -> None:  # constructeur
        super().__init__()  # initialise la partie nn.Module
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()  # ResNet18 pre-entraine sur ImageNet
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)  # debut du reseau
        self.layer2, self.layer3 = backbone.layer2, backbone.layer3  # les deux blocs dont on prend les sorties
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))  # moyenne ImageNet par canal
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))  # ecart-type ImageNet par canal
        for p in self.parameters():  # pour chaque poids
            p.requires_grad = False  # gele : aucun apprentissage

    @torch.no_grad()  # pas de gradient
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # image -> grille de vecteurs (N, 384, S/8, S/8)
        x = (x - self.mean) / self.std  # les images du Dataset sont en [0, 1] : normalisation ImageNet ici
        f2 = self.layer2(self.stem(x))  # (N, 128, S/8, S/8)
        f3 = F.interpolate(self.layer3(f2), size=f2.shape[-2:], mode="bilinear", align_corners=False)  # (N, 256, S/8, S/8)
        features = torch.cat([f2, f3], dim=1)  # colle les deux : 128 + 256 = 384 valeurs par zone
        return F.avg_pool2d(features, 3, stride=1, padding=1)  # voisinage 3x3 : chaque vecteur decrit une zone, pas un point


extractor = PatchFeatures().to(DEVICE)  # cree l'extracteur


def patch_vectors(loader: DataLoader) -> tuple[torch.Tensor, tuple[int, int]]:  # renvoie (tous les vecteurs, taille de la grille)
    """(N * h * w, C) : un vecteur par zone, pour toutes les images du loader."""
    chunks, grid = [], None  # morceaux a coller ; taille de la grille (connue apres le 1er lot)
    for images, _, _ in loader:  # pour chaque lot
        features = extractor(images.to(DEVICE))  # grille de vecteurs (N, C, h, w)
        grid = features.shape[-2:]  # (h, w)
        chunks.append(features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).cpu())  # -> (N * h * w, C) : une ligne par zone
    return torch.cat(chunks), tuple(grid)  # tous les vecteurs et la taille de grille


start = time.time()  # heure de depart
bank, grid = patch_vectors(make_loader(train_df, eval_transform, shuffle=False))  # sans augmentation : memoire deterministe
keep = torch.randperm(len(bank), generator=torch.Generator().manual_seed(0))[: len(bank) // 10]  # memoire : 10 % des vecteurs
memory_bank = bank[keep]  # la memoire : 10 % des vecteurs normaux, tires au hasard
# taille de la grille, nombre de vecteurs, taille d'un vecteur, taille de la memoire, duree
print(f"grille {grid} | {len(bank):,} vecteurs de {bank.shape[1]} valeurs -> memoire {len(memory_bank):,} | {time.time() - start:.0f}s")


def patchcore_scores(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:  # renvoie (scores image, cartes pixel)
    """Score pixel = distance au plus proche vecteur memorise, remontee a la taille de l'image et lissee ; score image = max."""
    scores, maps = [], []  # listes a remplir lot par lot
    kernel = gaussian_kernel(ANOMALY_CONFIG["gaussian_sigma"])  # noyau de flou gaussien
    for images, _, _ in loader:  # pour chaque lot
        # vecteurs de chaque image : (N, h * w, C), ramenes sur CPU
        features = extractor(images.to(DEVICE)).permute(0, 2, 3, 1).reshape(len(images), -1, bank.shape[1]).cpu()
        distances = torch.stack([torch.cdist(f, memory_bank).amin(dim=1) for f in features])  # (N, h*w)
        patch_map = distances.view(len(images), 1, *grid)  # remet les distances en grille (N, 1, h, w)
        pixel_map = F.interpolate(patch_map, size=(SIZE, SIZE), mode="bilinear", align_corners=False)  # agrandit a la taille de l'image
        radius = kernel.shape[-1] // 2  # rayon du noyau
        pixel_map = F.conv2d(F.pad(pixel_map, (radius,) * 4, mode="reflect"), kernel)  # flou gaussien
        maps.append(pixel_map[:, 0])  # garde les cartes (sans dimension canal)
        scores.append(distances.amax(dim=1))  # score image = distance de la zone la plus eloignee
    return torch.cat(scores).numpy(), torch.cat(maps).numpy()  # tout colle


start = time.time()  # heure de depart
pc_val_scores, _ = patchcore_scores(val_loader)  # scores des normales de validation (pour le seuil)
pc_test_scores, pc_test_maps = patchcore_scores(test_loader)  # scores et cartes du test
pc_threshold = float(np.quantile(pc_val_scores, ANOMALY_CONFIG["image_quantile"]))  # seuil calibre sur les normales, comme plus haut
pc_decisions = (pc_test_scores > pc_threshold).astype(int)  # decisions au seuil
pc_results = {  # memes metriques que pour l'auto-encodeur
    "AUROC image": roc_auc_score(test_labels, pc_test_scores),  # AUROC par image
    "AUROC pixel": roc_auc_score(test_masks.ravel(), pc_test_maps.ravel()),  # AUROC par pixel
    "rappel au seuil": pc_decisions[test_labels == 1].mean(),  # part des defauts signales
    "fausses alarmes": pc_decisions[test_labels == 0].mean(),  # part des normales signalees
}
# duree du scoring puis chaque metrique "nom valeur", separees par " | "
print(f"PatchCore simplifie : {time.time() - start:.0f}s de scoring | " + " | ".join(f"{k} {v:.3f}" for k, v in pc_results.items()))

# %% compare_methods
# tableau des resultats des deux methodes ; .T (transpose) : une ligne par methode
comparison = pd.DataFrame({"auto-encodeur (30 epoques)": ae_results, "PatchCore simplifie (0 epoque)": pc_results}).T
print(comparison.round(3).to_string())  # affiche le tableau
comparison.to_json(OUTPUT_DIR / "results.json", indent=2)  # sauvegarde la comparaison

# Une echelle de couleurs COMMUNE par methode (vmax = le max des quatre cartes) : avec une echelle par image,
# la carte d'une piece normale parait aussi "chaude" que celle d'un defaut.
fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))  # grille 2 x 4 : une ligne par methode
for col, i in enumerate(picks):  # pour chaque image choisie
    axes[0, col].imshow(test_maps[i], cmap="inferno", vmin=0, vmax=test_maps[picks].max())  # carte de l'auto-encodeur
    axes[0, col].set_title(f"AE : {test_df.iloc[i]['defect_type']}", fontsize=9)  # titre
    axes[1, col].imshow(pc_test_maps[i], cmap="inferno", vmin=0, vmax=pc_test_maps[picks].max())  # carte de PatchCore
    axes[1, col].set_title("PatchCore", fontsize=9)  # titre
    for r in (0, 1):  # pour les deux lignes
        if test_masks[i].any():  # s'il y a un vrai defaut
            axes[r, col].contour(test_masks[i], levels=[0.5], colors="cyan", linewidths=0.8)  # son contour en cyan
        axes[r, col].axis("off")  # cache les axes
plt.tight_layout()  # ajuste les marges
plt.show()  # affiche

# %% onnx_anomaly
# Etape 12 - Sauvegarder et exporter : on exporte le scoreur COMPLET (auto-encodeur + lissage + max). Trois sorties nommees.
# La production recoit directement la carte et le score, calcules par les memes operations qu'ici.
scorer.eval().cpu()  # mode evaluation, sur CPU pour l'export
onnx_path = OUTPUT_DIR / "anomaly_scorer.onnx"  # fichier ONNX a creer
torch.onnx.export(  # exporte le scoreur au format ONNX
    scorer,  # le modele a exporter
    (torch.zeros(2, 3, SIZE, SIZE),),  # exemple d'entree (2 images) : sert a tracer les calculs
    str(onnx_path),  # chemin du fichier
    input_names=["input"],  # nom donne a l'entree
    output_names=["reconstruction", "anomaly_map", "score"],  # noms des trois sorties
    dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=256)}},  # la dimension 0 (lot) peut varier de 1 a 256
    dynamo=True,  # utilise le nouvel exporteur de PyTorch
    external_data=False,  # tout dans un seul fichier (poids compris)
)
session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # charge le modele ONNX pour l'executer sur CPU
images, _, _ = next(iter(test_loader))  # un lot de test
with torch.no_grad():  # sans gradient
    _, torch_map, torch_score = scorer(images)  # resultats PyTorch
onnx_recon, onnx_map, onnx_score = session.run(None, {"input": images.numpy()})  # resultats ONNX (None : toutes les sorties)
# nom et forme de chaque sortie ; une dimension variable (non entiere) est affichee "batch"
print("sorties ONNX :", [(o.name, [d if isinstance(d, int) else "batch" for d in o.shape]) for o in session.get_outputs()])
# ecart maximal entre ONNX et PyTorch, pour le score et pour la carte : doit etre tres petit
print(f"ecart max score {np.abs(onnx_score - torch_score.numpy()).max():.2e} | carte {np.abs(onnx_map - torch_map.numpy()).max():.2e}")
assert np.allclose(onnx_score, torch_score.numpy(), atol=1e-4)  # arrete tout si les scores different
json.dump(  # ecrit la fiche du modele : ce qu'il faut pour l'utiliser ailleurs
    {  # contenu de la fiche
        "input": "input",  # nom de l'entree
        "shape": ["batch", 3, SIZE, SIZE],  # forme de l'entree
        "preprocessing": "RGB, resize bilineaire, /255, pas de normalisation",  # pretraitement a refaire avant
        "image_threshold": threshold,  # seuil de decision par image
        "pixel_threshold": pixel_threshold,  # seuil par pixel
    },
    open(OUTPUT_DIR / "model_card.json", "w"),  # fichier de destination
    indent=2,  # JSON indente
)

# Latence : ce qui compte en production, c'est le temps par image, lot de 1.
one = images[:1].numpy()  # une seule image
for _ in range(3):  # 3 executions de chauffe (les premieres sont plus lentes)
    session.run(None, {"input": one})  # execution non mesuree
start = time.time()  # heure de depart
for _ in range(20):  # 20 executions mesurees
    session.run(None, {"input": one})  # execution ONNX
onnx_ms = (time.time() - start) / 20 * 1000  # temps moyen en millisecondes
start = time.time()  # nouveau depart pour PyTorch
with torch.no_grad():  # sans gradient
    for _ in range(20):  # 20 executions
        scorer(images[:1])  # execution PyTorch
torch_ms = (time.time() - start) / 20 * 1000  # temps moyen en millisecondes
print(f"latence lot de 1 sur CPU : PyTorch {torch_ms:.0f} ms | onnxruntime {onnx_ms:.0f} ms")  # comparaison des vitesses


# %% embed_explore
# Etape 2 - Importer et explorer, sans etiquettes : un vecteur par image (ResNet18 ImageNet, sortie du pooling global),
# puis k-means. On utilise NEU-CLS en oubliant ses etiquettes, et on les ressort seulement pour juger les groupes.
# En pratique, c'est le moment du "pre-etiquetage" : on etiquette un groupe d'un coup, puis on corrige les exceptions.
# catalogue NEU-CLS : chemin de chaque .bmp et sa classe (nom du dossier parent)
neu_catalog = pd.DataFrame([{"path": p.as_posix(), "label": p.parent.name} for p in sorted(Path("data/raw/neu_cls_flat").rglob("*.bmp"))])
embedder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval().to(DEVICE)  # ResNet18 ImageNet, mode evaluation
embedder.fc = nn.Identity()  # on garde le vecteur de 512 valeurs, pas les 1000 classes ImageNet
embed_transform = v2.Compose(  # pretraitement des images
    [  # liste des etapes
        v2.ToImage(),  # image PIL -> tenseur image
        v2.Resize((128, 128), antialias=True),  # 128 x 128
        v2.ToDtype(torch.float32, scale=True),  # valeurs 0-1
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # normalisation ImageNet
    ]
)


class PathDataset(Dataset):  # jeu de donnees minimal : une liste de chemins -> images pretraitees (sans etiquette)
    def __init__(self, paths: list[str]) -> None:  # recoit la liste des chemins
        self.paths = paths  # la garde

    def __len__(self) -> int:  # nombre d'images
        return len(self.paths)  # une par chemin

    def __getitem__(self, i: int) -> torch.Tensor:  # renvoie l'image numero i pretraitee
        return embed_transform(Image.open(self.paths[i]).convert("RGB"))  # ouvre, RGB, pretraite


start = time.time()  # heure de depart
with torch.no_grad():  # sans gradient
    embeddings = torch.cat(  # embedding = vecteur qui resume une image ; on colle ceux de tous les lots
        [embedder(batch.to(DEVICE)).cpu() for batch in DataLoader(PathDataset(neu_catalog["path"].tolist()), batch_size=64)]  # lots de 64
    ).numpy()  # tenseur -> tableau numpy (N images, 512 valeurs)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)  # normalisation L2 : la distance devient un cosinus
print(f"{embeddings.shape[0]} images -> vecteurs de {embeddings.shape[1]} valeurs en {time.time() - start:.0f}s")  # resume

kmeans = KMeans(n_clusters=6, n_init=10, random_state=ANOMALY_CONFIG["seed"]).fit(embeddings)  # 6 groupes, 10 essais, garde le meilleur
neu_catalog["cluster"] = kmeans.labels_  # numero de groupe de chaque image
table = pd.crosstab(neu_catalog["cluster"], neu_catalog["label"])  # tableau croise groupes x vraies classes
purity = table.max(axis=1).sum() / len(neu_catalog)  # purete : pour chaque groupe, sa classe la plus frequente ; part totale
print(table)  # affiche le tableau croise
print(f"purete des groupes (part des images dans la classe majoritaire de leur groupe) : {purity:.3f}")  # 1 = groupes parfaits

coords = PCA(n_components=2, random_state=0).fit_transform(embeddings)  # 512 valeurs -> 2 coordonnees pour dessiner
fig, axes = plt.subplots(1, 2, figsize=(11, 4))  # deux nuages de points cote a cote
for ax, column, title in zip(axes, ("cluster", "label"), ("groupes k-means (sans etiquette)", "vraies classes (pour juger)")):  # 2 vues
    for value in sorted(neu_catalog[column].unique()):  # pour chaque groupe (ou chaque classe)
        sel = neu_catalog[column] == value  # images de ce groupe
        ax.scatter(coords[sel, 0], coords[sel, 1], s=6, label=str(value))  # leurs points, une couleur par groupe
    ax.set_title(title, fontsize=10)  # titre
    ax.legend(fontsize=7, markerscale=2)  # legende
    ax.set_xticks([])  # pas de graduations horizontales
    ax.set_yticks([])  # pas de graduations verticales
plt.tight_layout()  # ajuste les marges
plt.show()  # affiche


# %% gradcam
# Etape 11 - Evaluer : ou regarde un classifieur ? Grad-CAM sur le ResNet18 entraine par scripts/train.py (configs/resnet18.yaml) :
# les cartes de la derniere couche de convolution, ponderees par le gradient de la classe predite.
# Deux hooks : un pour garder les activations, un pour garder leur gradient. Rien d'autre.
saved = torch.load("outputs/resnet18/model.pt", map_location="cpu", weights_only=True)  # le modele sauve par scripts/train.py
clf = models.resnet18(weights=None)  # ResNet18 vide (les poids viennent du fichier)
clf.fc = nn.Linear(clf.fc.in_features, len(saved["class_names"]))  # tete adaptee au nombre de classes
clf.load_state_dict(saved["model_state"])  # charge les poids appris
clf.eval()  # mode evaluation
clf_transform = v2.Compose(  # meme pretraitement qu'a l'entrainement du classifieur
    [  # liste des etapes
        v2.ToImage(),  # image PIL -> tenseur image
        v2.Resize((saved["image_size"],) * 2, antialias=True),  # taille d'entrainement
        v2.ToDtype(torch.float32, scale=True),  # valeurs 0-1
        v2.Normalize(saved["mean"], saved["std"]),  # normalisation sauvegardee avec le modele
    ]
)
store: dict[str, torch.Tensor] = {}  # boite ou les hooks rangent activations et gradients


# hook (fonction appelee automatiquement) execute a chaque passage avant dans layer4
def keep_activations(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:  # range la sortie de layer4 ; ne renvoie rien
    store["activations"] = output  # garde les cartes de sortie de layer4
    output.register_hook(lambda grad: store.__setitem__("gradients", grad))  # appele au backward, avec d(score)/d(output)


clf.layer4.register_forward_hook(keep_activations)  # branche le hook sur le dernier bloc de convolution


def grad_cam(image: torch.Tensor) -> tuple[np.ndarray, int]:  # calcule la carte Grad-CAM d'une image
    """Carte (H, W) en [0, 1] pour la classe predite, et l'indice de cette classe."""
    clf.zero_grad()  # efface les gradients d'un appel precedent
    logits = clf(image[None])  # prediction (ajoute la dimension lot) ; le hook garde les activations
    predicted = int(logits.argmax())  # classe predite
    logits[0, predicted].backward()  # d(score de la classe) / d(activations de layer4)
    weights = store["gradients"].mean(dim=(2, 3), keepdim=True)  # un poids par canal : la moyenne de son gradient
    cam = F.relu((weights * store["activations"]).sum(dim=1))[0]  # (4, 4) pour une entree 128
    cam = F.interpolate(cam[None, None], size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]  # agrandit a la taille image
    return ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).detach().numpy(), predicted  # carte ramenee entre 0 et 1, classe


test_split = pd.read_csv("outputs/split.csv").query("split == 'test'")  # images de test du decoupage de scripts/train.py
fig, axes = plt.subplots(2, 6, figsize=(15, 5.2))  # grille 2 x 6 : une colonne par classe
for col, name in enumerate(saved["class_names"]):  # pour chaque classe
    path = test_split[test_split["label"] == name].iloc[0]["path"]  # premiere image de test de cette classe
    pil = Image.open(path).convert("RGB")  # ouvre l'image
    cam, predicted = grad_cam(clf_transform(pil))  # carte Grad-CAM et classe predite
    axes[0, col].imshow(pil, cmap="gray")  # ligne 1 : l'image
    axes[0, col].set_title(f"{name} -> {saved['class_names'][predicted]}", fontsize=8)  # vraie classe -> classe predite
    axes[1, col].imshow(pil.resize((saved["image_size"],) * 2))  # ligne 2 : l'image a la taille du modele...
    axes[1, col].imshow(cam, cmap="jet", alpha=0.45)  # ...avec la carte par-dessus (rouge = zone qui a decide)
for ax in axes.ravel():  # pour chaque case
    ax.axis("off")  # cache les axes
plt.tight_layout()  # ajuste les marges
plt.show()  # affiche
