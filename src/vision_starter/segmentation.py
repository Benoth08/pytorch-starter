"""Segmentation binaire : catalogue image + masque, Dataset, U-Net sur ResNet18, perte Dice + BCE, IoU.

Version fichiers du notebook 06 (chapitre Segmentation du cours). Par rapport a la classification,
seules quatre briques changent : le Dataset (image ET masque), les
transformations (identiques pour les deux), le modele (encodeur + decodeur) et
la perte / la metrique (par pixel). Tout le reste est le meme code.
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import logging  # logging : messages horodates du programme
import time  # time : mesure de la duree de chaque epoque
from pathlib import Path  # Path : represente un chemin de fichier

import numpy as np  # numpy : calcul sur des tableaux de nombres (un masque = un tableau de 0 et de 1)
import pandas as pd  # pandas : tableaux de donnees (le catalogue image + masque)
import torch  # PyTorch : tenseurs (tableaux de nombres sur CPU ou GPU) et calcul des gradients
import torch.nn as nn  # nn : briques de reseaux de neurones (couches, pertes)
import torch.nn.functional as F  # F : versions "fonction" des operations (ici interpolate pour agrandir une carte)
from PIL import Image  # Pillow : ouvre les fichiers image
from torch.utils.data import DataLoader, Dataset  # Dataset : un exemple par indice ; DataLoader : les regroupe en lots

# models : architectures connues (ResNet18) ; tv_tensors : types qui disent aux transformations "ceci est un masque"
from torchvision import models, tv_tensors
from torchvision.transforms import v2  # transformations d'images de torchvision

from vision_starter.config import SegmentationConfig  # la fiche de reglages de la segmentation

# statistiques d'ImageNet (normalisation) et decoupage stratifie deja ecrits pour la classification
from vision_starter.data import IMAGENET_MEAN, IMAGENET_STD, make_stratified_grouped_split, near_duplicate_groups

logger = logging.getLogger(__name__)  # journal propre a ce module


# --- Catalogue et decoupage (etapes 3 et 4) ---


# lit un masque PNG et le rend binaire ; renvoie un tableau de 0 (fond) et 1 (defaut)
def read_mask(path: str | Path, size: int | None = None) -> np.ndarray:
    """Masque 0/1. Seuil a 127 (bords anti-creneles du PNG) ; NEAREST pour rester binaire."""
    mask = Image.open(path).convert("L")  # ouvre le masque en niveaux de gris ("L" = un seul canal 0-255)
    if size is not None:  # si une taille est demandee
        mask = mask.resize((size, size), Image.NEAREST)  # NEAREST recopie le pixel le plus proche : pas de valeurs intermediaires
    return (np.array(mask) > 127).astype(np.uint8)  # pixel > 127 -> 1 (defaut), sinon 0 ; entiers sur 8 bits


# liste les images Magnetic Tile et leurs masques ; renvoie le catalogue (path, mask_path, label)
def catalog_magnetic_tile(data_dir: Path, n_free: int, seed: int) -> pd.DataFrame:
    """MT_<classe>/Imgs/x.jpg + x.png. Les images 'free' (sans defaut) sont sous-echantillonnees a n_free."""
    rows = []  # une ligne (dictionnaire) par image
    for class_dir in sorted(data_dir.glob("MT_*")):  # pour chaque dossier de classe (MT_Blowhole, MT_Crack, ...)
        label = class_dir.name.removeprefix("MT_").lower()  # "MT_Blowhole" -> "blowhole"
        for image_path in sorted((class_dir / "Imgs").glob("*.jpg")):  # pour chaque image de cette classe
            mask_path = image_path.with_suffix(".png")  # le masque a le meme nom avec l'extension .png
            if not mask_path.is_file():  # si le masque n'existe pas
                raise FileNotFoundError(f"masque manquant : {mask_path}")  # on s'arrete : image inutilisable sans masque
            rows.append({"path": image_path.as_posix(), "mask_path": mask_path.as_posix(), "label": label})  # chemins avec des /
    if not rows:  # si aucune image n'a ete trouvee
        raise FileNotFoundError(f"aucune image trouvee sous {data_dir}")  # erreur claire : mauvais dossier
    catalog = pd.DataFrame(rows)  # liste -> tableau pandas
    free = catalog[catalog["label"] == "free"]  # images sans defaut
    if len(free) > n_free:  # s'il y en a plus que voulu
        free = free.sample(n=n_free, random_state=seed)  # on en tire n_free au hasard (graine fixe = meme tirage a chaque fois)
    # reassemble images avec defaut + "free" gardees, trie par chemin et renumerote les lignes 0..n-1
    return pd.concat([catalog[catalog["label"] != "free"], free]).sort_values("path").reset_index(drop=True)


# coupe le catalogue en train / val / test ; renvoie le catalogue avec les colonnes group et split
def split_segmentation(catalog: pd.DataFrame, config: SegmentationConfig) -> pd.DataFrame:
    """Decoupage par groupes de quasi-doublons, equilibre par la classe de l'image (free compris) : l'etape 4, variante groupee.

    Ce jeu contient des images presque identiques (la meme tuile, recadree ou eclairee autrement). Separees entre train et
    test, elles feraient mesurer le modele sur des images qu'il a deja vues : on les garde ensemble.
    """
    catalog = catalog.copy()  # copie : on ne modifie pas le tableau recu
    # numero de groupe de chaque image : les quasi-doublons portent le meme
    catalog["group"] = near_duplicate_groups(catalog["path"].tolist(), config.duplicate_max_hamming, config.duplicate_min_correlation)
    return make_stratified_grouped_split(catalog, "group", config.val_fraction, config.test_fraction, config.seed)  # sans couper de groupe


# --- Dataset, transformations, DataLoader (etapes 5 a 7) ---


class SegmentationDataset(Dataset):  # jeu de donnees PyTorch : dataset[i] renvoie (image, masque)
    """(image (3, S, S) normalisee, masque (1, S, S) en 0/1). tv_tensors.Mask : meme geometrie, pas de normalisation sur le masque."""

    def __init__(self, frame: pd.DataFrame, transform) -> None:  # memorise les chemins et la transformation
        self.paths = frame["path"].tolist()  # chemins des images
        self.mask_paths = frame["mask_path"].tolist()  # chemins des masques, dans le meme ordre
        self.transform = transform  # transformation commune image + masque

    def __len__(self) -> int:  # nombre d'exemples
        return len(self.paths)  # une image par chemin

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:  # renvoie l'exemple numero index : (image, masque)
        image = Image.open(self.paths[index]).convert("RGB")  # gris -> 3 canaux : l'encodeur ImageNet attend du RGB
        # masque 0/1 ; [None] ajoute une dimension canal -> (1, H, W) ; Mask = redimensionne sans lissage, jamais normalise
        mask = tv_tensors.Mask(read_mask(self.mask_paths[index])[None])
        image, mask = self.transform(image, mask)  # MEME transformation (meme retournement) sur l'image et son masque
        return image, mask.float()  # masque en reels : la perte BCE attend des nombres a virgule


# construit les transformations ; renvoie (entrainement avec retournements, evaluation sans)
def build_segmentation_transforms(image_size: int) -> tuple[v2.Compose, v2.Compose]:
    common = [v2.ToImage(), v2.Resize((image_size, image_size), antialias=True)]  # debut commun : tenseur puis taille fixe
    # fin commune : entiers 0-255 -> reels 0-1, puis normalisation (pixel - moyenne) / ecart-type d'ImageNet
    tail = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    # entrainement : retournements aleatoires (augmentation) au milieu ; evaluation : sans hasard
    return v2.Compose(common + [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()] + tail), v2.Compose(common + tail)


# cree le DataLoader qui fournit les couples (images, masques) par lots ; renvoie ce DataLoader
def build_segmentation_loader(frame: pd.DataFrame, transform, config: SegmentationConfig, train: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)  # generateur aleatoire a graine fixe : melange reproductible
    return DataLoader(  # objet a parcourir avec for : chaque tour donne un lot
        SegmentationDataset(frame, transform),  # ou lire les exemples
        batch_size=config.batch_size,  # nombre d'images par lot (batch)
        shuffle=train,  # melange seulement pour l'entrainement
        drop_last=train,  # entrainement : jette le dernier lot incomplet (stabilite de la BatchNorm)
        num_workers=config.num_workers,  # processus de chargement en parallele
        generator=generator,  # generateur du melange
    )


# --- Modele (etape 8) ---


class UpBlock(nn.Module):  # bloc du decodeur : agrandit la carte et la combine avec les details de l'encodeur
    """x2 en resolution, concatenation avec le pont (skip), deux convolutions 3x3."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:  # cree les couches du bloc
        super().__init__()  # initialise la partie nn.Module
        self.conv = nn.Sequential(  # deux convolutions a la suite
            # 1re convolution 3x3 : prend les canaux du bas + ceux du pont, sort out_channels
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # BatchNorm : recentre chaque canal sur le lot, apprentissage plus stable
            nn.ReLU(inplace=True),  # ReLU : met a 0 les valeurs negatives (non-linearite)
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),  # 2e convolution 3x3, memes canaux
            nn.BatchNorm2d(out_channels),  # normalisation
            nn.ReLU(inplace=True),  # non-linearite
        )

    # calcul du bloc : x = sortie du bloc d'en dessous, skip = carte de l'encodeur de meme taille (ou None)
    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # double hauteur et largeur (interpolation douce)
        if skip is not None:  # s'il y a un pont a cet etage
            x = torch.cat([x, skip], dim=1)  # colle les canaux du pont a ceux de x (dim=1 = dimension des canaux)
        return self.conv(x)  # deux convolutions pour melanger les deux sources


class ResNetUNet(nn.Module):  # U-Net : encodeur qui reduit l'image, decodeur qui la reagrandit en une carte par pixel
    """U-Net : encodeur ResNet18 (pre-entraine) + decodeur avec ponts. Image (N, 3, S, S) -> logits (N, 1, S, S)."""

    def __init__(self, pretrained: bool = True) -> None:  # cree l'encodeur (repris de ResNet18) et le decodeur (neuf)
        super().__init__()  # initialise la partie nn.Module
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)  # ResNet18, poids ImageNet ou hasard
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 canaux, S/2
        self.pool = resnet.maxpool  # max-pooling du ResNet : taille encore divisee par 2 (S/4)
        # les 4 etages du ResNet : 64 canaux S/4, 128 canaux S/8, 256 canaux S/16, 512 canaux S/32
        self.enc1, self.enc2, self.enc3, self.enc4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.up4 = UpBlock(512, 256, 256)  # S/32 -> S/16, pont avec enc3 (256 canaux)
        self.up3 = UpBlock(256, 128, 128)  # S/16 -> S/8, pont avec enc2 (128 canaux)
        self.up2 = UpBlock(128, 64, 64)  # S/8 -> S/4, pont avec enc1 (64 canaux)
        self.up1 = UpBlock(64, 64, 32)  # S/4 -> S/2, pont avec enc0 (64 canaux)
        self.up0 = UpBlock(32, 0, 16)  # S/2 -> S, pas de pont (aucune carte de l'encodeur a pleine taille)
        self.head = nn.Conv2d(16, 1, 1)  # conv 1x1 : un seul logit par pixel (score "defaut" avant sigmoide)

    def encoder_parameters(self):  # renvoie la liste des poids de l'encodeur (noms commencant par "enc")
        return [p for name, p in self.named_parameters() if name.startswith("enc")]  # pour leur donner un petit pas d'apprentissage

    def decoder_parameters(self):  # renvoie la liste des poids du decodeur et de la tete (tout le reste)
        return [p for name, p in self.named_parameters() if not name.startswith("enc")]  # couches neuves : pas plus grand

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # calcul complet : images -> carte de logits de meme taille
        e0 = self.enc0(x)  # S/2, 64 canaux
        e1 = self.enc1(self.pool(e0))  # S/4, 64 canaux
        e2 = self.enc2(e1)  # S/8, 128 canaux
        e3 = self.enc3(e2)  # S/16, 256 canaux
        e4 = self.enc4(e3)  # S/32, 512 canaux : le plus abstrait, le plus petit
        d = self.up4(e4, e3)  # remonte a S/16 en recuperant les details de e3
        d = self.up3(d, e2)  # remonte a S/8
        d = self.up2(d, e1)  # remonte a S/4
        d = self.up1(d, e0)  # remonte a S/2
        return self.head(self.up0(d, None))  # remonte a S puis un logit par pixel : (N, 1, S, S)


# --- Perte et metriques (etapes 9 et 11) ---


# perte Dice : 1 - recouvrement entre prediction et masque ; renvoie un nombre (tenseur a 0 dimension)
def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """1 - Dice, calcule sur les probabilites : ne regarde que les pixels du defaut, pas le fond."""
    probs = torch.sigmoid(logits)  # sigmoide : transforme chaque logit en probabilite entre 0 et 1
    intersection = (probs * targets).sum(dim=(1, 2, 3))  # "pixels communs" par image (somme sur canal, hauteur, largeur)
    total = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))  # taille predite + taille vraie, par image
    # Dice = 2 x commun / total (1 = parfait) ; smooth evite 0/0 sur une image sans defaut ; moyenne sur le lot
    return (1 - (2 * intersection + smooth) / (total + smooth)).mean()


class DiceBCELoss(nn.Module):  # perte combinee utilisee pour entrainer le U-Net
    """BCE par pixel + Dice : la BCE pour des gradients stables, le Dice pour les defauts minuscules."""

    def __init__(self) -> None:  # cree la partie BCE
        super().__init__()  # initialise la partie nn.Module
        self.bce = nn.BCEWithLogitsLoss()  # BCE (entropie croisee binaire) pixel par pixel, sigmoide incluse

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # renvoie la perte totale
        return self.bce(logits, targets) + dice_loss(logits, targets)  # somme des deux pertes


# compte les pixels utiles a l'IoU et au Dice ; renvoie (pixels communs, pixels predits defaut, pixels vraiment defaut)
def confusion_pixels(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> tuple[int, int, int]:
    """(intersection, pixels predits, pixels vrais) : a cumuler sur tout un jeu avant de calculer IoU et Dice."""
    preds = (torch.sigmoid(logits) > threshold).float()  # pixel predit "defaut" si sa probabilite depasse le seuil
    return int((preds * targets).sum()), int(preds.sum()), int(targets.sum())  # trois comptes entiers


# calcule l'IoU et le Dice a partir des comptes ; renvoie (IoU, Dice), chacun entre 0 et 1
def iou_dice(intersection: int, predicted: int, truth: int) -> tuple[float, float]:
    union = predicted + truth - intersection  # union = predits + vrais - communs (comptes une seule fois)
    # IoU = commun / union ; Dice = 2 x commun / (predits + vrais) ; max(..., 1) evite une division par zero
    return intersection / max(union, 1), 2 * intersection / max(predicted + truth, 1)


# --- Entrainement et evaluation (etapes 10 et 11) ---


# entraine le U-Net une epoque ; renvoie la perte moyenne par image
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> float:
    model.train()  # mode entrainement : BatchNorm mise a jour avec le lot courant
    total, seen = 0.0, 0  # cumul des pertes et nombre d'images vues
    for images, masks in loader:  # pour chaque lot (images, masques)
        images, masks = images.to(device), masks.to(device)  # envoie le lot sur le device du modele (CPU ou GPU)
        optimizer.zero_grad()  # remet a zero les gradients du lot precedent
        loss = criterion(model(images), masks)  # prediction puis perte (erreur) par rapport aux masques
        loss.backward()  # retropropagation : calcule le gradient de la perte pour chaque poids
        optimizer.step()  # l'optimiseur corrige les poids d'un petit pas
        total += loss.item() * len(images)  # perte du lot x taille du lot (pour une moyenne juste par image)
        seen += len(images)  # images traitees jusqu'ici
    return total / seen  # perte moyenne de l'epoque


@torch.no_grad()  # pas de gradient pendant l'evaluation (plus rapide, moins de memoire)
# evalue le modele ; renvoie un dictionnaire {"loss", "iou", "dice"}
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device, threshold: float = 0.5) -> dict[str, float]:
    """Perte moyenne, IoU et Dice cumules sur tout le loader."""
    model.eval()  # mode evaluation : BatchNorm utilise ses statistiques apprises
    total, seen, inter, pred, truth = 0.0, 0, 0, 0, 0  # cumuls : perte, images, pixels communs, predits, vrais
    for images, masks in loader:  # pour chaque lot
        images, masks = images.to(device), masks.to(device)  # envoie le lot sur le device du modele
        logits = model(images)  # carte de logits (un par pixel)
        total += criterion(logits, masks).item() * len(images)  # perte du lot ponderee par sa taille
        seen += len(images)  # images traitees
        i, p, t = confusion_pixels(logits, masks, threshold)  # comptes de pixels de ce lot
        inter, pred, truth = inter + i, pred + p, truth + t  # cumul sur tout le jeu (plus juste qu'une moyenne de moyennes)
    iou, dice = iou_dice(inter, pred, truth)  # metriques globales
    return {"loss": total / seen, "iou": iou, "dice": dice}  # resultats dans un dictionnaire


# boucle d'entrainement complete du U-Net ; renvoie l'historique (une ligne par epoque)
def fit_segmentation(
    model: nn.Module,  # le U-Net a entrainer
    train_loader: DataLoader,  # lots d'entrainement
    val_loader: DataLoader,  # lots de validation
    config: SegmentationConfig,  # reglages (pas d'apprentissage, epoques, patience...)
    device: torch.device,  # CPU ou GPU
    checkpoint_path: Path,  # fichier du meilleur modele
) -> pd.DataFrame:  # renvoie l'historique sous forme de tableau pandas
    """Deux taux d'apprentissage (decodeur neuf, encodeur pre-entraine), meilleur modele selon la perte de validation, arret anticipe."""
    criterion = DiceBCELoss()  # la perte BCE + Dice
    optimizer = torch.optim.AdamW(  # AdamW : optimiseur (corrige les poids a partir des gradients) avec weight decay
        [  # deux groupes de poids, chacun avec son pas d'apprentissage
            {"params": model.decoder_parameters(), "lr": config.learning_rate},  # decodeur neuf : pas normal
            {"params": model.encoder_parameters(), "lr": config.encoder_learning_rate},  # encodeur pre-entraine : pas plus petit
        ],
        weight_decay=config.weight_decay,  # penalite sur les grands poids
    )
    # divise le pas d'apprentissage par 2 si la perte de validation ne baisse plus pendant 2 epoques
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    history, best_val, without_improvement = [], float("inf"), 0  # historique, meilleure perte (infini au depart), compteur
    for epoch in range(1, config.epochs + 1):  # pour chaque epoque, numerotee a partir de 1
        start = time.time()  # heure de debut
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)  # apprend sur le train
        val = evaluate(model, val_loader, criterion, device)  # mesure sur la validation
        scheduler.step(val["loss"])  # le scheduler regarde la perte de validation
        # ajoute une ligne a l'historique : epoque, pertes, IoU et Dice de validation
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"], "val_iou": val["iou"], "val_dice": val["dice"]})
        improved = val["loss"] < best_val  # vrai si meilleure validation jusqu'ici
        if improved:  # si le modele s'est ameliore
            best_val, without_improvement = val["loss"], 0  # nouveau record, compteur remis a zero
            # sauvegarde les poids du meilleur modele avec l'epoque et la perte
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val["loss"]}, checkpoint_path)
        else:  # sinon
            without_improvement += 1  # une epoque de plus sans progres
        logger.info(  # bilan de l'epoque ; chaque %... est remplace par la valeur correspondante ci-dessous
            "epoque %2d | train perte %.4f | val perte %.4f IoU %.3f Dice %.3f | %.0fs%s",  # modele du message
            epoch,  # numero d'epoque
            train_loss,  # perte d'entrainement
            val["loss"],  # perte de validation
            val["iou"],  # IoU de validation
            val["dice"],  # Dice de validation
            time.time() - start,  # duree de l'epoque en secondes
            " | meilleur" if improved else "",  # mention si meilleur modele
        )
        if without_improvement >= config.patience:  # si pas de progres depuis "patience" epoques
            logger.info("arret anticipe : %d epoques sans amelioration", config.patience)  # on le signale
            break  # et on arrete l'entrainement
    return pd.DataFrame(history)  # historique sous forme de tableau


@torch.no_grad()  # pas de gradient : on ne fait que predire
# rassemble tous les logits et masques d'un loader ; renvoie (logits, masques)
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Tous les logits et masques d'un loader, sur CPU (pour choisir un seuil ou tracer des figures)."""
    model.eval()  # mode evaluation
    # pour chaque lot : (logits ramenes sur CPU, masques) ; zip(*...) separe ensuite la liste en deux groupes
    logits, masks = zip(*[(model(images.to(device)).cpu(), m) for images, m in loader])
    return torch.cat(logits), torch.cat(masks)  # colle les lots : un seul tenseur de logits, un seul de masques


# essaie plusieurs seuils et garde le meilleur ; renvoie (meilleur seuil, {seuil: IoU})
def choose_threshold(logits: torch.Tensor, masks: torch.Tensor, candidates: list[float]) -> tuple[float, dict[float, float]]:
    """Le seuil qui maximise l'IoU sur la validation (jamais sur le test)."""
    sweep = {t: iou_dice(*confusion_pixels(logits, masks, t))[0] for t in candidates}  # IoU obtenue avec chaque seuil candidat
    return max(sweep, key=sweep.get), sweep  # seuil dont l'IoU est la plus grande, et le detail


# recharge un U-Net entraine depuis son fichier ; renvoie (modele pret, contenu du fichier)
def load_unet(model_file: Path, device: torch.device) -> tuple[ResNetUNet, dict]:
    """Recharge le U-Net et ses reglages depuis model.pt."""
    # map_location : charge sur le device voulu ; weights_only=True : ne lit que des tenseurs (pas de code, plus sur)
    saved = torch.load(model_file, map_location=device, weights_only=True)
    model = ResNetUNet(pretrained=False)  # architecture vide (inutile de telecharger les poids ImageNet : on les remplace)
    model.load_state_dict(saved["model_state"])  # copie les poids entraines dans l'architecture
    return model.to(device).eval(), saved  # modele sur le device, en mode evaluation, et le dictionnaire sauvegarde
