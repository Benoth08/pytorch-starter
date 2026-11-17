"""Detection d'anomalies par auto-encodeur : catalogue MVTec AD, Dataset avec masques, modele, score, calibration.

C'est la version fichiers du notebook 05 (chapitre Anomalies du cours). Le principe : on
n'apprend que sur des images normales ; l'erreur de reconstruction, lissee,
devient une carte d'anomalie, et son maximum le score de l'image. Le seuil est
un quantile des scores d'images normales de validation (aucun defaut n'est
necessaire pour le fixer).
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import logging  # logging : messages horodates du programme
import math  # math : fonctions mathematiques de base (ici ceil = arrondi au-dessus)
import time  # time : mesure de la duree de chaque epoque
from pathlib import Path  # Path : represente un chemin de fichier

import numpy as np  # numpy : calcul sur des tableaux de nombres (images, masques, scores)
import pandas as pd  # pandas : tableaux de donnees (le catalogue des images)
import torch  # PyTorch : tenseurs (tableaux de nombres sur CPU ou GPU) et calcul des gradients
import torch.nn as nn  # nn : briques de reseaux de neurones (couches)
import torch.nn.functional as F  # F : operations sous forme de fonctions (perte MSE, convolution, padding)
from PIL import Image  # Pillow : ouvre les fichiers image

# f1_score : compromis rappel / precision ; roc_auc_score : AUROC, qualite du classement des scores (1 = parfait, 0.5 = hasard)
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split  # coupe un tableau en deux parties au hasard
from torch.utils.data import DataLoader, Dataset  # Dataset : un exemple par indice ; DataLoader : les regroupe en lots
from torchvision import tv_tensors  # types qui disent aux transformations "ceci est un masque" (pas de lissage)
from torchvision.transforms import v2  # transformations d'images de torchvision

from vision_starter.config import AnomalyConfig  # la fiche de reglages de l'auto-encodeur

logger = logging.getLogger(__name__)  # journal propre a ce module


# --- Catalogue et decoupage (etapes 3 et 4) ---


# parcourt l'arborescence MVTec AD ; renvoie le catalogue (path, label, defect_type, mask_path, split)
def catalog_mvtec(data_dir: Path) -> pd.DataFrame:
    """Arborescence MVTec AD : train/good, test/good, test/<type> et ground_truth/<type>/<nom>_mask.png."""
    rows = []  # une ligne (dictionnaire) par image
    for subset in ("train", "test"):  # pour chacun des deux sous-ensembles officiels
        for class_dir in sorted((data_dir / subset).iterdir()):  # pour chaque element du sous-ensemble (good, broken_large...)
            if not class_dir.is_dir():  # si ce n'est pas un dossier (fichier parasite)
                continue  # on l'ignore et on passe au suivant
            for image_path in sorted(class_dir.glob("*.png")):  # pour chaque image PNG de ce dossier
                defect_type = class_dir.name  # type de defaut = nom du dossier ("good" = normale)
                # masque du defaut : ground_truth/<type>/<nom>_mask.png ; une image normale n'en a pas (None)
                mask_path = data_dir / "ground_truth" / defect_type / f"{image_path.stem}_mask.png" if defect_type != "good" else None
                rows.append(  # ajoute la ligne de cette image
                    {  # dictionnaire = une ligne du futur tableau
                        "path": image_path.as_posix(),  # chemin de l'image, ecrit avec des /
                        "label": "good" if defect_type == "good" else "defect",  # etiquette binaire : normale ou defaut
                        "defect_type": defect_type,  # type precis du defaut (pour les analyses par type)
                        "mask_path": mask_path.as_posix() if mask_path else "",  # chemin du masque, ou texte vide
                        "split": subset,  # sous-ensemble officiel (train ou test)
                    }
                )
    if not rows:  # si aucune image n'a ete trouvee
        raise FileNotFoundError(f"aucune image trouvee sous {data_dir}")  # erreur claire : mauvais dossier
    catalog = pd.DataFrame(rows)  # liste -> tableau pandas
    missing = [p for p in catalog["mask_path"] if p and not Path(p).is_file()]  # masques annonces mais absents du disque
    if missing:  # s'il en manque au moins un
        raise FileNotFoundError(f"{len(missing)} masques manquants, par exemple {missing[0]}")  # on s'arrete avec un exemple
    return catalog  # le catalogue complet


# met de cote une partie des normales d'entrainement pour la validation ; renvoie le catalogue mis a jour
def split_normals(catalog: pd.DataFrame, val_fraction: float, seed: int) -> pd.DataFrame:
    """La validation = une part des normales d'entrainement ; le test officiel (normales + defauts) est conserve."""
    catalog = catalog.copy()  # copie : on ne modifie pas le tableau recu
    train_good = catalog[catalog["split"] == "train"]  # images d'entrainement (toutes normales dans MVTec)
    _, val = train_test_split(train_good, test_size=val_fraction, random_state=seed)  # tire val_fraction d'entre elles au hasard
    catalog.loc[val.index, "split"] = "val"  # ces lignes passent en validation
    if (catalog.loc[catalog["split"] != "test", "label"] != "good").any():  # si une image "defect" se trouve hors du test
        raise ValueError("un defaut s'est glisse dans train ou val : l'auto-encodeur ne doit voir que des normales")  # on refuse
    return catalog  # catalogue avec train / val / test


# --- Dataset, transformations, DataLoader (etapes 5 a 7) ---


class AnomalyDataset(Dataset):  # jeu de donnees PyTorch : dataset[i] renvoie (image, masque, etiquette)
    """Image RGB en [0, 1] (pas de normalisation : la sortie sigmoide doit reproduire l'entree), masque (1, S, S) en 0/1, etiquette 0/1."""

    def __init__(self, frame: pd.DataFrame, transform) -> None:  # memorise chemins, masques, etiquettes et transformation
        self.paths = frame["path"].tolist()  # chemins des images
        self.mask_paths = frame["mask_path"].fillna("").tolist()  # chemins des masques ; vide (et pas NaN) quand il n'y en a pas
        self.targets = (frame["label"] == "defect").astype(int).tolist()  # etiquette 1 = defaut, 0 = normale
        self.transform = transform  # transformation commune image + masque

    def __len__(self) -> int:  # nombre d'exemples
        return len(self.paths)  # une image par chemin

    # renvoie l'exemple numero index : (image, masque, etiquette)
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image = Image.open(self.paths[index]).convert("RGB")  # ouvre l'image en 3 canaux
        if self.mask_paths[index]:  # si l'image a un masque (image avec defaut)
            mask = (np.array(Image.open(self.mask_paths[index]).convert("L")) > 127).astype(np.uint8)  # masque en 0/1
        else:  # sinon (image normale)
            mask = np.zeros((image.height, image.width), dtype=np.uint8)  # masque tout a 0 de la taille de l'image
        image, mask = self.transform(image, tv_tensors.Mask(mask[None]))  # meme geometrie pour les deux, masque en plus proche voisin
        return image, mask.float(), self.targets[index]  # masque en reels pour les calculs suivants


# construit les transformations ; renvoie (entrainement, evaluation)
def build_anomaly_transforms(image_size: int) -> tuple[v2.Compose, v2.Compose]:
    """Entrainement : retournements seulement (les normales doivent rester normales) ; evaluation : rien."""
    train = v2.Compose(  # Compose enchaine les etapes dans l'ordre
        [  # liste des etapes de l'entrainement
            v2.ToImage(),  # image PIL -> tenseur d'entiers (canaux, hauteur, largeur)
            v2.Resize((image_size, image_size), antialias=True),  # taille fixe
            v2.RandomHorizontalFlip(),  # miroir gauche-droite une fois sur deux (augmentation)
            v2.RandomVerticalFlip(),  # miroir haut-bas une fois sur deux
            v2.ToDtype(torch.float32, scale=True),  # entiers 0-255 -> reels 0-1 (pas de normalisation, voir la docstring)
        ]
    )
    # evaluation : tenseur, taille fixe, reels 0-1, sans hasard
    evaluation = v2.Compose([v2.ToImage(), v2.Resize((image_size, image_size), antialias=True), v2.ToDtype(torch.float32, scale=True)])
    return train, evaluation  # les deux chaines


# cree le DataLoader qui fournit (images, masques, etiquettes) par lots ; renvoie ce DataLoader
def build_anomaly_loader(frame: pd.DataFrame, transform, config: AnomalyConfig, train: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)  # generateur aleatoire a graine fixe : melange reproductible
    return DataLoader(  # objet a parcourir avec for : chaque tour donne un lot (batch) d'exemples
        AnomalyDataset(frame, transform),  # ou lire les exemples
        batch_size=config.batch_size,  # nombre d'images par lot
        shuffle=train,  # melange seulement pour l'entrainement
        num_workers=config.num_workers,  # processus de chargement en parallele
        generator=generator,  # generateur du melange (graine fixe)
    )


# --- Modele (etape 8) ---


# bloc descendant de l'encodeur ; renvoie un Sequential qui divise hauteur et largeur par 2
def down_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv 4x4 pas 2 : divise la resolution par 2."""
    return nn.Sequential(  # couches enchainees
        nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),  # convolution 4x4 qui saute 1 pixel sur 2 (stride=2)
        nn.BatchNorm2d(out_channels),  # BatchNorm : recentre chaque canal sur le lot, apprentissage plus stable
        nn.LeakyReLU(0.2, inplace=True),  # comme ReLU, mais laisse passer 20 % des valeurs negatives
    )


# bloc montant du decodeur ; renvoie un Sequential qui multiplie hauteur et largeur par 2
def up_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Convolution transposee 4x4 pas 2 : multiplie la resolution par 2."""
    return nn.Sequential(  # couches enchainees
        nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),  # "inverse" d'une convolution : agrandit x2
        nn.BatchNorm2d(out_channels),  # normalisation par canal
        nn.ReLU(inplace=True),  # met a 0 les valeurs negatives
    )


class ConvAutoencoder(nn.Module):  # auto-encodeur : compresse l'image puis tente de la reconstruire
    """Image (N, 3, S, S) en [0, 1] -> reconstruction de meme forme en [0, 1]. Le goulot est le reglage central."""

    # cree l'encodeur et le decodeur ; latent_channels = largeur du goulot, widths = largeurs des etages
    def __init__(self, latent_channels: int = 32, widths: tuple[int, ...] = (32, 64, 128, 256)) -> None:
        super().__init__()  # initialise la partie nn.Module
        encoder, previous = [], 3  # couches de l'encodeur a construire ; l'image d'entree a 3 canaux
        for width in widths:  # pour chaque largeur (32, 64, 128, 256)
            encoder.append(down_block(previous, width))  # un bloc qui divise la taille par 2
            previous = width  # sa sortie est l'entree du bloc suivant
        encoder.append(nn.Conv2d(previous, latent_channels, 1))  # conv 1x1 : reduit a latent_channels canaux (le goulot)
        self.encoder = nn.Sequential(*encoder)  # encodeur complet : (3, S, S) -> (latent, S/16, S/16)
        decoder = [nn.Conv2d(latent_channels, widths[-1], 1), nn.ReLU(inplace=True)]  # sortie du goulot : retour a 256 canaux
        # paires de largeurs successives en sens inverse : (256, 128), (128, 64), (64, 32)
        for width_in, width_out in zip(widths[::-1], widths[::-1][1:]):
            decoder.append(up_block(width_in, width_out))  # un bloc qui double la taille
        # derniere remontee vers 3 canaux, puis sigmoide : valeurs entre 0 et 1 comme l'image d'entree
        decoder += [nn.ConvTranspose2d(widths[0], 3, 4, stride=2, padding=1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder)  # decodeur complet : (latent, S/16, S/16) -> (3, S, S)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # calcul complet : image -> reconstruction
        return self.decoder(self.encoder(x))  # compresse puis reconstruit


# fabrique un filtre de flou gaussien ; renvoie un tenseur (1, 1, k, k) dont la somme vaut 1
def gaussian_kernel(sigma: float) -> torch.Tensor:
    """Noyau gaussien 2D normalise, forme (1, 1, k, k), k = 2 * ceil(3 sigma) + 1."""
    radius = max(1, math.ceil(3 * sigma))  # rayon du filtre : 3 sigma couvre presque toute la cloche gaussienne
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)  # positions -r ... 0 ... +r
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))  # courbe en cloche sur une dimension
    kernel_1d /= kernel_1d.sum()  # normalise : la somme vaut 1 (le flou ne change pas le niveau moyen)
    return torch.outer(kernel_1d, kernel_1d)[None, None]  # produit exterieur -> filtre 2D ; [None, None] ajoute 2 dimensions


class AnomalyScorer(nn.Module):  # enveloppe l'auto-encodeur pour sortir directement la carte et le score d'anomalie
    """Image -> (reconstruction, carte d'anomalie (N, 1, S, S), score image (N,)). Exportable en ONNX tel quel."""

    def __init__(self, autoencoder: nn.Module, sigma: float) -> None:  # memorise l'auto-encodeur et le filtre de lissage
        super().__init__()  # initialise la partie nn.Module
        self.autoencoder = autoencoder  # le modele entraine
        self.register_buffer("kernel", gaussian_kernel(sigma))  # buffer = tenseur du modele non entraine, suit .to(device) et l'export

    # calcul : images -> (reconstruction, carte d'anomalie, score par image)
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon = self.autoencoder(x)  # reconstruction de l'image
        error_map = ((x - recon) ** 2).mean(dim=1, keepdim=True)  # erreur au carre par pixel, moyennee sur les 3 canaux
        radius = self.kernel.shape[-1] // 2  # rayon du filtre (taille // 2)
        # lissage : marge "miroir" (reflect) sur les 4 bords pour garder la taille, puis convolution avec le filtre gaussien
        error_map = F.conv2d(F.pad(error_map, (radius,) * 4, mode="reflect"), self.kernel)
        return recon, error_map, error_map.flatten(1).amax(dim=1)  # score image = maximum de la carte lissee


# recharge un auto-encodeur entraine et le met dans un AnomalyScorer ; renvoie (scorer, contenu du fichier)
def load_scorer(model_file: Path, device: torch.device) -> tuple[AnomalyScorer, dict]:
    """Recharge l'auto-encodeur et ses reglages depuis model.pt (etape 13)."""
    saved = torch.load(model_file, map_location=device, weights_only=True)  # lit le fichier ; weights_only = donnees seulement (plus sur)
    autoencoder = ConvAutoencoder(saved["latent_channels"])  # architecture vide avec la meme largeur de goulot
    autoencoder.load_state_dict(saved["model_state"])  # copie les poids entraines
    return AnomalyScorer(autoencoder, saved["gaussian_sigma"]).to(device).eval(), saved  # scorer en mode evaluation + reglages


# --- Entrainement (etape 10) ---


# parcourt un loader une fois ; renvoie l'erreur de reconstruction moyenne (MSE)
def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None) -> float:
    """Une passe. Avec optimiseur : entrainement ; sans : evaluation. Renvoie la MSE moyenne de reconstruction."""
    model.train(optimizer is not None)  # mode entrainement si on a un optimiseur, sinon mode evaluation
    total, seen = 0.0, 0  # cumul des pertes et nombre d'images vues
    with torch.set_grad_enabled(optimizer is not None):  # gradients calcules seulement en entrainement
        for images, _, _ in loader:  # pour chaque lot (masques et etiquettes inutiles ici)
            images = images.to(device)  # envoie le lot sur le device du modele (CPU ou GPU)
            loss = F.mse_loss(model(images), images)  # MSE : moyenne des (reconstruction - image)^2 ; la cible est l'image elle-meme
            if optimizer is not None:  # si on entraine
                optimizer.zero_grad()  # remet a zero les gradients du lot precedent
                loss.backward()  # retropropagation : calcule le gradient de la perte pour chaque poids
                optimizer.step()  # l'optimiseur corrige les poids d'un petit pas
            total += loss.item() * len(images)  # perte du lot x taille du lot
            seen += len(images)  # images traitees
    return total / seen  # MSE moyenne par image


# boucle d'entrainement complete de l'auto-encodeur ; renvoie l'historique (une ligne par epoque)
def fit_autoencoder(
    model: nn.Module,  # l'auto-encodeur a entrainer
    train_loader: DataLoader,  # lots d'entrainement (images normales)
    val_loader: DataLoader,  # lots de validation (images normales aussi)
    config: AnomalyConfig,  # reglages (pas d'apprentissage, epoques, patience...)
    device: torch.device,  # CPU ou GPU
    checkpoint_path: Path,  # fichier ou sauvegarder le meilleur modele
) -> pd.DataFrame:  # renvoie l'historique sous forme de tableau pandas
    """Boucle complete : meilleur modele selon la MSE de validation (sur des normales), arret anticipe, historique."""
    # AdamW : optimiseur (corrige les poids a partir des gradients) avec weight decay (penalite sur les grands poids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # divise le pas d'apprentissage par 2 si la validation ne progresse plus pendant 4 epoques
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    history, best_val, without_improvement = [], float("inf"), 0  # historique, meilleure MSE (infini au depart), compteur
    for epoch in range(1, config.epochs + 1):  # pour chaque epoque (un passage sur tout le train), numerotee a partir de 1
        start = time.time()  # heure de debut
        train_loss = run_epoch(model, train_loader, device, optimizer)  # apprend sur le train
        val_loss = run_epoch(model, val_loader, device)  # mesure sur la validation (sans optimiseur = sans apprendre)
        scheduler.step(val_loss)  # le scheduler regarde la perte de validation
        # ajoute une ligne a l'historique : epoque, MSE train et validation, pas d'apprentissage actuel
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})
        improved = val_loss < best_val  # vrai si meilleure validation jusqu'ici
        if improved:  # si le modele s'est ameliore
            best_val, without_improvement = val_loss, 0  # nouveau record, compteur remis a zero
            # sauvegarde les poids avec l'epoque et la perte
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, checkpoint_path)
        else:  # sinon
            without_improvement += 1  # une epoque de plus sans progres
        logger.info(  # bilan de l'epoque ; chaque %... est remplace par la valeur correspondante ci-dessous
            "epoque %2d | train MSE %.5f | val MSE %.5f | %.0fs%s",  # modele du message
            epoch,  # numero d'epoque
            train_loss,  # MSE d'entrainement
            val_loss,  # MSE de validation
            time.time() - start,  # duree de l'epoque en secondes
            " | meilleur" if improved else "",  # mention si meilleur modele
        )
        if without_improvement >= config.patience:  # si pas de progres depuis "patience" epoques
            logger.info("arret anticipe : %d epoques sans amelioration", config.patience)  # on le signale
            break  # et on arrete
    return pd.DataFrame(history)  # historique sous forme de tableau


# --- Scores, calibration, metriques (etape 11) ---


@torch.no_grad()  # pas de gradient : on ne fait que predire
# note toutes les images d'un loader ; renvoie (scores, cartes, masques, etiquettes) en tableaux numpy
def score_loader(scorer: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Renvoie scores (N,), cartes (N, S, S), masques (N, S, S), etiquettes (N,) pour tout un loader."""
    scorer.eval()  # mode evaluation
    scores, maps, all_masks, labels = [], [], [], []  # une liste par sortie, remplie lot par lot
    for images, masks, targets in loader:  # pour chaque lot
        _, error_map, score = scorer(images.to(device))  # carte d'anomalie et score (la reconstruction ne sert pas ici)
        scores.append(score.cpu())  # scores du lot, ramenes sur CPU
        maps.append(error_map[:, 0].cpu())  # cartes du lot sans la dimension canal : (lot, S, S)
        all_masks.append(masks[:, 0])  # masques du lot sans la dimension canal
        labels.append(targets)  # etiquettes 0/1 du lot
    # colle les lots et convertit chaque resultat en tableau numpy
    return torch.cat(scores).numpy(), torch.cat(maps).numpy(), torch.cat(all_masks).numpy(), torch.cat(labels).numpy()


# fixe les seuils de decision a partir des normales de validation ; renvoie un dictionnaire de seuils
def calibrate_thresholds(val_scores: np.ndarray, val_maps: np.ndarray, image_quantile: float, pixel_quantile: float) -> dict[str, float]:
    """Seuils = quantiles des scores de normales : un taux de fausses alarmes choisi, sans aucun defaut."""
    return {  # dictionnaire des resultats
        "image_threshold": float(np.quantile(val_scores, image_quantile)),  # ex. quantile 0.99 : 1 % des normales depassent ce score
        "pixel_threshold": float(np.quantile(val_maps, pixel_quantile)),  # meme idee pixel par pixel
        "n_val": int(len(val_scores)),  # nombre d'images utilisees (un quantile sur peu d'images est fragile)
    }


# mesure la qualite de la detection ; renvoie un dictionnaire de metriques
def anomaly_metrics(
    scores: np.ndarray,  # score de chaque image (sortie de score_loader)
    maps: np.ndarray,  # carte d'anomalie de chaque image
    masks: np.ndarray,  # vrais masques des defauts (0/1 par pixel)
    labels: np.ndarray,  # vraies etiquettes (1 = defaut, 0 = normale)
    image_threshold: float,  # seuil calibre sur la validation
) -> dict[str, float]:  # renvoie {nom de la metrique: valeur}
    """AUROC image et pixel (sans seuil), puis rappel, fausses alarmes et F1 au seuil calibre."""
    decisions = (scores > image_threshold).astype(int)  # decision par image : 1 = anomalie si le score depasse le seuil
    metrics = {"auroc_image": float(roc_auc_score(labels, scores))}  # AUROC image : les defauts ont-ils des scores plus hauts ?
    if masks.any():  # s'il y a au moins un pixel de defaut dans les masques
        metrics["auroc_pixel"] = float(roc_auc_score(masks.ravel().astype(int), maps.ravel()))  # AUROC pixel par pixel (ravel = aplatir)
    metrics.update(  # ajoute les metriques qui dependent du seuil
        {  # dictionnaire des metriques au seuil
            # rappel : part des images avec defaut detectees (nan s'il n'y a aucun defaut)
            "recall": float(decisions[labels == 1].mean()) if (labels == 1).any() else float("nan"),
            # fausses alarmes : part des images normales signalees a tort (nan s'il n'y a aucune normale)
            "false_alarm_rate": float(decisions[labels == 0].mean()) if (labels == 0).any() else float("nan"),
            "f1": float(f1_score(labels, decisions)),  # F1 : moyenne harmonique de la precision et du rappel
            "n_images": int(len(labels)),  # nombre d'images evaluees
        }
    )
    return metrics  # toutes les metriques
