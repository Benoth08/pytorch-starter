"""Le moteur d'entrainement : une epoque, une evaluation, la boucle complete.

Ces fonctions ne savent rien des images ni du projet : elles recoivent un
modele, des DataLoader, une perte et un optimiseur. C'est ce qui les rend
reutilisables d'un projet a l'autre.
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import logging  # logging : messages horodates du programme
import time  # time : mesure de la duree de chaque epoque
from pathlib import Path  # Path : represente un chemin de fichier

import numpy as np  # numpy : tableaux de nombres (les predictions sont renvoyees sous cette forme)
import pandas as pd  # pandas : tableau de l'historique d'entrainement (une ligne par epoque)
import torch  # PyTorch : tenseurs (tableaux de nombres sur CPU ou GPU) et calcul des gradients
import torch.nn as nn  # nn : briques de reseaux de neurones (ici pour le type nn.Module)
from torch.utils.data import DataLoader  # DataLoader : fournit les images par lots (batch = paquet d'images traitees ensemble)

logger = logging.getLogger(__name__)  # journal propre a ce module


# entraine le modele sur toutes les images du train une fois ; renvoie (perte moyenne, accuracy)
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> tuple[float, float]:
    """Une passe sur le train ; renvoie (perte moyenne, accuracy)."""
    model.train()  # mode entrainement : dropout actif, BatchNorm mise a jour avec le lot courant
    total_loss, correct, seen = 0.0, 0, 0  # cumul des pertes, nombre de bonnes reponses, nombre d'images vues
    for images, targets in loader:  # pour chaque lot (images + numeros de classe attendus)
        images, targets = images.to(device), targets.to(device)  # envoie le lot sur le meme device que le modele (CPU ou GPU)
        optimizer.zero_grad()  # remet a zero les gradients du lot precedent (sinon PyTorch les additionne)
        logits = model(images)  # passe avant : logits = scores bruts par classe, avant transformation en probabilites
        loss = criterion(logits, targets)  # perte : nombre qui mesure l'erreur (plus petit = meilleur)
        loss.backward()  # retropropagation : calcule le gradient (sens dans lequel bouger chaque poids pour reduire la perte)
        optimizer.step()  # l'optimiseur corrige les poids d'un petit pas dans ce sens
        total_loss += loss.item() * len(targets)  # .item() extrait le nombre ; x taille du lot pour une moyenne juste par image
        correct += (logits.argmax(dim=1) == targets).sum().item()  # argmax = classe au plus haut score ; on compte les bonnes
        seen += len(targets)  # nombre d'images traitees jusqu'ici
    return total_loss / seen, correct / seen  # perte moyenne par image et accuracy (part de bonnes reponses)


@torch.no_grad()  # decorateur : aucun gradient calcule dans cette fonction (plus rapide, moins de memoire)
# mesure le modele sans l'entrainer ; renvoie (perte moyenne, accuracy, predictions, cibles)
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Une passe sans apprentissage ; renvoie perte, accuracy, predictions, cibles."""
    model.eval()  # mode evaluation : dropout coupe, BatchNorm utilise ses statistiques apprises
    total_loss, all_preds, all_targets = 0.0, [], []  # cumul des pertes ; listes des predictions et des cibles par lot
    for images, targets in loader:  # pour chaque lot
        images, targets = images.to(device), targets.to(device)  # envoie le lot sur le device du modele
        logits = model(images)  # scores bruts par classe
        total_loss += criterion(logits, targets).item() * len(targets)  # perte du lot ponderee par sa taille
        all_preds.append(logits.argmax(dim=1).cpu())  # classe predite pour chaque image, ramenee sur CPU
        all_targets.append(targets.cpu())  # vraies classes, ramenees sur CPU
    preds, targets = torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy()  # colle tous les lots puis convertit en numpy
    # perte moyenne par image, accuracy (moyenne des bonnes reponses), predictions et cibles pour la matrice de confusion
    return total_loss / len(targets), float((preds == targets).mean()), preds, targets


@torch.no_grad()  # pas de gradient : on ne fait que predire
# calcule les probabilites de chaque classe pour toutes les images ; renvoie un tableau (images, classes)
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Probabilites (softmax) pour toutes les images d'un loader."""
    model.eval()  # mode evaluation
    # pour chaque lot : softmax = transforme les logits en probabilites positives qui somment a 1 ; l'etiquette est ignoree (_)
    outputs = [torch.softmax(model(images.to(device)), dim=1).cpu() for images, _ in loader]
    return torch.cat(outputs).numpy()  # colle tous les lots en un seul tableau numpy


# boucle d'entrainement complete ; renvoie l'historique (une ligne par epoque) sous forme de tableau pandas
def fit(
    model: nn.Module,  # le reseau a entrainer
    train_loader: DataLoader,  # lots d'entrainement (melanges)
    val_loader: DataLoader,  # lots de validation (pour suivre la generalisation)
    criterion,  # la fonction de perte (ex. entropie croisee)
    optimizer,  # l'optimiseur : l'algorithme qui corrige les poids a partir des gradients (ex. AdamW)
    scheduler,  # ajuste le taux d'apprentissage au fil des epoques (ici d'apres la perte de validation)
    epochs: int,  # nombre maximal d'epoques (une epoque = un passage sur tout le train)
    patience: int,  # epoques sans amelioration toleres avant l'arret anticipe
    device: torch.device,  # CPU ou GPU
    checkpoint_path: Path,  # fichier ou sauvegarder le meilleur modele
) -> pd.DataFrame:  # renvoie l'historique sous forme de tableau pandas
    """Boucle complete : meilleur modele selon la perte de validation, arret anticipe, historique."""
    # historique vide, meilleure perte de validation (infini au depart), compteur d'epoques sans progres
    history, best_val_loss, epochs_without_improvement = [], float("inf"), 0
    for epoch in range(1, epochs + 1):  # pour chaque epoque, numerotee a partir de 1
        start = time.time()  # heure de debut, pour afficher la duree
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)  # apprend sur le train
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)  # mesure sur la validation
        scheduler.step(val_loss)  # le scheduler peut baisser le taux d'apprentissage si la validation stagne
        lr = optimizer.param_groups[0]["lr"]  # taux d'apprentissage actuel (1er groupe), pour le suivi
        history.append(  # ajoute une ligne a l'historique
            # numero d'epoque, pertes et accuracies train / validation, taux d'apprentissage
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "lr": lr}
        )
        improved = val_loss < best_val_loss  # vrai si c'est la meilleure validation jusqu'ici
        if improved:  # si le modele s'est ameliore
            best_val_loss, epochs_without_improvement = val_loss, 0  # nouveau record ; on remet le compteur a zero
            # sauvegarde les poids (state_dict = dictionnaire nom -> tenseur) avec l'epoque et la perte
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, checkpoint_path)
        else:  # sinon (pas de progres)
            epochs_without_improvement += 1  # une epoque de plus sans amelioration
        logger.info(  # affiche le bilan de l'epoque ; chaque %... est remplace par la valeur correspondante ci-dessous
            "epoque %2d | train perte %.4f acc %.3f | val perte %.4f acc %.3f | lr %.1e | %.0fs%s",  # modele du message
            epoch,  # numero d'epoque
            train_loss,  # perte d'entrainement
            train_acc,  # accuracy d'entrainement
            val_loss,  # perte de validation
            val_acc,  # accuracy de validation
            lr,  # taux d'apprentissage
            time.time() - start,  # duree de l'epoque en secondes
            " | meilleur" if improved else "",  # mention ajoutee si c'est le meilleur modele
        )
        if epochs_without_improvement >= patience:  # si la validation ne progresse plus depuis "patience" epoques
            logger.info("arret anticipe : %d epoques sans amelioration", patience)  # on le signale
            break  # et on sort de la boucle : continuer ne ferait que sur-apprendre
    return pd.DataFrame(history)  # historique sous forme de tableau (pratique pour tracer les courbes)
