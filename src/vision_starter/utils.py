"""Briques partagees par tous les scripts : graines, device, journalisation.

Ce module n'importe rien du projet : il est tout en bas de la pile des
dependances, ce qui evite les imports circulaires.
"""

from __future__ import annotations  # permet d'ecrire les types modernes (ex. Path | None) meme dans les annotations

import logging  # logging : affiche et enregistre des messages horodates (mieux que print)
import random  # random : generateur de nombres aleatoires de Python (melanges, tirages)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres ; il a son propre generateur aleatoire
import torch  # PyTorch : tenseurs (tableaux de nombres sur CPU ou GPU) et reseaux de neurones


# fixe toutes les graines aleatoires ; ne renvoie rien
def seed_everything(seed: int) -> None:
    """Fixe toutes les graines : deux executions donnent les memes resultats."""
    random.seed(seed)  # graine du hasard de Python : memes melanges a chaque lancement
    np.random.seed(seed)  # graine du hasard de numpy (utilise par certaines bibliotheques)
    torch.manual_seed(seed)  # graine de PyTorch sur CPU : memes poids initiaux et memes melanges de lots
    torch.cuda.manual_seed_all(seed)  # meme chose sur toutes les cartes graphiques (sans effet s'il n'y en a pas)


# donne a un processus de chargement du DataLoader sa propre graine pour numpy et random ; ne renvoie rien
def seed_worker(worker_id: int) -> None:
    """A passer au DataLoader (worker_init_fn) : une augmentation ecrite avec numpy ou random reste reproductible."""
    worker_seed = torch.initial_seed() % 2**32  # graine que PyTorch a donnee a ce processus, ramenee sur 32 bits
    np.random.seed(worker_seed)  # graine du hasard de numpy dans ce processus
    random.seed(worker_seed)  # graine du hasard de Python dans ce processus


# choisit ou faire les calculs et renvoie ce "device" (GPU ou CPU)
def get_device() -> torch.device:
    """GPU si disponible, sinon CPU. Le reste du code ne change pas."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # "cuda" = carte graphique NVIDIA, bien plus rapide


# configure les messages du programme (console + fichier eventuel) ; ne renvoie rien
def setup_logging(log_file: Path | None = None) -> None:
    """Logs sur la console et, si demande, dans un fichier (logging plutot que print)."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]  # un "handler" = une destination des messages ; ici la console
    if log_file is not None:  # si on a demande un fichier de log
        log_file.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier du fichier s'il n'existe pas encore
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))  # ajoute le fichier comme seconde destination
    # niveau INFO = on garde les messages normaux ; format = date | niveau | texte ; force=True remplace un reglage precedent
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers, force=True)
    quiet_libraries()  # coupe le bavardage des bibliotheques externes


# rend silencieuses les bibliotheques trop bavardes ; ne renvoie rien
def quiet_libraries() -> None:
    """Ramene au niveau WARNING les bibliotheques bavardes au niveau INFO (a rappeler apres leur import : certaines
    remettent leur niveau a INFO en s'important, Lightning par exemple)."""
    for noisy in ("onnxscript", "onnx_ir", "torch.onnx", "lightning", "lightning.pytorch", "anomalib"):  # pour chaque bibliotheque bavarde
        logging.getLogger(noisy).setLevel(logging.WARNING)  # elle n'affichera plus que les avertissements et les erreurs


# compte les nombres (poids) que le modele contient ; renvoie un entier
def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Nombre de parametres (entrainables par defaut)."""
    # numel() = nombre d'elements d'un tenseur ; requires_grad = le poids sera modifie par l'entrainement (non gele)
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
