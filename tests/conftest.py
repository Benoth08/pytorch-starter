"""Fixtures partagees : un petit jeu d'images synthetiques, sans dependre des vraies donnees."""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import numpy as np  # numpy : fabrique des images de bruit aleatoire
import pandas as pd  # pandas : type du catalogue renvoye
import pytest  # pytest : outil qui trouve et lance les tests (fonctions test_*)
from PIL import Image  # Pillow : enregistre les tableaux numpy en fichiers image


# fixture = fonction preparatoire : pytest l'execute et passe son resultat a tout test qui a un argument du meme nom
@pytest.fixture
# cree un dossier temporaire rempli d'images ; renvoie son chemin (tmp_path = dossier vide fourni par pytest)
def image_folder(tmp_path):
    """Cas B : 3 classes x 12 images 40x40 en niveaux de gris."""
    rng = np.random.default_rng(0)  # generateur aleatoire a graine fixe : memes images a chaque lancement
    for class_name in ("alpha", "beta", "gamma"):  # pour chacune des 3 classes inventees
        folder = tmp_path / class_name  # un sous-dossier par classe
        folder.mkdir()  # le cree
        for i in range(12):  # 12 images par classe
            # image 40x40 de bruit (entiers 0-254), enregistree en PNG sous un nom unique
            Image.fromarray(rng.integers(0, 255, (40, 40), dtype=np.uint8)).save(folder / f"{class_name}_{i}.png")
    return tmp_path  # le dossier racine, a donner aux tests


@pytest.fixture  # deuxieme fixture : elle utilise la premiere (argument image_folder)
def catalog(image_folder) -> pd.DataFrame:  # renvoie le catalogue (path, label) du dossier de test
    from vision_starter.data import catalog_from_class_folders  # la fonction testee, importee seulement quand on en a besoin

    return catalog_from_class_folders(image_folder)  # tableau de 36 lignes (3 classes x 12 images)
