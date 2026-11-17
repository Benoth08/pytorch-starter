"""Tests des donnees : catalogue, decoupage sans fuite, formes des tenseurs."""

from pathlib import Path  # Path : represente un chemin de fichier

import numpy as np  # numpy : fabrique les images de test
import pandas as pd  # pandas : tableaux de donnees (catalogue, tableau croise)
import pytest  # pytest : outil qui trouve et lance les tests ; pytest.raises verifie qu'une erreur est levee
import torch  # PyTorch : type des tenseurs a verifier
from PIL import Image  # Pillow : enregistre les images de test

from vision_starter.config import Config  # la fiche de reglages du projet (taille des lots, nombre de processus)

# les fonctions testees (voir src/vision_starter/data.py)
from vision_starter.data import (
    ImageDataset,  # le Dataset : une image et sa classe par indice
    build_loader,  # fabrique un DataLoader
    build_transforms,  # fabrique les transformations entrainement / evaluation
    catalog_from_csv,  # catalogue lu depuis un CSV
    check_group_leak,  # refuse un groupe a cheval sur deux sous-ensembles
    check_split,  # refuse doublons et sous-ensembles vides
    make_split,  # decoupage stratifie
    make_stratified_grouped_split,  # decoupage par groupes, equilibre par classe
    near_duplicate_groups,  # regroupe les images presque identiques
)


# test : le catalogue trouve les 36 images et leurs 3 classes (catalog = fixture definie dans conftest.py)
def test_catalog_lists_all_images_with_labels(catalog):
    # Assert
    assert len(catalog) == 36  # assert : le test echoue si la condition est fausse ; ici 3 classes x 12 images
    assert sorted(catalog["label"].unique()) == ["alpha", "beta", "gamma"]  # les 3 classes attendues
    assert catalog["path"].is_unique  # aucun chemin en double
    assert not catalog["path"].str.contains("\\\\").any()  # chemins ecrits avec des / : le catalogue se relit sur toute machine


# test : decrire le meme jeu par un CSV donne les memes etiquettes que le rangement par dossiers
def test_catalog_from_csv_matches_folder_layout(image_folder, catalog):
    # Arrange : le meme jeu, decrit par un CSV
    csv_path = image_folder / "labels.csv"  # emplacement du CSV a fabriquer
    # nom de fichier relatif au dossier, ex. "alpha/alpha_0.png", pour chaque image du catalogue
    filenames = [f"{label}/{Path(path).name}" for path, label in zip(catalog["path"], catalog["label"])]
    pd.DataFrame({"filename": filenames, "label": catalog["label"]}).to_csv(csv_path, index=False)  # ecrit le CSV

    # Act
    from_csv = catalog_from_csv(image_folder, csv_path)  # relit le jeu par la voie CSV

    # Assert
    assert from_csv["label"].tolist() == catalog["label"].tolist()  # memes etiquettes, dans le meme ordre


# test : le decoupage garde toutes les classes partout et ne met jamais une image dans deux sous-ensembles
def test_split_is_stratified_and_leak_free(catalog):
    # Act
    split = make_split(catalog, val_fraction=0.25, test_fraction=0.25, seed=1)  # decoupage 50 / 25 / 25
    check_split(split)  # ne doit pas lever d'erreur

    # Assert : chaque classe est presente partout, et aucune image n'est dans deux sous-ensembles
    counts = pd.crosstab(split["label"], split["split"])  # tableau : nombre d'images par classe et par sous-ensemble
    assert (counts > 0).all().all()  # toutes les cases sont positives (aucune classe absente d'un sous-ensemble)
    # aucun chemin commun entre train et test (pas de fuite)
    assert set(split[split["split"] == "train"]["path"]).isdisjoint(split[split["split"] == "test"]["path"])
    assert set(split[split["split"] == "val"]["path"]).isdisjoint(split[split["split"] == "test"]["path"])  # ni entre val et test


# test : meme graine = meme decoupage
def test_split_is_reproducible(catalog):
    a = make_split(catalog, 0.25, 0.25, seed=7)  # premier decoupage
    b = make_split(catalog, 0.25, 0.25, seed=7)  # second, avec la meme graine
    assert a["split"].tolist() == b["split"].tolist()  # identiques ligne a ligne


# test : check_split refuse un catalogue ou une image apparait deux fois
def test_check_split_rejects_duplicates(catalog):
    # ajoute une copie de la 1re ligne (head(1)) puis decoupe
    split = make_split(pd.concat([catalog, catalog.head(1)], ignore_index=True), 0.25, 0.25, seed=1)
    with pytest.raises(ValueError):  # le bloc doit lever une ValueError, sinon le test echoue
        check_split(split)  # doit detecter le doublon


# test : le Dataset renvoie un tenseur de la bonne forme et du bon type, avec le bon numero de classe
def test_dataset_returns_normalized_tensor_of_expected_shape(catalog):
    # Arrange
    class_to_idx = {"alpha": 0, "beta": 1, "gamma": 2}  # nom de classe -> numero
    _, eval_transform = build_transforms(32, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])  # transformation d'evaluation en 32x32
    dataset = ImageDataset(catalog, class_to_idx, eval_transform)  # le Dataset a tester

    # Act
    image, target = dataset[0]  # premier exemple

    # Assert
    assert image.shape == (3, 32, 32)  # 3 canaux (gris converti en RGB), 32x32
    assert image.dtype == torch.float32  # nombres a virgule, comme attendu par le modele
    assert target == 0  # la 1re image du catalogue est de la classe "alpha"


# test : une augmentation mal ecrite dans le YAML arrete le programme tout de suite, avec la liste des choix
def test_build_transforms_rejects_unknown_augmentation():
    with pytest.raises(ValueError, match="augmentations inconnues"):  # le bloc doit lever cette erreur, sinon le test echoue
        build_transforms(32, [0.5] * 3, [0.25] * 3, augment=["hflip", "flipp"])  # "flipp" : faute de frappe


# test : la liste augment du YAML decide des augmentations du train ; la transformation d'evaluation n'en a jamais
def test_build_transforms_follows_the_augment_list():
    train_transform, eval_transform = build_transforms(32, [0.5] * 3, [0.25] * 3, augment=["rotate"])  # une seule augmentation
    train_names = [type(step).__name__ for step in train_transform.transforms]  # noms des etapes de la chaine d'entrainement
    eval_names = [type(step).__name__ for step in eval_transform.transforms]  # noms des etapes de la chaine d'evaluation
    assert train_names == ["ToImage", "Resize", "RandomRotation", "ToDtype", "Normalize"]  # la rotation, entre taille et conversion
    assert eval_names == ["ToImage", "Resize", "ToDtype", "Normalize"]  # aucune etape aleatoire


# test : le DataLoader melange et jette le dernier lot incomplet pour le train seulement
def test_loader_shuffles_and_drops_last_only_for_train(catalog):  # catalog : 36 images synthetiques (fixture de conftest.py)
    # Arrange : 36 images, lots de 8, lecture dans le processus principal
    config = Config(batch_size=8, num_workers=0)  # num_workers=0 : un test doit rester rapide et lisible
    class_to_idx = {"alpha": 0, "beta": 1, "gamma": 2}  # nom de classe -> numero
    _, eval_transform = build_transforms(32, [0.5] * 3, [0.25] * 3)  # transformation sans hasard, images 32 x 32

    # Act : le meme catalogue, charge comme un train puis comme une validation
    train_loader = build_loader(catalog, class_to_idx, eval_transform, config, train=True)  # melange, dernier lot jete
    val_loader = build_loader(catalog, class_to_idx, eval_transform, config, train=False)  # ordre fixe, rien de jete
    val_targets = torch.cat([targets for _, targets in val_loader])  # toutes les cibles lues en validation
    train_targets = torch.cat([targets for _, targets in train_loader])  # toutes les cibles lues en entrainement

    # Assert
    assert len(val_targets) == 36  # evaluation : aucune image perdue (5 lots, le dernier de 4)
    assert len(train_targets) == 32  # train : 4 lots complets, le lot de 4 images est jete
    assert val_targets.tolist() == [0] * 12 + [1] * 12 + [2] * 12  # evaluation : l'ordre du catalogue
    assert train_targets.tolist() != val_targets[:32].tolist()  # train : l'ordre est melange
    images, _ = next(iter(val_loader))  # un lot
    assert images.shape == (8, 3, 32, 32) and images.dtype == torch.float32  # forme (N, C, H, W) et type attendus


# test : une copie un peu plus claire est rangee dans le groupe de son original, une image differente reste seule
def test_near_duplicate_groups_put_a_brighter_copy_with_its_original(tmp_path):  # tmp_path : dossier vide fourni par pytest
    # Arrange : deux images de bruit lisse differentes, et une copie eclaircie de la premiere
    rng = np.random.default_rng(0)  # generateur aleatoire a graine fixe : memes images a chaque lancement
    first = np.kron(rng.integers(0, 200, (16, 16)), np.ones((8, 8))).astype(np.uint8)  # 128 x 128 : damier de 16 x 16 paves au hasard
    other = np.kron(rng.integers(0, 200, (16, 16)), np.ones((8, 8))).astype(np.uint8)  # un autre damier, sans rapport
    brighter = np.clip(first.astype(int) + 12, 0, 255).astype(np.uint8)  # la premiere image, 12 niveaux de gris plus claire
    paths = []  # chemins des trois fichiers
    for name, array in (("first", first), ("other", other), ("brighter", brighter)):  # pour chaque image
        paths.append(str(tmp_path / f"{name}.png"))  # son chemin
        Image.fromarray(array).save(paths[-1])  # l'enregistre en PNG

    # Act
    groups = near_duplicate_groups(paths)  # un numero de groupe par image

    # Assert
    assert groups[0] == groups[2]  # l'original et sa copie eclaircie : meme groupe
    assert groups[1] != groups[0]  # l'autre image : un groupe a elle


# test : le decoupage groupe ne coupe aucun groupe, et check_group_leak refuse un decoupage qui en coupe un
def test_stratified_grouped_split_keeps_groups_together(catalog):  # catalog : 36 images, 3 classes (fixture de conftest.py)
    # Arrange : des groupes de 2 images consecutives (meme classe), comme deux vues d'une meme piece
    catalog = catalog.assign(group=[i // 2 for i in range(len(catalog))])  # 18 groupes de 2 images

    # Act
    split = make_stratified_grouped_split(catalog, "group", val_fraction=0.25, test_fraction=0.25, seed=1)  # decoupage groupe

    # Assert
    check_split(split)  # les trois sous-ensembles existent, aucun chemin en double
    check_group_leak(split, "group")  # aucun groupe a cheval : ne doit pas lever d'erreur
    assert (pd.crosstab(split["label"], split["split"]) > 0).all().all()  # chaque classe est presente partout
    leaky = split.copy()  # copie a abimer
    leaky.loc[leaky.index[0], "split"] = "test" if leaky["split"].iloc[0] != "test" else "train"  # separe la 1re image de sa jumelle
    with pytest.raises(ValueError, match="a cheval"):  # le bloc doit lever cette erreur, sinon le test echoue
        check_group_leak(leaky, "group")  # doit detecter le groupe coupe
