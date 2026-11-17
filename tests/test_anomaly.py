"""Tests du module anomalie : formes du scoreur, calibration, metriques, decoupage sans fuite de defaut."""

import numpy as np  # numpy : scores et cartes factices
import pandas as pd  # pandas : petit catalogue factice
import pytest  # pytest : outil qui trouve et lance les tests ; pytest.raises verifie qu'une erreur est levee
import torch  # PyTorch : images factices, comparaisons de tenseurs

# les classes et fonctions testees (voir src/vision_starter/anomaly.py)
from vision_starter.anomaly import AnomalyScorer, ConvAutoencoder, anomaly_metrics, calibrate_thresholds, gaussian_kernel, split_normals


# test : la reconstruction a la meme forme que l'entree et reste entre 0 et 1
def test_autoencoder_reconstructs_same_shape():
    model = ConvAutoencoder(latent_channels=8, widths=(8, 16, 32, 64)).eval()  # petit auto-encodeur, mode evaluation
    x = torch.rand(2, 3, 64, 64)  # 2 images aleatoires, valeurs entre 0 et 1
    with torch.no_grad():  # pas de gradient : simple prediction
        y = model(x)  # reconstruction
    assert y.shape == x.shape  # meme forme (2, 3, 64, 64)
    assert 0.0 <= y.min() and y.max() <= 1.0  # sigmoide en sortie


# test : le filtre gaussien a la bonne taille et sa somme vaut 1
def test_gaussian_kernel_is_normalized():
    kernel = gaussian_kernel(2.0)  # filtre pour sigma = 2
    assert kernel.shape == (1, 1, 13, 13)  # rayon 6 = ceil(3 * sigma)
    assert torch.isclose(kernel.sum(), torch.tensor(1.0))  # somme egale a 1, aux arrondis pres


# test : formes des sorties du scoreur, et score nul si la reconstruction est parfaite
def test_scorer_outputs_and_zero_error_on_perfect_reconstruction():
    identity = torch.nn.Identity()  # "auto-encodeur" parfait : la reconstruction est l'entree
    scorer = AnomalyScorer(identity, sigma=1.0).eval()  # scoreur autour de ce faux modele
    x = torch.rand(3, 3, 32, 32)  # 3 images aleatoires
    recon, error_map, score = scorer(x)  # reconstruction, carte d'erreur, score par image
    assert recon.shape == x.shape and error_map.shape == (3, 1, 32, 32) and score.shape == (3,)  # formes attendues
    assert torch.all(score == 0)  # aucune erreur -> score nul partout


# test : seuils calibres plausibles, et metriques parfaites sur un cas parfaitement separe
def test_calibration_and_metrics():
    rng = np.random.default_rng(0)  # generateur aleatoire a graine fixe
    val_scores = rng.uniform(0, 1, 200)  # 200 scores de "normales" entre 0 et 1
    val_maps = rng.uniform(0, 1, (10, 8, 8))  # 10 cartes 8x8 de valeurs entre 0 et 1
    # seuils : quantile 99 % des scores, quantile 99,9 % des pixels
    calibration = calibrate_thresholds(val_scores, val_maps, image_quantile=0.99, pixel_quantile=0.999)
    assert 0.95 < calibration["image_threshold"] <= 1.0 and calibration["n_val"] == 200  # seuil proche de 1, 200 images comptees

    # Defauts a 2.0, normales a 0.0 : separation parfaite -> AUROC 1, rappel 1, zero fausse alarme.
    labels = np.array([0, 0, 1, 1])  # 2 normales puis 2 defauts
    scores = np.array([0.0, 0.1, 2.0, 2.1])  # scores bas pour les normales, hauts pour les defauts
    maps = np.stack([np.zeros((4, 4)), np.zeros((4, 4)), np.ones((4, 4)), np.ones((4, 4))])  # cartes : 0 pour normales, 1 pour defauts
    masks = maps.copy()  # vrais masques identiques aux cartes (prediction parfaite)
    metrics = anomaly_metrics(scores, maps, masks, labels, image_threshold=calibration["image_threshold"])  # calcule les metriques
    assert metrics["auroc_image"] == 1.0 and metrics["recall"] == 1.0 and metrics["false_alarm_rate"] == 0.0  # tout est parfait


# test : split_normals refuse un defaut dans le train, et fonctionne quand le defaut est dans le test
def test_split_normals_refuses_defects_in_train():
    catalog = pd.DataFrame(  # catalogue de 4 images, toutes en train, dont un defaut (interdit)
        # chemins, etiquettes et sous-ensembles factices
        {"path": ["a", "b", "c", "d"], "label": ["good", "good", "good", "defect"], "split": ["train", "train", "train", "train"]}
    )
    with pytest.raises(ValueError):  # le bloc doit lever une ValueError, sinon le test echoue
        split_normals(catalog, val_fraction=0.34, seed=0)  # doit refuser le defaut present en train
    good = catalog[catalog["label"] == "good"].assign(split="train")  # garde seulement les 3 normales, en train
    # ajoute un defaut en test (autorise) puis decoupe
    out = split_normals(pd.concat([good, pd.DataFrame([{"path": "e", "label": "defect", "split": "test"}])]), val_fraction=0.34, seed=0)
    assert set(out["split"]) == {"train", "val", "test"}  # les trois sous-ensembles existent
