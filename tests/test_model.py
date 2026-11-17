"""Tests des modeles : forme de sortie, fabrique, gel du backbone."""

import pytest  # pytest : outil qui trouve et lance les tests ; pytest.raises verifie qu'une erreur est levee
import torch  # PyTorch : tenseurs d'entree factices

from vision_starter.config import Config  # fiche de reglages (pour tester la fabrique)

# les modeles et fonctions testes (voir src/vision_starter/model.py)
from vision_starter.model import (
    SimpleCNN,  # le CNN ecrit de zero
    SmallResNet,  # le petit ResNet du cours
    build_model,  # fabrique : nom du YAML -> modele
    build_optimizer,  # fabrique : cle optimizer du YAML -> optimiseur
    build_resnet18,  # ResNet18 avec une nouvelle tete
    rebuild_model,  # recree le modele enregistre dans model.pt
    set_backbone_trainable,  # gele ou degele tout sauf la tete
    unfreeze_layers,  # degele des blocs nommes
)


# test : la sortie a la forme (taille du lot, nombre de classes)
def test_simple_cnn_output_shape_is_batch_by_classes():
    model = SimpleCNN(num_classes=5)  # CNN a 5 classes
    out = model(torch.zeros(4, 3, 64, 64))  # lot de 4 images noires 64x64
    assert out.shape == (4, 5)  # 4 images, 5 scores chacune


# test : grace a la moyenne spatiale (AdaptiveAvgPool), le CNN accepte plusieurs tailles d'image
def test_simple_cnn_accepts_any_image_size():
    model = SimpleCNN(num_classes=3)  # CNN a 3 classes
    assert model(torch.zeros(1, 3, 96, 96)).shape == (1, 3)  # image 96x96 -> 3 scores
    assert model(torch.zeros(1, 3, 128, 128)).shape == (1, 3)  # image 128x128 -> 3 scores aussi


# test : un nom de modele inconnu provoque une erreur claire
def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):  # le bloc doit lever une ValueError, sinon le test echoue
        build_model(Config(model_name="nope"), num_classes=2)  # "nope" n'est pas dans le registre


# test : apres gel du backbone, seuls les poids de la tete (fc) restent entrainables
def test_frozen_resnet_trains_only_the_head():
    model = build_resnet18(num_classes=4, pretrained=False)  # ResNet18 sans poids pre-entraines (pas de telechargement)
    set_backbone_trainable(model, False)  # gele tout sauf la tete
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]  # noms des poids encore entrainables
    assert trainable == ["fc.weight", "fc.bias"]  # seulement les poids et le biais de la tete


# test : SmallResNet donne la bonne forme, et la fabrique le construit avec les reglages de extra
def test_small_resnet_shape_and_registry():
    # construit directement, mode evaluation, lot de 2 images 64x64 -> (2, 3)
    assert SmallResNet(num_classes=3, channels=(8, 16)).eval()(torch.zeros(2, 3, 64, 64)).shape == (2, 3)
    # par la fabrique, avec largeurs et dropout fournis dans extra
    model = build_model(Config(model_name="small_resnet", extra={"channels": [8, 16], "dropout": 0.0}), num_classes=4)
    assert model(torch.zeros(1, 3, 32, 32)).shape == (1, 4)  # 1 image 32x32 -> 4 scores


# test : degeler layer4 le rend entrainable en plus de la tete
def test_unfreeze_layers_adds_layer4_to_trainable():
    model = build_resnet18(num_classes=4, pretrained=False)  # ResNet18 sans poids pre-entraines
    set_backbone_trainable(model, False)  # gele tout sauf la tete
    unfreeze_layers(model, ["layer4"])  # degele le dernier bloc
    trainable = {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}  # premier morceau du nom des poids entrainables
    assert trainable == {"layer4", "fc"}  # exactement layer4 et la tete


# test : la cle optimizer du YAML choisit l'optimiseur, et une valeur inconnue est refusee
def test_build_optimizer_follows_config():
    model = SimpleCNN(num_classes=3)  # un petit modele suffit : on ne l'entraine pas
    assert isinstance(build_optimizer(model, Config()), torch.optim.AdamW)  # valeur par defaut : adamw
    assert isinstance(build_optimizer(model, Config(optimizer="sgd")), torch.optim.SGD)  # sgd demande dans la config
    with pytest.raises(ValueError):  # le bloc doit lever une ValueError, sinon le test echoue
        build_optimizer(model, Config(optimizer="adam_w"))  # faute de frappe : refusee tout de suite


# test : rebuild_model recree l'architecture notee dans le fichier, remet les poids, et rend le modele en mode evaluation
def test_rebuild_model_restores_architecture_and_weights():
    # Arrange : le contenu d'un model.pt, pour un CNN a 4 classes
    trained = SimpleCNN(num_classes=4)  # joue le role du modele entraine
    saved = {"model_state": trained.state_dict(), "model_name": "simple_cnn", "class_names": ["a", "b", "c", "d"]}  # comme train.py

    # Act : la config demande un autre modele, pre-entraine ; c'est le fichier qui doit decider
    model = rebuild_model(saved, Config(model_name="resnet18", pretrained=True))  # ne doit rien telecharger

    # Assert
    assert isinstance(model, SimpleCNN)  # l'architecture vient de model.pt, pas de la config
    assert not model.training  # mode evaluation
    x = torch.rand(2, 3, 64, 64)  # deux images au hasard
    assert torch.equal(model(x), trained.eval()(x))  # memes poids : memes sorties, au bit pres
