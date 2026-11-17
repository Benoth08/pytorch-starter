"""Tests du module segmentation : forme du U-Net, perte Dice + BCE, IoU cumule, choix du seuil."""

import torch  # PyTorch : tenseurs factices

# les classes et fonctions testees (voir src/vision_starter/segmentation.py)
from vision_starter.segmentation import DiceBCELoss, ResNetUNet, choose_threshold, confusion_pixels, dice_loss, iou_dice


# test : le U-Net sort un logit par pixel, et ses poids se repartissent bien entre encodeur et decodeur
def test_unet_outputs_one_logit_per_pixel():
    model = ResNetUNet(pretrained=False).eval()  # U-Net sans poids pre-entraines (pas de telechargement), mode evaluation
    with torch.no_grad():  # pas de gradient : simple prediction
        out = model(torch.zeros(2, 3, 64, 64))  # 2 images noires 64x64
    assert out.shape == (2, 1, 64, 64)  # une carte de logits de meme taille que l'image
    # chaque poids est dans exactement un des deux groupes (aucun oubli, aucun doublon)
    assert len(model.encoder_parameters()) + len(model.decoder_parameters()) == len(list(model.parameters()))


# test : perte Dice ~ 0 si la prediction est parfaite, ~ 1 si on predit "fond" partout
def test_dice_loss_is_zero_for_perfect_prediction_and_high_for_background_only():
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()  # masque aleatoire de 0 et de 1
    perfect_logits = target * 20 - 10  # sigmoide ~ 1 sur le defaut, ~ 0 ailleurs
    assert dice_loss(perfect_logits, target).item() < 0.02  # presque nulle
    assert dice_loss(torch.full_like(target, -10), target).item() > 0.9  # logits -10 partout = "aucun defaut" : perte proche de 1
    assert DiceBCELoss()(perfect_logits, target).item() < 0.03  # la perte combinee est aussi presque nulle


# test : formules de l'IoU et du Dice, et pas de division par zero
def test_iou_dice_from_pixel_counts():
    iou, dice = iou_dice(intersection=50, predicted=100, truth=100)  # 50 pixels communs sur 100 predits et 100 vrais
    assert abs(iou - 1 / 3) < 1e-9 and abs(dice - 0.5) < 1e-9  # IoU = 50 / 150, Dice = 100 / 200
    assert iou_dice(0, 0, 0) == (0.0, 0.0)  # rien a detecter, rien detecte : pas de division par zero


# test : le seuil choisi est celui qui donne la meilleure IoU
def test_choose_threshold_picks_best_iou_on_validation():
    target = torch.zeros(1, 1, 4, 4)  # masque 4x4 vide
    target[0, 0, :2] = 1  # moitie haute = defaut
    probs = torch.zeros(1, 1, 4, 4)  # probabilites predites, 0 par defaut
    probs[0, 0, :2] = 0.6  # le defaut a 0,6 ; un seuil de 0,7 ne detecte rien
    probs[0, 0, 2] = 0.45  # une ligne de fond a 0,45 : un seuil de 0,4 la prend a tort
    # probabilites -> logits (inverse de la sigmoide) ; clamp evite l'infini pour 0 et 1
    logits = torch.logit(probs.clamp(1e-4, 1 - 1e-4))
    best, sweep = choose_threshold(logits, target, [0.4, 0.5, 0.7])  # essaie 3 seuils
    assert best == 0.5 and sweep[0.5] == 1.0 and sweep[0.7] == 0.0  # 0,5 est parfait, 0,7 ne trouve rien
    assert confusion_pixels(logits, target, 0.5) == (8, 8, 8)  # au seuil 0,5 : 8 communs, 8 predits, 8 vrais
