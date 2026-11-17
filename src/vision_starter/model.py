"""Les modeles : un CNN fait de zero, un ResNet18 pre-entraine, et une fabrique.

La fabrique build_model(config, num_classes) est le seul point d'entree des
scripts : on change de modele dans le YAML, pas dans le code.
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

from dataclasses import replace  # replace : copie une fiche de reglages en changeant quelques champs

import torch  # PyTorch : tenseurs (tableaux de nombres sur CPU ou GPU) et calcul
import torch.nn as nn  # nn : briques de reseaux de neurones (couches, fonctions d'activation...)
from torchvision import models  # architectures connues (ResNet...) avec leurs poids pre-entraines


class ConvBlock(nn.Module):  # bloc de base du CNN simple ; nn.Module = classe mere de toute couche ou modele PyTorch
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2 : la taille spatiale est divisee par 2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:  # cree les couches du bloc (canaux en entree -> en sortie)
        super().__init__()  # initialise la partie nn.Module (obligatoire avant de declarer des couches)
        self.block = nn.Sequential(  # Sequential : enchaine les couches, la sortie de l'une est l'entree de la suivante
            # convolution 3x3 : petits filtres qui detectent des motifs locaux ; padding=1 garde la taille ;
            # bias=False car la BatchNorm qui suit a deja son propre decalage
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # BatchNorm : recentre les valeurs de chaque canal sur le lot, l'apprentissage est plus stable
            nn.ReLU(inplace=True),  # ReLU : met a 0 les valeurs negatives (non-linearite) ; inplace economise de la memoire
            nn.MaxPool2d(2),  # garde le maximum de chaque carre 2x2 : hauteur et largeur divisees par 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # calcul du bloc : appele quand on ecrit bloc(x)
        return self.block(x)  # passe x dans les 4 couches et renvoie le resultat


class SimpleCNN(nn.Module):  # petit reseau de convolution (CNN) ecrit de zero
    """4 blocs de convolution, moyenne spatiale, une couche de classification."""

    # cree le reseau : nombre de classes, largeur de chaque bloc, taux de dropout
    def __init__(self, num_classes: int, channels: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3) -> None:
        super().__init__()  # initialise la partie nn.Module
        blocks, in_channels = [], 3  # liste des blocs a construire ; l'image d'entree a 3 canaux (rouge, vert, bleu)
        for out_channels in channels:  # pour chaque largeur demandee (32, 64, 128, 256)
            blocks.append(ConvBlock(in_channels, out_channels))  # ajoute un bloc qui passe de in_channels a out_channels
            in_channels = out_channels  # la sortie de ce bloc est l'entree du suivant
        self.features = nn.Sequential(*blocks)  # extracteur de caracteristiques : les blocs a la suite (* deplie la liste)
        self.pool = nn.AdaptiveAvgPool2d(1)  # moyenne de chaque canal sur toute l'image : (C, H, W) -> (C, 1, 1)
        # tete : aplatit en vecteur, dropout (eteint au hasard 30 % des valeurs a l'entrainement contre le sur-apprentissage),
        # puis une couche lineaire qui sort un score (logit) par classe
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_channels, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # calcul complet : images (lot, 3, H, W) -> logits (lot, classes)
        return self.classifier(self.pool(self.features(x)))  # caracteristiques -> moyenne -> scores par classe


class ResidualBlock(nn.Module):  # bloc residuel, la brique des ResNet
    """Deux convolutions 3x3 et un raccourci : y = F(x) + x. Le reseau apprend une correction, pas une transformation complete."""

    # cree le bloc ; stride=2 divise la taille de l'image par 2
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()  # initialise la partie nn.Module
        self.body = nn.Sequential(  # le chemin principal F(x)
            # 1re convolution 3x3 (divise la taille par 2 si stride=2)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # normalise chaque canal sur le lot
            nn.ReLU(inplace=True),  # non-linearite
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),  # 2e convolution, taille conservee
            nn.BatchNorm2d(out_channels),  # normalisation ; la ReLU finale est appliquee apres l'addition
        )
        # Le raccourci doit avoir la meme forme que la sortie : conv 1x1 si les canaux ou la taille changent.
        self.shortcut = nn.Identity()  # par defaut le raccourci recopie x tel quel
        if stride != 1 or in_channels != out_channels:  # si la taille ou le nombre de canaux change dans le corps
            self.shortcut = nn.Sequential(  # le raccourci doit faire le meme changement pour qu'on puisse additionner
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),  # conv 1x1 : ajuste canaux et taille
                nn.BatchNorm2d(out_channels),  # puis normalisation, comme dans le corps
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # calcul du bloc : appele quand on ecrit bloc(x)
        return torch.relu(self.body(x) + self.shortcut(x))  # y = ReLU(F(x) + raccourci(x))


class SmallResNet(nn.Module):  # petit ResNet maison (sans poids pre-entraines)
    """Modele custom : tige de convolution, 4 etages residuels (taille / 2 a chaque etage), moyenne spatiale, tete.

    Les largeurs et le dropout viennent de config.extra, pour pouvoir les regler dans le YAML sans toucher au code.
    """

    # cree le reseau : nombre de classes, largeur de chaque etage, taux de dropout
    def __init__(self, num_classes: int, channels: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3) -> None:
        super().__init__()  # initialise la partie nn.Module
        self.stem = nn.Sequential(  # "tige" : 1re convolution qui passe de 3 canaux a channels[0], taille conservee
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False),  # convolution 3x3, taille conservee
            nn.BatchNorm2d(channels[0]),  # normalisation par canal
            nn.ReLU(inplace=True),  # non-linearite
        )
        stages, in_channels = [], channels[0]  # liste des etages a construire ; nombre de canaux en sortie de la tige
        for out_channels in channels:  # pour chaque largeur demandee
            stages.append(ResidualBlock(in_channels, out_channels, stride=2))  # un etage residuel qui divise la taille par 2
            in_channels = out_channels  # la sortie de cet etage est l'entree du suivant
        self.stages = nn.Sequential(*stages)  # les etages a la suite
        self.pool = nn.AdaptiveAvgPool2d(1)  # moyenne de chaque canal sur toute l'image
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_channels, num_classes))  # tete : vecteur -> logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # calcul complet : images -> logits (un score par classe)
        return self.classifier(self.pool(self.stages(self.stem(x))))  # tige -> etages -> moyenne -> tete


# charge un ResNet18 (poids ImageNet si pretrained) et remplace sa derniere couche ; renvoie le modele
def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet18 ImageNet avec une nouvelle tete de classification."""
    # DEFAULT = meilleurs poids publies (appris sur ImageNet, 1,2 million de photos) ; None = poids au hasard
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # nouvelle tete : 512 entrees -> nos classes (au lieu des 1000 d'ImageNet)
    return model  # le modele pret a etre entraine


# gele (trainable=False) ou degele tout le reseau sauf la tete ; ne renvoie rien
def set_backbone_trainable(model: nn.Module, trainable: bool, head_prefix: str = "fc.") -> None:
    """Gele ou degele tout sauf la tete."""
    for name, parameter in model.named_parameters():  # pour chaque poids du modele, avec son nom (ex. "layer1.0.conv1.weight")
        if not name.startswith(head_prefix):  # si ce poids n'appartient pas a la tete
            parameter.requires_grad = trainable  # requires_grad=False : pas de gradient, le poids ne bouge plus (gele)


# degele certains blocs nommes (ex. layer4) ; ne renvoie rien
def unfreeze_layers(model: nn.Module, prefixes: list[str]) -> None:
    """Degele les blocs dont le nom commence par un des prefixes (ex. ["layer4"]) : fine-tuning progressif."""
    for name, parameter in model.named_parameters():  # pour chaque poids du modele
        if any(name.startswith(prefix) for prefix in prefixes):  # si son nom commence par un des prefixes demandes
            parameter.requires_grad = True  # il redevient entrainable


MODEL_REGISTRY = {  # registre : nom du modele dans le YAML -> fonction qui le fabrique (config, nombre de classes)
    "simple_cnn": lambda config, n: SimpleCNN(n),  # CNN simple, reglages par defaut
    "small_resnet": lambda config, n: SmallResNet(  # petit ResNet dont les reglages viennent de config.extra
        n,  # nombre de classes
        tuple(config.extra.get("channels", (32, 64, 128, 256))),  # largeurs des etages lues dans extra (ou valeurs par defaut)
        config.extra.get("dropout", 0.3),  # taux de dropout lu dans extra (ou 0.3)
    ),
    "resnet18": lambda config, n: build_resnet18(n, pretrained=config.pretrained),  # ResNet18 pre-entraine ou non
}


# fabrique le modele choisi dans le YAML ; renvoie le modele (eventuellement partiellement gele)
def build_model(config, num_classes: int) -> nn.Module:
    """Fabrique : le nom dans le YAML choisit la classe."""
    if config.model_name not in MODEL_REGISTRY:  # si le nom n'est pas dans le registre
        raise ValueError(f"model_name inconnu : {config.model_name} (choix : {sorted(MODEL_REGISTRY)})")  # erreur avec les choix possibles
    model = MODEL_REGISTRY[config.model_name](config, num_classes)  # appelle la fonction de fabrication correspondante
    if config.model_name == "resnet18" and config.freeze_backbone:  # transfert d'apprentissage avec backbone gele
        set_backbone_trainable(model, False)  # gele tout sauf la tete
        unfreeze_layers(model, config.unfreeze_layers)  # puis degele les blocs demandes dans le YAML
    return model  # le modele pret


# recree le modele enregistre dans model.pt et y remet ses poids ; renvoie le modele en mode evaluation
def rebuild_model(saved: dict, config) -> nn.Module:
    """L'architecture vient de model.pt (model_name), sans poids pre-entraines : ceux du fichier vont les remplacer."""
    # copie des reglages avec l'architecture notee dans le fichier, et sans telechargement de poids ImageNet
    config = replace(config, model_name=saved["model_name"], pretrained=False)
    model = build_model(config, len(saved["class_names"]))  # meme architecture, autant de sorties que de classes
    model.load_state_dict(saved["model_state"])  # recopie les poids appris
    return model.eval()  # mode evaluation : dropout coupe, BatchNorm figee


# prepare les groupes de poids pour l'optimiseur, chacun avec son taux d'apprentissage ; renvoie une liste de dictionnaires
def param_groups(model: nn.Module, config) -> list[dict]:
    """Deux taux d'apprentissage : la tete neuve apprend vite, le backbone doucement."""
    head = [p for n, p in model.named_parameters() if n.startswith("fc.") and p.requires_grad]  # poids entrainables de la tete
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.") and p.requires_grad]  # poids entrainables du reste
    # 1er groupe : la tete au taux normal ; sans tete "fc." (CNN maison), tout le modele dans ce groupe
    groups = [{"params": head or list(model.parameters()), "lr": config.learning_rate}]
    if head and backbone:  # s'il y a une tete ET des couches pre-entrainees degelees
        groups.append({"params": backbone, "lr": config.backbone_learning_rate})  # elles apprennent avec un pas plus petit
    return groups  # liste a donner a l'optimiseur (ex. torch.optim.AdamW(groups))


# fabrique l'optimiseur choisi dans le YAML (cle optimizer) ; renvoie l'optimiseur
def build_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """adamw (par defaut) ou sgd, avec les groupes de taux d'apprentissage de param_groups."""
    groups = param_groups(model, config)  # un groupe de poids par taux d'apprentissage
    if config.optimizer == "adamw":  # AdamW : le choix par defaut, peu sensible au taux d'apprentissage
        return torch.optim.AdamW(groups, weight_decay=config.weight_decay)  # weight_decay : penalite sur les grands poids
    if config.optimizer == "sgd":  # descente de gradient avec elan : plus de reglages, parfois meilleure en fin d'entrainement
        return torch.optim.SGD(groups, momentum=config.momentum, weight_decay=config.weight_decay)  # momentum : l'elan
    raise ValueError(f"optimizer inconnu : {config.optimizer} (choix : adamw, sgd)")  # faute de frappe dans le YAML
