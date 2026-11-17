# %% [markdown]
# # References : pertes et metriques, hyperparametres, gel des poids
#
# Trois complements de reference, avec du code execute : les pertes et les hyperparametres
# (etape 9 du cours), les metriques (etape 11), le gel des poids (etape 8). Ce notebook reutilise le
# modele entraine par 01_projet_complet.py (outputs/notebook/cnn_final.pt).

# %% imports
from __future__ import annotations  # permet d'ecrire les types (tuple[int, int]) sans souci de version de Python

import copy  # copie complete d'un objet (ici : un modele entier)
import math  # fonctions mathematiques de base (log...)
import random  # hasard de Python, fixe plus bas
import time  # mesure du temps ecoule
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import matplotlib.pyplot as plt  # matplotlib : graphiques
import numpy as np  # numpy : calcul sur des tableaux de nombres
import pandas as pd  # pandas : tableaux de donnees (DataFrame)
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU) et apprentissage profond
import torch.nn as nn  # nn : briques de reseau (couches, fonctions de perte)
import torch.nn.functional as F  # F : les memes operations sous forme de fonctions simples (pertes, redimensionnement)
from PIL import Image  # PIL (Pillow) : ouvrir les fichiers images
from sklearn.metrics import (  # scikit-learn : metriques d'evaluation toutes faites
    average_precision_score,  # AP : aire sous la courbe precision-rappel
    balanced_accuracy_score,  # exactitude equilibree : moyenne du rappel de chaque classe
    f1_score,  # F1 : combine precision et rappel
    log_loss,  # perte logarithmique (la CrossEntropy) calculee sur des probabilites
    precision_recall_curve,  # points de la courbe precision-rappel pour tous les seuils
    precision_score,  # precision : parmi les images annoncees de la classe, part de justes
    recall_score,  # rappel : parmi les vraies images de la classe, part retrouvee
    roc_auc_score,  # ROC AUC : capacite a classer les positifs avant les negatifs (1 = parfait, 0.5 = hasard)
    top_k_accuracy_score,  # top-k : la bonne classe est-elle parmi les k plus probables ?
)
from torch.utils.data import DataLoader, Dataset  # Dataset : une image a la fois ; DataLoader : les regroupe en lots
from torchvision import models  # modeles connus (ResNet...) avec poids pre-entraines
from torchvision.transforms import v2  # transformations d'images (redimensionner, normaliser...)

OUTPUT_DIR = Path("outputs/notebook")  # dossier ou le notebook 01 a range ses resultats
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # carte graphique (GPU) si elle existe, sinon CPU


def seed_everything(seed: int) -> None:  # fixe toutes les sources de hasard ; ne renvoie rien
    random.seed(seed)  # hasard de Python
    np.random.seed(seed)  # hasard de numpy
    torch.manual_seed(seed)  # hasard de PyTorch sur CPU
    torch.cuda.manual_seed_all(seed)  # hasard de PyTorch sur GPU


seed_everything(42)  # resultats reproductibles

# %% losses_classification
# Etape 9 - Perte et optimiseur : les pertes de classification, sur un petit exemple a la main.
logits = torch.tensor([[2.0, 0.5, -1.0], [0.2, 0.1, 0.0]])  # 2 images, 3 classes, sorties brutes du modele (logits = scores non normalises)
targets = torch.tensor([0, 2])  # etiquettes entieres : la vraie classe de chaque image

ce = nn.CrossEntropyLoss()(logits, targets)  # = -log(softmax)[classe vraie], moyenne sur le lot ; softmax = scores -> probabilites
manual = -torch.log_softmax(logits, dim=1)[torch.arange(2), targets].mean()  # meme calcul a la main : -log de la proba de la vraie classe
print(f"CrossEntropy = {ce:.4f} (a la main : {manual:.4f})")  # les deux valeurs doivent etre egales
print(f"  image 1, sure et juste  : -log p = {-torch.log_softmax(logits, 1)[0, 0]:.3f}")  # bonne reponse sure : perte faible
# image hesitante (3 scores proches) : perte proche de ln 3, celle d'un tirage au hasard entre 3 classes
print(f"  image 2, hesitante      : -log p = {-torch.log_softmax(logits, 1)[1, 2]:.3f}  (proche de ln 3 = {math.log(3):.3f})")
print(  # label smoothing : la cible n'est plus 100 % sur la vraie classe, un peu est reparti sur les autres
    f"label smoothing 0.1     : {nn.CrossEntropyLoss(label_smoothing=0.1)(logits, targets):.4f}  (cibles 0.933 / 0.033 / 0.033 : le modele ne force pas p = 1)"
)
weights = torch.tensor([1.0, 1.0, 3.0])  # la classe 2 est rare : ses erreurs comptent 3 fois
print(f"ponderee par classe     : {nn.CrossEntropyLoss(weight=weights)(logits, targets):.4f}")  # perte avec ces poids par classe

# %% losses_other
# Binaire et multi-etiquettes : une sigmoide par sortie, BCEWithLogits (jamais Sigmoid + BCELoss separes : instable).
logit = torch.tensor([[1.5], [-0.5], [0.0]])  # 3 exemples, 1 sortie brute chacun (sigmoide = score -> probabilite entre 0 et 1)
label = torch.tensor([[1.0], [0.0], [1.0]])  # verites : 1 = positif, 0 = negatif
print(f"BCEWithLogits           : {nn.BCEWithLogitsLoss()(logit, label):.4f}")  # perte binaire : sigmoide + entropie croisee en une fois
print(f"  pos_weight=4 (positifs rares) : {nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]))(logit, label):.4f}")  # positifs x4


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:  # renvoie la focal loss moyenne du lot
    """Focal loss : la CrossEntropy attenuee pour les exemples deja bien classes (p proche de 1)."""
    log_p = torch.log_softmax(logits, dim=1)[torch.arange(len(targets)), targets]  # log de la proba de la vraie classe, par image
    p = log_p.exp()  # la proba elle-meme
    return (-((1 - p) ** gamma) * log_p).mean()  # (1 - p)^gamma : proche de 0 si l'image est facile, donc elle compte peu


print(f"focal (gamma=2)         : {focal_loss(logits, targets):.4f}  (l'image 1, facile, ne pese presque plus rien)")  # a comparer a ce

# Regression (un angle, une distance, un comptage) : MSE, L1, Huber.
pred, truth = torch.tensor([2.0, 4.0, 10.0]), torch.tensor([2.5, 4.0, 4.0])  # la 3e valeur est aberrante
# MSE : moyenne des ecarts au carre ; L1 : moyenne des ecarts absolus ; Huber : au carre si petit ecart, lineaire au-dela de delta
print(
    f"MSE {F.mse_loss(pred, truth):.3f} | L1 {F.l1_loss(pred, truth):.3f} | Huber {F.huber_loss(pred, truth, delta=1.0):.3f}  (Huber : l'aberration pese lineairement, pas au carre)"
)


# Segmentation : CrossEntropy par pixel + Dice (recouvrement), qui resiste au desequilibre fond / objet.
def dice_loss(logits: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:  # renvoie 1 - Dice (0 = recouvrement parfait)
    """Dice binaire : 1 - 2|A inter B| / (|A| + |B|), sur les probabilites."""
    p = torch.sigmoid(logits)  # probabilite "objet" de chaque pixel
    inter = (p * mask).sum()  # recouvrement entre prediction et masque vrai
    return 1 - (2 * inter + eps) / (p.sum() + mask.sum() + eps)  # eps evite une division par 0 si tout est vide


mask = torch.zeros(1, 1, 16, 16)  # masque vrai : image 16 x 16, tout a 0 (fond)
mask[..., 4:8, 4:8] = 1  # un objet de 16 pixels sur 256 : le fond domine
good = (mask * 8 - 4).clone()  # logits confiants au bon endroit (+4 sur l'objet, -4 sur le fond)
print(  # compare : bonne prediction, puis prediction "tout fond" jugee par Dice et par BCE
    f"Dice (prediction correcte) {dice_loss(good, mask):.3f} | Dice (tout fond) {dice_loss(torch.full_like(mask, -4.0), mask):.3f} | BCE (tout fond) {F.binary_cross_entropy_with_logits(torch.full_like(mask, -4.0), mask):.3f}"
)


# %% metrics_setup
# Etape 11 - Evaluer : les metriques, sur de vraies predictions (le CNN du notebook 01, sur le test).
class ImageDataset(Dataset):  # jeu de donnees PyTorch : renvoie une image transformee et son numero de classe
    def __init__(self, frame, class_to_idx, transform=None):  # recoit le tableau des images, les numeros de classe et la transformation
        # garde la liste des chemins, la liste des numeros de classe et la transformation
        self.paths, self.targets, self.transform = frame["path"].tolist(), [class_to_idx[x] for x in frame["label"]], transform

    def __len__(self):  # nombre d'images du jeu
        return len(self.paths)  # une image par chemin

    def __getitem__(self, i):  # renvoie l'image numero i et sa classe
        image = Image.open(self.paths[i]).convert("RGB")  # ouvre le fichier, force 3 canaux couleur
        return (self.transform(image) if self.transform else image), self.targets[i]  # image transformee + classe


class ConvBlock(nn.Module):  # bloc de base : convolution, normalisation, activation, reduction de taille
    def __init__(self, cin, cout):  # cin : canaux en entree, cout : en sortie
        super().__init__()  # initialise la partie nn.Module
        self.block = nn.Sequential(  # enchaine les couches dans l'ordre
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),  # convolution 3x3 : filtres qui detectent des motifs locaux
            nn.BatchNorm2d(cout),  # BatchNorm : remet les valeurs a une echelle stable
            nn.ReLU(inplace=True),  # ReLU : garde les valeurs positives, met 0 sinon
            nn.MaxPool2d(2),  # MaxPool : divise hauteur et largeur par 2
        )

    def forward(self, x):  # passage avant du bloc
        return self.block(x)  # applique les couches dans l'ordre


class SimpleCNN(nn.Module):  # le meme petit CNN que dans le notebook 01 (necessaire pour recharger ses poids)
    def __init__(self, num_classes, channels=(32, 64, 128, 256), dropout=0.3):  # nb de classes, canaux des blocs, taux de dropout
        super().__init__()  # initialise la partie nn.Module
        blocks, cin = [], 3  # blocs a construire ; l'image a 3 canaux
        for cout in channels:  # pour chaque nombre de canaux voulu
            blocks.append(ConvBlock(cin, cout))  # ajoute un bloc cin -> cout
            cin = cout  # la sortie devient l'entree du bloc suivant
        self.features = nn.Sequential(*blocks)  # les blocs a la suite : extraction des caracteristiques
        self.pool = nn.AdaptiveAvgPool2d(1)  # moyenne de chaque canal sur toute l'image
        # aplatit, dropout (eteint des valeurs au hasard contre le par-coeur), puis un score par classe
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, num_classes))

    def forward(self, x):  # passage avant : image -> logits (un score par classe)
        return self.classifier(self.pool(self.features(x)))  # caracteristiques -> moyenne -> classifieur


saved = torch.load(OUTPUT_DIR / "cnn_final.pt", map_location=DEVICE, weights_only=True)  # relit le fichier sauve par le notebook 01
class_names = saved["class_names"]  # noms des classes, dans l'ordre du modele
class_to_idx = {c: i for i, c in enumerate(class_names)}  # nom -> numero
model = SimpleCNN(len(class_names))  # recree le reseau vide
model.load_state_dict(saved["model_state"])  # y charge les poids appris
model.to(DEVICE).eval()  # sur le bon appareil, en mode evaluation (dropout coupe)
SIZE = (saved["image_size"], saved["image_size"])  # taille d'image utilisee a l'entrainement
eval_transform = v2.Compose(  # meme pretraitement qu'a l'entrainement, sans augmentation
    # tenseur image, redimensionnement, valeurs 0-1, normalisation avec la moyenne et l'ecart-type sauvegardes
    [v2.ToImage(), v2.Resize(SIZE, antialias=True), v2.ToDtype(torch.float32, scale=True), v2.Normalize(saved["mean"], saved["std"])]
)
catalog = pd.read_csv(OUTPUT_DIR / "split.csv")  # le catalogue avec le decoupage train / val / test
test_df = catalog[catalog["split"] == "test"].reset_index(drop=True)  # seulement les images de test, renumerotees a partir de 0
test_loader = DataLoader(ImageDataset(test_df, class_to_idx, eval_transform), batch_size=32)  # lots de 32 images de test

with torch.no_grad():  # pas de gradient : on ne fait que predire
    # pour chaque lot : softmax -> probabilites, ramenees sur CPU ; puis tous les lots colles en un tableau numpy
    probabilities = torch.cat([torch.softmax(model(x.to(DEVICE)), dim=1).cpu() for x, _ in test_loader]).numpy()
y_true = np.array([class_to_idx[x] for x in test_df["label"]])  # vraies classes, en numeros
y_pred = probabilities.argmax(axis=1)  # classe predite = la plus probable
print("probabilites :", probabilities.shape, "| predictions :", y_pred.shape)  # (nb images, nb classes) et (nb images,)

# %% metrics_classification
# Les metriques de classification, calculees avec scikit-learn a partir de y_true, y_pred et des probabilites.
rows = {  # nom de la metrique -> valeur
    "accuracy": (y_pred == y_true).mean(),  # exactitude : part de bonnes reponses
    "balanced accuracy": balanced_accuracy_score(y_true, y_pred),  # moyenne des rappels par classe : insensible au desequilibre
    "precision macro": precision_score(y_true, y_pred, average="macro"),  # precision de chaque classe, puis moyenne simple
    "rappel macro": recall_score(y_true, y_pred, average="macro"),  # rappel de chaque classe, puis moyenne simple
    "F1 macro": f1_score(y_true, y_pred, average="macro"),  # F1 de chaque classe, puis moyenne simple (chaque classe pese pareil)
    "F1 micro (= accuracy)": f1_score(y_true, y_pred, average="micro"),  # F1 sur toutes les images melangees : egal a l'accuracy
    "F1 pondere": f1_score(y_true, y_pred, average="weighted"),  # moyenne des F1 ponderee par la taille de chaque classe
    "top-2 accuracy": top_k_accuracy_score(y_true, probabilities, k=2),  # la vraie classe est dans les 2 plus probables
    "log loss (perte CE sur le test)": log_loss(y_true, probabilities),  # la CrossEntropy calculee sur le test
    # ROC AUC calculee pour chaque classe contre toutes les autres ("ovr"), puis moyenne simple
    "ROC AUC (un contre tous, macro)": roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"),
}
print(pd.Series(rows).round(4).to_string())  # affiche le tableau des metriques, arrondies a 4 decimales

# %% metrics_imbalance
# Pourquoi l'accuracy trompe : on simule un test desequilibre (45 "rolled" contre 3 de chaque autre classe).
rng = np.random.default_rng(0)  # generateur de hasard (pas utilise dans cette cellule)
# indices gardes : les 45 premieres images "rolled" et les 3 premieres de chaque autre classe
keep = np.concatenate([np.where(y_true == i)[0][: (45 if name == "rolled" else 3)] for i, name in enumerate(class_names)])
y_true_imb, y_pred_imb = y_true[keep], y_pred[keep]  # verites et predictions du test desequilibre
majority = np.full_like(y_true_imb, class_to_idx["rolled"])  # un "modele" qui repond toujours la classe majoritaire
for name, pred in (("notre CNN", y_pred_imb), ("toujours 'rolled'", majority)):  # pour chacun des deux "modeles"
    print(  # accuracy flatteuse pour le modele idiot, balanced accuracy et F1 macro le demasquent
        f"{name:18s} accuracy {(pred == y_true_imb).mean():.3f} | balanced accuracy {balanced_accuracy_score(y_true_imb, pred):.3f}"
        f" | F1 macro {f1_score(y_true_imb, pred, average='macro'):.3f}"
    )

# %% metrics_threshold
# Seuil de decision et courbe precision-rappel : pour UNE classe (inclusion), vue comme un probleme binaire.
target = class_to_idx["inclusion"]  # numero de la classe etudiee
is_inclusion = (y_true == target).astype(int)  # verite binaire : 1 si l'image est une inclusion, 0 sinon
score = probabilities[:, target]  # probabilite "inclusion" donnee par le modele
precision, recall, thresholds = precision_recall_curve(is_inclusion, score)  # precision et rappel pour chaque seuil possible
print(  # AP : aire sous la courbe precision-rappel ; ROC AUC : bonne separation inclusion / reste
    f"AP (aire sous precision-rappel) = {average_precision_score(is_inclusion, score):.3f} | ROC AUC = {roc_auc_score(is_inclusion, score):.3f}"
)
for t in (0.1, 0.5, 0.9):  # pour trois seuils de decision
    decided = (score >= t).astype(int)  # decision : inclusion si la probabilite depasse le seuil
    # seuil bas : on rate peu d'inclusions (rappel haut) ; seuil haut : moins de fausses alertes (precision haute)
    print(f"  seuil {t:.1f} : precision {precision_score(is_inclusion, decided):.3f} | rappel {recall_score(is_inclusion, decided):.3f}")
fig, ax = plt.subplots(figsize=(5, 4))  # une figure, un graphique
ax.plot(recall, precision)  # courbe : precision en fonction du rappel
ax.set_xlabel("rappel")  # nom de l'axe horizontal
ax.set_ylabel("precision")  # nom de l'axe vertical
ax.set_title("inclusion : precision-rappel selon le seuil")  # titre
ax.grid(alpha=0.3)  # quadrillage leger
plt.tight_layout()  # ajuste les marges
plt.show()  # affiche


# %% lr_range_test
# Etape 9 - Perte et optimiseur (hyperparametres) : le test de plage de taux d'apprentissage. On augmente lr de facon exponentielle, un lot a la fois.
def build_loader_64(frame, train):  # cree un DataLoader d'images 64 x 64 (petites : essais rapides)
    transform = v2.Compose(  # pretraitement
        [  # liste des etapes
            v2.ToImage(),  # image PIL -> tenseur image
            v2.Resize((64, 64), antialias=True),  # images 64 x 64 pixels
            v2.ToDtype(torch.float32, scale=True),  # valeurs 0-255 -> 0-1
            v2.Normalize(saved["mean"], saved["std"]),  # normalisation par canal
        ]
    )
    return DataLoader(ImageDataset(frame, class_to_idx, transform), batch_size=32, shuffle=train, drop_last=train)  # melange si train


train_df, val_df = catalog[catalog["split"] == "train"], catalog[catalog["split"] == "val"]  # images d'entrainement et de validation
train_64, val_64 = build_loader_64(train_df, True), build_loader_64(val_df, False)  # leurs DataLoaders en 64 px

seed_everything(42)  # meme point de depart a chaque execution
probe = SimpleCNN(len(class_names)).to(DEVICE)  # un reseau neuf, juste pour le test
optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-5)  # optimiseur (met a jour les poids) ; lr change a chaque pas plus bas
criterion = nn.CrossEntropyLoss()  # perte de classification
lrs, losses = [], []  # taux essayes et pertes obtenues
lr_min, lr_max, n_steps = 1e-5, 1.0, len(train_64)  # une epoque : 39 lots, de 1e-5 a 1
probe.train()  # mode entrainement
for step, (images, targets) in enumerate(train_64):  # pour chaque lot (step : son numero)
    lr = lr_min * (lr_max / lr_min) ** (step / (n_steps - 1))  # lr monte de facon exponentielle de lr_min a lr_max
    for group in optimizer.param_groups:  # pour chaque groupe de parametres de l'optimiseur
        group["lr"] = lr  # impose le nouveau taux
    optimizer.zero_grad()  # gradients a zero
    loss = criterion(probe(images.to(DEVICE)), targets.to(DEVICE))  # passage avant et perte
    loss.backward()  # gradients
    optimizer.step()  # mise a jour des poids avec ce lr
    lrs.append(lr)  # note le taux
    losses.append(loss.item())  # note la perte
best = int(np.argmin(losses))  # pas ou la perte est la plus basse
# affiche le lr de la perte minimale et le conseil : prendre 3 a 10 fois plus petit (avant que la perte n'explose)
print(f"perte minimale {losses[best]:.3f} a lr = {lrs[best]:.1e} ; un bon choix est 3 a 10 fois plus petit, vers {lrs[best] / 5:.0e}")
fig, ax = plt.subplots(figsize=(6, 3.5))  # un graphique
ax.plot(lrs, losses)  # perte en fonction du taux
ax.set_xscale("log")  # axe horizontal logarithmique (les lr vont de 1e-5 a 1)
ax.set_xlabel("taux d'apprentissage")  # nom de l'axe horizontal
ax.set_ylabel("perte du lot")  # nom de l'axe vertical
ax.set_title("test de plage : la perte descend, puis explose")  # titre
ax.grid(alpha=0.3)  # quadrillage leger
plt.tight_layout()  # ajuste les marges
plt.show()  # affiche


# %% random_search
# Recherche aleatoire sur (lr, weight_decay, dropout) : 4 tirages, 3 epoques chacun, images 64 px, meme graine.
def train_short(lr: float, weight_decay: float, dropout: float, epochs: int = 3) -> dict:  # entrainement court ; renvoie reglages + scores
    seed_everything(42)  # meme depart pour chaque essai : seule la configuration change
    net = SimpleCNN(len(class_names), dropout=dropout).to(DEVICE)  # reseau neuf avec ce dropout
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)  # optimiseur avec ce lr et ce weight_decay
    for _ in range(epochs):  # pour chaque epoque
        net.train()  # mode entrainement
        for images, targets in train_64:  # pour chaque lot
            opt.zero_grad()  # gradients a zero
            criterion(net(images.to(DEVICE)), targets.to(DEVICE)).backward()  # passage avant, perte, gradients
            opt.step()  # mise a jour des poids
    net.eval()  # mode evaluation
    with torch.no_grad():  # pas de gradient pour l'evaluation
        outputs = [(net(x.to(DEVICE)).cpu(), y) for x, y in val_64]  # (logits, verites) de chaque lot de validation
    logits_val = torch.cat([o for o, _ in outputs])  # tous les logits en un tenseur
    y_val = torch.cat([y for _, y in outputs])  # toutes les verites en un tenseur
    return {  # resultat de l'essai
        "lr": lr,  # taux d'apprentissage essaye
        "weight_decay": weight_decay,  # weight decay essaye (pousse les poids vers 0)
        "dropout": dropout,  # dropout essaye
        "val_loss": criterion(logits_val, y_val).item(),  # perte de validation
        "val_acc": (logits_val.argmax(1) == y_val).float().mean().item(),  # exactitude de validation
    }


rng = np.random.default_rng(1)  # generateur de hasard pour tirer les configurations
trials = []  # resultats des essais
start = time.time()  # heure de depart, pour mesurer la duree
for _ in range(4):  # 4 essais
    config = {  # une configuration tiree au hasard
        "lr": float(10 ** rng.uniform(-4, -2)),  # lr entre 1e-4 et 1e-2, tire sur une echelle logarithmique
        "weight_decay": float(10 ** rng.uniform(-5, -2)),  # weight_decay entre 1e-5 et 1e-2
        "dropout": float(rng.choice([0.0, 0.3, 0.5])),  # dropout parmi 3 valeurs
    }
    trials.append(train_short(**config))  # entraine avec cette configuration et garde le resultat
    print(  # affiche la configuration et ses scores de validation
        f"lr {config['lr']:.1e} | wd {config['weight_decay']:.1e} | dropout {config['dropout']:.1f} -> val perte {trials[-1]['val_loss']:.3f} acc {trials[-1]['val_acc']:.3f}"
    )
print(f"4 essais en {time.time() - start:.0f} s ; meilleur :", min(trials, key=lambda t: t["val_loss"]))  # meilleur = plus petite val_loss

# %% freeze_basics
# Etape 8 - Modele (gel des poids) : geler = requires_grad=False. Un parametre gele ne recoit pas de gradient et ne bouge pas.
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # ResNet18 pre-entraine sur ImageNet (1000 classes)
resnet.fc = nn.Linear(resnet.fc.in_features, len(class_names))  # remplace la derniere couche (tete) : une sortie par classe a nous


def count(model: nn.Module) -> tuple[int, int]:  # renvoie (nb de parametres entrainables, nb total de parametres)
    # numel() : nombre de valeurs d'un parametre ; premiere somme sur les entrainables, seconde sur tous
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())


def freeze(model: nn.Module, prefixes: tuple[str, ...]) -> None:  # gele des parties du modele ; ne renvoie rien
    """Gele tous les parametres dont le nom commence par un des prefixes (ex. 'conv1', 'layer1.')."""
    for name, parameter in model.named_parameters():  # pour chaque parametre et son nom (ex. "layer1.0.conv1.weight")
        if name.startswith(prefixes):  # si le nom commence par un des prefixes
            parameter.requires_grad = False  # gele : plus de gradient, plus de mise a jour


print("tout entrainable          :", count(resnet))  # au depart, tout est entrainable
freeze(resnet, ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"))  # tout sauf fc
print("backbone gele, tete seule :", count(resnet))  # backbone = partie qui extrait les caracteristiques ; seule la tete reste
for name, parameter in list(resnet.named_parameters())[:3] + list(resnet.named_parameters())[-2:]:  # 3 premiers et 2 derniers parametres
    print(f"  {name:22s} requires_grad={parameter.requires_grad}")  # leur nom et s'ils sont entrainables

# %% freeze_verify
# La preuve : apres une etape d'optimisation, les poids geles sont identiques, la tete a bouge.
images, targets = next(iter(train_64))  # un lot d'images 64 px
images = F.interpolate(images, size=128).to(DEVICE)  # ResNet veut un peu plus que 64 px : agrandit a 128 x 128
resnet.to(DEVICE)  # le modele sur le bon appareil
before = {name: p.detach().clone() for name, p in resnet.named_parameters()}  # copie de tous les poids avant l'etape
optimizer = torch.optim.AdamW([p for p in resnet.parameters() if p.requires_grad], lr=1e-3)  # seulement les entrainables
resnet.train()  # mode entrainement
optimizer.zero_grad()  # gradients a zero
criterion(resnet(images), targets.to(DEVICE)).backward()  # passage avant, perte, gradients
optimizer.step()  # une mise a jour
changed = [name for name, p in resnet.named_parameters() if not torch.equal(before[name], p.detach())]  # noms des poids qui ont bouge
print("parametres modifies par l'etape :", changed)  # attendu : seulement fc.weight et fc.bias
print("gradient d'un poids gele :", resnet.layer4[1].conv2.weight.grad)  # None : jamais calcule

# %% freeze_batchnorm
# Ce que le gel ne gele PAS : les statistiques glissantes des BatchNorm, mises a jour en mode train meme sans gradient.
bn = resnet.layer4[1].bn2  # une couche BatchNorm du dernier bloc (gelee)
mean_before = bn.running_mean.clone()  # copie de sa moyenne glissante (statistique apprise sur les donnees)
resnet.train()  # mode train : les BatchNorm mettent a jour leurs statistiques a chaque passage
with torch.no_grad():  # sans gradient
    resnet(images)  # un simple passage avant en mode train
print("running_mean d'une BatchNorm gelee a change :", not torch.equal(mean_before, bn.running_mean))  # True : elle a bouge


def set_train_mode(model: nn.Module) -> None:  # met le modele en mode train en protegeant les BatchNorm gelees ; ne renvoie rien
    """Mode train, sauf les BatchNorm dont les poids sont geles : elles restent en eval (statistiques figees)."""
    model.train()  # tout en mode train
    for module in model.modules():  # pour chaque couche du modele (y compris celles dans les blocs)
        if isinstance(module, nn.BatchNorm2d) and not module.weight.requires_grad:  # si c'est une BatchNorm gelee
            module.eval()  # mode eval : ses statistiques ne bougent plus


mean_before = bn.running_mean.clone()  # nouvelle copie de la moyenne glissante
set_train_mode(resnet)  # mode train "protege"
with torch.no_grad():  # sans gradient
    resnet(images)  # passage avant
print("avec set_train_mode, running_mean a change     :", not torch.equal(mean_before, bn.running_mean))  # False : figee

# %% freeze_progressive
# Degel progressif avec des taux d'apprentissage differents par groupe : la tete vite, le dernier bloc doucement.
for parameter in resnet.layer4.parameters():  # pour chaque parametre du dernier bloc
    parameter.requires_grad = True  # le degele
optimizer = torch.optim.AdamW(  # optimiseur avec deux groupes de parametres
    [  # liste des groupes, chacun avec son taux
        {"params": resnet.fc.parameters(), "lr": 1e-3},  # la tete : taux normal
        {"params": resnet.layer4.parameters(), "lr": 1e-4},  # dernier bloc : taux 10 fois plus petit, pour garder le pre-entrainement
    ],
    weight_decay=1e-4,  # weight decay commun aux deux groupes
)
print("entrainables apres degel de layer4 :", count(resnet))  # plus de parametres entrainables qu'avant
for i, group in enumerate(optimizer.param_groups):  # pour chaque groupe de l'optimiseur
    print(f"  groupe {i} : {sum(p.numel() for p in group['params']):8d} parametres, lr {group['lr']:.0e}")  # taille et taux du groupe

# Variante : tout degeler avec un lr qui decroit avec la profondeur (discriminative learning rates).
groups, lr = [], 1e-3  # groupes a construire ; taux de depart (pour la tete)
for name in ("fc", "layer4", "layer3", "layer2", "layer1"):  # de la tete vers l'entree du reseau
    groups.append({"params": getattr(resnet, name).parameters(), "lr": lr})  # un groupe par bloc, avec le taux courant
    lr /= 3  # le bloc suivant (plus proche de l'entree) apprend 3 fois moins vite
print("lr par bloc, de la tete vers l'entree :", [f"{g['lr']:.1e}" for g in groups])  # affiche les taux

# %% freeze_vs_nograd
# A ne pas confondre : torch.no_grad() (pas de graphe du tout, pour l'inference) et le gel (certains poids seulement).
copy_model = copy.deepcopy(resnet).eval()  # copie independante du modele, en mode evaluation
with torch.no_grad():  # aucun calcul de gradient
    out = copy_model(images)  # prediction (inference)
print("sortie sous no_grad : requires_grad =", out.requires_grad, "-> impossible d'appeler backward, rien n'apprend")  # False
copy_model.train()  # repasse en mode entrainement
out = copy_model(images)  # passage avant normal : PyTorch enregistre les calculs
print("sortie normale      : requires_grad =", out.requires_grad, "-> seuls les parametres non geles recevront un gradient")  # True
