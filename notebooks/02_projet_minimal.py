# Le projet complet en version minimale : la fiche a savoir reecrire de tete.
# Memes briques que 01_projet_complet.py, sans les variantes ni les affichages.
# Meme fil en 13 etapes que le cours : un titre "# === Etape N - nom ===" ouvre chaque etape.

# %% imports_config  (etape 1)
# === Etape 1 - Imports et reglages ===
import random  # hasard de Python, fixe plus bas pour des resultats reproductibles
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres
import pandas as pd  # pandas : tableaux de donnees (DataFrame), ici la liste des images et leurs classes
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU) et apprentissage profond
import torch.nn as nn  # nn : briques de reseau (couches, fonctions de perte)
from PIL import Image  # PIL (Pillow) : ouvrir les fichiers images
from sklearn.metrics import f1_score  # F1 : note qui combine precision et rappel, utile si les classes sont desequilibrees
from sklearn.model_selection import train_test_split  # coupe un tableau en deux parties (ex. entrainement / test)
from torch.utils.data import DataLoader, Dataset  # Dataset : donne une image a la fois ; DataLoader : les regroupe en lots
from torchvision.transforms import v2  # transformations d'images (redimensionner, retourner, normaliser...)

CONFIG = {  # tous les reglages au meme endroit, pour les changer sans chercher dans le code
    "data_dir": "data/raw/neu_cls_flat",  # dossier des images : un sous-dossier par classe
    "output_dir": "outputs/minimal",  # dossier ou ecrire les resultats (split, modeles)
    "image_size": 128,  # cote des images donnees au reseau (128 x 128 pixels)
    "batch_size": 32,  # 32 images traitees ensemble a chaque pas : plus grand = plus rapide mais plus de memoire
    "epochs": 5,  # nombre d'epoques (une epoque = un passage complet sur toutes les images d'entrainement)
    "learning_rate": 1e-3,  # taux d'apprentissage : taille du pas de chaque mise a jour des poids
    "patience": 3,  # arret anticipe apres 3 epoques sans amelioration sur la validation
    "seed": 42,  # graine du hasard : memes resultats a chaque execution
}
OUTPUT_DIR = Path(CONFIG["output_dir"])  # chemin du dossier de sortie
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # cree le dossier (et ses parents) s'il n'existe pas encore


def seed_everything(seed):  # fixe toutes les sources de hasard ; ne renvoie rien
    random.seed(seed)  # hasard de Python
    np.random.seed(seed)  # hasard de numpy
    torch.manual_seed(seed)  # hasard de PyTorch sur CPU (poids initiaux, melange des lots)
    torch.cuda.manual_seed_all(seed)  # hasard de PyTorch sur les cartes graphiques (sans effet si pas de GPU)


seed_everything(CONFIG["seed"])  # applique la graine tout de suite, avant de creer quoi que ce soit d'aleatoire
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # calcule sur la carte graphique (GPU) si elle existe, sinon CPU

# %% catalog_split  (etapes 3 et 4)
# === Etape 2 - Importer et explorer ===
# (pas de code dans cette version courte : compter et regarder les images, cellules explore_* de 01_projet_complet.py)
# === Etape 3 - Cataloguer ===
EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}  # extensions de fichiers acceptees comme images
rows = [  # une ligne par image : son chemin et sa classe (le nom du sous-dossier)
    {"path": p.as_posix(), "label": d.name}  # ce qu'on garde pour chaque image
    for d in sorted(Path(CONFIG["data_dir"]).iterdir())  # pour chaque element du dossier de donnees (trie : ordre stable)
    if d.is_dir()  # seulement les sous-dossiers (= les classes)
    for p in sorted(d.iterdir())  # pour chaque fichier du sous-dossier
    if p.suffix.lower() in EXTENSIONS  # seulement les fichiers images
]
catalog = pd.DataFrame(rows)  # le catalogue : un tableau pandas avec les colonnes path et label
class_names = sorted(catalog["label"].unique())  # liste triee des classes differentes
class_to_idx = {c: i for i, c in enumerate(class_names)}  # nom de classe -> numero (0, 1, 2...) : le reseau ne manipule que des nombres

# === Etape 4 - Decouper ===
# decoupe 1 : 15 % des images pour le test ; stratify garde les memes proportions de classes dans chaque partie
train_val, test_df = train_test_split(catalog, test_size=0.15, stratify=catalog["label"], random_state=CONFIG["seed"])
# decoupe 2 : dans le reste, 15 % du total (0.15 / 0.85 du reste) pour la validation, le reste pour l'entrainement
train_df, val_df = train_test_split(train_val, test_size=0.15 / 0.85, stratify=train_val["label"], random_state=CONFIG["seed"])
assert set(train_df["path"]).isdisjoint(test_df["path"])  # verifie qu'aucune image de test n'est aussi en entrainement (sinon note faussee)
# ajoute une colonne split au catalogue : "train", "val" ou "test" selon la partie de chaque image
catalog.loc[train_df.index, "split"], catalog.loc[val_df.index, "split"], catalog.loc[test_df.index, "split"] = "train", "val", "test"
catalog.to_csv(OUTPUT_DIR / "split.csv", index=False)  # sauvegarde le decoupage pour pouvoir le retrouver plus tard
print(catalog["split"].value_counts().to_dict())  # affiche le nombre d'images dans chaque partie


# %% dataset_loader  (etapes 5 a 7)
# === Etape 7 - Charger par lots ===
# (la classe Dataset s'ecrit en premier ; les transformations des etapes 5 et 6 lui sont donnees plus bas)
class ImageDataset(Dataset):  # jeu de donnees PyTorch : renvoie une image transformee et son numero de classe
    def __init__(self, frame, class_to_idx, transform=None):  # recoit le tableau des images, les numeros de classe et la transformation
        # garde la liste des chemins, la liste des numeros de classe et la transformation a appliquer
        self.paths, self.targets, self.transform = frame["path"].tolist(), [class_to_idx[x] for x in frame["label"]], transform

    def __len__(self):  # nombre d'images du jeu (utilise par le DataLoader)
        return len(self.paths)  # une image par chemin

    def __getitem__(self, i):  # renvoie l'image numero i et sa classe
        image = Image.open(self.paths[i]).convert("RGB")  # ouvre le fichier et force 3 canaux couleur (meme pour une image en gris)
        return (self.transform(image) if self.transform else image), self.targets[i]  # image transformee (tenseur) + classe


# === Etape 5 - Pretraiter ===
# taille commune et normalisation : ToImage, Resize, ToDtype, Normalize (dans train_transform ET eval_transform)
SIZE, MEAN, STD = (CONFIG["image_size"],) * 2, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]  # taille (128, 128), moyenne et ecart-type par canal
# === Etape 6 - Augmenter ===
# les deux retournements aleatoires (Random...Flip), dans train_transform seulement
train_transform = v2.Compose(  # transformations de l'entrainement, appliquees dans l'ordre
    [  # liste des etapes
        v2.ToImage(),  # image PIL -> tenseur image (canaux, hauteur, largeur)
        v2.Resize(SIZE, antialias=True),  # met toutes les images a la meme taille (antialias : lisse pour eviter les escaliers)
        v2.RandomHorizontalFlip(),  # augmentation : retourne gauche-droite une fois sur deux, pour varier les exemples
        v2.RandomVerticalFlip(),  # augmentation : retourne haut-bas une fois sur deux
        v2.ToDtype(torch.float32, scale=True),  # entiers 0-255 -> nombres a virgule 0-1
        v2.Normalize(MEAN, STD),  # normalisation : (valeur - moyenne) / ecart-type par canal, valeurs autour de 0
    ]
)
# transformations de validation / test : les memes SANS augmentation aleatoire, pour une note stable
eval_transform = v2.Compose([v2.ToImage(), v2.Resize(SIZE, antialias=True), v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])
# === Etape 7 - Charger par lots (suite) ===
train_loader = DataLoader(  # DataLoader : fournit les images par lots (batch = paquet d'images traitees ensemble)
    ImageDataset(train_df, class_to_idx, train_transform),  # le jeu d'entrainement, avec augmentation
    batch_size=CONFIG["batch_size"],  # taille des lots
    shuffle=True,  # melange l'ordre des images a chaque epoque
    drop_last=True,  # jette le dernier lot s'il est incomplet (evite un lot minuscule pour BatchNorm)
)
val_loader = DataLoader(ImageDataset(val_df, class_to_idx, eval_transform), batch_size=CONFIG["batch_size"])  # lots de validation
test_loader = DataLoader(ImageDataset(test_df, class_to_idx, eval_transform), batch_size=CONFIG["batch_size"])  # lots de test, sans melange
images, targets = next(iter(train_loader))  # prend un premier lot pour verifier les formes
print(images.shape, targets.shape)  # attendu : (32, 3, 128, 128) et (32,)


# %% model  (etape 8)
# === Etape 8 - Modele ===
class ConvBlock(nn.Module):  # un bloc de base du reseau : convolution, normalisation, activation, reduction de taille
    def __init__(self, cin, cout):  # cin : nombre de canaux en entree, cout : en sortie
        super().__init__()  # initialise la partie nn.Module (obligatoire)
        self.block = nn.Sequential(  # enchaine les couches dans l'ordre
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),  # convolution 3x3 : filtres qui detectent des motifs locaux
            nn.BatchNorm2d(cout),  # BatchNorm : remet les valeurs a une echelle stable, l'apprentissage est plus facile
            nn.ReLU(inplace=True),  # ReLU (activation) : garde les valeurs positives, met 0 sinon
            nn.MaxPool2d(2),  # MaxPool : garde le max de chaque carre 2x2, divise hauteur et largeur par 2
        )

    def forward(self, x):  # passage avant : ce que fait le bloc sur un lot x
        return self.block(x)  # applique les couches dans l'ordre


class SimpleCNN(nn.Module):  # petit reseau convolutif (CNN) : blocs de convolution puis un classifieur
    def __init__(self, num_classes, channels=(32, 64, 128, 256)):  # num_classes sorties ; channels : canaux de chaque bloc
        super().__init__()  # initialise la partie nn.Module
        blocks, cin = [], 3  # liste des blocs a construire ; l'image d'entree a 3 canaux (RVB)
        for cout in channels:  # pour chaque nombre de canaux voulu
            blocks.append(ConvBlock(cin, cout))  # ajoute un bloc qui passe de cin a cout canaux
            cin = cout  # la sortie de ce bloc est l'entree du suivant
        self.features = nn.Sequential(*blocks)  # extracteur de caracteristiques : les blocs a la suite
        self.pool = nn.AdaptiveAvgPool2d(1)  # moyenne de chaque canal sur toute l'image : (N, 256, h, w) -> (N, 256, 1, 1)
        # aplatit en (N, 256), Dropout (eteint 30 % des valeurs au hasard pendant l'entrainement, contre le par-coeur),
        # puis couche lineaire qui donne un score par classe
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(cin, num_classes))

    def forward(self, x):  # passage avant : image -> scores par classe (logits = scores bruts, pas encore des probabilites)
        return self.classifier(self.pool(self.features(x)))  # caracteristiques -> moyenne -> classifieur


model = SimpleCNN(len(class_names)).to(DEVICE)  # cree le reseau (une sortie par classe) et le place sur CPU ou GPU
assert model(torch.zeros(2, 3, *SIZE, device=DEVICE)).shape == (2, len(class_names))  # test rapide : 2 images factices -> 2 x nb classes


# %% engine  (etapes 9 et 10)
# === Etape 10 - Entrainer ===
# (les deux fonctions d'abord ; la perte et l'optimiseur de l'etape 9 viennent juste apres, puis la boucle)
def train_one_epoch(model, loader, criterion, optimizer, device):  # entraine une epoque ; renvoie (perte moyenne, exactitude)
    model.train()  # mode entrainement : Dropout et BatchNorm actifs
    total, correct, seen = 0.0, 0, 0  # cumul de la perte, nombre de bonnes reponses, nombre d'images vues
    for images, targets in loader:  # pour chaque lot d'images et ses classes
        images, targets = images.to(device), targets.to(device)  # envoie le lot sur le meme appareil que le modele
        optimizer.zero_grad()  # remet les gradients a zero (sinon ils s'ajoutent d'un lot a l'autre)
        logits = model(images)  # passage avant : scores bruts par classe
        loss = criterion(logits, targets)  # perte : mesure l'erreur entre scores et vraies classes
        loss.backward()  # calcule les gradients (dans quel sens bouger chaque poids pour baisser la perte)
        optimizer.step()  # optimiseur : met a jour les poids avec les gradients
        total += loss.item() * len(targets)  # ajoute la perte du lot (ponderee par sa taille)
        correct += (logits.argmax(1) == targets).sum().item()  # classe predite = score le plus haut ; compte les bonnes
        seen += len(targets)  # compte les images vues
    return total / seen, correct / seen  # perte moyenne et exactitude (part de bonnes reponses)


@torch.no_grad()  # pas de calcul de gradient dans cette fonction : plus rapide, moins de memoire
def evaluate(model, loader, criterion, device):  # evalue sans apprendre ; renvoie (perte, exactitude, predictions, verites)
    model.eval()  # mode evaluation : Dropout coupe, BatchNorm utilise ses statistiques apprises
    total, preds, trues = 0.0, [], []  # cumul de la perte, listes des predictions et des vraies classes
    for images, targets in loader:  # pour chaque lot
        images, targets = images.to(device), targets.to(device)  # envoie le lot sur l'appareil du modele
        logits = model(images)  # scores bruts par classe
        total += criterion(logits, targets).item() * len(targets)  # ajoute la perte du lot
        preds.append(logits.argmax(1).cpu())  # garde les classes predites (ramenees sur CPU)
        trues.append(targets.cpu())  # garde les vraies classes
    preds, trues = torch.cat(preds).numpy(), torch.cat(trues).numpy()  # colle tous les lots en un seul tableau numpy
    return total / len(trues), float((preds == trues).mean()), preds, trues  # perte moyenne, exactitude, predictions, verites


# === Etape 9 - Perte et optimiseur ===
criterion = nn.CrossEntropyLoss()  # perte de classification : applique un softmax (scores -> probabilites) puis penalise l'erreur
# optimiseur AdamW : ajuste le pas de chaque poids ; weight_decay pousse les poids vers 0 (limite le par-coeur)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
# divise le taux d'apprentissage par 2 si la perte de validation ne baisse plus pendant 2 epoques
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
# === Etape 10 - Entrainer (suite) ===
# la boucle : une epoque d'entrainement, une evaluation, sauvegarde du meilleur modele, arret anticipe
best, bad_epochs = float("inf"), 0  # meilleure perte de validation vue (infini au depart) ; epoques sans progres
for epoch in range(1, CONFIG["epochs"] + 1):  # pour chaque epoque, numerotees a partir de 1
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)  # apprend sur le train
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)  # mesure sur la validation (images jamais apprises)
    scheduler.step(val_loss)  # le scheduler regarde la perte de validation pour decider de baisser le pas
    print(f"epoque {epoch} | train {train_loss:.3f} / {train_acc:.3f} | val {val_loss:.3f} / {val_acc:.3f}")  # suivi : perte / exactitude
    if val_loss < best:  # si la validation s'ameliore
        best, bad_epochs = val_loss, 0  # nouveau record, compteur remis a zero
        torch.save({"model_state": model.state_dict(), "epoch": epoch}, OUTPUT_DIR / "best.pt")  # sauvegarde les poids du meilleur modele
    else:  # sinon (pas d'amelioration)
        bad_epochs += 1  # une epoque de plus sans progres
        if bad_epochs >= CONFIG["patience"]:  # si la patience est epuisee
            print("arret anticipe")  # on previent
            break  # on arrete l'entrainement : continuer ferait apprendre par coeur

# %% test_save_predict  (etapes 11 a 13)
# === Etape 11 - Evaluer ===
model.load_state_dict(torch.load(OUTPUT_DIR / "best.pt", weights_only=True)["model_state"])  # recharge les poids du meilleur modele
test_loss, test_acc, preds, trues = evaluate(model, test_loader, criterion, DEVICE)  # note finale sur le test, jamais vu avant
print(f"test accuracy {test_acc:.3f} | F1 macro {f1_score(trues, preds, average='macro'):.3f}")  # F1 macro : moyenne des F1 des classes
# === Etape 12 - Sauvegarder et exporter ===
torch.save(  # sauvegarde tout ce qu'il faut pour reutiliser le modele plus tard
    # poids, noms des classes, taille et normalisation : la prediction doit refaire exactement le meme pretraitement
    {"model_state": model.state_dict(), "class_names": class_names, "image_size": CONFIG["image_size"], "mean": MEAN, "std": STD},
    OUTPUT_DIR / "model.pt",  # fichier de destination
)


# === Etape 13 - Predire ===
@torch.no_grad()  # prediction seule : pas de gradient
def predict_image(path, model, transform, class_names, device):  # predit la classe d'un fichier image ; renvoie (classe, probabilite)
    batch = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)  # ouvre, transforme, ajoute la dimension lot -> (1, 3, H, W)
    probabilities = torch.softmax(model(batch), dim=1)[0]  # softmax : scores -> probabilites qui somment a 1 ; [0] : la seule image du lot
    return class_names[int(probabilities.argmax())], float(probabilities.max())  # nom de la classe la plus probable et sa probabilite


# predit la premiere image de test et affiche aussi sa vraie classe pour comparer
print(predict_image(test_df["path"].iloc[0], model, eval_transform, class_names, DEVICE), "| vrai :", test_df["label"].iloc[0])
