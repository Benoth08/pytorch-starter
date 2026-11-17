# %% [markdown]
# # Detection d'objets avec YOLO (Ultralytics) sur NEU-DET, brique par brique
#
# Meme fil en 13 etapes que la classification. Dans ce notebook : reglages (etape 1) ->
# catalogue, ici des boites (etape 3) -> decoupage (etape 4) -> donnees au format YOLO (etape 7) ->
# entrainement (etape 10) -> evaluation, mAP (etape 11) -> prediction (etape 13) ->
# export ONNX avec post-traitement verifie (etape 12).
# Le pretraitement, l'augmentation, le modele et la perte (etapes 5, 6, 8 et 9) sont dans Ultralytics.

# %% imports
from __future__ import annotations  # permet d'ecrire les types (tuple[int, ...]) sans souci de version de Python

import random  # hasard de Python, fixe plus bas pour des resultats reproductibles
import shutil  # copie de fichiers
import xml.etree.ElementTree as ET  # lecture des fichiers XML (les annotations Pascal VOC sont en XML)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import cv2  # OpenCV : lecture, dessin et redimensionnement d'images
import matplotlib.pyplot as plt  # matplotlib : affichage d'images et de graphiques
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import onnxruntime as ort  # onnxruntime : execute un modele exporte au format ONNX, sans PyTorch
import pandas as pd  # pandas : tableaux de donnees (DataFrame), ici le catalogue des images
import torch  # PyTorch : tenseurs et apprentissage profond (ici surtout pour la graine et la NMS)
import yaml  # lecture / ecriture de fichiers YAML (le fichier de configuration attendu par Ultralytics)
from sklearn.model_selection import train_test_split  # coupe un tableau en deux parties (ex. entrainement / test)
from torchvision.ops import nms  # NMS (non-maximum suppression) : supprime les boites en double sur un meme objet
from ultralytics import YOLO  # Ultralytics : bibliotheque qui fournit les modeles YOLO et leur entrainement

DET_CONFIG = {  # tous les reglages de la detection au meme endroit
    "data_dir": "data/raw/neu_det",  # IMAGES/*.jpg + ANNOTATIONS/*.xml (Pascal VOC)
    "yolo_dir": "data/processed/neu_det_yolo",  # le meme jeu, range comme Ultralytics l'attend
    "output_dir": "outputs/yolo",  # dossier des resultats (split, entrainements, export)
    "weights": "outputs/yolo/pretrained/yolo11n.pt",  # poids COCO officiels (telecharges par Ultralytics si absents)
    "image_size": 320,  # cote des images donnees au reseau (320 x 320 pixels)
    "epochs": 2,  # verification rapide ; 60 epoques pour un vrai entrainement (epoque = un passage sur tout le train)
    "batch_size": 16,  # 16 images traitees ensemble a chaque pas (lot / batch)
    "seed": 42,  # graine du hasard : memes resultats a chaque execution
}
OUTPUT_DIR = Path(DET_CONFIG["output_dir"])  # chemin du dossier de sortie
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # cree le dossier s'il n'existe pas encore
random.seed(DET_CONFIG["seed"])  # fixe le hasard de Python
np.random.seed(DET_CONFIG["seed"])  # fixe le hasard de numpy
torch.manual_seed(DET_CONFIG["seed"])  # fixe le hasard de PyTorch


# %% catalog_boxes
# Etape 3 - Cataloguer : le catalogue d'un jeu de detection = une image + son fichier d'annotations.
# Format Pascal VOC : un XML par image, des coins (xmin, ymin, xmax, ymax) en pixels, un nom de classe par boite.
def parse_voc(xml_path: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:  # lit un XML VOC ; renvoie taille + boites
    """Fichier VOC -> (largeur, hauteur, [(classe, xmin, ymin, xmax, ymax), ...])."""
    root = ET.parse(xml_path).getroot()  # lit le fichier XML et prend sa balise racine
    width, height = int(root.findtext("size/width")), int(root.findtext("size/height"))  # taille de l'image notee dans le XML
    boxes = []  # liste des boites trouvees
    for obj in root.iter("object"):  # pour chaque objet annote (une balise <object> par defaut)
        name = obj.findtext("name").strip()  # nom de la classe, sans espaces autour
        box = obj.find("bndbox")  # la balise qui contient les coordonnees de la boite
        boxes.append((name, *(float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))))  # (classe, 4 coins en pixels)
    return width, height, boxes  # largeur, hauteur et boites de l'image


DATA_DIR = Path(DET_CONFIG["data_dir"])  # dossier des donnees brutes
rows = []  # une ligne par image pour le catalogue
for image_path in sorted((DATA_DIR / "IMAGES").glob("*.jpg")):  # pour chaque image .jpg (triee : ordre stable)
    xml_path = DATA_DIR / "ANNOTATIONS" / f"{image_path.stem}.xml"  # son fichier d'annotations porte le meme nom
    width, height, boxes = parse_voc(xml_path)  # lit la taille et les boites
    rows.append(  # ajoute la ligne de cette image
        {  # les informations gardees
            "path": image_path.as_posix(),  # chemin de l'image, ecrit avec des /
            "annotation": str(xml_path),  # chemin du XML
            "label": image_path.stem.rsplit("_", 1)[0],  # la classe est dans le nom du fichier : crazing_12.jpg
            "n_boxes": len(boxes),  # nombre de boites (de defauts) dans l'image
            "width": width,  # largeur en pixels
            "height": height,  # hauteur en pixels
        }
    )
catalog = pd.DataFrame(rows)  # le catalogue : un tableau pandas, une ligne par image
class_names = sorted(catalog["label"].unique())  # liste triee des classes (6 types de defauts)
print(len(catalog), "images |", catalog["n_boxes"].sum(), "boites | classes :", class_names)  # resume du jeu
print(catalog.groupby("label")["n_boxes"].agg(["count", "sum", "mean"]).round(2))  # par classe : nb d'images, nb de boites, moyenne


# %% show_boxes
# Toujours dessiner quelques annotations : une boite decalee ou un format mal lu se voit tout de suite.
def draw_boxes(image: np.ndarray, boxes: list[tuple[str, float, float, float, float]]) -> np.ndarray:  # renvoie une copie annotee
    """Dessine des boites (nom, xmin, ymin, xmax, ymax) sur une image BGR OpenCV."""
    canvas = image.copy()  # copie : on ne dessine pas sur l'image d'origine
    for name, x_min, y_min, x_max, y_max in boxes:  # pour chaque boite
        cv2.rectangle(canvas, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)  # rectangle vert, 2 pixels d'epaisseur
        # ecrit le nom au-dessus de la boite (au moins a 10 pixels du haut pour rester visible)
        cv2.putText(canvas, name, (int(x_min), max(int(y_min) - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return canvas  # l'image avec les boites dessinees


fig, axes = plt.subplots(1, 6, figsize=(16, 3))  # une figure avec 6 cases cote a cote (une par classe)
for ax, name in zip(axes, class_names):  # pour chaque case et chaque classe
    row = catalog[catalog["label"] == name].iloc[0]  # premiere image de cette classe
    image = cv2.imread(row["path"])  # OpenCV lit en BGR (bleu, vert, rouge), pas en RGB
    _, _, boxes = parse_voc(Path(row["annotation"]))  # ses boites (la taille ne sert pas ici)
    ax.imshow(cv2.cvtColor(draw_boxes(image, boxes), cv2.COLOR_BGR2RGB))  # dessine les boites, repasse en RGB pour matplotlib, affiche
    ax.set_title(f"{name} ({len(boxes)} boites)", fontsize=9)  # titre : classe et nombre de boites
    ax.axis("off")  # cache les axes gradues
plt.tight_layout()  # ajuste les marges
plt.show()  # affiche la figure

# %% split_det
# Etape 4 - Decouper : decoupage stratifie par la classe de l'image, enregistre. Aucune image ne doit etre dans deux ensembles.
# decoupe 1 : 15 % pour le test ; stratify garde les memes proportions de classes dans chaque partie
train_val, test_df = train_test_split(catalog, test_size=0.15, stratify=catalog["label"], random_state=DET_CONFIG["seed"])
# decoupe 2 : dans le reste, 15 % du total pour la validation, le reste pour l'entrainement
train_df, val_df = train_test_split(train_val, test_size=0.15 / 0.85, stratify=train_val["label"], random_state=DET_CONFIG["seed"])
# ajoute la colonne split au catalogue : "train", "val" ou "test"
catalog.loc[train_df.index, "split"], catalog.loc[val_df.index, "split"], catalog.loc[test_df.index, "split"] = "train", "val", "test"
catalog.to_csv(OUTPUT_DIR / "split.csv", index=False)  # sauvegarde le decoupage
assert catalog["path"].is_unique  # verifie qu'aucune image n'apparait deux fois
print(pd.crosstab(catalog["label"], catalog["split"]))  # tableau croise : nombre d'images par classe et par partie

# %% to_yolo
# Etape 7 - Charger par lots : Ultralytics attend une arborescence precise et des etiquettes "classe cx cy w h" NORMALISEES.
#   images/train/a.jpg  <->  labels/train/a.txt   (le mot "images" est remplace par "labels" pour trouver l'etiquette)
YOLO_DIR = Path(DET_CONFIG["yolo_dir"])  # dossier du jeu au format YOLO
class_to_idx = {name: i for i, name in enumerate(class_names)}  # nom de classe -> numero (YOLO n'utilise que des numeros)
for _, row in catalog.iterrows():  # pour chaque image du catalogue
    image_dir, label_dir = YOLO_DIR / "images" / row["split"], YOLO_DIR / "labels" / row["split"]  # images/<split> et labels/<split>
    image_dir.mkdir(parents=True, exist_ok=True)  # cree le dossier des images si besoin
    label_dir.mkdir(parents=True, exist_ok=True)  # cree le dossier des etiquettes si besoin
    shutil.copy2(row["path"], image_dir / Path(row["path"]).name)  # copie l'image a sa place
    width, height, boxes = parse_voc(Path(row["annotation"]))  # relit taille et boites
    lines = []  # une ligne de texte par boite
    for name, x_min, y_min, x_max, y_max in boxes:  # pour chaque boite
        x_min, x_max = np.clip([x_min, x_max], 0, width)  # quelques boites VOC debordent d'un pixel : on les ramene dans l'image
        y_min, y_max = np.clip([y_min, y_max], 0, height)  # idem en hauteur
        cx, cy = (x_min + x_max) / 2 / width, (y_min + y_max) / 2 / height  # centre de la boite, divise par la taille : entre 0 et 1
        w, h = (x_max - x_min) / width, (y_max - y_min) / height  # largeur et hauteur de la boite, aussi entre 0 et 1
        lines.append(f"{class_to_idx[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")  # format YOLO : "classe cx cy w h"
    (label_dir / f"{Path(row['path']).stem}.txt").write_text("\n".join(lines) + "\n")  # ecrit le fichier .txt du meme nom que l'image

data_yaml = {  # description du jeu pour Ultralytics
    "path": str(YOLO_DIR.resolve()),  # dossier racine du jeu (chemin absolu)
    "train": "images/train",  # images d'entrainement (relatif a path)
    "val": "images/val",  # images de validation
    "test": "images/test",  # images de test
    "names": dict(enumerate(class_names)),  # numero -> nom de classe
}
(YOLO_DIR / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))  # ecrit data.yaml (sort_keys=False : garde l'ordre)
print((YOLO_DIR / "data.yaml").read_text())  # relit et affiche pour verifier
example = sorted((YOLO_DIR / "labels" / "train").glob("*.txt"))[0]  # premier fichier d'etiquettes d'entrainement
print(example.name, "->", example.read_text().strip())  # montre son contenu : une ligne "classe cx cy w h" par boite

# %% train_yolo
# Etape 10 - Entrainer : la boucle d'entrainement est dans Ultralytics. On lui donne le YAML et les hyperparametres.
model = YOLO(DET_CONFIG["weights"])  # YOLO11n pre-entraine sur COCO ; la tete est adaptee a nos 6 classes
results = model.train(  # lance l'entrainement ; renvoie les metriques de validation
    data=str(YOLO_DIR / "data.yaml"),  # description du jeu
    epochs=DET_CONFIG["epochs"],  # nombre d'epoques
    imgsz=DET_CONFIG["image_size"],  # taille des images
    batch=DET_CONFIG["batch_size"],  # taille des lots
    device="cpu",  # calcul sur le processeur
    seed=DET_CONFIG["seed"],  # graine du hasard
    project=str((OUTPUT_DIR / "runs").resolve()),  # chemin absolu, sinon Ultralytics prefixe runs/detect/
    name="train",  # sous-dossier de cet entrainement
    exist_ok=True,  # reutilise le dossier s'il existe deja (pas de train2, train3...)
    workers=2,  # 2 processus pour charger les images en parallele
    close_mosaic=1,  # derniere epoque sans mosaic (augmentation qui colle 4 images ; 10 par defaut, avec 60 epoques)
    warmup_epochs=1,  # 1 epoque de "chauffe" : le taux d'apprentissage monte doucement au debut
    plots=False,  # pas de graphiques sauvegardes (plus rapide)
    verbose=False,  # moins de texte affiche
)
best_weights = Path(results.save_dir) / "weights" / "best.pt"  # fichier des meilleurs poids, ecrit par Ultralytics
print("meilleur modele :", best_weights)  # affiche son chemin
# mAP (mean Average Precision) : note de detection entre 0 et 1 ; @0.5 = une boite compte si son IoU (recouvrement) avec la vraie >= 0.5
print(f"validation : mAP@0.5 = {results.box.map50:.3f} | mAP@0.5:0.95 = {results.box.map:.3f}")

# %% val_test
# Etape 11 - Evaluer : mAP sur le test, une seule fois. conf=0.001 : toute la courbe precision-rappel est evaluee.
model = YOLO(best_weights)  # recharge le meilleur modele
metrics = model.val(  # evalue le modele ; renvoie les metriques
    data=str(YOLO_DIR / "data.yaml"),  # description du jeu
    split="test",  # sur la partie test (jamais vue pendant l'entrainement)
    imgsz=DET_CONFIG["image_size"],  # meme taille d'image qu'a l'entrainement
    batch=16,  # taille des lots
    conf=0.001,  # seuil de confiance tres bas : on garde presque toutes les boites pour calculer la mAP
    iou=0.7,  # seuil IoU de la NMS : deux boites qui se recouvrent a plus de 70 % = doublon
    device="cpu",  # calcul sur le processeur
    plots=False,  # pas de graphiques
    verbose=False,  # moins de texte
)
print(f"test : mAP@0.5 = {metrics.box.map50:.3f} | mAP@0.5:0.95 = {metrics.box.map:.3f}")  # notes globales sur le test
for i, name in enumerate(class_names):  # pour chaque classe
    print(f"  {name:16s} AP@0.5 = {metrics.box.ap50[i]:.3f} | AP@0.5:0.95 = {metrics.box.ap[i]:.3f}")  # note de cette classe (AP)

# %% predict_draw
# Etape 13 - Predire : prediction sur des images de test, et dessin des boites. conf=0.25 pour la production.
fig, axes = plt.subplots(1, 6, figsize=(16, 3))  # 6 cases cote a cote
for ax, name in zip(axes, class_names):  # pour chaque case et chaque classe
    path = test_df[test_df["label"] == name].iloc[0]["path"]  # premiere image de test de cette classe
    # predit les boites : conf=0.25 ne garde que les boites assez sures ; [0] : resultat de la seule image
    result = model.predict(path, imgsz=DET_CONFIG["image_size"], conf=0.25, iou=0.7, device="cpu", verbose=False)[0]
    boxes = [  # met les detections au format de draw_boxes : ("classe score", xmin, ymin, xmax, ymax)
        (class_names[int(c)] + f" {s:.2f}", *b)  # texte "classe score" puis les 4 coins
        for b, c, s in zip(result.boxes.xyxy.numpy(), result.boxes.cls.numpy(), result.boxes.conf.numpy())  # coins, classe, score
    ]
    ax.imshow(cv2.cvtColor(draw_boxes(cv2.imread(path), boxes), cv2.COLOR_BGR2RGB))  # dessine et affiche
    ax.set_title(f"{Path(path).name} : {len(boxes)} detection(s)", fontsize=8)  # titre : fichier et nombre de detections
    ax.axis("off")  # cache les axes
plt.tight_layout()  # ajuste les marges
plt.show()  # affiche la figure


# %% onnx_yolo
# Etape 12 - Sauvegarder et exporter : export ONNX (sans NMS) + post-traitement ecrit a la main, verifie contre Ultralytics.
# ONNX : format de fichier standard pour un modele, executable sans PyTorch
# dynamic=True : taille de lot libre ; nms=False : la NMS sera faite a la main plus bas
onnx_path = Path(model.export(format="onnx", imgsz=DET_CONFIG["image_size"], dynamic=True, nms=False, simplify=False, device="cpu"))
session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # charge le modele ONNX pour l'executer sur CPU
SIZE = DET_CONFIG["image_size"]  # cote de l'image carree attendue par le modele


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:  # met l'image dans un carre sans la deformer
    """Redimensionne sans deformer, complete en gris 114 pour obtenir un carre. Renvoie (image, ratio, (pad_x, pad_y))."""
    height, width = image.shape[:2]  # taille d'origine
    ratio = min(size / height, size / width)  # facteur d'echelle pour que le plus grand cote tienne dans le carre
    new_w, new_h = round(width * ratio), round(height * ratio)  # nouvelle taille, proportions gardees
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # redimensionne (interpolation lineaire)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2  # marges a gauche et en haut pour centrer l'image
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)  # carre rempli de gris 114 (la valeur utilisee par Ultralytics)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized  # colle l'image redimensionnee au centre
    return canvas, ratio, (pad_x, pad_y)  # le carre, et de quoi revenir aux coordonnees d'origine


def preprocess(image_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:  # image OpenCV -> tableau pret pour le modele
    """BGR -> letterbox -> RGB -> [0, 1] -> NCHW float32 (le pretraitement exact d'Ultralytics)."""
    boxed, ratio, pad = letterbox(image_bgr, SIZE)  # carre sans deformation
    tensor = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # BGR -> RGB, puis valeurs 0-255 -> 0-1
    return tensor.transpose(2, 0, 1)[np.newaxis], ratio, pad  # (H, W, C) -> (C, H, W), puis ajoute le lot : (1, C, H, W)


# transforme la sortie brute du modele en boites finales dans l'image d'origine
def postprocess(output: np.ndarray, ratio: float, pad: tuple[int, int], conf: float = 0.25, iou: float = 0.7) -> np.ndarray:
    """Sortie brute (1, 4 + C, A) -> boites (n, 6) : xmin, ymin, xmax, ymax, score, classe, dans l'image d'origine."""
    pred = output[0].T  # (A, 4 + C) : pour chaque ancre, cx cy w h puis un score par classe (deja decodes, en pixels)
    scores, classes = pred[:, 4:].max(axis=1), pred[:, 4:].argmax(axis=1)  # pour chaque ancre : meilleur score et sa classe
    keep = scores > conf  # garde seulement les ancres assez sures
    pred, scores, classes = pred[keep], scores[keep], classes[keep]  # filtre les trois tableaux de la meme facon
    boxes = np.column_stack(  # passe de (centre, largeur, hauteur) a (coins) : xmin, ymin, xmax, ymax
        # xmin = cx - w/2, ymin = cy - h/2, xmax = cx + w/2, ymax = cy + h/2
        [pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2, pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2]
    )
    # NMS par classe : on decale chaque classe dans son propre coin pour qu'elles ne se chevauchent pas.
    offsets = classes[:, None] * (SIZE + 1)  # decalage different pour chaque classe (plus grand que l'image)
    kept = nms(torch.from_numpy(boxes + offsets).float(), torch.from_numpy(scores).float(), iou).numpy()  # indices des boites gardees
    boxes = (boxes[kept] - np.array([pad[0], pad[1], pad[0], pad[1]])) / ratio  # on annule le letterbox : retire marges, remet a l'echelle
    return np.column_stack([boxes, scores[kept], classes[kept]])  # tableau (n, 6) : coins, score, classe


path = test_df.iloc[0]["path"]  # une image de test
image = cv2.imread(path)  # lue en BGR
tensor, ratio, pad = preprocess(image)  # pretraitement identique a Ultralytics
raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]  # execute le modele ONNX (None : toutes les sorties) ; [0] : la premiere
print("sortie ONNX brute :", raw.shape, "= (lot, 4 + nombre de classes, nombre d'ancres)")  # ancre = position candidate pour une boite
onnx_boxes = postprocess(raw, ratio, pad)  # boites finales calculees a la main
# reference : les boites d'Ultralytics sur la meme image, avec les memes seuils
reference = model.predict(path, imgsz=SIZE, conf=0.25, iou=0.7, device="cpu", verbose=False)[0].boxes.xyxy.numpy()
print(len(onnx_boxes), "detections ONNX |", len(reference), "detections Ultralytics")  # le nombre doit etre le meme


def box_iou(a: np.ndarray, b: np.ndarray) -> float:  # IoU de deux boites : surface commune / surface totale (1 = identiques)
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))  # aire commune (0 si disjointes)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter  # aire de l'union : les 2 aires moins l'aire commune
    return inter / union if union > 0 else 0.0  # IoU ; 0 si les boites sont vides


# pour chaque boite ONNX, son meilleur IoU avec une boite d'Ultralytics (liste vide si Ultralytics ne trouve rien)
ious = [max(box_iou(box[:4], ref) for ref in reference) for box in onnx_boxes] if len(reference) else []
print("IoU moyen entre les boites ONNX et celles d'Ultralytics :", round(float(np.mean(ious)), 4) if ious else "n/a")  # proche de 1 attendu
assert not ious or np.mean(ious) > 0.99, "le post-traitement ONNX ne reproduit pas Ultralytics"  # arrete tout si l'ecart est trop grand
