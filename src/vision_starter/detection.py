"""Detection d'objets avec YOLO (Ultralytics) : les briques autour de la librairie (chapitre Detection du cours).

Ultralytics fournit le Dataset, le DataLoader, les augmentations, la boucle et la
validation. Ce qui reste a ecrire, et ou les projets de detection echouent, est ici :
le catalogue (image + fichier d'annotations), la conversion des boites au format
YOLO, le data.yaml, et le pre/post-traitement ONNX reproduit a la main.

Ce module n'importe pas ultralytics : les scripts le font, pour que les tests et
les fonctions pures tournent sans l'extra `detection`.
"""

from __future__ import annotations  # permet d'ecrire les types modernes dans les annotations

import shutil  # shutil : copie de fichiers
import xml.etree.ElementTree as ET  # ElementTree : lit les fichiers XML (format des annotations Pascal VOC)
from collections import Counter  # Counter : compte les occurrences (ici, combien de boites par classe)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux de donnees (le catalogue des images)
import torch  # PyTorch : tenseurs ; utilise ici seulement pour la fonction nms
import yaml  # PyYAML : ecrit le fichier data.yaml attendu par Ultralytics
from torchvision.ops import nms  # NMS (non-maximum suppression) : garde la meilleure boite parmi celles qui se chevauchent

Box = tuple[str, float, float, float, float]  # (classe, xmin, ymin, xmax, ymax) en pixels

# --- Catalogue : une image + son fichier d'annotations (format Pascal VOC) ---


# lit un fichier d'annotations VOC ; renvoie (largeur, hauteur, liste des boites)
def parse_voc(xml_path: Path) -> tuple[int, int, list[Box]]:
    """Fichier VOC -> (largeur, hauteur, [(classe, xmin, ymin, xmax, ymax), ...])."""
    root = ET.parse(xml_path).getroot()  # lit le XML et prend la balise racine <annotation>
    width, height = int(root.findtext("size/width")), int(root.findtext("size/height"))  # taille de l'image lue dans <size>
    boxes: list[Box] = []  # boites trouvees, a remplir
    for obj in root.iter("object"):  # pour chaque objet annote (une balise <object> par defaut encadre)
        name = obj.findtext("name").strip()  # nom de la classe, sans espaces autour
        box = obj.find("bndbox")  # balise <bndbox> qui contient les coins de la boite
        # ajoute (classe, xmin, ymin, xmax, ymax) : chaque coordonnee est lue comme texte puis convertie en nombre reel
        boxes.append((name, *(float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))))
    return width, height, boxes  # taille de l'image et ses boites


# parcourt le dossier d'images et leurs annotations ; renvoie le catalogue (une ligne par image)
def catalog_voc(data_dir: Path, images_subdir: str = "IMAGES", annotations_subdir: str = "ANNOTATIONS") -> pd.DataFrame:
    """Un tableau (path, annotation, label, n_boxes, width, height). label = classe majoritaire des boites, pour stratifier."""
    rows = []  # une ligne (dictionnaire) par image
    for image_path in sorted((data_dir / images_subdir).glob("*.jpg")):  # pour chaque image .jpg, dans l'ordre alphabetique
        xml_path = data_dir / annotations_subdir / f"{image_path.stem}.xml"  # son annotation porte le meme nom (stem = sans extension)
        if not xml_path.exists():  # si l'annotation manque
            raise FileNotFoundError(f"annotation absente pour {image_path.name} : {xml_path}")  # on s'arrete : image inutilisable
        width, height, boxes = parse_voc(xml_path)  # lit la taille et les boites
        # classe la plus frequente parmi les boites (sert a equilibrer le decoupage) ; "background" s'il n'y a aucune boite
        label = Counter(name for name, *_ in boxes).most_common(1)[0][0] if boxes else "background"
        rows.append(  # ajoute la ligne de cette image
            # chemins, classe majoritaire, nombre de boites, taille de l'image
            {
                "path": image_path.as_posix(),  # chemin de l'image, ecrit avec des /
                "annotation": xml_path.as_posix(),  # chemin du fichier d'annotations
                "label": label,  # classe majoritaire de l'image (sert a stratifier le decoupage)
                "n_boxes": len(boxes),  # nombre de boites
                "width": width,  # largeur de l'image en pixels
                "height": height,  # hauteur de l'image en pixels
            }
        )
    if not rows:  # si aucune image n'a ete trouvee
        raise FileNotFoundError(f"aucune image .jpg dans {data_dir / images_subdir}")  # erreur claire : mauvais dossier
    return pd.DataFrame(rows)  # la liste devient un tableau pandas


# liste toutes les classes presentes dans les annotations ; renvoie une liste triee
def class_names_of_boxes(catalog: pd.DataFrame) -> list[str]:
    """Toutes les classes presentes dans les boites (pas seulement la majoritaire), triees."""
    names: set[str] = set()  # ensemble : chaque nom n'y figure qu'une fois
    for annotation in catalog["annotation"]:  # pour chaque fichier d'annotations
        names.update(name for name, *_ in parse_voc(Path(annotation))[2])  # ajoute les classes de ses boites ([2] = la liste des boites)
    return sorted(names)  # ordre alphabetique : il fixe le numero de chaque classe


# --- Conversion au format YOLO : "classe cx cy w h" normalises dans [0, 1] ---


# convertit les boites VOC (coins en pixels) en lignes de texte YOLO ; renvoie la liste des lignes
def voc_to_yolo_lines(width: int, height: int, boxes: list[Box], class_to_idx: dict[str, int]) -> list[str]:
    """Coins en pixels -> centre et taille relatifs. Les boites qui debordent d'un pixel sont ramenees dans l'image."""
    lines = []  # une ligne de texte par boite
    for name, x_min, y_min, x_max, y_max in boxes:  # pour chaque boite de l'image
        x_min, x_max = np.clip([x_min, x_max], 0, width)  # clip : force les x entre 0 et la largeur (pas de debordement)
        y_min, y_max = np.clip([y_min, y_max], 0, height)  # meme chose pour les y avec la hauteur
        cx, cy = (x_min + x_max) / 2 / width, (y_min + y_max) / 2 / height  # centre de la boite, divise par la taille -> entre 0 et 1
        w, h = (x_max - x_min) / width, (y_max - y_min) / height  # largeur et hauteur de la boite, relatives a l'image
        lines.append(f"{class_to_idx[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")  # "numero_classe cx cy w h" avec 6 decimales
    return lines  # les lignes du fichier .txt de l'image


# recopie le jeu de donnees dans le rangement attendu par Ultralytics ; renvoie le chemin du data.yaml
def export_yolo_dataset(catalog: pd.DataFrame, class_names: list[str], yolo_dir: Path) -> Path:
    """Ecrit images/<split>/x.jpg, labels/<split>/x.txt et data.yaml. Renvoie le chemin du data.yaml."""
    class_to_idx = {name: i for i, name in enumerate(class_names)}  # nom de classe -> numero (0, 1, 2...)
    for _, row in catalog.iterrows():  # pour chaque image du catalogue (iterrows donne (indice, ligne))
        image_dir, label_dir = yolo_dir / "images" / row["split"], yolo_dir / "labels" / row["split"]  # dossiers de son sous-ensemble
        image_dir.mkdir(parents=True, exist_ok=True)  # cree le dossier des images s'il n'existe pas
        label_dir.mkdir(parents=True, exist_ok=True)  # cree le dossier des etiquettes s'il n'existe pas
        source = Path(row["path"])  # chemin de l'image d'origine
        shutil.copy2(source, image_dir / source.name)  # copie l'image (copy2 garde aussi la date du fichier)
        width, height, boxes = parse_voc(Path(row["annotation"]))  # lit ses boites VOC
        lines = voc_to_yolo_lines(width, height, boxes, class_to_idx)  # les convertit au format YOLO
        # ecrit x.txt : une ligne par boite, retour a la ligne final s'il y a des boites (fichier vide = image sans objet)
        (label_dir / f"{source.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    data_yaml = {  # contenu du data.yaml : la "carte" du jeu de donnees pour Ultralytics
        "path": str(yolo_dir.resolve()),  # dossier racine, en chemin absolu
        "train": "images/train",  # images d'entrainement (relatif a path)
        "val": "images/val",  # images de validation
        "test": "images/test",  # images de test
        "names": dict(enumerate(class_names)),  # numero -> nom de classe, ex. {0: "crazing", 1: "inclusion"}
    }
    (yolo_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")  # ecrit le fichier
    return yolo_dir / "data.yaml"  # chemin a donner a model.train(data=...)


# --- Pre et post-traitement ONNX : ce qu'un programme sans Ultralytics doit reproduire ---


# letterbox : met l'image au format carre sans la deformer ; renvoie (image carree, ratio, (marge x, marge y))
def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Redimensionne sans deformer, complete en gris 114 pour obtenir un carre. Renvoie (image, ratio, (pad_x, pad_y))."""
    import cv2  # OpenCV : traitement d'images ; importe ici car c'est une dependance optionnelle

    height, width = image.shape[:2]  # hauteur et largeur (les 2 premieres dimensions du tableau)
    ratio = min(size / height, size / width)  # facteur d'echelle : le plus grand cote tiendra pile dans le carre
    new_w, new_h = round(width * ratio), round(height * ratio)  # nouvelles dimensions, proportions conservees
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # redimensionne (OpenCV attend largeur, hauteur)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2  # marges a gauche et en haut pour centrer l'image
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)  # carre rempli de gris 114 (la valeur utilisee par Ultralytics)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized  # colle l'image redimensionnee au centre du carre
    return canvas, ratio, (pad_x, pad_y)  # ratio et marges servent a ramener les boites dans l'image d'origine


# prepare une image OpenCV pour le modele ONNX ; renvoie (tenseur pret, ratio, marges)
def preprocess(image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """BGR (OpenCV) -> letterbox -> RGB -> [0, 1] -> NCHW float32 : le pretraitement exact d'Ultralytics."""
    boxed, ratio, pad = letterbox(image_bgr, size)  # image carree sans deformation
    tensor = boxed[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB ; puis reels entre 0 et 1
    # transpose : (H, W, C) -> (C, H, W) ; np.newaxis ajoute la dimension du lot -> (1, C, H, W) = format NCHW ;
    # ascontiguousarray range les nombres en memoire dans le nouvel ordre (ONNX Runtime l'exige)
    return np.ascontiguousarray(tensor.transpose(2, 0, 1))[np.newaxis], ratio, pad


# transforme la sortie brute du modele en boites finales ; renvoie un tableau (n, 6)
def postprocess(output: np.ndarray, ratio: float, pad: tuple[int, int], size: int, conf: float = 0.25, iou: float = 0.7) -> np.ndarray:
    """Sortie brute (1, 4 + C, A) -> boites (n, 6) : xmin, ymin, xmax, ymax, score, classe, dans l'image d'origine."""
    pred = output[0].T  # (A, 4 + C) : pour chaque ancre, cx cy w h (pixels de l'image letterboxee) puis un score par classe
    scores, classes = pred[:, 4:].max(axis=1), pred[:, 4:].argmax(axis=1)  # meilleur score de chaque ancre et sa classe
    keep = scores > conf  # masque vrai / faux : on garde les ancres assez sures d'elles
    pred, scores, classes = pred[keep], scores[keep], classes[keep]  # filtre les trois tableaux de la meme facon
    if len(pred) == 0:  # si plus aucune boite ne passe le seuil
        return np.zeros((0, 6), dtype=np.float32)  # renvoie un tableau vide, mais avec la bonne forme
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]  # centre et taille de chaque boite
    boxes = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])  # centre/taille -> coins (xmin, ymin, xmax, ymax)
    offsets = classes[:, None] * (size + 1)  # NMS par classe : chaque classe est decalee dans son propre coin
    # NMS : parmi des boites qui se chevauchent (IoU > iou), garde la mieux notee ; renvoie les indices gardes
    kept = nms(torch.from_numpy(boxes + offsets).float(), torch.from_numpy(scores).float(), iou).numpy()
    boxes = (boxes[kept] - np.array([pad[0], pad[1], pad[0], pad[1]])) / ratio  # on annule le letterbox
    return np.column_stack([boxes, scores[kept], classes[kept]]).astype(np.float32)  # une ligne par boite : coins, score, classe


# IoU (intersection sur union) : recouvrement de deux boites, de 0 (disjointes) a 1 (identiques) ; renvoie un reel
def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU de deux boites (xmin, ymin, xmax, ymax)."""
    # aire de l'intersection = largeur commune x hauteur commune (0 si elles ne se touchent pas)
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter  # aire de a + aire de b - partie commune
    return float(inter / union) if union > 0 else 0.0  # protege contre une division par zero (boites d'aire nulle)


# dessine des boites et leur nom sur une copie de l'image ; renvoie l'image annotee
def draw_boxes(image_bgr: np.ndarray, boxes: list[Box]) -> np.ndarray:
    """Dessine des boites (nom, xmin, ymin, xmax, ymax) sur une copie d'une image BGR."""
    import cv2  # OpenCV : dessin de rectangles et de texte

    canvas = image_bgr.copy()  # copie : on ne dessine pas sur l'image d'origine
    for name, x_min, y_min, x_max, y_max in boxes:  # pour chaque boite
        # rectangle vert (0, 255, 0 en BGR) d'epaisseur 2 ; OpenCV veut des coordonnees entieres
        cv2.rectangle(canvas, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        # ecrit le nom juste au-dessus de la boite (au moins a y=10 pour rester dans l'image), petite police verte
        cv2.putText(canvas, name, (int(x_min), max(int(y_min) - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return canvas  # l'image annotee
