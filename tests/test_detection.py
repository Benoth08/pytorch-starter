"""Tests du module detection : lecture VOC, conversion YOLO, letterbox, post-traitement. Sans Ultralytics ni vraies donnees."""

from pathlib import Path  # Path : represente un chemin de fichier

import numpy as np  # numpy : images et sorties factices
import pytest  # pytest : outil qui trouve et lance les tests ; fixtures et comparaisons approchees
from PIL import Image  # Pillow : enregistre les images factices

# les fonctions testees (voir src/vision_starter/detection.py)
from vision_starter.detection import (
    box_iou,  # recouvrement de deux boites
    catalog_voc,  # catalogue images + annotations
    export_yolo_dataset,  # ecriture au format YOLO
    letterbox,  # mise au carre sans deformation
    parse_voc,  # lecture d'un fichier VOC
    postprocess,  # sortie brute -> boites finales
    preprocess,  # image -> tenseur pret pour ONNX
    voc_to_yolo_lines,  # conversion des boites en lignes YOLO
)

# fichier d'annotations VOC factice : image 200x100, une boite "crack" et une boite "hole" qui deborde (xmax 205 > 200)
VOC = """<annotation><size><width>200</width><height>100</height></size>
<object><name>crack</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>60</ymax></bndbox></object>
<object><name>hole</name><bndbox><xmin>150</xmin><ymin>0</ymin><xmax>205</xmax><ymax>40</ymax></bndbox></object>
</annotation>"""


# fixture = fonction preparatoire : pytest passe son resultat a tout test qui a un argument voc_folder
@pytest.fixture
def voc_folder(tmp_path: Path) -> Path:  # cree un mini jeu VOC dans un dossier temporaire ; renvoie ce dossier
    """Deux images 200x100 avec le meme fichier VOC : assez pour un catalogue et un export."""
    (tmp_path / "IMAGES").mkdir()  # dossier des images
    (tmp_path / "ANNOTATIONS").mkdir()  # dossier des annotations
    for name in ("a_1", "a_2"):  # deux images
        Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)).save(tmp_path / "IMAGES" / f"{name}.jpg")  # image noire 200x100
        (tmp_path / "ANNOTATIONS" / f"{name}.xml").write_text(VOC)  # son annotation (le XML factice)
    return tmp_path  # le dossier racine


# test : lecture d'un fichier VOC et construction du catalogue
def test_parse_voc_and_catalog(voc_folder):
    width, height, boxes = parse_voc(voc_folder / "ANNOTATIONS" / "a_1.xml")  # lit le XML
    # bonne taille, 2 boites, premiere boite lue correctement
    assert (width, height) == (200, 100) and len(boxes) == 2 and boxes[0] == ("crack", 10.0, 20.0, 50.0, 60.0)
    catalog = catalog_voc(voc_folder)  # catalogue du dossier
    assert list(catalog.columns) == ["path", "annotation", "label", "n_boxes", "width", "height"]  # colonnes attendues, dans l'ordre
    assert len(catalog) == 2 and catalog["n_boxes"].tolist() == [2, 2]  # 2 images, 2 boites chacune


# test : conversion YOLO normalisee entre 0 et 1, boite qui deborde ramenee dans l'image
def test_voc_to_yolo_normalizes_and_clips():
    _, _, boxes = parse_voc_from_text()  # boites du XML factice (taille ignoree avec _)
    lines = voc_to_yolo_lines(200, 100, boxes, {"crack": 0, "hole": 1})  # conversion avec les numeros de classes
    assert lines[0] == "0 0.150000 0.400000 0.200000 0.400000"  # centre (30, 40), taille (40, 40) sur 200 x 100
    cls, cx, cy, w, h = lines[1].split()  # decoupe la 2e ligne en ses 5 champs
    assert cls == "1" and float(cx) + float(w) / 2 <= 1.0  # xmax 205 ramene a 200


# fonction d'aide (pas un test : son nom ne commence pas par test_) ; renvoie le resultat de parse_voc sur le XML factice
def parse_voc_from_text(tmp_path=None):
    """Le meme XML, parse par la fonction du projet (via un fichier temporaire)."""
    import tempfile  # tempfile : cree des fichiers temporaires

    # cree un fichier .xml temporaire (delete=False : il survit a la fermeture, pour pouvoir le relire)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as handle:
        handle.write(VOC)  # y ecrit le XML factice
    return parse_voc(Path(handle.name))  # le relit avec la fonction du projet


# test : l'export YOLO cree l'arborescence, les fichiers d'etiquettes et le data.yaml
def test_export_yolo_dataset_writes_tree_and_yaml(voc_folder, tmp_path):
    catalog = catalog_voc(voc_folder).assign(split=["train", "val"])  # catalogue avec a_1 en train et a_2 en val
    data_yaml = export_yolo_dataset(catalog, ["crack", "hole"], tmp_path / "yolo")  # export ; renvoie le chemin du data.yaml
    assert (tmp_path / "yolo" / "images" / "train" / "a_1.jpg").exists()  # l'image a ete copiee au bon endroit
    assert (tmp_path / "yolo" / "labels" / "val" / "a_2.txt").read_text().count("\n") == 2  # 2 lignes = 2 boites
    assert "names:" in data_yaml.read_text() and "1: hole" in data_yaml.read_text()  # data.yaml liste les classes numerotees


# test : letterbox (taille, ratio, marges, couleur de remplissage) et forme du tenseur pretraite
def test_letterbox_and_preprocess():
    image = np.zeros((100, 200, 3), dtype=np.uint8)  # image noire 200 de large, 100 de haut
    boxed, ratio, (pad_x, pad_y) = letterbox(image, 64)  # mise au carre 64x64
    # carre 64x64 ; ratio = 64 / 200 ; l'image fait 64x32 : 16 pixels de marge en haut et en bas
    assert boxed.shape == (64, 64, 3) and ratio == 0.32 and (pad_x, pad_y) == (0, 16)
    assert boxed[0, 0, 0] == 114  # bande de remplissage
    tensor, _, _ = preprocess(image, 64)  # pretraitement complet
    assert tensor.shape == (1, 3, 64, 64) and tensor.dtype == np.float32 and tensor.max() <= 1.0  # NCHW, reels, entre 0 et 1


# test : post-traitement = filtrage par score, NMS par classe et retour aux coordonnees de l'image d'origine
def test_postprocess_filters_nms_and_undoes_letterbox():
    # 4 ancres, 2 classes : deux boites quasi identiques de classe 0 (NMS garde la meilleure), une de classe 1, une sous le seuil.
    size = 64  # cote de l'image letterboxee
    anchors = np.array(  # sortie brute factice : une ligne par ancre
        [  # liste des ancres
            [32, 32, 20, 20, 0.9, 0.1],  # cx cy w h, score classe 0, score classe 1
            [33, 32, 20, 20, 0.8, 0.1],  # recouvre la premiere -> supprimee par la NMS
            [10, 10, 8, 8, 0.1, 0.7],  # classe 1
            [50, 50, 8, 8, 0.1, 0.1],  # sous conf
        ]
    ).T[np.newaxis]  # (1, 4 + C, A)
    boxes = postprocess(anchors, ratio=0.5, pad=(0, 16), size=size, conf=0.25, iou=0.7)  # post-traitement a tester
    assert boxes.shape == (2, 6)  # 2 boites gardees, 6 valeurs chacune
    assert boxes[0, 5] == 0 and boxes[0, 4] == pytest.approx(0.9)  # la 1re est de classe 0 avec le score 0,9 (approx : arrondis)
    assert np.allclose(boxes[0, :4], [(22 - 0) / 0.5, (22 - 16) / 0.5, (42 - 0) / 0.5, (42 - 16) / 0.5])  # letterbox annule
    assert box_iou(np.array([0, 0, 10, 10]), np.array([5, 5, 15, 15])) == pytest.approx(25 / 175)  # intersection 25, union 175
    assert postprocess(anchors, 0.5, (0, 0), size, conf=0.95).shape == (0, 6)  # seuil trop haut : aucune boite, forme (0, 6)
