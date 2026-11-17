# vision-starter : projet de depart pour la classification d'images en PyTorch

Compagnon du cours "La vision par ordinateur en 13 etapes, avec PyTorch". Deux versions du meme projet :

- `notebooks/01_projet_complet.py` : les 13 etapes du cours, cellule par cellule (mode cellules VS Code, `# %%`) ;
- `src/` + `scripts/` + `configs/` : le meme projet industrialise (chapitre "Du notebook au projet" du cours).

## Le fil en 13 etapes

Le cours, les notebooks et les scripts suivent le meme fil. Dans `01_projet_complet.py` et `02_projet_minimal.py`,
un commentaire `# === Etape N - nom ===` ouvre chaque etape.

| Phase | Etape | Cellules de `01_projet_complet.py` | Version scripts |
|---|---|---|---|
| Donnees | 1 - Imports et reglages | `imports`, `config`, `seed_device` | `configs/*.yaml`, `config.py`, `utils.py` |
| Donnees | 2 - Importer et explorer | `explore_count`, `explore_show`, `explore_sizes` | `scripts/get_neu_cls.py` (recuperer les images) |
| Donnees | 3 - Cataloguer | `catalog`, `catalog_variants` | `data.py`, `scripts/make_split.py` |
| Donnees | 4 - Decouper | `split`, `split_check` | `data.py`, `scripts/make_split.py` |
| Donnees | 5 - Pretraiter | `mean_std`, `transforms` (taille, normalisation) | `data.py` (`compute_mean_std`, `build_transforms`) |
| Donnees | 6 - Augmenter | `transforms` (retournements aleatoires) | `data.py` (`build_transforms`) |
| Donnees | 7 - Charger par lots | `dataset`, `dataloader`, `show_batch`, `class_weights` | `data.py` (`ImageDataset`, `build_loader`, `class_weights`) |
| Modele | 8 - Modele | `model_mlp`, `model_cnn`, `model_check`, `model_summary`, puis `pretrained_direct`, `transfer_*`, `feature_extraction`, `load_foreign_weights` | `model.py` |
| Modele | 9 - Perte et optimiseur | debut de `train_step`, fin de `fit` | `scripts/train.py` |
| Apprentissage | 10 - Entrainer | `train_step`, `sanity_*`, `engine`, `fit`, `curves` | `engine.py`, `scripts/train.py` |
| Apprentissage | 11 - Evaluer | `test_metrics`, `confusion`, `worst_images` | `scripts/evaluate.py` |
| Livraison | 12 - Sauvegarder et exporter | `save_model`, `resume`, `onnx_export` | `scripts/train.py` (`model.pt`), `scripts/export_onnx.py` |
| Livraison | 13 - Predire | `load_predict` | `scripts/predict.py` |

Les autres notebooks reprennent ce fil, avec les memes numeros d'etape dans leurs commentaires :

| Notebook | Dans le cours |
|---|---|
| `00_bases_tenseurs.py` | chapitre "Avant de commencer" : tenseurs et gradients |
| `02_projet_minimal.py` | les 13 etapes sur une page : la fiche a reecrire de tete |
| `03_detection_yolo.py` | chapitre "Detection" (YOLO) |
| `04_references.py` | complements des etapes 9 (pertes, hyperparametres), 11 (metriques) et 8 (gel des poids) |
| `05_anomalie.py` | chapitre "Anomalies" (auto-encodeur, PatchCore maison), puis etape 2 (explorer sans etiquettes) et etape 11 (Grad-CAM) |
| `05b_anomalib_patchcore.py` | chapitre "Anomalies" (PatchCore avec anomalib) |
| `06_segmentation.py` | chapitre "Segmentation" (U-Net) |

## Installation

```
uv sync --extra dev                 # cree .venv et installe tout (PyTorch version CPU)
uv run python -c "import torch; print(torch.__version__)"
uv sync --extra dev --extra detection --extra anomaly   # en plus : ultralytics + OpenCV (chapitre Detection), anomalib (chapitre Anomalies)
```

`uv sync` ecrit `uv.lock` (les versions exactes installees) a cote de `pyproject.toml`, au premier lancement : le fichier
n'est pas fourni, car il se calcule sur ton poste, avec acces a l'index de PyTorch. Mets-le dans git avec le code : c'est lui
qui permet de reinstaller exactement les memes versions plus tard (`uv sync --frozen`).

Sans uv : `python -m venv .venv`, activer, `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`, `pip install -e ".[dev]"`.

## Donnees

```
uv run python scripts/get_neu_cls.py
```

Telecharge NEU-CLS (1800 images, 6 classes) et le range selon les trois organisations du cours :
`data/raw/neu_cls_split` (train/val/test), `data/raw/neu_cls_flat` (un dossier par classe), `data/raw/neu_cls_csv` (images + CSV).

Pour tes propres images : les ranger dans un dossier par classe sous `data/raw/mes_images/` et changer `data_dir` dans `configs/default.yaml`
(ou `data_layout: split_folders` / `csv` selon le cas, voir `configs/layout_*.yaml`).

Pour la detection (chapitre "Detection" du cours) : `uv run python scripts/get_neu_det.py` puis `notebooks/03_detection_yolo.py` ou les scripts `*_yolo.py` (ci-dessous).

Pour les anomalies (chapitre "Anomalies") : `uv run python scripts/get_mvtec_bottle.py` (MVTec AD, categorie bottle, environ 300 Mo).
Pour la segmentation (chapitre "Segmentation") : `uv run python scripts/get_magnetic_tile.py` (Magnetic Tile Defect, environ 110 Mo).

## Les cinq scripts, dans l'ordre

```
uv run python scripts/make_split.py  --config configs/default.yaml   # etapes 3 et 4 : cataloguer, decouper
uv run python scripts/train.py       --config configs/default.yaml   # etapes 5 a 10 : des transformations a l'entrainement
uv run python scripts/evaluate.py    --config configs/default.yaml   # etape 11 : evaluer sur le test
uv run python scripts/export_onnx.py --config configs/default.yaml   # etape 12 : exporter en ONNX, verifier, ecrire la carte
uv run python scripts/predict.py     --config configs/default.yaml --source chemin/image_ou_dossier   # etape 13 : predire
```

Les scripts suivent l'ordre des etapes. L'export vient apres l'evaluation : la carte du modele (`model_card.json`) recopie les
chiffres de `test_metrics.json`.

Resultats dans `outputs/<nom de la config>/` : `config_used.yaml`, `train.log`, `history.csv`, `curves.png`, `model.pt`,
`test_metrics.json`, `classification_report.txt`, `confusion_matrix.png`, `model.onnx`, `model_card.json`.

Ce qui se regle dans le YAML, sans toucher au code (cles validees par `src/vision_starter/config.py`) :

| Etape | Cles |
|---|---|
| 3 et 4 | `data_dir`, `data_layout`, `labels_csv`, `split_file`, `val_fraction`, `test_fraction`, `seed` |
| 5 | `image_size`, `normalization` |
| 6 | `augment` : liste parmi `hflip`, `vflip`, `rotate`, `brightness` (`[]` = aucune augmentation) |
| 7 | `batch_size`, `num_workers` |
| 8 | `model_name`, `pretrained`, `freeze_backbone`, `unfreeze_layers` |
| 9 | `optimizer` (`adamw` ou `sgd`), `momentum`, `learning_rate`, `backbone_learning_rate`, `weight_decay`, `scheduler_factor`, `scheduler_patience`, `class_weighting` |
| 10 | `epochs`, `patience` |
| 12 | `onnx_opset`, `output_dir` |

`test_metrics.json` donne le F1 macro avec son intervalle de confiance a 95 % (bootstrap) et le nombre d'erreurs.
`evaluate.py`, `export_onnx.py` et `predict.py` recreent l'architecture notee dans `model.pt`, sans retelecharger de poids.

Autres configurations : `configs/resnet18.yaml` (transfer learning, tete seule), `configs/resnet18_finetune.yaml` (layer4 degele, deux taux),
`configs/small_resnet.yaml` (un modele custom enregistre dans `MODEL_REGISTRY`), `configs/quick_check.yaml` (2 epoques, pour verifier la chaine).

## Les autres taches (chapitres "Detection", "Anomalies" et "Segmentation" du cours)

```
uv run python scripts/train_anomaly.py        --config configs/anomaly_ae.yaml      # auto-encodeur : split, entrainement, seuils
uv run python scripts/evaluate_anomaly.py     --config configs/anomaly_ae.yaml      # AUROC image / pixel, rappel au seuil, cartes
uv run python scripts/export_onnx_anomaly.py  --config configs/anomaly_ae.yaml      # scoreur complet en ONNX
uv run python scripts/train_patchcore.py      --config configs/patchcore.yaml       # anomalib : memoire, test, export (extra anomaly)
uv run python scripts/train_segmentation.py   --config configs/segmentation.yaml    # U-Net : split, entrainement, seuil
uv run python scripts/evaluate_segmentation.py --config configs/segmentation.yaml   # IoU / Dice par classe, figure
uv run python scripts/export_onnx_segmentation.py --config configs/segmentation.yaml
```

Detection avec YOLO11 / Ultralytics (extra `detection`) ; `configs/yolo_quick.yaml` (2 epoques) pour verifier la chaine :

```
uv run python scripts/prepare_yolo.py     --config configs/yolo.yaml   # catalogue VOC, decoupage, arborescence YOLO + data.yaml (une fois)
uv run python scripts/train_yolo.py       --config configs/yolo.yaml   # Ultralytics : 60 epoques (2 h sur CPU)
uv run python scripts/evaluate_yolo.py    --config configs/yolo.yaml   # mAP sur le test, AP par classe
uv run python scripts/predict_yolo.py     --config configs/yolo.yaml --source chemin/image_ou_dossier
uv run python scripts/export_onnx_yolo.py --config configs/yolo.yaml   # ONNX sans NMS + post-traitement maison verifie
```

Resultats dans `outputs/anomaly_ae/`, `outputs/patchcore/`, `outputs/segmentation/`, `outputs/yolo/`.

## Qualite

```
uv run pytest
uv run ruff check . && uv run ruff format .
```

## Resultats de reference (CPU 2 coeurs)

| Configuration | Epoques | Test accuracy | F1 macro | Duree |
|---|---|---|---|---|
| `default.yaml` (SimpleCNN de zero) | 15 | 0.970 | 0.970 | environ 8 min |
| `resnet18.yaml` (tete seule, backbone gele) | 6 | 0.970 | 0.970 | environ 2 min |
| `resnet18_finetune.yaml` (tete + layer4 degele, deux taux) | 8 | 1.000 | 1.000 | environ 2,5 min |
| `small_resnet.yaml` (modele custom, blocs residuels, 1,2 M parametres) | 15 | 0.948 | 0.948 | environ 8 min |
| notebook, ResNet18 tete puis layer4 degele | 3 + 3 | 0.993 | 0.993 | environ 2,5 min |
| notebook, ResNet18 gele + regression logistique | - | 0.996 | - | 20 s |

Anomalies et segmentation (MVTec AD bottle : 209 normales, test 20 normales + 63 defauts ; Magnetic Tile : 542 images, 78 en test, decoupage par groupes de quasi-doublons) :

| Configuration | Epoques | Metriques sur le test | Duree |
|---|---|---|---|
| `anomaly_ae.yaml` (auto-encodeur, scripts) | 30 | AUROC image 0.776, pixel 0.719 ; rappel 0.46 a 0 fausse alarme | 8 min |
| notebook 05, auto-encodeur | 30 | AUROC image 0.790, pixel 0.747 | 7 min |
| notebook 05, PatchCore maison (ResNet18, memoire 10 % au hasard) | 0 | AUROC image 1.000, pixel 0.982 | 30 s |
| `patchcore.yaml` (anomalib, ResNet18, coreset 1 %) | 0 | AUROC image 1.000, pixel 0.982, F1 image 0.992 | 5 min (coreset) |
| `segmentation.yaml` (U-Net ResNet18, scripts) | 12 | IoU 0.769, Dice 0.869 (seuil 0.4) | 18 min |
| notebook 06, U-Net | 12 | IoU 0.732, Dice 0.845 (seuil 0.3) | 19 min |

Les deux entrainements de segmentation ne sont pas converges (12 epoques) : ils verifient la chaine, un vrai entrainement en demande 30 a 40.
A 12 epoques et 256 pixels, aucune fissure (`crack`) n'est trouvee (IoU 0.00) : lire le detail par classe de `test_metrics.json`, pas seulement le chiffre global.

Le decoupage de segmentation regroupe d'abord les quasi-doublons (`near_duplicate_groups` dans `data.py`, cles `duplicate_max_hamming` et
`duplicate_min_correlation` du YAML), puis decoupe sans separer un groupe (`make_stratified_grouped_split`, controle par `check_group_leak`).
La cle `min_defect_area` regle la decision par image.

Detection (NEU-DET, 1 770 images, 266 en test) :

| Configuration | Epoques | Test mAP@0.5 | mAP@0.5:0.95 | Duree |
|---|---|---|---|---|
| `yolo_quick.yaml` (YOLO11n, verification de la chaine) | 2 | 0.40 | 0.18 | 5 min |
| `yolo.yaml` (YOLO11n ; chiffres du depot vision-reference, meme jeu) | 60 | 0.706 | 0.410 | 1 h 56 |
