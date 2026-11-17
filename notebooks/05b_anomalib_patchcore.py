# %% [markdown]
# # PatchCore avec anomalib : la version bibliotheque
#
# Le notebook 05 a construit un PatchCore simplifie a la main. anomalib (Intel, Apache-2.0) contient la version
# complete : coreset, k plus proches voisins, seuils calibres, visualisation, export. Ce notebook refait le meme
# travail avec la bibliotheque, sur les memes images (MVTec AD bottle), pour comparer et pour apprendre a lire
# ce que la bibliotheque cache. Installation : `uv sync --extra anomaly` (ou `pip install anomalib`).

# %% anomalib_imports
from __future__ import annotations  # permet d'ecrire les types modernes sans souci de version de Python

import json  # ecriture de fichiers JSON (ici la fiche du modele exporte)
import logging  # reglage des messages affiches par les bibliotheques
import time  # mesure du temps ecoule
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import anomalib  # anomalib : bibliotheque de detection d'anomalies (images normales vs defauts)
import matplotlib.pyplot as plt  # matplotlib : affichage d'images et de graphiques
import numpy as np  # numpy : calcul sur des tableaux de nombres
import onnxruntime as ort  # onnxruntime : execute un modele exporte au format ONNX (format standard, sans PyTorch)
import pandas as pd  # pandas : tableaux de donnees (DataFrame)
import torch  # PyTorch : tenseurs (tableaux de nombres) et apprentissage profond
from anomalib.data import Folder, MVTecAD  # chargeurs de donnees : un dossier quelconque (Folder) ou le jeu MVTec AD
from anomalib.deploy import ExportType  # liste des formats d'export possibles (ONNX, OpenVINO...)
from anomalib.engine import Engine  # le "moteur" qui lance fit (apprentissage), test, predict et export
from anomalib.models import Patchcore  # le modele PatchCore : compare chaque morceau d'image a une memoire de morceaux normaux
from lightning import seed_everything  # fixe toutes les graines de hasard d'un coup (Lightning est utilise par anomalib)
from PIL import Image  # PIL (Pillow) : ouvrir les fichiers images
from torchvision import models  # modeles connus (ResNet...) avec poids pre-entraines

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)  # Lightning est bavard ; on garde nos propres prints
logging.getLogger("anomalib").setLevel(logging.ERROR)  # idem pour anomalib : seulement les erreurs
torch.manual_seed(42)  # fixe le hasard de PyTorch

PC_CONFIG = {  # tous les reglages de PatchCore au meme endroit
    "data_root": "data/raw",  # le dossier qui contient mvtec_bottle/ (scripts/get_mvtec_bottle.py)
    "category": "mvtec_bottle",  # sous-dossier du jeu (une categorie d'objet : des bouteilles)
    "output_dir": "outputs/anomalib",  # dossier des resultats
    "image_size": 256,  # cote des images donnees au modele
    "batch_size": 16,  # 16 images traitees ensemble (un lot / batch)
    "backbone": "resnet18",  # le meme que le PatchCore maison, pour isoler ce que la bibliotheque ajoute
    "layers": ["layer2", "layer3"],  # couches du ResNet dont on prend les caracteristiques (niveau intermediaire)
    # memoire = 1 % des vecteurs, choisis par coreset (pas au hasard) ; 10 % = 40 min de selection sur CPU
    "coreset_sampling_ratio": 0.01,
    "num_neighbors": 9,  # score image = distance ponderee aux 9 voisins, pas seulement au plus proche
    "seed": 42,  # graine du hasard
}
OUTPUT_DIR = Path(PC_CONFIG["output_dir"])  # chemin du dossier de sortie
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # cree le dossier s'il n'existe pas
print("anomalib", anomalib.__version__, "| torch", torch.__version__)  # versions utilisees (utile si un resultat differe)

# %% anomalib_data
# Le datamodule remplace catalogue + decoupage + Dataset + DataLoader (etapes 3 a 7). MVTecAD lit l'arborescence
# officielle ; root/category doit exister, sinon il telecharge tout MVTec AD (5 Go).
datamodule = MVTecAD(  # le "datamodule" : sait lire, decouper et servir les images par lots
    root=PC_CONFIG["data_root"],  # dossier racine des donnees
    category=PC_CONFIG["category"],  # categorie (sous-dossier) a utiliser
    train_batch_size=PC_CONFIG["batch_size"],  # taille des lots d'apprentissage
    eval_batch_size=PC_CONFIG["batch_size"],  # taille des lots d'evaluation
    num_workers=0,  # 0 : chargement dans le processus principal (plus simple, marche partout)
    seed=PC_CONFIG["seed"],  # graine du hasard
)
datamodule.setup()  # lit l'arborescence et prepare les jeux train / test
# Dessous, c'est un DataFrame : le meme catalogue qu'a l'etape 3 du notebook 05, avec les memes colonnes.
# colle les tableaux train et test en un seul, avec une colonne split pour savoir d'ou vient chaque ligne
samples = pd.concat([datamodule.train_data.samples.assign(split="train"), datamodule.test_data.samples.assign(split="test")])
for column in ("image_path", "mask_path"):  # chemins absolus chez anomalib : relatifs au projet pour l'affichage
    samples[column] = samples[column].map(lambda p: str(Path(p).relative_to(Path.cwd())) if p else p)  # garde vide si pas de masque
print(samples[["image_path", "label", "label_index", "mask_path", "split"]].iloc[[0, 1, -1]].to_string())  # 3 lignes d'exemple
print(pd.crosstab(samples["label"], samples["split"]))  # nombre d'images normales / anormales dans train et test
batch = next(iter(datamodule.train_dataloader()))  # un premier lot, pour voir sa forme
print(  # affiche le type du lot, la forme et le type des images, et leurs valeurs min / max
    "un lot :",  # texte affiche
    type(batch).__name__,  # nom du type d'objet renvoye par anomalib
    "| image",  # texte affiche
    tuple(batch.image.shape),  # forme (N, C, H, W) : lot, canaux, hauteur, largeur
    batch.image.dtype,  # type des valeurs (float32)
    f"[{batch.image.min():.2f}, {batch.image.max():.2f}]",  # plage des valeurs : 0 a 1 attendu
)

# Pour TES images, sans arborescence MVTec : Folder prend un dossier de normales, des dossiers de defauts,
# et (facultatif) les masques. Memes noms de fichiers entre un defaut et son masque.
own_images = Folder(  # meme role que MVTecAD, mais pour une arborescence choisie
    name="bottle",  # nom du jeu
    root=Path(PC_CONFIG["data_root"]) / PC_CONFIG["category"],  # dossier racine
    normal_dir="train/good",  # images normales pour apprendre
    normal_test_dir="test/good",  # images normales pour tester
    abnormal_dir=["test/broken_large", "test/broken_small", "test/contamination"],  # dossiers d'images avec defauts
    mask_dir=["ground_truth/broken_large", "ground_truth/broken_small", "ground_truth/contamination"],  # masques (blanc = defaut)
    train_batch_size=PC_CONFIG["batch_size"],  # taille des lots d'apprentissage
    eval_batch_size=PC_CONFIG["batch_size"],  # taille des lots d'evaluation
    num_workers=0,  # chargement dans le processus principal
    seed=PC_CONFIG["seed"],  # graine du hasard (decoupage validation / test)
)
own_images.setup()  # lit les dossiers et prepare les jeux
print(  # combien d'images dans chaque partie
    "Folder :",  # texte affiche
    len(own_images.train_data),  # nombre d'images normales d'apprentissage
    "normales pour apprendre |",  # texte affiche
    len(own_images.val_data),  # nombre d'images de validation
    "en validation |",  # texte affiche
    len(own_images.test_data),  # nombre d'images de test
    "en test",  # texte affiche
)

# %% anomalib_model
# Le modele est un LightningModule : il contient le reseau, le pre-traitement (resize + normalisation ImageNet),
# le post-traitement (seuils, normalisation des scores) et les metriques. Pas d'epoques : fit = extraire les
# vecteurs des normales, puis en garder 1 % par coreset (les plus representatifs, pas un tirage au hasard).
# backbone : soit un nom timm ("resnet18", "wide_resnet50_2"... poids telecharges depuis Hugging Face), soit un
# nn.Module a toi. Ici le ResNet18 de torchvision (etape 8 du cours) : exactement les memes poids que le PatchCore maison.
# Le reglage du papier : wide_resnet50_2 (vecteurs de 1 536 valeurs au lieu de 384, memoire 4 x plus grosse, 3 a 4 x
# plus lent sur CPU) : backbone=models.wide_resnet50_2(weights="DEFAULT"), ou backbone="wide_resnet50_2" via timm.
backbone = getattr(models, PC_CONFIG["backbone"])(weights="DEFAULT")  # equivaut a models.resnet18(weights="DEFAULT") : poids ImageNet
model = Patchcore(  # cree le modele PatchCore
    backbone=backbone,  # reseau qui extrait les caracteristiques (backbone = "colonne vertebrale")
    layers=PC_CONFIG["layers"],  # couches utilisees
    coreset_sampling_ratio=PC_CONFIG["coreset_sampling_ratio"],  # part des vecteurs gardes en memoire
    num_neighbors=PC_CONFIG["num_neighbors"],  # nombre de voisins pour le score
)
print("pre-traitement :", model.pre_processor.transform)  # montre le pretraitement que le modele applique lui-meme
engine = Engine(  # le moteur qui pilote l'apprentissage et l'evaluation
    default_root_dir=OUTPUT_DIR,  # dossier ou il ecrit ses fichiers (images, modele)
    accelerator="cpu",  # calcul sur le processeur
    devices=1,  # un seul appareil
    logger=False,  # pas de journal d'entrainement
    enable_progress_bar=False,  # pas de barre de progression
    enable_model_summary=False,  # pas de resume du modele a l'ecran
)
# le coreset part d'un vecteur tire au hasard : graine fixee juste avant, pour retrouver la meme memoire
seed_everything(PC_CONFIG["seed"])
start = time.time()  # heure de depart
engine.fit(model, datamodule=datamodule)  # "apprentissage" : extrait les vecteurs des images normales et remplit la memoire
fit_seconds = time.time() - start  # duree du fit en secondes
bank = model.model.memory_bank  # la memoire : un tableau (nombre de vecteurs, taille d'un vecteur)
print(f"fit en {fit_seconds:.0f}s | memoire : {tuple(bank.shape)} vecteurs (sur {len(datamodule.train_data)} images)")  # resume

# %% anomalib_test
# test = scores sur le test, seuils, metriques. Attention au protocole par defaut : la validation EST le test
# (val_split_mode="same_as_test"), donc le seuil F1 est choisi sur le test. L'AUROC ne depend pas du seuil ;
# le F1 affiche est optimiste. Protocole propre : val_split_mode="from_test" (une moitie du test pour le seuil).
start = time.time()  # heure de depart
results = engine.test(model, datamodule=datamodule)[0]  # evalue sur le test ; [0] : dictionnaire des metriques
print(f"test en {time.time() - start:.0f}s")  # duree du test
for name in ("image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"):  # pour chaque metrique, par image puis par pixel
    print(f"  {name:14s} {results[name]:.3f}")  # AUROC : 1 = defauts toujours mieux notes que les normales ; F1 : au seuil choisi
post = model.post_processor  # le post-traitement : contient les seuils appris
print(f"seuil image (brut) {float(post.image_threshold):.3f} | seuil pixel (brut) {float(post.pixel_threshold):.3f}")  # avant normalisation

# %% anomalib_predict
# predict renvoie, par lot, les scores normalises (0,5 = le seuil), l'etiquette predite, la carte et le masque.
predictions = engine.predict(model, datamodule=datamodule)  # predictions sur le test, une entree par lot
scores = torch.cat([b.pred_score for b in predictions]).numpy()  # score d'anomalie de chaque image (tous les lots colles)
labels = torch.cat([b.pred_label for b in predictions]).numpy().astype(int)  # decision : 1 = defaut, 0 = normale
truth = torch.cat([b.gt_label for b in predictions]).numpy().astype(int)  # verite (gt = ground truth)
paths = [p for b in predictions for p in b.image_path]  # chemins des images, dans le meme ordre
maps = torch.cat([b.anomaly_map for b in predictions]).numpy()  # (N, S, S) : carte d'anomalie, un score par pixel
masks = torch.cat([b.gt_mask for b in predictions]).numpy().astype(bool)  # vrais masques des defauts (True = pixel defectueux)
print(  # scores min / median / max des normales puis des defauts : les defauts doivent etre plus hauts
    "scores normalises : normales",  # texte affiche
    np.round(np.percentile(scores[truth == 0], [0, 50, 100]), 3),  # percentiles 0, 50, 100 des images normales
    "| defauts",  # texte affiche
    np.round(np.percentile(scores[truth == 1], [0, 50, 100]), 3),  # idem pour les images avec defaut
)
# rappel : part des defauts detectes ; fausses alarmes : part des normales signalees a tort
print(f"au seuil 0,5 : rappel {labels[truth == 1].mean():.3f} | fausses alarmes {labels[truth == 0].mean():.3f}")

fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))  # grille 2 lignes x 4 colonnes
for col, defect in enumerate(("good", "broken_large", "broken_small", "contamination")):  # une colonne par type d'image
    i = [k for k, p in enumerate(paths) if f"/{defect}/" in p.replace("\\", "/")][1]  # la 2e image de ce type ("\\" : chemins Windows)
    axes[0, col].imshow(Image.open(paths[i]).convert("RGB").resize((PC_CONFIG["image_size"],) * 2))  # ligne 1 : l'image, en 256 x 256
    axes[0, col].set_title(f"{defect} | score {scores[i]:.2f} -> {'defaut' if labels[i] else 'normale'}", fontsize=8)  # score et decision
    axes[1, col].imshow(maps[i], cmap="inferno", vmin=0, vmax=1)  # cartes normalisees par anomalib : 0,5 = le seuil pixel
    if masks[i].any():  # si l'image a un vrai defaut
        axes[1, col].contour(masks[i], levels=[0.5], colors="cyan", linewidths=0.8)  # dessine le contour du vrai defaut en cyan
for ax in axes.ravel():  # pour chaque case de la grille
    ax.axis("off")  # cache les axes
plt.tight_layout()  # ajuste les marges
plt.show()  # affiche

# anomalib ecrit aussi ses propres images (image | carte | masque predit) dans default_root_dir.
saved = sorted(OUTPUT_DIR.rglob("images/**/*.png"))  # cherche les .png produits, dans tous les sous-dossiers
print(len(saved), "images produites par anomalib, par exemple", saved[0].relative_to(OUTPUT_DIR) if saved else "-")  # nombre et un exemple

# %% anomalib_export
# L'export embarque pre-traitement (resize + normalisation ImageNet), memoire, kNN, lissage et normalisation des
# scores : le fichier prend une image en [0, 1] et renvoie des scores deja comparables a 0,5. Deux fichiers :
# model.onnx (le graphe) et model.onnx.data (les poids et la memoire). On verifie TOUJOURS les formes et les sorties.
# opset_version=18 : celui que l'exporteur de torch 2.15 produit ; la valeur par defaut d'anomalib (14) declenche une
# tentative de conversion qui echoue bruyamment (sans consequence : le modele reste en opset 18).
onnx_path = engine.export(  # exporte le modele ; renvoie le chemin du fichier .onnx
    model,  # le modele entraine
    export_type=ExportType.ONNX,  # format ONNX
    export_root=OUTPUT_DIR / "export",  # dossier de destination
    input_size=(PC_CONFIG["image_size"],) * 2,  # taille d'entree (256, 256)
    onnx_kwargs={"opset_version": 18},  # opset : version du jeu d'operations ONNX
)
data_path = onnx_path.with_name(onnx_path.name + ".data")  # le second fichier : model.onnx.data
print(  # chemin de l'export et taille totale des deux fichiers en Mo
    "export :",  # texte affiche
    onnx_path.relative_to(OUTPUT_DIR),  # chemin relatif au dossier de sortie
    "|",  # separateur affiche
    round((onnx_path.stat().st_size + data_path.stat().st_size) / 1e6, 1),  # taille des deux fichiers, en Mo
    "Mo avec",  # texte affiche
    data_path.name,  # nom du fichier de donnees
)
session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # charge le modele ONNX pour l'executer sur CPU
print("entrees :", [(i.name, i.shape) for i in session.get_inputs()])  # nom et forme de chaque entree
print("sorties :", [(o.name, o.shape) for o in session.get_outputs()])  # nom et forme de chaque sortie
# anomalib 2.7 + torch 2.15 : la dimension du lot est figee a 2 par l'export (torch.export la specialise).
# En production : des lots de 2 (dupliquer l'image s'il n'y en a qu'une), ou le checkpoint Lightning et engine.predict.
batch_size_onnx = session.get_inputs()[0].shape[0]  # taille de lot imposee par le fichier ONNX

# Le datamodule livre les images a leur taille native (900 x 900) : on les ramene a 256 en [0, 1], sans normaliser
# (la normalisation ImageNet est dans le graphe). Le modele PyTorch recoit la meme entree : les deux sont comparables.
from torchvision.transforms.v2 import functional as TF  # fonctions de transformation d'images (resize...)

test_batch = next(iter(datamodule.test_dataloader()))  # un lot de test
# redimensionne en 256 x 256 et garde autant d'images que la taille de lot du fichier ONNX
images = TF.resize(test_batch.image, [PC_CONFIG["image_size"]] * 2, antialias=True)[:batch_size_onnx]
onnx_out = session.run(None, {session.get_inputs()[0].name: images.numpy()})  # execute le modele ONNX ; None : toutes les sorties
onnx_scores = onnx_out[[o.name for o in session.get_outputs()].index("pred_score")]  # prend la sortie nommee "pred_score"
with torch.no_grad():  # sans gradient
    torch_scores = model(images).pred_score.numpy()  # memes images dans le modele PyTorch
print(  # les deux scores doivent etre quasi egaux (ecart max tres petit)
    f"scores ONNX {np.round(onnx_scores, 3)} | PyTorch {np.round(torch_scores, 3)} | ecart max {np.abs(onnx_scores - torch_scores).max():.2e}"
)
json.dump(  # ecrit une fiche du modele : tout ce qu'il faut savoir pour l'utiliser ailleurs
    {  # contenu de la fiche
        "onnx_file": onnx_path.name,  # nom du fichier ONNX
        "input": {  # description de l'entree
            "name": "input",  # nom de l'entree
            "shape": list(session.get_inputs()[0].shape),  # forme attendue
            "preprocessing": "RGB, resize bilineaire 256, /255 ; normalisation ImageNet DANS le graphe",  # pretraitement a faire avant
        },
        "outputs": {o.name: list(o.shape) for o in session.get_outputs()},  # nom -> forme de chaque sortie
        "score_threshold": 0.5,  # seuil de decision sur les scores normalises
        "image_threshold_raw": float(model.post_processor.image_threshold),  # seuil image avant normalisation
        "pixel_threshold_raw": float(model.post_processor.pixel_threshold),  # seuil pixel avant normalisation
        "backbone": PC_CONFIG["backbone"],  # reseau utilise
        "memory_bank": list(model.model.memory_bank.shape),  # taille de la memoire
    },
    open(OUTPUT_DIR / "model_card.json", "w"),  # fichier de destination, ouvert en ecriture
    indent=2,  # JSON indente, lisible par un humain
)
one = images.numpy()  # le lot d'images au format numpy, pour mesurer la vitesse
for _ in range(3):  # 3 executions de "chauffe" (les premieres sont plus lentes)
    session.run(None, {session.get_inputs()[0].name: one})  # execution non mesuree
start = time.time()  # heure de depart
for _ in range(10):  # 10 executions mesurees
    session.run(None, {session.get_inputs()[0].name: one})  # execution du modele ONNX
print(  # temps moyen par lot et par image, en millisecondes
    f"latence sur CPU (onnxruntime, lot de {batch_size_onnx}) : {(time.time() - start) / 10 * 1000:.0f} ms, soit {(time.time() - start) / 10 * 1000 / batch_size_onnx:.0f} ms par image"
)
