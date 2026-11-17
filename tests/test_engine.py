"""Tests du moteur : une etape fait baisser la perte, et un lot peut etre appris par coeur."""

import torch  # PyTorch : tenseurs aleatoires, optimiseur
import torch.nn as nn  # nn : fonction de perte
from torch.utils.data import DataLoader, TensorDataset  # TensorDataset : Dataset fait directement de tenseurs deja en memoire

from vision_starter.engine import evaluate, train_one_epoch  # les fonctions testees
from vision_starter.model import SimpleCNN  # petit modele pour les tests


# fabrique un petit jeu jouet ; renvoie un DataLoader qui donne un seul lot de n images
def _toy_loader(n: int = 16):
    torch.manual_seed(0)  # graine fixe : memes donnees a chaque lancement
    images = torch.randn(n, 3, 32, 32)  # n "images" de bruit 3x32x32
    targets = torch.arange(n) % 2  # etiquettes alternees 0, 1, 0, 1...
    return DataLoader(TensorDataset(images, targets), batch_size=n)  # un seul lot contenant tout


# test : une epoque renvoie une perte positive et une accuracy entre 0 et 1
def test_one_epoch_returns_loss_and_accuracy():
    model, loader = SimpleCNN(num_classes=2, channels=(8, 16)), _toy_loader()  # petit modele (2 blocs) et donnees jouets
    # une epoque d'entrainement avec la perte d'entropie croisee et l'optimiseur Adam (pas 1e-3), sur CPU
    loss, acc = train_one_epoch(model, loader, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), 1e-3), torch.device("cpu"))
    assert loss > 0 and 0.0 <= acc <= 1.0  # valeurs plausibles


# test : le modele doit pouvoir apprendre par coeur un seul lot (sinon la boucle ou le modele ont un bug)
def test_model_can_overfit_one_batch():
    # Arrange
    model, loader = SimpleCNN(num_classes=2, channels=(16, 32)), _toy_loader()  # modele et donnees jouets
    criterion, optimizer = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), 5e-3)  # perte et optimiseur (pas 5e-3)

    # Act : 80 passes sur le meme lot
    for _ in range(80):  # 80 fois (_ = le numero de tour ne sert pas)
        loss, _ = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))  # une passe ; on garde la perte
    _, acc, _, _ = evaluate(model, loader, criterion, torch.device("cpu"))  # accuracy sur ce meme lot

    # Assert
    assert loss < 0.2  # la perte est devenue petite
    assert acc == 1.0  # toutes les images du lot sont bien classees
