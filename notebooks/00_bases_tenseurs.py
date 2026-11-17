# %% [markdown]
# # Les deux notions a comprendre avant le projet : tenseurs et gradients
#
# Court volontairement : le strict necessaire pour lire le reste du cours.
# Dans le cours : chapitre "Avant de commencer", juste avant l'etape 1 du fil.

# %% tensor_basics
import numpy as np  # numpy : calcul sur des tableaux de nombres (OpenCV et PIL donnent des tableaux numpy)
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU) et apprentissage profond

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # cree un tenseur 2 lignes x 3 colonnes a partir d'une liste Python
print(x.shape, x.dtype, x.device)  # forme (2, 3), type float32 (nombres a virgule), memoire CPU
print(torch.zeros(2, 3).shape, torch.ones(4).shape, torch.rand(2, 2).shape)  # autres facons de creer : que des 0, que des 1, au hasard
print(x.sum().item(), x.mean(dim=0), x.max(dim=1).values)  # somme totale, moyenne par colonne, max par ligne ; .item() : un nombre Python

# %% tensor_image
# Une image pour PyTorch : (canaux, hauteur, largeur), puis un lot (plusieurs images ensemble) : (N, C, H, W).
image_hwc = np.random.randint(0, 255, size=(200, 300, 3), dtype=np.uint8)  # fausse image couleur, rangee comme OpenCV / PIL : H, W, C
image_chw = torch.from_numpy(image_hwc).permute(2, 0, 1)  # tableau numpy -> tenseur, puis reordonne les axes H, W, C -> C, H, W
image_float = image_chw.float() / 255.0  # entiers uint8 [0, 255] -> float32 [0, 1] : les reseaux calculent en nombres a virgule
batch = image_float.unsqueeze(0)  # ajoute la dimension du lot en tete : (1, 3, 200, 300), le reseau attend toujours un lot
print(image_hwc.shape, "->", tuple(image_chw.shape), "->", tuple(batch.shape), batch.dtype)  # affiche les formes successives
back_to_numpy = (batch[0].permute(1, 2, 0) * 255).to(torch.uint8).numpy()  # le chemin inverse (C, H, W -> H, W, C, 0-255), pour afficher
print(back_to_numpy.shape, back_to_numpy.dtype)  # verifie : (200, 300, 3) uint8, comme au depart

# %% tensor_broadcast
# Normalisation par canal : (3, H, W) - (3, 1, 1) -> le broadcasting etend mean sur H et W.
mean = torch.tensor([0.5, 0.4, 0.3]).view(3, 1, 1)  # une moyenne par canal (R, G, B), mise en forme (3, 1, 1)
std = torch.tensor([0.2, 0.2, 0.2]).view(3, 1, 1)  # un ecart-type par canal, meme forme
normalized = (image_float - mean) / std  # normalisation : centre et met a l'echelle chaque canal (valeurs autour de 0)
print(normalized.shape, round(normalized[0].mean().item(), 3))  # la forme ne change pas ; moyenne du canal 0 apres normalisation

# %% autograd
# Un parametre a apprendre : requires_grad=True. backward() calcule d(loss)/d(w).
w = torch.tensor(2.0, requires_grad=True)  # requires_grad=True : PyTorch retient les calculs faits avec w pour deriver ensuite
loss = (w * 3 - 12) ** 2  # perte (erreur a rendre petite) : minimum en w = 4
loss.backward()  # calcule le gradient (pente de la perte par rapport a w) et le range dans w.grad
print("gradient :", w.grad.item())  # 2 * (w*3 - 12) * 3 = -36 : negatif, donc augmenter w fait baisser la perte
# Les gradients S'ACCUMULENT : sans remise a zero, le prochain backward s'ajoute au precedent.
w.grad.zero_()  # remet le gradient a 0 (le _ final : modifie le tenseur sur place)
print("apres zero_ :", w.grad.item())  # verifie : 0.0

# %% regression_manual
# Regression lineaire y = a x + b, ecrite entierement a la main.
torch.manual_seed(0)  # fixe le hasard : memes nombres aleatoires a chaque execution
x = torch.linspace(-1, 1, 100).unsqueeze(1)  # 100 valeurs regulieres entre -1 et 1, forme (100, 1)
y = 3 * x + 1 + 0.1 * torch.randn_like(x)  # cibles a retrouver : verite a = 3, b = 1, plus un peu de bruit
a, b = torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=True)  # les deux parametres a apprendre, partent de 0
lr = 0.1  # taux d'apprentissage (learning rate) : taille du pas a chaque mise a jour
for _ in range(200):  # 200 pas d'apprentissage sur toutes les donnees
    prediction = a * x + b  # 1. passage avant : calcule les predictions avec les a et b actuels
    loss = ((prediction - y) ** 2).mean()  # 2. perte : erreur quadratique moyenne entre prediction et verite
    loss.backward()  # 3. gradients : remplit a.grad et b.grad
    with torch.no_grad():  # 4. mise a jour SANS enregistrer d'operations (on ne veut pas deriver la mise a jour)
        a -= lr * a.grad  # descente de gradient : on bouge a dans le sens qui fait baisser la perte
        b -= lr * b.grad  # idem pour b
        a.grad.zero_()  # 5. remise a zero, sinon les gradients s'additionnent d'un pas a l'autre
        b.grad.zero_()  # idem pour b
print(f"a = {a.item():.3f}, b = {b.item():.3f}, perte = {loss.item():.4f}")  # a doit etre proche de 3 et b de 1

# %% regression_module
# La meme chose avec les briques de PyTorch : nn.Linear + optimiseur.
import torch.nn as nn  # nn : briques toutes faites pour construire des reseaux (couches, pertes)

torch.manual_seed(0)  # meme hasard que plus haut, pour comparer
model = nn.Linear(1, 1)  # couche lineaire 1 entree -> 1 sortie : a et b sont dedans (model.weight, model.bias)
# optimiseur : applique la mise a jour des parametres a notre place (SGD = descente de gradient simple)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # il recoit les parametres du modele et le pas lr
criterion = nn.MSELoss()  # perte toute faite : erreur quadratique moyenne
for _ in range(200):  # 200 pas d'apprentissage, comme la version a la main
    optimizer.zero_grad()  # 5 (fait avant) : gradients a zero
    loss = criterion(model(x), y)  # 1 + 2 : passage avant et perte
    loss.backward()  # 3 : gradients
    optimizer.step()  # 4 : mise a jour
print(f"a = {model.weight.item():.3f}, b = {model.bias.item():.3f}, perte = {loss.item():.4f}")  # meme resultat qu'a la main
