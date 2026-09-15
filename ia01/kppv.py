from ia01.utils import *
from ia01.majoritaire import *

def distance2(x1: list[float], x2: list[float], p: float = 2) -> float:
    """Calcule la distance de Minkowski de paramètre p entre les vecteurs x1 et x2

    Paramètres
    ----------
    x1, x2 : list[float]
        Vecteurs de dimension d
    p : float, default = 2
        Paramètre de la distance de Minkowski p > 0, pour la distance de Tchebychev p = float('inf')

    Sorties
    -------
    dist : float
        Distance entre x1 et x2
    """

    assert len(x1) == len(x2), "Les vecteurs x1 et x2 doivent être de même dimension."
    assert p > 0, "Le paramètre p doit être strictement supérieur à 0."
    
    d = len(x1)
    
    if p < float("inf"):
        s = 0
        for i in range(d):
            s += abs(x1[i] - x2[i])**p
        return s**(1/p)
    else:
        return max([abs(x1[i] - x2[i]) for i in range(d)])


def distance(x: list[float], X_train: list[list], p: float = 2) -> list[float]:
    """Calcule la distance de Minkowski de paramètre p entre le vecteur x 
    et tous les éléments de X_train.

    Paramètres
    ----------
    X_train : list[list]
        Liste de vecteurs de dimension d
    x : list[float]
        Vecteur de dimension d
    p : float, default = 2
        Paramètre de la distance de Minkowski
        p > 0, pour la distance de Tchebychev p = float('inf')

    Sorties
    -------
    dist : list[float]
        Distances entre x et tous les éléments de X_train
    """

    dist = []
    for i in range(len(X_train)):
        dist.append(distance2(x, X_train[i], p))
    return dist


def kppv(X: list[list], X_train: list[list], y_train: list, k: int, p: float = 2, reg: bool = False, pond: bool = False) -> list:
    """Méthode des k plus proches voisins

    Paramètres
    ----------
    X : list[list]
        Liste de vecteurs sur lesquels appliquer la méthode des k-ppv
    X_train : list[list]
        Liste des vecteurs de l'ensemble d'apprentissage
    y_train : list
        Liste des prédictions associées aux éléments de X_train
    k : int
        Nombre de voisins
    p : float, default = 2
        Paramètre de la distance de Minkowski
        p > 0, pour la distance de Tchebychev p = float('inf')
    reg : bool, default = False
        Indique s'il s'agit d'un problème de régression (True) ou de classification (False)

    Sorties
    -------
    y_pred : list
        Liste des prédictions associées aux éléments de X
    """

    assert isinstance(k, int) and k > 0, "k doit être un entier strictement positif"

    N = len(X)
    Dist = [distance(x, X_train, p) for x in X]
    y_pred = []

    for i in range(N):
        index_i = argsort(Dist[i])[:k]
        liste_voisins = [y_train[j] for j in index_i]
        distance_voisins = [Dist[i][j] for j in index_i]
        if pond:
            y_pred.append(vote_majoritaire(liste_voisins, reg, pond=True, distance=distance_voisins))
        else :
            y_pred.append(vote_majoritaire(liste_voisins, reg))
    
    return y_pred
