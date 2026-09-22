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
        return sum([abs(x1i - x2i) ** p for x1i, x2i in zip(x1, x2)]) ** (1/p)
    else:
        return max([abs(x1i - x2i) for x1i, x2i in zip(x1, x2)])


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

    return [distance2(x, xi, p) for xi in X_train]


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

    y_pred = []
    for xi in X:
        dist = distance(xi, X_train, p)
        idx = argsort(dist)
        y_pred.append(vote_majoritaire([y_train[idx[j]] for j in range(k)], reg))
    return y_pred
