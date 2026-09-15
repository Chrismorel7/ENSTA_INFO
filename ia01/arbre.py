from ia01.utils import *
from ia01.majoritaire import *


def score(y: list, reg:bool) -> float:
    return variance(y) if reg else gini(y)


def coupe(X: list[list], y: list, d: int, s: float) -> tuple[list[list], list, list[list], list]:
    """Partitionnement d'un ensemble sur le dimension d par rapport à un seuil s

    Paramètres
    ----------
    X : list[list]
        Liste de vecteurs à partitionner
    y : list
        Liste des prédictions associées à X
    d : int
        Dimension selon laquelle faire la coupe
    s : float
        Seuil pour faire la coupe

    Sorties
    -------
    X_inf, y_inf, X_sup, y_sup :
        X_inf, y_inf : partie des éléments tels que x[d] <= s
        X_sup, y_sup : partie des éléments tels que x[d] > s
    """
    assert (isinstance(d, int) and d >= 0), "Le paramètre `d` doit être un entier positif."

    X_inf, y_inf, X_sup, y_sup = [], [], [], []
    
    for i in range(len(X)):
        x = X[i]
        yi = y[i]
        if x[d] <= s:
            X_inf.append(x)
            y_inf.append(yi)
        else:
            X_sup.append(x)
            y_sup.append(yi)
    
    return X_inf, y_inf, X_sup, y_sup


def score_coupe(X: list[list], y: list, d: int, s: float, reg: bool) -> float:
    """Calcul le score résultant d'une coupe

    Paramètres
    ----------
    X : list[list]
        Liste de vecteurs à partitionner
    y : list
        Liste des prédictions associées à X
    d : int
        Dimension selon laquelle faire la coupe
    s : float
        Seuil pour faire la coupe
    reg : bool
        Indique s'il s'agit d'un problème de régression (True) ou de classification (False)

    Sorties
    -------
    score : float
        Score résultant de la coupe        
    """
    
    X_inf, y_inf, X_sup, y_sup = coupe(X, y, d, s)
    
    n_inf, n_sup = len(X_inf), len(X_sup)
    
    score_val = (n_inf*score(y_inf, reg) + n_sup*score(y_sup, reg)) / (n_inf+n_sup)
    
    return score_val


def seuil_coupe(X: list[list], d: int) -> list:
    """Calcul des seuils auxquels partitionner un ensemble sur la dimension d

    Paramètres
    ----------
    X : list[list]
        Liste de vecteurs à partitionner
    d : int
        Dimension selon laquelle faire la coupe        

    Sorties
    -------
    seuils : list
        Seuils pour faire les coupes
    """
    assert (isinstance(d, int) and d >= 0), "Le paramètre `d` doit être un entier positif."

    x = sorted(list(set([vecteur[d] for vecteur in X])))
    
    seuils = [(x[i] + x[i+1]) / 2 for i in range(len(x) - 1)]
    
    return seuils




def meilleure_coupe(X: list[list], y: list, reg: bool) -> tuple[int, float, list[list], list, list[list], list]:
    """Calcul des seuils auxquels partitionner un ensemble sur la dimension d

    Paramètres
    ----------
    X : list[list]
        Liste de vecteurs à partitionner
    y : list
        Liste des prédictions associées à X
    reg : bool
        Indique s'il s'agit d'un problème de régression (True) ou de classification (False)

    Sorties
    -------
    best_dim : int
        Meilleure dimension pour faire la coupe
    best_seuil : float
        Meilleure seuils pour faire la coupe
    X_inf, y_inf, X_sup, y_sup : list
        Partitionnement résultant de la coupe
    """

    D = len(X[0])
    best_seuil, best_dim, best_score = 0, 0, 1e10
    for d in range(D):
        liste_seuils = seuil_coupe(X, d)
        for s in liste_seuils:
            score_d_s = score_coupe(X, y, d, s, reg)
            if score_d_s <= best_score:
                best_seuil, best_dim, best_score = s, d, score_d_s
    
    X_inf, y_inf, X_sup, y_sup = coupe(X, y, best_dim, best_seuil)
    
    return (best_dim, best_seuil, X_inf, y_inf, X_sup, y_sup)




def arbre_train(X_train: list[list], y_train: list, reg: bool = False, max_prof: int = float("inf"), profondeur: int = 0) -> dict:
    """
    Apprentissage d'un arbre de décision

    Paramètres
    ----------
    X_train : list[list]
        Liste des vecteurs de l'ensemble d'apprentissage
    y_train : list
        Liste des prédictions associées aux éléments de X_train
    reg : bool, default = False
        Indique s'il s'agit d'un problème de régression (True) 
        ou de classification (False)
    max_prof : int, default = float("inf")
        Profondeur maximale de l'arbre de décision
    profondeur : int
        Profondeur courante du noeud de l'arbre, paramètre utilisé par récurrence

    Sorties
    -------
    arbre : dict
        Structure d'un arbre binaire, chaque noeud est un dictionnaire contenant 
        un champ "info" et un champ "coupe".
        Dans le champ "info", il y a l'information de profondeur ("profondeur"), 
        le score associé ("score") et une prédiction si elle est faite 
        au niveau de ce noeud ("prediction").
        Le champ "coupe" est nul ("None") si le noeud est une feuille, sinon il
        contient la dimension ("dimension") et le seuil ("seuil") de la coupe ainsi
        que les deux sous-arbres résultants de la coupe ("arbre_inf" et "arbre_sup").
    """
    arbre = {
        "info": {
            "profondeur": profondeur,
            "score": score(y_train, reg),
            "prediction": vote_majoritaire(y_train, reg),
        },
        "coupe": None,
    }
    if profondeur < max_prof:
        d, s, X_inf, y_inf, X_sup, y_sup = meilleure_coupe(X_train, y_train, reg)
        if X_inf:
            arbre["coupe"] = {
                "dimension": d,
                "seuil": s,
                "arbre_inf": arbre_train(X_inf, y_inf, reg, max_prof, profondeur + 1),
                "arbre_sup": arbre_train(X_sup, y_sup, reg, max_prof, profondeur + 1),
            }
    return arbre


def arbre_pred(X: list[list], arbre: dict, max_prof: int = float("inf")) -> list:
    """
    Prédiction à partir d'un arbre de décision

    Paramètres
    ----------
    X : list[list]
        Liste de vecteurs sur lesquels appliquer l'arbre de décision
    arbre : dict
        Arbre de décision
    max_prof : int, default = float("inf")
        Profondeur maximale d'exploration de l'arbre de décision

    Sorties
    -------
    y_pred : list
        Liste des prédictions associées aux éléments de X
    """

    def arbre_pred_single(x: list, arbre: dict, max_prof: int) -> dict:
        if arbre["coupe"] is None or arbre["info"]["profondeur"] >= max_prof:
            return arbre["info"]["prediction"]
        else:
            d = arbre["coupe"]["dimension"]
            s = arbre["coupe"]["seuil"]
            if x[d] <= s:
                return arbre_pred_single(x, arbre["coupe"]["arbre_inf"], max_prof)
            else:
                return arbre_pred_single(x, arbre["coupe"]["arbre_sup"], max_prof)

    return [arbre_pred_single(x, arbre, max_prof) for x in X]
