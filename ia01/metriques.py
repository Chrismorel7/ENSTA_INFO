from ia01.utils import moyenne

def taux_erreur(y_true: list, y_pred: list) -> float:
    """Taux d'erreur pour un problème de classification

    Paramètres
    ----------
    y_true : list
        Liste contenant les vraies valeurs
    y_pred : list
        Liste contenant les valeurs prédites par un classifieur

    Sorties
    -------
    err : float [0,1]
        Ratio (entre 0 et 1) d'éléments où y_true et y_pred sont différents.
    """
    return len([yt for yt, yp in zip(y_true, y_pred) if yt != yp]) / len(y_true)

def eqm(y_true: list[float], y_pred: list[float]) -> float:
    """Erreur quadratique moyenne pour un problème de regression

    Paramètres
    ----------
    y_true : list[float]
        Liste contenant les vraies valeurs
    y_pred : list[float]
        Liste contenant les valeurs prédites

    Sorties
    -------
    e : float
        Erreur quadratique moyenne
    """
    return moyenne([(yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)])

def reqm(y_true: list[float], y_pred: list[float]) -> float:
    """Racine de l'erreur quadratique moyenne pour un problème de regression

    Paramètres
    ----------
    y_true : list[float]
        Liste contenant les vraies valeurs
    y_pred : list[float]
        Liste contenant les valeurs prédites

    Sorties
    -------
    e : float
        Racine de l'erreur quadratique moyenne
    """
    return (eqm(y_true, y_pred))**(1/2)