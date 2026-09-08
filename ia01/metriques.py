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
    N = len(y_true)
    nombre_juste = 0
    for i in range(N):
        if y_true[i] == y_pred[i]:
            nombre_juste += 1
    return nombre_juste / N

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
    N, s = len(y_true), 0
    for i in range(N):
        s += (y_true[i] - y_pred[i])**2
    return s / N

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