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

def precision(y_true: list, y_pred: list, label_pos: int | str) -> float:
    """Précision

    Paramètres
    ----------
    y_true : list
        Liste contenant les vraies valeurs
    y_pred : list
        Liste contenant les valeurs prédites par un classifieur
    label_pos :
        Label de la classe considérée comme positive

    Sorties
    -------
    prec : float [0,1]
        prec = VP / (VP + FP)
        Si VP + FP = 0, alors prec = 0
    """
    VP = len([yt for yt, yp in zip(y_true, y_pred) if yt == yp and yp == label_pos])
    FP = len([yt for yt, yp in zip(y_true, y_pred) if yt != yp and yp == label_pos])
    if VP + FP ==0: return 0
    else : return VP / (VP + FP)

def rappel(y_true: list, y_pred: list, label_pos: int | str) -> float:
    """Rappel

    Paramètres
    ----------
    y_true : list
        Liste contenant les vraies valeurs
    y_pred : list
        Liste contenant les valeurs prédites par un classifieur
    label_pos :
        Label de la classe considérée comme positive

    Sorties
    -------
    rap : float [0,1]
        rap = VP / (VP + FN)
        Si VP + FN = 0, alors rap = 0
    """
    VP = len([yt for yt, yp in zip(y_true, y_pred) if yt == yp and yp == label_pos])
    FN = len([yt for yt, yp in zip(y_true, y_pred) if yt != yp and yp != label_pos])
    if VP + FN ==0: return 0
    else : return VP / (VP + FN)

def f_score(y_true: list, y_pred: list, label_pos: int | str, beta: float = 1) -> float:
    """F-score

    Paramètres
    ----------
    y_true : list
        Liste contenant les vraies valeurs
    y_pred : list
        Liste contenant les valeurs prédites par un classifieur
    label_pos :
        Label de la classe considérée comme positive
    beta : float, default = 1
        Paramètre beta du score, calcul F1 par défaut

    Sorties
    -------
    f : float [0,1]
        f = ((1 + beta**2)*(prec * rap)) / (beta**2 * prec + rap)
        Si prec = rap = 0, alors f = 0
    """
    prec = precision(y_true, y_pred, label_pos)
    rap = rappel(y_true, y_pred, label_pos)
    if prec == rappel and rappel == 0: return 0
    else: return ((1 + beta**2)*(prec * rap)) / (beta**2 * prec + rap)

def matrice_confusion(y_true: list, y_pred: list, labels: list) -> list[list]:
    """Matrice de confusion

    Paramètres
    ----------
    y_true : list
        Liste contenant les vraies valeurs
    y_pred : list
        Liste contenant les valeurs prédites par un classifieur
    labels : list
        List des labels du problème de classification

    Sorties
    -------
    mat : list[list]
        Matrice de confusion
        mat[i][j] donne le nombre d'éléments de la classe i ayant
        été prédits comme appartenant à la classe j
    """
    Mat = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
    for k in range(len(y_pred)):
        i = labels.index(y_true[k])
        j = labels.index(y_pred[k])
        Mat[i][j] += 1
    return Mat
