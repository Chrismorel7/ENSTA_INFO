from ia01.utils import compte
from ia01.utils import moyenne


def vote_majoritaire(y: list[int | str | float], reg: bool = False) -> int | str | float:
    """Applique le vote majoritaire à la liste y.

    Paramètres
    ----------
    y : list[int | str | float]
        Liste des labels pour l'ensemble des données
        Pour un problème de classification : 
            les labels sont encodés par un entier (int) 
            ou une chaîne de caractères (str)
        Pour un problème de régression : 
            les labels sont des nombres flotants (float)
    reg : bool, default = False
        Indique s'il s'agit d'un problème de régression (True) ou de classification (False)
        Par défaut, on considère qu'il s'agit d'un problème de classification (reg=False)

    Sorties
    -------
    label : int | str | float
        Classification : label le plus représenté dans la liste y
        Regression : moyenne empirique des éléments de y
    """
    if reg:
        return moyenne(y)
    else:
        nombre = compte(y)
        return max(nombre, key=nombre.get)
        