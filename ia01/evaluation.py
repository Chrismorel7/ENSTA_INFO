def partition_train_val(X: list[list], y: list, r: float = 0.2) -> tuple[list[list], list, list[list], list]:
    """Partitionne un ensemble X, y en un ensemble train et val.

    Paramètres
    ----------
    X : list[list]
        Liste de vecteurs à partitionner
    y : list
        Liste des prédictions associées à X
    r : float [0, 1], default = 1/5
        Ratio de (X, y) à mettre dans l'ensemble de validation
        On choisit r tel que r = 1/K avec K un entier
        L'ensemble de validation X_val, y_val contient tous les éléments de X, y
        dont l'indice modulo K est égal à zéro, les autres éléments sont dans
        l'ensemble X_train, y_train.
        X_val, y_val contiennent les éléments 0, K, 2K, 3K, etc.

    Sorties
    -------
    X_train, y_train : list[list], list
        Ensemble d'apprentissage
    X_val, y_val : list[list], list
        Ensemble de validation
    """

    K = round(1 / r)
    X_train = [X[i] for i in range(len(X)) if i%K != 0]
    y_train = [y[i] for i in range(len(y)) if i%K != 0]
    X_val = [X[i] for i in range(len(X)) if i%K == 0]
    y_val = [y[i] for i in range(len(y)) if i%K == 0]
    return X_train, y_train, X_val, y_val


def partition_val_croisee(X: list[list], y: list, K: int = 5) -> tuple[list[list[list]], list[list]]:
    """Partitionne un ensemble X, y en K sous-ensembles.

    Paramètres
    ----------
    X : list[list]
        Liste de vecteurs à partitionner
    y : list
        Liste des prédictions associées à X
    K : int, default = 5
        Nombre de partitions

    Sorties
    -------
    X_K, y_K : list[list[list]], list[list]
        Liste comprenant les K sous-ensembles de X et y
        X_K[k], y_val[k] contiennent tous les éléments de X, y
        dont l'indice modulo K est égal à k.
    """
    X_K = [[X[i] for i in range(len(X)) if i%K == k] for k in range(K)]
    y_K = [[y[i] for i in range(len(y)) if i%K == k] for k in range(K)]

    return X_K, y_K