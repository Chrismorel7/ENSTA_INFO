def test_utils():
    """Fonction de test"""
    print("Test du module utils du package ia01 !")


def compte(y: list[int | str]) -> dict[int | str, int]:
    """Compte le nombre d'occurance de chaque label de la liste y.

    Paramètres
    ----------
    y : list[int | str]
        Liste de labels encodés par un entier (int) 
            ou une chaîne de caractères (str)

    Sorties
    -------
    nombre : dict(int | str, int)
        nombre[x] donne le nombre d'occurrence de l'élément x dans la liste y
    """
    nombre: dict[int | str, int] = {}
    for element in y:
        nombre[element] = nombre.get(element, 0) + 1
    return nombre


def lecture_csv(fichier: str, sep: str=",") -> list[dict]:
    """Lecture d'un fichier texte sous format CSV.
    La première ligne du fichier donne le nom des champs.

    Paramètres
    ----------
    fichier : str
        Chemin vers le fichier à lire
    sep : str, default = ","
        Caractère pour séparer les champs

    Sorties
    -------
    data : list[dict]
        Liste des éléments du fichier CSV.
        Chaque élément est stocké dans un dictionnaire dont les champs
        sont donnés par la première ligne du fichier CSV.
        Les champs sont convertis en float si possible, sinon ils restent des chaînes de caractères.
    """
    data = []
    with open(fichier, "r") as f:
        lines = f.read().splitlines()
        keys = lines[0].split(sep)
        for line in lines[1:]:
            values = line.split(sep)
            for i, v in enumerate(values):
                try:
                    values[i] = float(v)
                except ValueError:
                    pass
            data.append(dict(zip(keys, values)))
    return data

def moyenne(y: list[float]) -> float:
    """Calcul de la moyenne d'une liste de valeurs numériques.

    Paramètres
    ----------
    y : list[float]
        Liste de valeurs numériques

    Sorties
    -------
    moy : float
        Moyenne des valeurs de la liste y
    """
    N, s = len(y), 0
    for e in y:
        s += e
    return s / N


def gini(y: list) -> float:
    """Calcul de l'impureté de Gini

    Paramètres
    ----------
    y : list
        Liste de labels

    Sorties
    -------
    g : float
        Impureté de Gini
    """
    g = 1
    C = set(y)
    for i in C:
        pi = len([x for x in y if x == i]) / len(y)
        g -= pi**2
    
    return g


def variance(x: list[float]) -> float:
    """Variance d'une liste de nombres

    Paramètres
    ----------
    x : list[float]
        Liste de nombres

    Sorties
    -------
    v : float
        Variance
    """
    
    E = moyenne(x)
    v = 0
    for i in range(len(x)):
        v += (x[i] - E)**2
    return v / len(x)


def ecart_type(x: list[float]) -> float:
    """Variance d'une liste de nombres
    
        Paramètres
        ----------
        x : list[float]
            Liste de nombres
    
        Sorties
        -------
        var : float
            Variance
        """
    return variance(x)**(1/2)


def argsort(x: list[float], reverse: bool = False) -> list[int]:
    """Calcul l'indice de chaque élément dans l'ordre trié.

    Paramètres
    ----------
    x : list[float]
        Liste d'éléments non trié
    reverse : bool
        False (par défaut): tri par ordre croissant
        True : tri par ordre décroissant

    Sorties
    -------
    idx : list[int]
        Liste des indices dans l'ordre
        Si tri croissant, alors x[idx[0]] est le plus petit élément de x
    """
    return sorted(range(len(x)), key=x.__getitem__, reverse=reverse)


def normalisation(X: list[list], loc: list[float], scale: list[float]) -> list[list]:
    """Méthodes de normalisation de vecteurs

    Paramètres
    ----------
    X : list[list]
        Liste de vecteurs sur lesquels appliquer la normalisation
    loc, scale : list
        Liste de paramètres de normalisation pour chaque dimension
        Pour un vecteur x de la liste X, la normalisation pour chaque dimension est :
        x[j] = (x[j] - loc[j]) / scale[j]

    Sorties
    -------
    Xnorm : list[list]
        Liste des vecteurs normalisés
    """
    
    Xnorm = []
    for i in range(len(X)):
        xnormi = []
        for j in range(len(X[i])):
            xj = (X[i][j] - loc[j]) / scale[j]
            xnormi.append(xj)
        Xnorm.append(xnormi)
    
    return Xnorm


def norm_param(X: list[list], methode: str = "echelle") -> tuple[list[float], list[float]]:
    """Calcul des paramètres de normalisation

    Paramètres
    ----------
    X : list[list]
        Liste de vecteurs à partir desquels calculer les paramètres de normalisation
    methode : str, default = "echelle"
        Méthode de normalisation utilisée : "echelle" ou "centre"

    Sorties
    -------
    loc, scale : list
        Liste de paramètres de normalisation pour chaque dimension
    """

    assert (methode == "echelle" or methode == "centre"), "Le paramètre `methode` doit valoir `echelle` ou `centre`."
    
    loc, scale = [], []
    
    if methode == "echelle":
        for j in range(len(X[0])):
            colonne = [vecteur[j] for vecteur in X]
            mini = min(colonne)
            maxi = max(colonne)
            loc.append(mini)
            scale.append(maxi - mini)
    
    else:
        for j in range(len(X[0])):
            colonne = [vecteur[j] for vecteur in X]
            loc.append(moyenne(colonne))
            scale.append(ecart_type(colonne))
    
    return loc, scale