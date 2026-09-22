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
    label = []
    for i in range(len(y)):
        if y[i] not in label:
            label.append(y[i])
    nombre = dict()
    for k in range(len(label)):
        nombre[label[k]] = 0
    for i in range(len(y)):
        nombre[y[i]] += 1
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
    return sum(y) / len(y)


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
    c = compte(y)
    n = len(y)
    return 1 - sum([(ci/n) ** 2 for ci in c.values()])


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
    x_moy = moyenne(x)
    return moyenne([(xi - x_moy) ** 2 for xi in x]))


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
    
    return [[(xij - l) / s for xij, l, s in zip(xi, loc, scale)] for xi in X]


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
    
    d = len(X[0])
    if methode == "echelle":
        loc = [min([xi[j] for xi in X]) for j in range(d)]
        scale = [max([xi[j] for xi in X]) - min([xi[j] for xi in X]) for j in range(d)]
    else:
        loc = [moyenne([xi[j] for xi in X]) for j in range(d)]
        scale = [ecart_type([xi[j] for xi in X]) for j in range(d)]

    return loc, scale