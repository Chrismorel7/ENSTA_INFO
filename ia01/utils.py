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
