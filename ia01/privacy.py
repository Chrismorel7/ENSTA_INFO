
#? Question 1.1.
def est_identique(d1, d2, attributs):
    """Calcule si d1 et d2 sont identiques selon une liste d'attributs

    Paramètres
    ----------
    d1 : dict
        Individu 1 décrit un par dictionnaire
    d2 : dict
        Individu 2 décrit un par dictionnaire
    attributs : list
        Liste des attributs selon lesquels d1 et d2 sont comparés

    Sorties
    -------
    boolean
        True si d1[a] == d2[a] pour tout a dans attributs
    """
    for key in attributs:
        if d1[key] != d2[key]:
            return False
    return True

#? Question 1.2.
def groupe(data, attributs):
    """Regroupe les individus partageant les mêmes attributs

    Paramètres
    ----------
    data : list[dict]
        Liste d'individus décrits par un dictionnaire
    attributs : list
        Liste des attributs selon lesquels les individus sont comparés

    Sorties
    -------
    G : list
        Liste contenant pour chaque individu un indice representant son groupe
        Tous les individus du même groupe sont égaux selon les attributs considérés
        Les groupes sont indexés de 1 jusqu'à max(G) et il y a exactement max(G) groupes différents
    """
    indices = {}
    G = []
    for d in data:
        cle = tuple(d[a] for a in attributs)
        if cle not in indices:
            indices[cle] = len(indices) + 1
        G.append(indices[cle])
    return G

#? Question 1.3.
def k_anonymite(data, attributs):
    """k-anonymité d'un jeu de données selon une liste d'attributs

    Paramètres
    ----------
    data : list[dict]
        Liste d'individus décrits par un dictionnaire
    attributs : list
        Liste des attributs selon lesquels les individus sont comparés

    Sorties
    -------
    k : int
        k-anonymité de data selon la liste d'attributs
    """
    G = sorted(groupe(data, attributs))
    if not G:
        return 0

    k = len(data) + 1
    groupe_courant = G[0]
    taille = 0

    for val in G:
        if val == groupe_courant:
            taille += 1
        else:
            k = min(k, taille)
            groupe_courant = val
            taille = 1

    return min(k, taille)

#? Question 1.9.
def discret_seuils(X, n):
    """Calcule les seuils de discrétisation de X pour avoir au moins n éléments par interval

    Paramètres
    ----------
    X : list
        Liste de valeurs à discrétiser
    n : int
        Nombre minimal d'éléments par intervalle

    Sorties
    -------
    seuils : list
        Le nombre d'élements de X tels que seuils[i] <= x < seuils[i+1] est supérieur ou égal à n
        Le dernier élément de seuils est float("inf")
    """
    if n <= 0:
        raise ValueError("n doit être strictement positif")
    X = sorted(X)
    if len(X) < n:
        raise ValueError("Pas assez de valeurs pour former un intervalle de taille n")
    seuils = [X[0]]
    dernier_idx = 0
    for i in range(1, len(X)):
        if i - dernier_idx >= n and X[i] != X[i-1] and len(X) - i >= n:
            seuils.append(X[i])
            dernier_idx = i
    seuils.append(float("inf"))
    return seuils

#? Question 1.10.
def discretisation(x, seuils):
    """Discrétise une valeur selon des seuils

    Paramètres
    ----------
    x : float
        Une valeur à discrétiser    
    seuils : list
        Seuils de discrétisation

    Sorties
    -------
    i : int
        Indice tel que seuils[i] <= x < seuils[i+1] 
    """
    for i in range(len(seuils)-1):
        if seuils[i] <= x and x < seuils[i+1]:
            return i
    return None


#? Question 2.1.
def l_diversite(data, attributs, sensible):
    """l-diversité d'un jeu de données selon une liste d'attributs

    Paramètres
    ----------
    data : list[dict]
        Liste d'individus décrits par un dictionnaire
    attributs : list
        Liste des attributs selon lesquels les individus sont comparés
    sensible :
        Attribut sensible sur lequel calculer la diversité

    Sorties
    -------
    l : int
        l-diversité de data selon la liste d'attributs
    """
    if not data:
        return 0

    G = groupe(data, attributs)
    l = len(data)
    for g in range(1, max(G) + 1):
        # Pour chaque groupe, on récupère les valeurs distinctes de l'attribut sensible
        # parmi les individus appartenant à ce groupe.
        valeurs = set(data[i][sensible] for i in range(len(data)) if G[i] == g)

        # On garde le minimum entre l-diversité actuelle et le nombre de valeurs
        # distinctes trouvées dans ce groupe.
        l = min(l, len(valeurs))
    return l
    