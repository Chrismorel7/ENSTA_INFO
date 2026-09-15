from ia01.utils import compte
from ia01.utils import moyenne


def vote_majoritaire(y: list[int | str | float], reg: bool = False, pond: bool = False, distance: list[float] = None) -> int | str | float:
    """_summary_

    Args:
        y (list[int  |  str  |  float]): _description_
        reg (bool, optional): _description_. Defaults to False.
        pond (bool, optional): _description_. Defaults to False.
        distance (list[float], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        int | str | float: _description_
    """
    if pond and (distance is None or len(distance) != len(y)):
        raise ValueError("distance doit contenir une valeur pour chaque label")

    if reg:
        if pond:
            somme_valeurs = 0.0
            somme_poids = 0.0
            for val, d in zip(y, distance):
                if d < 0:
                    raise ValueError("Les distances doivent être positives")
                if d == 0:
                    return val
                w = 1.0 / d
                somme_valeurs += w * val
                somme_poids += w
            return somme_valeurs / somme_poids
        else:
            return moyenne(y)
    else:
        if pond:
            poids_dict = {}
            for label, d in zip(y, distance):
                if d < 0:
                    raise ValueError("Les distances doivent être positives")
                if d == 0:
                    return label
                poids_dict[label] = poids_dict.get(label, 0.0) + (1.0 / d)
            return max(poids_dict, key=poids_dict.get)
        else:
            nombre = compte(y)
            return max(nombre, key=nombre.get)
        