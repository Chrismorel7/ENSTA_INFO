### --- Bibiothèque --- ###
from ia01.utils import *
from ia01.majoritaire import *
from ia01.arbre import *
from ia01.kppv import *
from ia01.evaluation import *
from ia01.metriques import *
from ia01.privacy import *

### --- Constante --- ###
L_CLE_INUTILE = ['sexe', 'app_smartphone', 'occupation', 'deficit_sommeil']
L_CLE_PRIVE = ['age', 'sexe', 'occupation']
L_CHRONOTYPE = {"Matinal":[1,0,0], "Intermediaire":[0,1,0], "Couche_tard":[0,0,1]}
K = 5
L_K = [1,2,3,3,5,6,7,8,9,10,15,20,50]

### --- Fonction Utiles --- ###
def ecriture_csv_projet(y_pred: list[tuple[float, int]], fichier: str = "y_test.csv"):
    assert len(y_pred) == 2000, "Les prédictions doivent contenir 2000 données."
    with open(fichier, "w") as f:
        f.write("score_fatigue, deficit_sommeil")
        f.write("\n")
        for y in y_pred:
            f.write(str(y[0]) + ", " + str(y[1]))
            f.write("\n")

def encoder_donnees(X, seuils_age, cles_inutiles):
    """Nettoie, discrétise et encode un jeu de données (Train ou Test).

    Args:
        X (_type_): Jeu de donnée
        seuils_age (_type_): Seuils pour discretiser les ages afin d'assurer l'anonymite
        cles_inutiles (_type_): Clé intuile à retirer du jeu de données

    Returns:
        _type_: _description_
    """
    X_encod = []
    for i in range(len(X)):
        X_encod.append([])
        L_cle = list(X[i].keys())
        for j in range(len(L_cle)):
            cle = L_cle[j]
            val = X[i][cle]
            if cle in cles_inutiles:
                continue # Ignorer les variables inutiles (filtrage)
            elif cle == 'age':
                idx = discretisation(val, seuils_age)
                if idx is None: 
                    idx = len(seuils_age) - 2 if val >= seuils_age[-1] else 0 # Sécurité si un âge du set de test déborde des seuils
                X_encod[i].append(float(idx)) # Discrétiser l'âge
            elif cle == 'chronotype':
                X_encod[i].extend(L_CHRONOTYPE[val]) # Encoder le chronotype en One-Hot
            else:
                X_encod[i].append(val) # Garder les autres variables telles quelles
                
    return X_encod

def afficher_performance(y_true, y_pred, label_pos=1, beta=1, label_model='Defaut'):
    """Affiche les performances d'un modèle

    Args:
        y_true (_type_): Valeures réelles considérées comme vraie
        y_pred (_type_): Valeures prédites par le modèle
        label_pos (int, optional): Label considéré comme succès. Defaults to 1.
        beta (int, optional): Beta pour claculer le F-Score. Defaults to 1.
        label_model (str, optional): Nom du modèle pour personaliser l'affichage. Defaults to 'Defaut'.
    """
    print(f"Essai du modèle : {label_model}")
    print(f"Taux d'erreur : {taux_erreur(y_true, y_pred)}")
    print(f"Precision : {precision(y_true, y_pred, label_pos)}")
    print(f"True Positive Rate (Rappel) : {TPR(y_true, y_pred, label_pos)}")
    print(f"False Positive Rate : {FPR(y_true, y_pred, label_pos)}")
    print(f"F-Score : {f_score(y_true, y_pred, label_pos, beta)}")
    
    

### --- Extraction et Traitement des données --- ###
data_brute = lecture_csv('data/X_train.csv')
y_brut = lecture_csv('data/y_train.csv')
data_test_brute = lecture_csv('data/X_test.csv')

# --- Analyse et Calcul des Seuils (uniquement sur le Train) --- #
for i in range(len(data_brute)):
    data_brute[i]['deficit_sommeil'] = y_brut[i]['deficit_sommeil']
n_optimal = 5
for n in range(5, len(data_brute) + 1):
    seuils_temp = discret_seuils([d['age'] for d in data_brute], n)
    data_anonyme = [dict(d) for d in data_brute]
    for d in data_anonyme:
        i = discretisation(d['age'], seuils_temp)
        d['age'] = f"[{seuils_temp[i]}, {seuils_temp[i+1]}["
    if k_anonymite(data_anonyme, L_CLE_PRIVE) >= 5:
        n_optimal = n
        break
seuils_age = discret_seuils([d['age'] for d in data_brute], n_optimal)

# --- Préparation des jeux d'entrainement et de test --- #
X = encoder_donnees(data_brute, seuils_age, L_CLE_INUTILE)
y = [ligne['deficit_sommeil'] for ligne in y_brut]
X_train, y_train, X_verif, y_verif = partition_train_val(X, y, r=0.2)
X_test = encoder_donnees(data_test_brute, seuils_age, L_CLE_INUTILE)


#!######################!#
#!  DEFICIT DE SOMMEIL  !#
#!######################!#

### --- Methode 1 : Arbre --- ###

# --- Calcul de la profondeur maximale de l'abre --- #
X_K, y_K = partition_val_croisee(X, y, K)
prof = list(range(12)) + [float("inf")]
erreur_arbre = [0] * len(prof)
for i in range(K):
    X_val, y_val = X_K[i], y_K[i]
    X_t, y_t = [], []
    for j in range(K):
        if j != i:
            X_t += X_K[j]
            y_t += y_K[j]
    arbre = arbre_train(X_t, y_t)
    for j, p in enumerate(prof):
        y_pred_val = arbre_pred(X_val, arbre, max_prof=p)
        erreur_arbre[j] += f_score(y_val, y_pred_val, 1) / K
meilleur_prof = argsort(erreur_arbre)[-1]

# --- Entrainement de l'abre --- #
arbre_final = arbre_train(X_train, y_train, False, max_prof=meilleur_prof)
y_pred_arbre = arbre_pred(X_verif, arbre_final, 5)

### --- Methode 2 : KPPV --- ###

# --- Normalisation des données --- #
loc, scale = norm_param(X)
X_train_norm = normalisation(X_train, loc, scale)
X_verif_norm = normalisation(X_verif, loc, scale)

# --- Calcul du K optimal maximisant le score F1 --- #
erreur_kppv = [0] * len(L_K)
for i in range(len(L_K)):
    y_pred_kppv = kppv(X_verif_norm, X_train_norm, y_train, L_K[i])
    erreur_kppv[i] = f_score(y_verif, y_pred_kppv, 1)
meilleur_k = L_K[argsort(erreur_kppv)[-1]]

# --- Application des KPPV --- #
y_pred_kppv = kppv(X_verif_norm, X_train_norm, y_train, meilleur_k)


#!######################!#
#!  AFFICHAGE TERMINAL  !#
#!######################!#
print("\n----------\n1. Traitement des données\n----------")
print("k-anonymité après discrétisation :", k_anonymite(data_anonyme, L_CLE_PRIVE))
print("l-diversité après discrétisation :", l_diversite(data_anonyme, L_CLE_PRIVE, 'deficit_sommeil'))

print("\n----------\n3. Classification\n----------")
print("\nMéthode 1 : Arbre")
print(f"Profondeur optimale : prof = {meilleur_prof}")
afficher_performance(y_verif, y_pred_arbre)

print("\nMéthode 2 : KPPV")
print(f"K optimal : K = {meilleur_k}")
afficher_performance(y_verif, y_pred_kppv)
