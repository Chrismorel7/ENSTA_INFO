from ia01.utils import *
from ia01.majoritaire import *
from ia01.arbre import *
from ia01.kppv import *
import random


data = lecture_csv("data/dorade.csv")
random.shuffle(data)
X = [[d["longueur"], d["poids"]] for d in data]
y = [d["espece"] for d in data]


X_train, y_train = X[:160], y[:160]
X_test, y_test = X[160:], y[160:]

#X_train, y_train = X, y
#X_test, y_test = X, y

loc_echelle, scale_echelle = norm_param(X, "echelle")
X_train_ech = normalisation(X_train, loc_echelle, scale_echelle)
X_test_ech = normalisation(X_test, loc_echelle, scale_echelle)

loc_centre, scale_centre = norm_param(X, "centre")
X_train_cen = normalisation(X_train, loc_centre, scale_centre)
X_test_cen = normalisation(X_test, loc_centre, scale_centre)


print("\n\n---\nPROBLEME 1 : ARBRE DE CLASSIFICATION")

data = lecture_csv("data/dorade.csv")
PROF = [2, 5, 10, 20, 30]
for prof in PROF:
    arbre = arbre_train(X_train, y_train, reg=False, max_prof=prof)
    predictions = arbre_pred(X_test, arbre)
    erreurs = sum(1 for vrai, pred in zip(y_test, predictions) if vrai != pred)
    taux_erreur = erreurs / len(y)
    print(f"Taux d'erreur pour prof = {prof}: {taux_erreur:.2%}")


print("\n\n---\nPROBLEME 2 : K PLUS PROCHE VOISINS")

liste_K = [1, 3, 5, 7, 200]
for k in liste_K:
    predictions = kppv(X_test, X_train, y_train, k)
    erreurs = sum(1 for vrai, pred in zip(y_test, predictions) if vrai != pred)
    taux_erreur = erreurs / len(y_test)
    print(f"Taux d'erreur pour k = {k}: {taux_erreur:.2%}")


print("\n\n---\nBONUS 3 : PONDERATION DE LA DISTANCE")

liste_K = [1, 3, 5, 7, 200]
for k in liste_K:
    predictions = kppv(X_test, X_train, y_train, k, pond=True)
    erreurs = sum(1 for vrai, pred in zip(y_test, predictions) if vrai != pred)
    taux_erreur = erreurs / len(y_test)
    print(f"Taux d'erreur pour k = {k}: {taux_erreur:.2%}")





print("\n\n---\nBILAN")


bilan_arbre = []
for prof in [2, 5, 10, 20, 30]:
    arbre = arbre_train(X_train, y_train, reg=False, max_prof=prof)
    predictions = arbre_pred(X_test, arbre)
    erreur = sum(1 for v, p in zip(y_test, predictions) if v != p) / len(y_test)
    bilan_arbre.append((f"prof={prof}", erreur))

bilan_kppv = []
for k in [1, 3, 5, 7, 200]:
    pred_brut = kppv(X_test, X_train, y_train, k)
    bilan_kppv.append((f"k={k}", "Non", "Non", sum(1 for v, p in zip(y_test, pred_brut) if v != p) / len(y_test)))
    
    pred_ech = kppv(X_test_ech, X_train_ech, y_train, k)
    bilan_kppv.append((f"k={k}", "Echelle", "Non", sum(1 for v, p in zip(y_test, pred_ech) if v != p) / len(y_test)))
    
    pred_cen = kppv(X_test_cen, X_train_cen, y_train, k)
    bilan_kppv.append((f"k={k}", "Centre", "Non", sum(1 for v, p in zip(y_test, pred_cen) if v != p) / len(y_test)))
    
    pred_pond = kppv(X_test, X_train, y_train, k, pond=True)
    bilan_kppv.append((f"k={k}", "Non", "Oui", sum(1 for v, p in zip(y_test, pred_pond) if v != p) / len(y_test)))

print("\n--- TABLEAU BILAN : ARBRE ---")
print(f"{'Paramètre':<15} | {'Erreur'}")
print("-" * 30)
for ligne in bilan_arbre:
    print(f"{ligne[0]:<15} | {ligne[1]:.2%}")

print("\n--- TABLEAU BILAN : KPPV ---")
print(f"{'Paramètre':<10} | {'Normalisation':<15} | {'Pondération':<12} | {'Erreur'}")
print("-" * 55)
for ligne in bilan_kppv:
    print(f"{ligne[0]:<10} | {ligne[1]:<15} | {ligne[2]:<12} | {ligne[3]:.2%}")