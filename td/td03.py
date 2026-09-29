from ia01.utils import *
from ia01.majoritaire import *
from ia01.arbre import *
from ia01.kppv import *
from ia01.evaluation import *

data = lecture_csv("data/dorade_test.csv")
X = [[d["longueur"], d["poids"]] for d in data]
y = [d["espece"] for d in data]
X_train, y_train, X_val, y_val = partition_train_val(X, y)


print("#"*50)
print("\t\tCLASSIFICATION")
print("#"*50)

liste_K = [1, 3, 5, 7, 200]

print("\n---\nQUESTION 1.1. ENSEMBLE TRAIN\n---")
for k in liste_K:
    predictions = kppv(X_train, X_train, y_train, k)
    erreurs = sum(1 for vrai, pred in zip(y_train, predictions) if vrai != pred)
    taux_erreur = erreurs / len(y_train)
    print(f"Taux d'erreur pour k = {k}: {taux_erreur:.2%}")
    
print("\n---\nQUESTION 1.3. PARTITION 1/5\n---")
for k in liste_K:
    predictions = kppv(X_val, X_train, y_train, k)
    erreurs = sum(1 for vrai, pred in zip(y_val, predictions) if vrai != pred)
    taux_erreur = erreurs / len(y_val)
    print(f"Taux d'erreur pour k = {k}: {taux_erreur:.2%}")

print("\n---\nQUESTION 1.5. PARTITION VALIDATION CROISEE\n---")
X_K, y_K = partition_val_croisee(X, y, K=5)

for k in liste_K:
    erreurs_vc = []
    for i in range(len(X_K)):
        # Séparation du bloc de validation et des blocs d'apprentissage
        X_val_vc = X_K[i]
        y_val_vc = y_K[i]
        X_train_vc = [x for j in range(len(X_K)) if j != i for x in X_K[j]]
        y_train_vc = [val for j in range(len(y_K)) if j != i for val in y_K[j]]
        # Évaluation
        predictions = kppv(X_val_vc, X_train_vc, y_train_vc, k)
        erreurs = sum(1 for vrai, pred in zip(y_val_vc, predictions) if vrai != pred)
        erreurs_vc.append(erreurs / len(y_val_vc))
    taux_erreur_moyen = sum(erreurs_vc) / len(erreurs_vc)
    print(f"Taux d'erreur croisé moyen pour k = {k}: {taux_erreur_moyen:.2%}")


print("#"*50)
print("\t\tARBRE")
print("#"*50)

print("\n---\nQUESTION 1.1. ENSEMBLE TRAIN\n---")
arbre = arbre_train(X_train, y_train)
predictions = arbre_pred(X_train, arbre)
erreurs = sum(1 for vrai, pred in zip(y_train, predictions) if vrai != pred)
taux_erreur_train = erreurs / len(y_train)
print(f"Taux d'erreur : {taux_erreur_train:.2%}")

print("\n---\nQUESTION 1.3. PARTITION 1/5\n---")
arbre = arbre_train(X_train, y_train)
predictions = arbre_pred(X_val, arbre)
erreurs = sum(1 for vrai, pred in zip(y_val, predictions) if vrai != pred)
taux_erreur_val = erreurs / len(y_val)
print(f"Taux d'erreur : {taux_erreur_val:.2%}")

print("\n---\nQUESTION 1.3. PARTITION VALIDATION CROISEE\n---")
X_K, y_K = partition_val_croisee(X, y, K=5)
erreurs_vc = []
for i in range(len(X_K)):
    X_val_vc = X_K[i]
    y_val_vc = y_K[i]
    X_train_vc = [x for j in range(len(X_K)) if j != i for x in X_K[j]]
    y_train_vc = [val for j in range(len(y_K)) if j != i for val in y_K[j]]
    arbre_vc = arbre_train(X_train_vc, y_train_vc)
    predictions = arbre_pred(X_val_vc, arbre_vc)
    erreurs = sum(1 for vrai, pred in zip(y_val_vc, predictions) if vrai != pred)
    erreurs_vc.append(erreurs / len(y_val_vc))
taux_erreur_moyen = sum(erreurs_vc) / len(erreurs_vc)
print(f"Taux d'erreur moyen {taux_erreur_moyen:.2f}")


print("#"*50)
print("\t\tLOGEMENT")
print("#"*50)

champs_descr = [
    "annee_construction",
    "surface_habitable",
    "nombre_niveaux",
    "surface_baies_orientees_nord",
    "surface_baies_orientees_est_ouest",
    "surface_baies_orientees_sud",
    "surface_planchers_hauts_deperditifs",
    "surface_planchers_bas_deperditifs",
    "surface_parois_verticales_opaques_deperditives",
    "longitude",
    "latitude",
    "tr001_modele_dpe_type_libelle",
    "tr002_type_batiment_libelle",
]

one_hot_dpe_type = {"Copropriete": [1,0,0,0], "Location": [0,1,0,0], "Neuf":[0,0,1,0],"Vente":[0,0,0,1]}
one_hot_bat_type = {"Appartement": [1,0,0], "Logements collectifs":[0,1,0], "Maison":[0,0,1]}

data = lecture_csv("data/dep_48_DPE.csv")


y = [0 for _ in range(len(data))]
classe_passoire = ['F', 'G']
for i in range(len(data)):
    if data[i]['classe_consommation_energie'] in classe_passoire or data[i]['classe_estimation_ges'] in classe_passoire:
        y[i] = 1
print(y)

X = [[0,0,0,0,0,0,0,0,0,0,0,[0,0,0,0],[0,0,0]] for _ in range(len(data))]
for i in range(len(data)):
    for j in range(len(champs_descr)-2):
        X[i][j] = data[i][champs_descr[j]]
    X[i][11] = one_hot_dpe_type[data[i]["tr001_modele_dpe_type_libelle"]]
    X[i][12] = one_hot_bat_type[data[i]["tr002_type_batiment_libelle"]]

X_train, y_train, X_val, y_val = partition_train_val(X, y, r=0.25)