from ia01.utils import compte
from ia01.utils import lecture_csv
from ia01.metriques import taux_erreur
from ia01.metriques import eqm
from ia01.metriques import reqm
from ia01.majoritaire import vote_majoritaire



print("\n\n---\nPROBLEME 1 : CLASSIFICATION")

print("\n\n---\nExercice 1.1\n")
print(compte([1, 1, 2, 3, 3, 3, 3]))
print(compte(["chat", "chien", "chien", "chat", "chat", "chat"]))

print("\n\n---\nExercice 1.4\n")
data_dorade = lecture_csv("data/dorade.csv")
especes_dorades = [data_dorade[i]['espece'] for i in range(len(data_dorade))]
espece_predite = vote_majoritaire(especes_dorades)
print("Expece predite :", espece_predite)
print("Taux d'erreur :", taux_erreur(especes_dorades, [espece_predite] * len(especes_dorades)))



print("\n\n---\nPROBLEME 2 : REGRESSION")

print("\n\n---\nExercice 2.4\n")
poids_dorades = [data_dorade[i]['poids'] for i in range(len(data_dorade))]
poids_predit = vote_majoritaire(poids_dorades, True)
print("Poids predit :", poids_predit)
print("Erreur quadratique moyenne :", eqm(poids_dorades, [poids_predit] * len(especes_dorades)))

print("\n\n---\nExercice 2.4\n")
print("Racine de l'erreur quadratique moyenne :", reqm(poids_dorades, [poids_predit] * len(especes_dorades)))


print("\n\n---\nBONUS 3 : REGRESSION PAR CLASSIFICATION")
liste_classe_poids = [250, 750, 1250, 1750]
poids_rpc = []
for poids in poids_dorades:
    match poids:
        case _ if 0 <= poids <= 500: poids_rpc.append(250)
        case _ if 500 < poids <= 1000: poids_rpc.append(750)
        case _ if 1000 < poids <= 1500: poids_rpc.append(1250)
        case _ if 1500 < poids <= 2000: poids_rpc.append(1750)
poids_predit_rpc = vote_majoritaire(poids_rpc)
print("Poids predit :", poids_predit_rpc)
print("Taux d'erreur :", taux_erreur(poids_rpc, [poids_predit_rpc] * len(especes_dorades)))

