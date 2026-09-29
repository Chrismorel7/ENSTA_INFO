from ia01.utils import *
from ia01.majoritaire import *
from ia01.arbre import *
from ia01.kppv import *
from ia01.evaluation import *
from ia01.metriques import *

#? Exercice 1.1.
DATA = lecture_csv("data/compas.csv")
N_data = len(DATA)
DATA = [DATA[i] for i in range(N_data) if DATA[i]['race'] in ["African-American", "Caucasian"]]
N_data = len(DATA)


#? Exercice 1.2.
X = [DATA[i]['decile_score'] for i in range(N_data)]
Y_TRUE = [DATA[i]['two_year_recid'] for i in range(N_data)]

G = [1] * N_data
y = [0] * N_data
for i in range(N_data):
    if DATA[i]['race'] == 'African-American':
        G[i] = 2
    if X[i] >= 8:
        y[i] = 1


#? Exercice 1.3.
Y_TRUE_c = [Y_TRUE[i] for i in range(N_data) if DATA[i]['race'] == "Caucasian"]
y_c = [y[i] for i in range(N_data) if DATA[i]['race'] == "Caucasian"]
N_c = len(y_c)
Y_TRUE_am = [Y_TRUE[i] for i in range(N_data) if DATA[i]['race'] == "African-American"]
y_am = [y[i] for i in range(N_data) if DATA[i]['race'] == "African-American"]
N_am = len(y_am)

VP_c = sum(1 for vrai, pred in zip(Y_TRUE_c, y_c) if vrai == pred and vrai == 1)
FP_c = sum(1 for vrai, pred in zip(Y_TRUE_c, y_c) if vrai == 0 and pred == 1)
VP_am = sum(1 for vrai, pred in zip(Y_TRUE_am, y_am) if vrai == pred and vrai == 1)
FP_am = sum(1 for vrai, pred in zip(Y_TRUE_am, y_am) if vrai == 0 and pred == 1)

score_parite_c = (VP_c + FP_c) / N_c
score_parite_am = (VP_am + FP_am) / N_am
print("\n--- Question 1.3. ---\n")
print(f"Score de Parité :\nCaucasian :\t\t{round(score_parite_c,3)}\nAfrican-American : \t{round(score_parite_am,3)}")

#? Exercice 1.4.
TPR_c = TPR(Y_TRUE_c, y_c, 1)
FPR_c = FPR(Y_TRUE_c, y_c, 1)
TPR_am = TPR(Y_TRUE_am, y_am, 1)
FPR_am = FPR(Y_TRUE_am, y_am, 1)
print("\n--- Question 1.4. ---\n")
print(f"True Positive Rate :\nCaucasian :\t\t{round(TPR_c,3)}\nAfrican-American : \t{round(TPR_am,3)}")
print(f"\nFalse Positive Rate :\nCaucasian :\t\t{round(FPR_c,3)}\nAfrican-American : \t{round(FPR_am,3)}")

#? Exercice 1.6.
LISTE_SEUILS = list(range(1, 12))
print("\n--- Question 1.6. ---\n")
print(f"Liste des seuils : {LISTE_SEUILS}")
print(f"True Positive Rate pour les seuils : {ROC(Y_TRUE, X, 1, LISTE_SEUILS)[0]}")
print(f"False Positive Rate pour les seuils : {ROC(Y_TRUE, X, 1, LISTE_SEUILS)[1]}")


#? Exercice 1.7.
ALPHA = 0.05
couple_possible = []

X_c = [DATA[i]['decile_score'] for i in range(N_data) if DATA[i]['race'] == 'Caucasian']
X_am = [DATA[i]['decile_score'] for i in range(N_data) if DATA[i]['race'] == 'African-American']

L_TPR_c, L_FPR_c = ROC(Y_TRUE_c, X_c, 1, LISTE_SEUILS)
L_TPR_am, L_FPR_am = ROC(Y_TRUE_am, X_am, 1, LISTE_SEUILS)

for t1 in range(len(LISTE_SEUILS)):
    TPR_t1, FPR_t1 = L_TPR_c[t1], L_FPR_c[t1]
    for t2 in range(len(LISTE_SEUILS)):
        TPR_t2, FPR_t2 = L_TPR_am[t2], L_FPR_am[t2]
        if abs(TPR_t1 - TPR_t2) <= ALPHA and abs(FPR_t1 - FPR_t2) <= ALPHA:
            couple_possible.append((LISTE_SEUILS[t1], LISTE_SEUILS[t2]))

print("\n--- Question 1.7. ---\n")
print(f"Couple possible pour avoir equite avec alpha = {ALPHA} : \t{couple_possible}")

#? Exercice 1.8.
BETA = 1
best_f, best_couple = 0, (0,0)
for (t1, t2) in couple_possible:
    Y_PRED = [0] * N_data
    for i in range(N_data):
        if DATA[i]['race'] == 'Caucasian' and X[i] >= t1:
            Y_PRED[i] = 1
        if DATA[i]['race'] == 'African-American' and X[i] >= t2:
            Y_PRED[i] = 1
    ft1t2 = f_score(Y_TRUE, Y_PRED, 1, BETA)
    if ft1t2 >= best_f:
        best_f = ft1t2
        best_couple = (t1, t2)
print("\n--- Question 1.8. ---\n")
print(f"Score F1 maximal pour le couple {best_couple} : {round(best_f,3)}")

#? Exercice 1.9.
print("\n--- Question 1.9. ---\n")
LISTE_BETA = [0.33, 1, 3]
for Beta in LISTE_BETA:
    best_f, best_couple = 0, (0,0)
    for (t1, t2) in couple_possible:
        Y_PRED = [0] * N_data
        for i in range(N_data):
            if DATA[i]['race'] == 'Caucasian' and X[i] >= t1:
                Y_PRED[i] = 1
            if DATA[i]['race'] == 'African-American' and X[i] >= t2:
                Y_PRED[i] = 1
        ft1t2 = f_score(Y_TRUE, Y_PRED, 1, Beta)
        if ft1t2 >= best_f:
            best_f = ft1t2
            best_couple = (t1, t2)
    Y_PRED = [0] * N_data
    for i in range(N_data):
        if DATA[i]['race'] == 'Caucasian' and X[i] >= best_couple[0]:
            Y_PRED[i] = 1
        if DATA[i]['race'] == 'African-American' and X[i] >= best_couple[1]:
            Y_PRED[i] = 1
    print(f"Score F-{Beta} maximal pour le couple {best_couple} : {round(best_f,3)}\nPrecisions : {round(precision(Y_TRUE, Y_PRED, 1),3)}\t\tRappel : {round(rappel(Y_TRUE, Y_PRED, 1),3)}\n")