from ia01.privacy import *
from ia01.utils import *

data = lecture_csv('data/compas-privacy.csv')

print(f"\n---\nQuestion 1.4.\n---")
print(f"K-anonymite pour l'attribut 'sex'   :\t{k_anonymite(data, ['sex'])}")
print(f"K-anonymite pour l'attribut 'age'   :\t{k_anonymite(data, ['age'])}")
print(f"K-anonymite pour l'attribut 'group' :\t{k_anonymite(data, ['group'])}")

print(f"\n---\nQuestion 1.5.\n---")
print(f"K-anonymite pour l'attribut ('sex', 'group') :\t{k_anonymite(data, ['sex', 'group'])}")
print(f"K-anonymite pour l'attribut ('sex', 'age')   :\t{k_anonymite(data, ['sex', 'age'])}")
print(f"K-anonymite pour l'attribut ('age, 'group')  :\t{k_anonymite(data, ['age', 'group'])}")

print(f"\n---\nQuestion 1.6.\n---")
print(f"K-anonymite pour l'attribut ('sex', 'age', 'group') :\t{k_anonymite(data, ['sex', 'age', 'group'])}")

print(f"\n---\nQuestion 1.7.\n---")
print(f"Nombre d'individus par groupe ethnique :\t{compte([individu['group'] for individu in data])}")

print(f"\n---\nQuestion 1.8.\n---")
groupe_maj = ['African-American', 'Caucasian', 'Other']
for individu in data:
    if individu['group'] not in groupe_maj:
        individu['group'] = 'Other'
print(f"K-anonymite pour l'attribut ('sex', 'group') :\t{k_anonymite(data, ['sex', 'group'])}")

print(f"\n---\nQuestion 1.11.\n---")
for n in range(5, len(data) + 1):
    seuils = discret_seuils([d['age'] for d in data], n)
    data_anonyme = [dict(d) for d in data]
    for d in data_anonyme:
        i = discretisation(d['age'], seuils)
        d['age'] = f"[{seuils[i]}, {seuils[i+1]}["
    if k_anonymite(data_anonyme, ['sex', 'group', 'age']) >= 5:
        print(f"Valeur de n nécessaire pour la 5-anonymité : {n}")
        break
else:
    print("Aucune discrétisation ne donne la 5-anonymité")
print(discret_seuils((indiv['age'] for indiv in data), n))

print(f"\n---\nQuestion 2.2.\n---")
attributs = ['sex', 'group', 'age']
print(f"k-anonymité des données utilisées : {k_anonymite(data_anonyme, attributs)}")
l_div = l_diversite(data_anonyme, attributs, 'charge')
print(f"Niveau de l-diversité : {l_div}")