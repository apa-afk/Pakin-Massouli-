# Projet Python

*Analyse des accidents de la route en France (2005-2024).*

# Table des matières
1. [Définitions](#definitions)
2. [Objectifs](#objectifs)
3. [Sources des données](#sources)
4. [Présentation du dépôt](#pres)
5. [Exécution](#execution)
6. [Licence](#licence)

## 1. Définitions <a name="definitions">

**BAAC (Base de données annuelles des accidents corporels) :**

Ensemble de jeux de données publics décrivant les accidents corporels de la circulation. Chaque ligne correspond à un accident et inclut des informations temporelles, géographiques et contextuelles (conditions, usagers, véhicules).

**Alcoolisation ponctuelle importante (API) :**

Indicateur régional issu des enquêtes de santé publique, mesurant les épisodes d’alcoolisation importante sur une période donnée.

**Richesse régionale (proxy) :**

Indicateur construit comme le rapport entre le montant total d’imposition et le nombre de foyers fiscaux par région et par année.

## 2. Objectifs <a name="objectifs">

Ce projet vise à analyser les accidents de la route en France à partir des données BAAC (2005-2024) et à mettre en évidence les facteurs associés à la gravité des accidents. L’étude combine :

- une analyse descriptive des accidents et des variables contextuelles ;
- une analyse exploratoire multivariée (ACP) ;
- une étude régionale croisant accidents, alcoolisation et richesse.

## 3. Sources des données <a name="sources">

Les données sont principalement issues des sources publiques suivantes :

- data.gouv.fr (BAAC : accidents corporels 2005-2024) ;
- Sante publique France - ODiSSe (alcoolisation ponctuelle importante et consommation quotidienne) ;
- impots.gouv.fr (données fiscales régionales : foyers et montants).

## 4. Présentation du dépôt <a name="pres">

- `main.ipynb` : notebook principal (rapport).
- `modules/` : fonctions utilitaires et graphiques (`module_BAAC.py`, `module_region.py`, `graph_baac.py`, `graph_region.py`).
- `requirements.txt` : dépendances Python.

## 5. Exécution <a name="execution">

Pour exécuter le projet, il suffit de lancer `main.ipynb`.
