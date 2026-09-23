# Workflow d'extension du catalogue

Ce workflow est le point d'entrée pour ajouter un modèle, une courbe, un
produit, un sample ou un prix. Une extension est terminée lorsque son code, sa
recette, ses tests et son entrée publique sont cohérents, avec une validation
indépendante lorsqu'elle publie un prix.

Les règles détaillées restent dans leurs contrats propriétaires :

- [paramètres modèle et produit](model-and-product-parameter-dataset-generation.md) ;
- [samples modèle](model-sample-dataset-generation.md) ;
- [validation indépendante des prix](independent-price-validation-pipeline.md) ;
- [contrats CUDA](cuda/README.md) ;
- [génération automatique des bindings](../tools/codegen/pricing_bindings/README.md).

Générer un prix et le certifier sont deux opérations distinctes. Une génération
écrit toujours `validation.status: pending` et `verified: false`, avec le
chemin prévu du cache indépendant, même si une ancienne version était certifiée.
Elle ne lance aucun moteur de référence et ne crée pas de notebook de validation
adjacent. Le YAML décrit les données réellement générées : modifier seulement
le nombre de trajectoires ou `verified` à la main ne met pas à jour les données
ni leurs preuves. Le lien vers un validateur explique le contrôle à effectuer ;
il ne remplace jamais son résultat.

## 1. Classer l'extension

Identifier les couches réellement nouvelles avant de créer un fichier :

| Élément | Propriétaire runtime | Recette |
|---|---|---|
| Modèle equity | `src/model/equity/<markovian|rough>/<model>` | `catalog/model/equity/<family>/<model>` |
| Modèle fixed income | `src/model/fixed_income/<model>` | `catalog/model/fixed_income/<model>` |
| Courbe | `src/curve/<curve>` | `catalog/curve/<curve>` |
| Produit | `src/product/<product>` | `catalog/product/<product>` |
| Composition modèle-produit | `<model>/product/[<curve>/]<product>.{cuh,cu}` | `<model>/prices/[<curve>/]<product>/<dataset_id>` |
| Outil partagé | `tools/<responsibility>` | aucune recette concrète |

`catalog` et `datasets` reprennent exactement le préfixe canonique de `src`.
La famille equity est toujours visible. Une courbe apparaît dans le chemin
fixed-income uniquement lorsqu'elle appartient au contrat de pricing.

Ne pas recréer un produit pour un nouveau modèle, une dynamique pour un
nouveau payoff, ou une infrastructure partagée dans une recette.

## 2. Déclarer le contrat

Avant l'implémentation :

- choisir le nom canonique `snake_case` et le nom d'affichage ;
- déclarer les couples modèle, courbe, produit et variante attendus ;
- fixer temps, jours ouvrés, paiements, exercice, notionnel, strike,
  actualisation et mesure de pricing ;
- distinguer dates contractuelles et grille numérique ;
- choisir formule fermée, Monte Carlo, Volterra FFT, lift N-facteurs ou
  Longstaff--Schwartz ;
- identifier le contrat CUDA et l'implémentation voisine les plus proches ;
- définir les domaines numériques normaux, stress et limites.

Les dates du calendrier modèle sont des jours ouvrés entiers sous la convention
globale `days_per_year: 252`. Un schéma discrétisé dérive son `dt` du
`steps_per_year` déclaré ; une transition exacte ne reçoit pas de sous-pas
artificiels.

## 3. Implémenter le runtime minimal

### Modèle

Créer uniquement les responsabilités applicables :

- `parameters.hpp` pour la ligne compacte transférable au GPU ;
- `dataset.hpp/.cpp` pour le chargement et la validation host ;
- `dynamics.cuh` et `dynamics_impl.cuh` pour une dynamique simulée ;
- `analytics.cuh` et `analytics_impl.cuh` pour les quantités réutilisables ;
- une préparation explicitement qualifiée lorsqu'un moteur l'exige.

Respecter le [contrat des dynamiques](cuda/model-dynamics-contract.md) et le
[contrat des analytics](cuda/model-analytics-contract.md). Une erreur de ligne
cite son identifiant et les loaders préservent l'ordre des données.

### Courbe

Une courbe possède sa ligne, son loader et les fonctions term-structure
applicables : taux zéro, facteur d'actualisation, forward instantané, dérivée
du forward et forward de période. Les modèles ajustés composent cette API sans
recopier la courbe.

### Produit

Créer sous `src/product/<product>` :

- `parameters.hpp` et `dataset.hpp/.cpp` ;
- `pricing_policy.cuh` ;
- `schedule.cuh` ou `continuation_state.cuh` seulement si nécessaires.

Les paramètres décrivent le contrat financier, pas un modèle. Le payoff, le
calendrier et les invariants sont validés une seule fois dans leur propriétaire.

### Composition pricing

Le dossier `product/` du modèle contient uniquement la composition mince entre
modèle, courbe éventuelle et produit. Les moteurs, workspaces, réductions et
schedules génériques restent partagés.

Pour call/put ou payer/receiver, utiliser la spécialisation template publique
et ses instanciations explicites. Ne pas ajouter un branchement de côté dans le
hot path ni dupliquer un produit dont seuls le signe ou l'orientation changent.

Suivre le [contrat closed form et Monte Carlo](cuda/closed-form-and-monte-carlo-pricing-contract.md)
ou le [contrat American/Bermudan](cuda/american-and-bermudan-pricing-contract.md).

## 4. Déclarer la capacité et générer

Mettre à jour le manifeste typé avant d'écrire un binding ou une recette
mécanique. Il doit résoudre sans ambiguïté :

```text
(model, optional curve, product, variant)
    -> engine -> binding -> CMake target -> catalog recipe
```

Le codegen produit les bindings pricing, bindings sample, recettes répétitives,
enregistrement CMake et empreintes. Les algorithmes, distributions de
paramètres et exceptions mathématiques restent explicites chez leur
propriétaire.

Vérifier la dérive sans modifier le dépôt :

```bash
python3 tools/codegen/pricing_bindings/generate.py \
  --family all \
  --output /tmp/ai_factory-pricing-bindings \
  --compare-root .
```

Un fichier généré n'est jamais corrigé à la main.

## 5. Ajouter les datasets

### Paramètres modèle, courbe ou produit

La recette définit les distributions, grilles et contraintes. Le générateur
écrit le JSON et un `generation.yaml` minimal ; le contrôleur enrichit ce reçu
avec la provenance, puis le publie en dernier. Le générateur recharge l'artefact
avec le loader de production avant de réussir.

Les datasets modèle et produit suivent l'ordre contractuel de 900 lignes core
puis 100 lignes stress. Utiliser le
[contrat de génération des paramètres](model-and-product-parameter-dataset-generation.md)
pour les bornes, domaines Philox et contrôles obligatoires.

### Samples modèle

Les deux recettes publiées par modèle produisent chacune trois millions de
lignes selon les formes contractuelles `12 000 x 250` et `3 000 000 x 1`.
Bindings, helpers et recettes sont générés depuis le manifeste ; aucune recette
ad hoc ne réimplémente la dynamique.

Chaque générateur expose `--smoke-test` et `--preflight`. Le
[contrat des samples](model-sample-dataset-generation.md) possède la forme des
lignes, les seeds, la mémoire, le replay et les métadonnées.

### Prix

Une recette de prix charge les datasets d'entrée, applique une construction
`Aligned` ou `CartesianProduct`, lance le pricer public, puis écrit prix, erreur
standard applicable et timings. Le JSON est relu et validé immédiatement.

Le YAML reste compact et pointe vers les entrées et la référence indépendante.
Les détails de backend et les comparaisons appartiennent au dataset de
référence, pas à la recette publiée.

## 6. Valider indépendamment un prix

Appliquer le [pipeline de validation indépendant](independent-price-validation-pipeline.md)
sans le recopier dans le validateur modèle-produit :

1. inventorier tous les moteurs Premia compatibles dans tous les menus ;
2. ordonner les candidats Premia par compatibilité, robustesse et temps mesuré ;
3. essayer ensuite QuantLib spécialisé, puis QuantLib Monte Carlo ;
4. conserver `not_available` si aucune référence fiable n'existe.

Un échec technique peut descendre dans cette hiérarchie. Une divergence finie
est une erreur à comprendre ; elle ne choisit jamais rétrospectivement la
référence la plus proche.

Les 900 lignes core et 100 lignes stress doivent toutes disposer d'une
référence, respecter leurs budgets et passer avant `verified: true`. Le mode
normal est cache-only ; les moteurs externes ne sont lancés que par une
régénération explicite.

## 7. Enregistrer le build et les tests

- ajouter les unités runtime au module CMake propriétaire ;
- laisser les artefacts générés au CMake généré ;
- construire le target le plus étroit ;
- tester loaders, domaines invalides, limites et symétries ;
- exécuter les tests CUDA du moteur et du produit concernés ;
- exécuter `pricing_binding_codegen`, `model_source_layout` et les contrôles de
  catalogue ;
- lancer les validations numériques concernées ;
- exécuter les suites agrégées avant publication.

Les changements sensibles aux kernels comparent avant/après temps, registres,
spills, shared memory et occupation sur le même GPU et toolchain. Une valeur
SM89 ne devient jamais un défaut portable.

## 8. Publier

- générer ensemble JSON et YAML ;
- vérifier identifiants, chemins, URLs, nombres de lignes et empreintes ;
- conserver les gros JSON hors Git ;
- ne stocker aucun secret dans le catalogue ou le site statique ;
- ajouter ou mettre à jour l'entrée correspondante dans le projet du site ;
- tester le téléchargement et l'affichage des métadonnées.

## Définition de terminé

Une extension est terminée lorsque :

- le manifeste déclare une composition unique et le codegen est sans diff ;
- le runtime respecte les contrats de son domaine ;
- les recettes reproduisent leurs JSON et YAML sans édition manuelle ;
- les tests ciblés et contrôles d'architecture passent ;
- chaque prix publié possède une validation indépendante core et stress ;
- les ressources et performances sont qualifiées si le hot path change ;
- la documentation et le site pointent vers les nouveaux propriétaires sans
  recopier d'inventaire dérivé.
