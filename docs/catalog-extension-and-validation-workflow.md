# Workflow d'extension et de validation du catalogue

Ce document est la checklist de référence pour toute extension du catalogue.
Une extension n'est terminée que lorsque le code, les datasets, les tests, la
validation indépendante et le site ont tous été mis à jour.

## Regles non negociables

- [ ] Reprendre l'ossature de l'exemple existant le plus proche.
- [ ] Conserver les memes noms, signatures, ordre des fonctions et commentaires
      lorsque la semantique est identique.
- [ ] Ne mettre dans `src/` que le chargement, les mathematiques et le pricing.
- [ ] Mettre les helpers de generation et de validation de datasets dans
      `tools/datasets/`.
- [ ] Generer les JSON et YAML avec le code; ne pas les corriger a la main.
- [ ] Terminer chaque generateur de modele, courbe, produit ou prix par le
      validateur de structure correspondant au fichier qu'il vient d'ecrire.
- [ ] Faire preceder les validations metier de chaque `load_*` par la validation
      de l'ossature JSON commune a sa famille de dataset.
- [ ] Ajouter chaque nouveau modele, courbe, produit et prix au site.
- [ ] Appliquer, dans cet ordre, la meilleure validation disponible: pricer
      specialise Premia, pricer specialise QuantLib, Monte-Carlo QuantLib,
      puis `none` si aucune reference independante fiable n'existe.
- [ ] Decider la disponibilite de Premia sur le couple `(modele, produit)` et
      son moteur compatible, jamais sur la methode numerique employee par le
      generateur AI_factory.
- [ ] Inscrire obligatoirement le resultat et le backend dans le bloc YAML
      racine `validation`.
- [ ] Ne jamais activer CUDA fast math (`--use_fast_math`). Le projet exige des
      résultats reproductibles et n'expose volontairement aucune option de
      compilation correspondante.
- [ ] Ne pas ajouter `__launch_bounds__` sans une conception explicitement
      approuvée, adaptée aux architectures CUDA ciblées, puis validée par les
      registres, les spills, l'occupation et les temps mesurés sur chaque cible.
- [ ] Préserver le mapping déterministe des lignes et des chemins, l'ordre des
      réductions, ainsi que les accumulations FP64 déjà utilisées.

`validation.reference: "none"` est un etat explicite, pas une validation. Il
permet de publier une V1 clairement etiquetee lorsque Premia et QuantLib ne
fournissent aucune reference comparable. La selection s'effectue par regime et
par ligne selon
[`independent-price-validation-pipeline.md`](independent-price-validation-pipeline.md).
Un echec technique Premia autorise un repli documente pour les seules lignes
concernees; une divergence apres un calcul Premia reussi reste un echec et ne
doit jamais etre masquee par QuantLib.

## Identifier la nature de l'extension

Avant de creer un dossier, separer les couches reellement nouvelles:

- une famille de parametres produit appartient a `src/product/` et a
  `catalog/product/`;
- une dynamique ou des analytiques appartiennent a
  `src/model/<asset_class>/<model>/` et a `catalog/model/`;
- une courbe appartient a `src/curve/` et a `catalog/curve/`;
- le pricing d'un couple modele-produit appartient au modele, dans
  `src/model/<asset_class>/<model>/[<curve>/]`;
- une base de prix conserve son propre generateur, YAML, rapport et notebook
  sous `catalog/price/`, meme lorsque son code de pricing est partage;
- l'orchestration de validation appartient a `validation/model/`, tandis que
  les conversions propres a Premia et QuantLib restent dans leurs backends.

Une extension peut ne concerner qu'une de ces couches. Ne pas dupliquer une
structure produit parce qu'un nouveau modele la price, ni une dynamique parce
qu'un nouveau payoff l'utilise.

## Avant de coder

- [ ] Definir le nom canonique en `snake_case` et le nom affiche sur le site.
- [ ] Lister les combinaisons modele, courbe et produit a supporter.
- [ ] Identifier l'implementation existante la plus proche a copier.
- [ ] Fixer les conventions financieres: temps, paiements, exercice, notionnel,
      strike, actualisation et mesure de pricing.
- [ ] Utiliser `Actual/360` pour les temps du catalogue; employer
      `target_dt = 1 / 360` seulement lorsqu'une grille quotidienne ou un schema
      numerique est effectivement necessaire.
- [ ] Choisir la methode: formule exacte, Monte-Carlo, Longstaff-Schwartz, etc.
- [ ] Identifier immediatement le pricer Premia ou QuantLib de reference et son
      domaine de validite.
- [ ] Chercher le produit financier dans tout le catalogue Premia, meme si son
      moteur est continu, PDE ou Monte Carlo alors que le prix CUDA utilise une
      approximation discrete; documenter ensuite l'ecart de contrat.

## Ajouter un modele

- [ ] Creer `src/model/<asset_class>/<model>/dataset.hpp` et `dataset.cpp`.
- [ ] Declarer une structure de parametres compacte, explicite et adaptee au GPU.
- [ ] Implementer `load_models(...)` en preservant l'ordre des lignes.
- [ ] Valider la structure JSON avant de lire les lignes.
- [ ] Valider chaque parametre; toute erreur doit citer l'identifiant de ligne.
- [ ] Ajouter `dynamics.cuh/.cu` seulement si le modele doit etre simule.
- [ ] Reprendre les interfaces communes pertinentes: `prepare_model`,
      `one_step_transition`, `simulate_terminal_state` et
      `simulate_on_regular_grid`.
- [ ] Ajouter `analytics.cuh/.cu` pour les quantites analytiques reutilisables.
- [ ] Conserver les noms et l'ordre des fonctions des modeles voisins.
- [ ] Ajouter les sources et les tests au `CMakeLists.txt`.

### Dataset du modele

- [ ] Ajouter les helpers reutilisables dans `tools/datasets/`.
- [ ] Creer `catalog/model/<asset_class>/<model>/<dataset_id>/generator.cpp`.
- [ ] Creer `catalog/model/<asset_class>/<model>/<dataset_id>/dataset.yaml` par le generateur.
- [ ] Utiliser des plages financieres raisonnables.
- [ ] Placer les cas extremes en queue de distribution, pas au centre.
- [ ] Rejeter les lignes mathematiquement ou numeriquement invalides.
- [ ] Verifier `database_id`, `model_family`, `catalog`, `url`, `row_count` et
      `models` dans le JSON genere.
- [ ] Recharger le JSON genere avec `load_models(...)` avant de terminer.

## Ajouter une courbe

- [ ] Creer `src/curve/<curve>/dataset.hpp` et `dataset.cpp`.
- [ ] Implementer `load_curves(...)` avec validation de structure et de lignes.
- [ ] Creer `term_structure.cuh/.cu` avec l'interface commune pertinente:
      `zero_rate`, `log_discount_factor`, `discount_factor`,
      `instantaneous_forward`, `forward_derivative` et `forward_rate`.
- [ ] Garder exactement les memes noms entre courbes lorsque les objets
      financiers calcules sont identiques.
- [ ] Ajouter les helpers reutilisables dans `tools/datasets/`.
- [ ] Creer `catalog/curve/<curve>/<dataset_id>/generator.cpp` et
      `dataset.yaml`.
- [ ] Controler les taux et forwards sur une grille de maturites.
- [ ] Rejeter les courbes absurdes selon les bornes documentees.
- [ ] Recharger et valider le JSON genere.

### Modeles ajustes a la courbe

- [ ] Creer `src/model/<asset_class>/<model>/<curve>/` a partir de la courbe deja supportee la
      plus proche.
- [ ] Modifier uniquement le namespace, le type de courbe et les termes qui
      dependent reellement de sa parametrisation.
- [ ] Verifier que les fonctions analytiques communes gardent la meme interface.
- [ ] Ajouter toutes les combinaisons produit supportees, leurs generateurs et
      leurs validations QuantLib.

## Ajouter un produit

- [ ] Creer `src/product/<product>/dataset.hpp` et `dataset.cpp`.
- [ ] Declarer uniquement les parametres necessaires au payoff ou au contrat.
- [ ] Implementer `load_products(...)` avec validation de structure et de lignes.
- [ ] Verifier au minimum les valeurs finies, positivites, maturites croissantes,
      calendriers et conventions propres au produit.
- [ ] Ajouter les helpers reutilisables dans `tools/datasets/`.
- [ ] Creer `catalog/product/<asset_class>/<product>/<dataset_id>/generator.cpp`
      et `dataset.yaml`.
- [ ] Generer des strikes, maturites et tenors plausibles pour le produit.
- [ ] Recharger et valider le JSON genere.

### Familles call/put

Lorsque call et put different uniquement par l'orientation du payoff:

- [ ] Utiliser une seule famille produit, par exemple `european_options`, et un
      seul couple de fichiers de pricing `<product>.cuh/.cu`.
- [ ] Selectionner `OptionSide::call` ou `OptionSide::put` par template public;
      ne pas stocker un signe dans chaque ligne et ne pas brancher dans chaque
      trajectoire.
- [ ] Instancier explicitement les deux versions dans le `.cu`, afin que les
      generateurs C++ puissent les lier sans inclure l'implementation CUDA.
- [ ] Conserver deux dossiers sous `catalog/price/`: les prix call et put sont
      deux datasets publics distincts, meme s'ils partagent les parametres.
- [ ] Ne creer des bases de parametres propres au call ou au put que si leur
      construction differe reellement; les ranger alors dans la meme famille
      produit, comme les deux grilles de gap options.

Une barriere montante et une barriere descendante, ou une knock-in et une
knock-out, ne sont pas automatiquement la meme famille d'implementation: leur
etat de chemin et leur contrat doivent d'abord etre identiques a un simple
changement de signe.

### Pricing du produit

- [ ] Ajouter `<product>.cuh/.cu` dans chaque modele compatible.
- [ ] Copier l'ossature du produit numeriquement le plus proche.
- [ ] Garder `PreparedRow`, `prepare_row`, `evaluate_price` ou `evaluate_path`,
      le kernel, la validation et le launch dans le meme ordre.
- [ ] Choisir une topologie adaptee sans casser l'uniformite:
      un thread par prix pour une formule fermee, un ou plusieurs blocs par prix
      pour le Monte-Carlo selon la reduction et le workspace necessaires.
- [ ] Verifier les acces globaux contigus, la pression registre, les reductions,
      l'occupation et les allocations temporaires.
- [ ] Ajouter le generateur de prix pour chaque dataset public distinct, meme
      lorsqu'il ne fait que choisir une specialisation call/put.
- [ ] Ajouter le validateur modele-produit mince et les adaptateurs de backend
      necessaires avant de considerer le produit publiable.

Pour les signatures, l'ordre des fonctions et la strategie de kernel, suivre
le contrat
[`cuda-closed-form-and-monte-carlo-pricing-contract.md`](cuda-closed-form-and-monte-carlo-pricing-contract.md)
ou, pour l'exercice anticipe,
[`cuda-american-and-bermudan-pricing-contract.md`](cuda-american-and-bermudan-pricing-contract.md).

## Ajouter une base de prix

- [ ] Creer `catalog/price/<asset_class>/<model>/[<curve>/]<product>/<dataset_id>/`.
- [ ] Ajouter `generator.cpp` et `dataset.yaml` dans ce meme dossier.
- [ ] Charger les datasets modele, courbe si necessaire, et produit.
- [ ] Verifier la construction `Aligned` ou `CartesianProduct` et le nombre de
      lignes attendu.
- [ ] Garder le bloc de configuration separe de la logique CUDA et des metadonnees.
- [ ] Allouer, copier, lancer, synchroniser et liberer toutes les ressources CUDA.
- [ ] Verifier les erreurs CUDA et la validite de la configuration de lancement.
- [ ] Ecrire prix, erreur standard si pertinente et timings.
- [ ] Valider la structure du JSON de prix juste apres son ecriture.
- [ ] Verifier que tous les prix et erreurs standards attendus sont finis.

### Hiérarchie de validation obligatoire

- [ ] Separer explicitement les 900 lignes `core` des 100 lignes `stress`.
- [ ] Chercher d'abord un pricer specialise Premia compatible avec le modele,
      le produit et chaque ligne du domaine de parametres.
- [ ] Ne declarer Premia indisponible que si aucun moteur n'existe pour ce
      couple modele-produit, ou si le moteur rejette techniquement la ligne;
      une difference discret/continu n'est pas une absence de moteur.
- [ ] A defaut, chercher un pricer specialise QuantLib.
- [ ] A defaut, construire une simulation Monte-Carlo QuantLib independante.
- [ ] Si aucun backend fiable n'existe, conserver `status: "not_available"`,
      `verified: false`, `reference: "none"` dans le YAML.
- [ ] Ajouter le validateur unifie sous
      `validation/model/<asset_class>/<model>/[<curve>/]<product>.py`, avec la
      meme ossature que les produits voisins; omettre la courbe en equity.
- [ ] Exposer la commande canonique `python -m <module> DATASET REPORT`; elle
      seule orchestre les backends, ecrit le rapport et synchronise le YAML.
- [ ] Declarer les emplacements Premia specialise, QuantLib specialise et
      QuantLib Monte Carlo dans cet ordre, avec un adaptateur ou une raison
      d'indisponibilite explicite.
- [ ] Laisser `validation/hierarchy.py` transmettre au moteur suivant les seules
      exceptions techniques ligne par ligne; ne jamais y envoyer une divergence.
- [ ] Conserver les adaptateurs de backend reutilisables sous
      `validation/premia/` et `validation/quantlib/`.
- [ ] Mettre les conversions reutilisables dans un fichier commun au modele.
- [ ] Lire le JSON de prix produit par le vrai generateur CUDA.
- [ ] Reconstruire chaque ligne dans le backend avec les memes conventions.
- [ ] Comparer toutes les lignes, pas seulement un echantillon favorable.
- [ ] En cas d'echec technique Premia, conserver ligne, statut et raison, puis
      appliquer QuantLib uniquement a cette ligne.
- [ ] Ne jamais basculer vers QuantLib lorsque Premia a calcule un prix fini et
      comparable qui diverge: enregistrer une `comparison failure` et corriger
      la cause.
- [ ] Rejeter comme exception technique toute sortie pourtant finie qui viole
      une borne de non-arbitrage; conserver le diagnostic avant le repli.
- [ ] Controler erreur absolue, erreur relative, erreur maximale et taux d'echec.
- [ ] Controler le biais signe moyen pour detecter une erreur systematique.
- [ ] Expliquer les tolerances par la precision FP32 ou la statistique Monte-Carlo;
      ne pas les elargir uniquement pour faire passer le test.
- [ ] Ecrire `validation_report.json` a cote du notebook, avec ses sections
      `core` et `stress`, le plan complet des moteurs et l'empreinte canonique
      des prix et de leur configuration numerique.
- [ ] Generer le rapport et le bloc YAML exclusivement depuis l'execution du
      validateur; ne jamais rediger leurs resultats a la main.
- [ ] Faire charger et afficher ce rapport par le notebook sans relancer de
      pricer de reference.
- [ ] Executer et livrer le notebook afin que sa sortie compilee corresponde au
      rapport courant.
- [ ] Si aucun moteur n'est compatible, ecrire un rapport `reference: "none"`
      sans statistiques inventees et conserver `verified: false` dans le YAML.
- [ ] Ajouter le test au `CMakeLists.txt` avec le label `premia` ou `quantlib`.
- [ ] Ajouter aussi le label du modele et un timeout adapte au backend; les
      modules directs sous `validation/premia/` et `validation/quantlib/` sont
      des outils de diagnostic, pas des commandes de publication.
- [ ] Executer le validateur isole, puis la suite CTest complete.

Le generateur ecrit d'abord un bloc obligatoire non verifie:

```yaml
validation:
  status: "pending"
  verified: false
  reference: "none"
  notebook: "catalog/price/<asset_class>/<model>/[<curve>/]<product>/<dataset_id>/validation.ipynb"
```

Le validateur synchronise ensuite ce bloc depuis le rapport reel. Une reference
fusionne son moteur et sa methode, par exemple `Premia (specialized pricer)` ou
`QuantLib (Monte Carlo)`; `mixed` est reserve a des regimes couverts par des
references principales differentes. Premia reste prioritaire lorsqu'il produit
un prix comparable, meme si QuantLib est aussi disponible.

Le champ `notebook` est toujours place apres `reference` et pointe vers le
notebook compile adjacent au YAML. Il est calcule par le validateur; il n'est
pas redige manuellement.

La comparaison d'une fonction CUDA avec une reimplementation des memes formules
dans le projet n'est pas une validation independante et ne remplace ni Premia
ni QuantLib.

## Mise a jour obligatoire du site

Le site est un projet local separe et ignore par ce depot. Sa mise a jour est
obligatoire pour publier le catalogue, mais elle est commitee et livree depuis
son propre projet: elle ne doit pas etre ajoutee a la PR de la bibliotheque.

- [ ] Ajouter le modele, la courbe, le produit et la base de prix dans
      `AI_factory_website/static/catalog-data.js`.
- [ ] Respecter la navigation: classe d'actifs, modele, courbe si necessaire,
      produit, puis dataset.
- [ ] Afficher le nombre de lignes, la methode de pricing et les liens essentiels.
- [ ] Ajouter le lien de telechargement du dataset.
- [ ] Ajouter le lien `Code on GitHub` vers le dossier contenant YAML et generateur.
- [ ] Ajouter le lien vers la documentation du code.
- [ ] Documenter les formules, le nom des fonctions CUDA et leurs fichiers.
- [ ] Ajouter ou mettre a jour l'equation TeX du modele, de la courbe ou du payoff.
- [ ] Regenerer les images avec `AI_factory_website/equations/build.sh`.
- [ ] Verifier les cartes, les bandeaux, les liens et la navigation sur desktop et
      mobile.
- [ ] Mettre a jour les compteurs et listes de modeles, courbes et produits.

L'ajout au code ou au catalogue sans ajout correspondant au site est incomplet.

## Verification finale

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
ctest --test-dir build --output-on-failure
git diff --check
git status --short
```

- [ ] Executer tous les generateurs ajoutes.
- [ ] Recharger tous les JSON generes avec les loaders de production.
- [ ] Executer les tests CUDA sur le GPU cible.
- [ ] Executer toutes les validations Premia et QuantLib concernees.
- [ ] Executer la suite CTest complete sans echec.
- [ ] Comparer les performances au cas existant le plus proche.
- [ ] Verifier qu'aucun dataset volumineux, build, site local ou note interne
      n'entre dans le commit.
- [ ] Mettre a jour le README public si l'architecture visible a change.

## Definition de termine

Une extension peut etre consideree terminee uniquement si:

- [ ] le code est uniforme, lisible, compile et teste;
- [ ] les datasets sont generes, documentes et valides au chargement;
- [ ] chaque YAML indique explicitement le meilleur validateur qui a passe, ou
      `none` lorsqu'aucune reference independante fiable n'existe;
- [ ] les erreurs et le biais sont dans les tolerances justifiees;
- [ ] le catalogue et le site exposent la nouvelle extension;
- [ ] les equations, liens, documentation et images du site sont verifies;
- [ ] la suite CTest complete passe;
- [ ] le commit ne contient que les sources et catalogues destines a GitHub.
