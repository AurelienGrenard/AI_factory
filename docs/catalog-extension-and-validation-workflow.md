# Workflow d'extension et de validation du catalogue

Ce document est la checklist de référence pour toute extension du catalogue.
Une extension n'est terminée que lorsque le code, les datasets, les tests, la
validation indépendante et le site ont tous été mis à jour.

## Regles non negociables

- [ ] Reprendre l'ossature de l'exemple existant le plus proche.
- [ ] Conserver les memes noms, signatures, ordre des fonctions et commentaires
      lorsque la semantique est identique.
- [ ] Ne mettre dans `src/` que le chargement, les mathematiques, le pricing et
      les bibliotheques generatives reutilisables; les recettes restent dans
      `catalog/`.
- [ ] Mettre les helpers de generation et de validation de datasets dans
      `tools/datasets/`.
- [ ] Generer les JSON et YAML avec le code; ne pas les corriger a la main.
- [ ] Terminer chaque generateur de modele, courbe, produit ou prix par le
      validateur de structure correspondant au fichier qu'il vient d'ecrire.
- [ ] Faire preceder les validations metier de chaque `load_*` par la validation
      de l'ossature JSON commune a sa famille de dataset.
- [ ] Ajouter chaque nouveau modele, courbe, produit et prix au site.
- [ ] Inventorier d'abord toutes les methodes Premia du couple `(modele,
      produit)` dans tous les menus Premia; ne selectionner un moteur qu'apres
      cet inventaire exhaustif.
- [ ] Ordonner les moteurs Premia compatibles par robustesse et performance
      mesuree, puis appliquer: liste Premia complete, pricer specialise
      QuantLib, Monte-Carlo QuantLib, enfin `none`.
- [ ] Decider la disponibilite de Premia sur l'existence d'au moins un moteur
      compatible, jamais sur la methode numerique employee par le generateur
      AI_factory.
- [ ] Garder le YAML de validation minimal: il pointe vers le dataset de
      reference; les details backend restent dans le JSON.
- [ ] Ne jamais activer CUDA fast math (`--use_fast_math`). Le projet exige des
      résultats reproductibles et n'expose volontairement aucune option de
      compilation correspondante.
- [ ] Ne pas ajouter `__launch_bounds__` sans une conception explicitement
      approuvée, adaptée aux architectures CUDA ciblées, puis validée par les
      registres, les spills, l'occupation et les temps mesurés sur chaque cible.
- [ ] Préserver le mapping déterministe des lignes et des chemins, l'ordre des
      réductions, ainsi que les accumulations FP64 déjà utilisées.

Un statut de reference `not_available` est un etat explicite, pas une
validation. Il interdit `verified: true` lorsque Premia et QuantLib ne
fournissent aucune reference comparable. La selection s'effectue ligne par
ligne sur les regimes core et stress selon
[`independent-price-validation-pipeline.md`](independent-price-validation-pipeline.md).
Un echec technique Premia autorise un repli documente pour les seules lignes
concernees, d'abord vers les autres moteurs Premia compatibles et seulement
ensuite vers QuantLib. Une divergence apres un calcul Premia reussi reste un
echec et ne doit jamais etre masquee par le choix retrospectif d'une reference
plus proche.

## Identifier la nature de l'extension

Avant de creer un dossier, separer les couches reellement nouvelles:

- une famille de parametres produit appartient a `src/product/` et a
  `catalog/product/`;
- une dynamique ou des analytiques appartiennent a
  `src/model/<asset_class>/<model>/`; ses recettes appartiennent a
  `catalog/model/<asset_class>/<model>/parameters/` ou `samples/`;
- une courbe appartient a `src/curve/` et a `catalog/curve/`;
- le pricing d'un couple modele-produit appartient au modele, dans
  `src/model/<asset_class>/<model>/[<curve>/]`;
- une base de prix conserve son propre generateur et YAML sous
  `catalog/model/<asset_class>/<model>/prices/`; un dataset migre ne stocke ni
  rapport ni notebook dans ce dossier, ses references vivent sous
  `validation/datasets/price/`;
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
- [ ] Stocker les dates contractuelles comme des jours ouvrés entiers sous la
      convention globale `days_per_year: 252`; employer une grille numérique
      distincte, actuellement `dt = 1 / 504`, seulement lorsqu'un schéma ou un
      monitoring fin est effectivement nécessaire.
- [ ] Choisir la methode: formule exacte, Monte-Carlo, Longstaff-Schwartz, etc.
- [ ] Enumerer toutes les methodes Premia du couple, dans tous les menus/classes
      d'actifs, et relever pour chacune le nom natif exact (`CF_*`, `AP_*`,
      `FD_*`, `TR_*`, `MC_*`), son domaine et ses conventions.
- [ ] Chercher le contrat direct et les reductions exactes composees de moteurs
      Premia, meme si le moteur est continu, PDE ou Monte Carlo alors que le
      prix CUDA utilise une approximation discrete; documenter ensuite l'ecart
      de contrat.
- [ ] Sonder les candidats sur des lignes representatives du `core`,
      puis choisir le moteur principal le plus rapide parmi ceux qui sont
      compatibles et robustes; conserver les autres comme replis Premia.

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
- [ ] Creer
      `catalog/model/<asset_class>/<model>/parameters/<dataset_id>/generator.cpp`.
- [ ] Creer
      `catalog/model/<asset_class>/<model>/parameters/<dataset_id>/dataset.yaml`
      par le generateur.
- [ ] Utiliser des plages financieres raisonnables.
- [ ] Placer les cas extremes en queue de distribution, pas au centre.
- [ ] Rejeter les lignes mathematiquement ou numeriquement invalides.
- [ ] Verifier `database_id`, `model_family`, `catalog`, `url`, `row_count` et
      `models` dans le JSON genere.
- [ ] Recharger le JSON genere avec `load_models(...)` avant de terminer.

### Dataset de samples du modele

- [ ] Creer
      `catalog/model/<asset_class>/<model>/samples/<dataset_id>/generator.cpp`.
- [ ] Generer le JSON complet sous
      `datasets/model/<asset_class>/<model>/samples/<dataset_id>.json`.
- [ ] Produire le YAML adjacent exclusivement depuis le generateur.
- [ ] Considerer le generateur comme source de verite: le YAML documente la
      recette executee et ne sert jamais d'entree au generateur.
- [ ] Generer les parametres plausibles directement dans un vecteur type
      contigu avec un flux Philox par ligne; ne pas ecrire puis recharger un
      dataset de parametres intermediaire.
- [ ] Reprendre uniquement le regime core de 90% du dataset de parametres de
      pricing, sans sa queue stress de 10%.
- [ ] Faire appeler au generateur la dynamique de reference placee sous
      `src/model/<asset_class>/<model>/`; ne pas reimplementer le modele dans la
      recette.
- [ ] Produire exactement 3 000 000 de samples d'entrainement.
- [ ] Fournir les deux recettes par modele: `samples_01` avec
      `12 000 * 250 = 3 000 000`, puis `samples_02` avec
      `3 000 000 * 1 = 3 000 000` et des seeds independantes.
- [ ] Tirer independamment chaque maturite selon la loi uniforme discrete sur
      `{90/360, ..., 720/360}` et utiliser `target_dt = 1 / 360` pour les
      schemas discretises.
- [ ] Declarer les parametres, la loi de `T`, les observables, la methode
      numerique et les trois seeds dans le YAML.
- [ ] Ecrire une liste plate de 3M lignes autonomes contenant chacune les
      parametres, `maturity_days`, `T` et les valeurs terminales; accepter la
      repetition des parametres dans les paquets de 250 de `samples_01`.
- [ ] Ajouter un test `--smoke-test` de 1 000 samples, avec relecture du JSON,
      controle des dimensions et rejet de toute valeur non finie.

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
- [ ] Conserver deux dossiers sous
      `catalog/model/<asset_class>/<model>/prices/`: les prix call et put sont
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

- [ ] Creer
      `catalog/model/<asset_class>/<model>/prices/[<curve>/]<product>/<dataset_id>/`.
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

### Hierarchie de validation obligatoire

- [ ] Separer explicitement les 900 lignes `core` des 100 lignes `stress` et
      exiger une reference independante sur les deux regimes.
- [ ] Inventorier exhaustivement tous les moteurs Premia compatibles avec le
      modele et le produit, dans tous les menus Premia, avant d'en choisir un.
- [ ] Inclure les contrats directs et les reductions exactes fondees sur des
      moteurs Premia; une difference discret/continu n'est pas une absence de
      moteur.
- [ ] Mesurer les candidats sur un echantillon representatif du `core`,
      puis les ordonner par compatibilite, robustesse et vitesse.
- [ ] Ne declarer Premia indisponible que si aucun moteur compatible n'existe;
      pour une ligne, n'autoriser la sortie de Premia qu'apres l'echec technique
      de tous les moteurs Premia declares.
- [ ] A defaut, chercher un pricer specialise QuantLib.
- [ ] A defaut, construire une simulation Monte-Carlo QuantLib independante.
- [ ] Si aucun backend fiable n'existe, conserver la reference
      `not_available` et interdire la publication avec `verified: true`.
- [ ] Ajouter le validateur unifie sous
      `validation/model/<asset_class>/<model>/[<curve>/]<product>.py`, avec la
      meme ossature que les produits voisins; omettre la courbe en equity.
- [ ] Reutiliser le `reference_pipeline.py` de la classe d'actifs et conserver
      les memes fonctions, leur ordre et leurs signatures que le modele voisin.
- [ ] Exposer `python -m <module> DATASET REFERENCE_DATASET`;
      la commande est cache-only par defaut et `--generate` est la seule voie
      qui relance un backend externe.
- [ ] Declarer la liste ordonnee complete des moteurs Premia, puis les
      emplacements QuantLib specialise et QuantLib Monte Carlo, avec un
      adaptateur ou une raison d'indisponibilite explicite.
- [ ] Laisser `validation/hierarchy.py` transmettre au moteur suivant les seules
      exceptions techniques ligne par ligne; ne jamais y envoyer une divergence.
- [ ] Conserver les adaptateurs de backend reutilisables sous
      `validation/premia/` et `validation/quantlib/`.
- [ ] Mettre les conversions reutilisables dans un fichier commun au modele.
- [ ] Pour un modele equity stochastique, reutiliser
      `validation/model/equity/stochastic_equity.py`: le fichier du modele
      declare seulement produits, moteurs et noms natifs; les fichiers produits
      restent des wrappers CLI minces.
- [ ] Lire le JSON de prix produit par le vrai generateur CUDA.
- [ ] Reconstruire chaque ligne dans le backend avec les memes conventions.
- [ ] Comparer les 900 lignes core et les 100 lignes stress, pas seulement un
      echantillon favorable.
- [ ] En cas d'echec technique du moteur Premia principal, conserver ligne,
      statut et raison, puis essayer successivement chaque autre moteur Premia;
      appliquer QuantLib uniquement si tous ont techniquement echoue.
- [ ] Ne jamais basculer vers QuantLib lorsque Premia a calcule un prix fini et
      comparable qui diverge: enregistrer une `comparison failure` et corriger
      la cause.
- [ ] Rejeter comme exception technique toute sortie pourtant finie qui viole
      une borne de non-arbitrage; conserver le diagnostic avant le repli.
- [ ] Controler erreur absolue, erreur relative, erreur maximale et taux d'echec.
- [ ] Controler le biais signe moyen pour detecter une erreur systematique.
- [ ] Lorsqu'une relation mathematique continu/discret justifie le biais,
      conserver `systematic_bias: true`, ajouter une explication non vide et
      verifier la borne ligne par ligne; ne jamais masquer un biais inexplique.
- [ ] Expliquer les tolerances par la precision FP32 ou la statistique Monte-Carlo;
      ne pas les elargir uniquement pour faire passer le test.
- [ ] Ecrire les 1 000 prix sous `validation/datasets/price`, avec empreintes
      semantiques,
      provenance `reference_pricer_id`, `row_priced`, version du backend
      utilisee et verification core/stress.
- [ ] Supprimer tout `validation_report.json` ou
      `validation.ipynb` adjacent au YAML et ne publier dans celui-ci que
      `status`, `verified` et `dataset`.
- [ ] Generer le cache et le bloc YAML exclusivement depuis l'execution du
      generateur de references; ne jamais rediger les resultats a la main.
- [ ] Ajouter au `CMakeLists.txt` un test court portant le label
      `cached_reference`; les modules Premia et QuantLib directs restent des
      outils de regeneration et de diagnostic.
- [ ] Ajouter un test qui bloque les imports Premia/QuantLib sur le chemin
      cache-only.
- [ ] Executer le validateur isole, puis la suite CTest complete.

Le YAML publie uniquement:

```yaml
validation:
  status: "available"
  verified: true
  dataset: "validation/datasets/price/<asset_class>/.../<database_id>.json"
```

Le YAML ne repete aucun moteur. Le JSON sous `validation/datasets` contient les
trois emplacements ordonnes `premia`, `quantlib_specialized` et
`quantlib_monte_carlo`. Seules les methodes effectivement utilisees portent un
identifiant, une version, un nom natif et `row_priced`; une methode disponible
mais inutilisee ne contient que son statut.

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
- [ ] chaque rapport nomme la fonction ou methode de pricing exacte pour tous
      les moteurs disponibles et tous les replis executes;
- [ ] les erreurs et le biais du core sont dans les tolerances justifiees;
- [ ] le catalogue et le site exposent la nouvelle extension;
- [ ] les equations, liens, documentation et images du site sont verifies;
- [ ] la suite CTest complete passe;
- [ ] le commit ne contient que les sources et catalogues destines a GitHub.
