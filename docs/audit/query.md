# Referentiel des audits d'architecture C++/CUDA

Version du referentiel : 7, 2026-08-30.

## Objet

Ce document contient la liste stable des audits a effectuer sur l'architecture
C++/CUDA du projet. Il formule les questions a poser et les invariants a
verifier ; il ne contient aucun resultat d'audit.

`status.md` conserve la date, la revision, le perimetre, les exclusions et les
preuves du dernier passage de chaque audit, qu'il ait produit ou non un
constat. Une section vide de `response.md` ne doit donc jamais servir a
deduire qu'un audit est recent ou complet.

La provenance doit identifier a la fois la version de ce referentiel et le
contenu reellement audite. Sur un worktree propre, la revision Git suffit. Sur
un worktree modifie, `status.md` conserve aussi l'etat porcelain, un manifeste
des fichiers non suivis et une empreinte du diff suivi ; si le contenu non
suivi entre dans le perimetre, son manifeste doit egalement porter une
empreinte de contenu. Le passage initial et l'etat apres remediation sont deux
snapshots distincts. Lorsqu'un snapshot exact n'est pas reconstructible, le
statut le dit explicitement et aucune revision seule ne doit etre presentee
comme sa source complete.

Tous les constats non resolus sont inscrits dans `response.md`, y compris ceux
dont le traitement est explicitement reporte. Toute section qui contient un
constat doit reprendre le meme nom et le meme ordre que dans ce document.
Chaque constat conserve un identifiant stable, un etat courant, une preuve
dans le code ou les mesures et un critere de cloture.

`closed.md` est le registre compact de tous les constats corriges, refutes par
mesure, fusionnes ou devenus inapplicables. Avant de creer un identifiant,
l'auditeur doit rechercher dans `response.md` et `closed.md` une signature
equivalente. L'identite d'un constat est definie par sa cause et son perimetre,
pas par la correction envisagee ni par la formulation du titre.

Le passage d'une version du referentiel a une autre ne reouvre pas les constats
fermes et ne change jamais leur identifiant. Si une signature reapparait sous
une nouvelle section, reutiliser l'identifiant historique. Une
reclassification documentaire n'est ni une fermeture ni un nouveau probleme.

Lorsqu'un constat est clos, transferer toute decision durable vers le contrat
d'implementation concerne, puis le deplacer de `response.md` vers `closed.md`.
Son entree fermee conserve un titre explicite, la nature de la cloture, la
signature initiale, la resolution ou decision, la preuve et la condition de
reouverture. Les rapports volumineux ne sont pas recopies si une preuve durable
plus precise peut etre liee.

Si le meme probleme reapparait, reutiliser son identifiant et redeplacer
l'entree vers `response.md`, en conservant la cloture precedente et la raison
de la reouverture. Un nouvel identifiant n'est permis que si la cause ou le
perimetre differe reellement. Un identifiant possede exactement un etat
courant : non resolu dans `response.md` ou ferme dans `closed.md`. `status.md`
conserve la comptabilite et les preuves, sans devenir un troisieme registre.

L'audit transversal des references independantes, caches, notebooks,
fingerprints et artefacts publies reste separe dans `../validation/query.md`.
Il n'est pas execute pendant un passage ordinaire de ce referentiel. Les tests
cibles necessaires aux audits Homogeneity, Numerical robustness, CUDA safety,
CMake ou Performance restent dans leur perimetre.

`validation/**` et `docs/validation/**` sont hors perimetre de tous les axes du
present referentiel, y compris Structure, Naming et Code generation. Ils
peuvent etre lus ou executes comme consommateurs, preuves de non-regression ou
outils de comparaison, mais leur arborescence, leur nommage, leurs abstractions,
leurs moteurs, leurs caches et leur reproductibilite ne produisent jamais un
constat dans `docs/audit/response.md`. Tout probleme dont la cause primaire vit
dans ces arbres appartient exclusivement a `../validation/query.md`. Une
remediation principale peut devoir adapter un consommateur sous `validation/**`
pour conserver le build ou le contrat public ; cela ne transfere pas son audit
interne dans le present referentiel.

Les composants qui produisent une decision du present audit ne sont pas des
outils de validation externe. Les sources de benchmarks et leurs fixtures,
manifestes et baselines principales sont possedes par `tests/performance` ;
les runners, checkers, outils de profilage et procedures de retuning le sont
par `tools/performance`. Leur protocole durable vit sous `docs`. Un composant
autoritatif de gate Performance place uniquement sous `validation/**` ne peut
pas servir de preuve principale : l'audit ouvre alors un constat Structure sur
l'absence ou le mauvais ownership de l'infrastructure principale, sans auditer
pour autant l'implementation interne de la validation externe. Une sortie de
`validation/**` peut rester une preuve complementaire, jamais l'unique source
d'une decision du present referentiel.

Un audit distingue explicitement :

- les incoherences prouvees par le code ou les artefacts compiles ;
- les risques qui necessitent une mesure pour etre confirmes ;
- les choix satisfaisants qu'une correction doit preserver ;
- les exceptions mathematiques necessaires et leur justification.

Un constat transversal est inscrit une seule fois sous la section qui porte sa
cause primaire. Les autres audits sont references depuis ce constat. Un choix
satisfaisant n'est pas inscrit dans `response.md` : s'il exprime une decision
durable, il est documente dans le contrat d'implementation proprietaire. Une
exception necessaire suit la meme regle.

L'auditeur part du code, des dependances CMake, des tests et, pour la
performance, des artefacts compiles. L'arborescence et les noms sont toutefois
une interface de navigation a part entiere : ils doivent permettre de predire
le role d'un fichier avant de l'ouvrir. Le commentaire d'en-tete confirme et
precise ce role; il ne peut jamais compenser un mauvais placement ou un nom
ambigu. Noms, dossiers et commentaires ne constituent pas seuls une preuve de
responsabilite ou de factorisation. L'auditeur ne propose aucune abstraction
sans identifier ses consommateurs reels.

Pour chaque audit, renseigner d'abord dans `status.md` la matrice de couverture
effectivement inspectee. Chaque constat indique ensuite :

- son **etat** : ouvert ou reporte par decision explicite ;
- sa **severite** : critique, haute, moyenne ou faible ;
- sa **priorite** : haute, moyenne ou basse ;
- sa **confiance** : prouvee, forte ou a mesurer ;
- fichiers et symboles, preuve, impact, correction minimale et test ou mesure
  permettant la cloture ;
- references de preuves, date de derniere verification et proprietaire ou
  mention `non attribue`.

Un constat reporte indique date et auteur de la decision, motif et evenement
de reexamen. Un report sans condition de reprise n'est pas durable.

Une preuve issue d'un binaire, profil ou benchmark est attachee au snapshot, a
la configuration et a l'artefact exacts. Elle est marquee `a rafraichir` des
qu'un changement peut modifier cet artefact. Une decision chiffree doit rester
reproductible par commande, environnement, manifeste de workloads, sortie
conservee ou artefact versionne.

Le statut `complet` signifie que toutes les preuves attendues de la section ont
ete couvertes sur le perimetre declare. Des tests cibles d'une remediation ne
transforment pas retroactivement un audit partiel en audit complet.

Chaque section possede en outre une **condition de completude**. Si une case
obligatoire est absente, non executee ou inconnue, le passage reste `partiel`
pour cette section, meme si aucun defaut n'a ete trouve. Une preuve manquante
est une exclusion de couverture, pas automatiquement un constat. Inversement,
une exclusion ne permet jamais de conclure positivement sur la case omise.

Lorsque deux constats partagent fichiers, mesures ou migration, ajouter un
champ **Coordination** citant les identifiants et precisant si le traitement
commun est obligatoire, recommande ou ordonne. Cela ne fusionne pas deux
causes ni leurs criteres de cloture.

Une opportunite non prouvee reste une hypothese `a mesurer`. Sa cloture
consiste d'abord a accepter ou rejeter l'hypothese, pas a imposer une
implementation.

## Principes de classement

Les trois questions suivantes sont separees :

- **Structure** : ou vit un composant, qui le possede et comment il se nomme ;
- **Homogeneity** : deux composants qui jouent le meme role respectent-ils le
  meme contrat ;
- **Factorization** : des composants qui partagent les memes invariants
  reutilisent-ils une implementation au bon niveau.

Une difference de nom ou de chemin appartient a Structure. Une difference de
signature entre dynamics homologues appartient a Homogeneity. Une boucle de
simulation recopiee appartient a Factorization. Le cout chiffre de
l'abstraction est ensuite mesure par Performance et CMake.

## Gates P0 obligatoires

Avant les checklists detaillees, `status.md` reproduit cette table compacte en
ajoutant pour chaque gate applicable un etat `pass`, `partial` ou `excluded` et
une preuve durable. Un gate applicable `partial`, `excluded` sans justification
contractuelle ou sans preuve maintient les sections concernees partielles.

| Gate | Condition obligatoire |
|---|---|
| `P0-SNAPSHOT` | Revision, dirty worktree, fichiers non suivis, empreintes, build, toolchain et artefacts mesures identifient un snapshot exact. |
| `P0-INVENTORY` | Un inventaire canonique exhaustif precede l'echantillonnage et chaque element du perimetre possede un etat. |
| `P0-SCOPE` | Chaque preuve et chaque outil autoritatif ont un proprietaire dans le perimetre principal ; `validation/**` ne porte aucun gate principal. |
| `P0-COVERAGE` | Une valeur requise absente, inconnue, stale, indisponible ou non executee echoue fermement la case au lieu d'etre assimilee a un succes. |
| `P0-IDENTITY` | `response.md` et `closed.md` ont ete recherches par cause et perimetre avant toute creation ou reouverture d'identifiant. |
| `P0-NUMERICS` | Domaines, references et budgets numeriques sont fixes independamment du resultat candidat et couvrent core, stress et limites. |
| `P0-PERF-STATS` | Protocole, campagnes completes, agregation et exclusions sont predeclares ; aucun best-of-N ni assemblage opportuniste n'est admis. |
| `P0-REBASELINE` | Toute nouvelle baseline conserve l'ancienne, explique chaque delta et ne transforme pas sa propre sortie en preuve de conformite. |
| `P0-EVIDENCE` | Toute conclusion renvoie a une commande, un artefact, un environnement et une sortie conserves et mutuellement compatibles. |

Ces gates ne remplacent aucune condition de completude plus stricte. Ils sont
le minimum lisible et automatisable qui empeche une conclusion positive
lorsqu'une preuve critique manque.

## Sommaire

- [Gates P0 obligatoires](#gates-p0-obligatoires)
- [Structure, conventions and ownership](#structure-conventions-and-ownership)
  - [Repository tree and domain ownership](#repository-tree-and-domain-ownership)
  - [File responsibilities and granularity](#file-responsibilities-and-granularity)
  - [Naming and semantic conventions](#naming-and-semantic-conventions)
  - [Dependency boundaries](#dependency-boundaries)
  - [Cleanup and extension locality](#cleanup-and-extension-locality)
  - [Navigation exercises and completion gate](#navigation-exercises-and-completion-gate)
- [Contract homogeneity](#contract-homogeneity)
  - [Dynamics](#dynamics)
  - [Analytics](#analytics)
  - [Products, policies and exercise](#products-policies-and-exercise)
- [Factorization pyramid](#factorization-pyramid)
  - [Markovian factorization](#markovian-factorization)
  - [Rough factorization](#rough-factorization)
  - [Rough-Markovian factorization](#rough-markovian-factorization)
  - [Closed-form factorization](#closed-form-factorization)
  - [Fixed-income and equity factorization](#fixed-income-and-equity-factorization)
  - [Factorization cost and limits](#factorization-cost-and-limits)
- [Code generation and extension cost](#code-generation-and-extension-cost)
  - [Minimum hand-written model and product](#minimum-hand-written-model-and-product)
  - [Canonical capability manifest](#canonical-capability-manifest)
  - [Generated bindings and catalogue recipes](#generated-bindings-and-catalogue-recipes)
  - [Model-sample bindings and recipes](#model-sample-bindings-and-recipes)
  - [Parameter dataset generation](#parameter-dataset-generation)
  - [Regeneration, drift and exceptions](#regeneration-drift-and-exceptions)
- [CMake and build graph](#cmake-and-build-graph)
- [Portability and hardware tuning](#portability-and-hardware-tuning)
  - [Supported architecture matrix](#supported-architecture-matrix)
  - [Portable defaults and retuning](#portable-defaults-and-retuning)
  - [Cross-architecture evidence](#cross-architecture-evidence)
- [Numerical robustness and reproducibility](#numerical-robustness-and-reproducibility)
  - [Precision and mathematical domains](#precision-and-mathematical-domains)
  - [Solvers and linear algebra](#solvers-and-linear-algebra)
  - [Stochastic dynamics and convergence](#stochastic-dynamics-and-convergence)
  - [Determinism and random-number mapping](#determinism-and-random-number-mapping)
  - [Failure propagation and completion gate](#failure-propagation-and-completion-gate)
- [CUDA execution and memory safety](#cuda-execution-and-memory-safety)
- [Performance](#performance)
  - [Common performance protocol and kernel strategy](#common-performance-protocol-and-kernel-strategy)
  - [Generic CUDA performance](#generic-cuda-performance)
  - [Model-sample performance](#model-sample-performance)
  - [Early-exercise performance](#early-exercise-performance)
  - [Rough performance](#rough-performance)

## Structure, conventions and ownership

Auditer l'arborescence, les responsabilites, les noms et les frontieres de
dependances comme un contrat de lisibilite architecturale. Cette section dit
ou un composant doit vivre et comment il doit se nommer ; elle ne conclut pas
qu'une implementation doit etre partagee, ce qui releve de Factorization.

**Perimetre :** dossiers, fichiers, namespaces, types, fonctions, variables,
targets, identifiants de catalogue, includes et liens entre domaines sous
`src`, `tools`, `catalog`, les chemins logiques de `datasets`, `tests`, `cmake`,
les CMake racine et la documentation principale.

**Hors perimetre :** differences d'API entre composants homologues et
duplication d'implementation, sauf si elles prouvent un mauvais ownership.

**Preuves attendues :** arborescence complete par racine, graphes d'includes et
de liens, inventaire exhaustif des conventions et exceptions, consommateurs,
tailles des fichiers, exercices de navigation et nombre de lieux modifies par
une extension representative.

**Livrable :** carte des domaines et dependances, contradictions de nommage ou
placement, elements orphelins et localite mesuree d'un ajout modele/produit.
Le passage inventorie exhaustivement les chemins inspectes et leur classe
(`runtime-infrastructure`, `model-product-binding`, `offline-tool`,
`catalog-recipe`, `generated`, `test`, `build` ou `documentation`); un
echantillon de noms ne suffit pas a declarer cet axe complet.

La grammaire attendue des chemins est auditee explicitement :

| Racine | Responsabilite autorisee | Signal de mauvais ownership |
|---|---|---|
| `src/common` | Primitives runtime/CUDA/numeriques reutilisables | Modele, courbe, produit ou publication concret |
| `src/model/<asset_class>/<family?>/<model>` | Parametres, dataset, dynamics, analytics et sampling du modele | Payoff ou binding produit hors `product/` |
| `src/model/**/product/[<curve>/]<product>` | Composition mince modele-produit | Dynamics, analytics, moteur generique ou outil offline |
| `src/curve/<curve>` | Parametres, loader et analytics de courbe | Produit ou modele concret non requis par la courbe |
| `src/product/<asset_class?>/<product>` | Parametres et semantique propres au produit | Dynamics ou launcher lie a un modele concret |
| `src/generative` | Code generatif independant d'une methode financiere | Copie d'un moteur pricing/sampling existant |
| `tools/<responsibility>` | Codegen, construction de datasets, orchestration et publication offline | Formule runtime canonique ou dependance circulaire vers `catalog` |
| `catalog/<domain>/.../<dataset_id>` | Recette et metadonnees d'un dataset | Infrastructure partagee ou variante implicite |
| `datasets/<domain>/...` | Artefacts locaux suivant la meme taxonomie d'entites que `src` et `catalog` | Hierarchie propre, famille omise ou renommage d'entite |
| `tests/<domain-or-contract>` | Preuve nommee du proprietaire ou contrat teste | Seconde implementation de production ou fichier fourre-tout |
| `cmake` et CMake racine | Definition declarative du graphe de build | Inventaire concurrent de capacites metier |
| `docs` hors `docs/validation` | Contrats, workflows et registres principaux | Description obsolete d'une implementation supprimee |

Une racine peut conserver quelques primitives canoniques directement a son
niveau, mais la coexistence d'un fichier et d'un dossier homonymes, une liste
croissante de primitives sans domaine, ou un niveau `common/core/utils` sans
frontiere testable doit etre justifie fichier par fichier.

`src` est l'unique taxonomie canonique des courbes, modeles et produits.
`catalog` et `datasets` en heritent ; ils n'en definissent pas une seconde. Pour
chaque entite, le prefixe relatif est identique dans les trois arbres :

```text
src/model/<asset_class>/<family?>/<model>
catalog/model/<asset_class>/<family?>/<model>/<dataset_kind>/<dataset_id>
datasets/model/<asset_class>/<family?>/<model>/<dataset_kind>/<dataset_id>.json

src/curve/<curve>
catalog/curve/<curve>/<dataset_id>
datasets/curve/<curve>/<dataset_id>.json

src/product/<relative_product_path>/<product>
catalog/product/<relative_product_path>/<product>/<dataset_id>
datasets/product/<relative_product_path>/<product>/<dataset_id>.json
```

Les suffixes propres aux datasets (`parameters`, `samples`, `prices`, courbe,
produit, variante et `dataset_id`) commencent seulement apres ce prefixe. Si
`src` distingue `equity/rough`, `equity/markovian`, `fixed_income` ou une
future famille, `catalog` et `datasets` reproduisent exactement cette decision
au lieu d'omettre le niveau ou de la reencoder dans un nom. Singularisation,
pluralisation, alias historique et chemin special dans un renderer ne doivent
pas changer l'identite de l'entite. Une exception exige un mapping declaratif,
un motif durable et un test ; la convention implicite n'est pas une exception.

### Repository tree and domain ownership

Verifier notamment :

- que `src/common`, `src/model`, `src/curve`, `src/product`, `src/generative`,
  `tools`, `catalog`, `datasets`, `tests`, `cmake`, les CMake racine, `docs` hors
  `docs/validation` et la frontiere du site ne portent que leurs
  responsabilites declarees ;
- que `src/model/equity/markovian`, `src/model/equity/rough` et
  `src/model/fixed_income` classent les modeles par famille mathematique et non
  par commodite du moteur actuellement employe ;
- que chaque niveau de dossier represente un axe durable et nomme : domaine,
  famille mathematique, modele, responsabilite puis variante eventuelle; un
  niveau cree seulement pour contourner un nom ambigu, un dossier fourre-tout
  ou une exception historique est un constat ;
- que les dossiers modele restent centres sur parametres, datasets, dynamics,
  analytics, sampling et primitives propres; chaque unite de composition
  modele-produit vit exclusivement sous le sous-dossier `product/`, qui ne
  contient reciproquement aucune infrastructure de modele ;
- que le dossier `product/` est lui-meme plat pour une composition directe et
  n'ajoute qu'un niveau de courbe explicitement nomme lorsqu'une composition
  fixed-income la possede; produit, infrastructure et strategie de pricing ne
  se melangent jamais au meme niveau ;
- que contrats produit, schedules, payoffs, pricing policies et continuation
  states restent sous `src/product` lorsqu'ils ne dependent pas d'un modele ;
- que les primitives mathematiques ou d'execution reellement neutres vivent
  sous `src/common`, sans type concret de modele, courbe ou produit, et que ses
  sous-dossiers nomment une responsabilite stable plutot qu'une accumulation
  historique de headers a la racine ;
- que la racine de `src/common` ne conserve que les primitives veritablement
  transversales ; des familles telles que simulation, Volterra, Monte-Carlo,
  sample, payoff, closed form ou fixed income ont chacune un ownership nomme,
  sans fichier homonyme partage entre racine et sous-dossier ;
- que chaque `src/product/<product>` expose une ossature predicible
  (`parameters`, `dataset`, `schedule`, `pricing_policy`, continuation si
  necessaire) et que tout fichier supplementaire nomme sa phase ou son
  invariant au lieu de `core`, `impl`, `helper` ou `common` non qualifie ;
- que `tools` possede sampling offline, orchestration, serialisation,
  publication et codegen, tandis que `catalog` contient les recettes et
  metadonnees reproductibles ;
- que `tools/codegen`, `tools/datasets`, outils CUDA, runners, checkers et
  profils de tuning sont ranges par responsabilite et consommateur ; un meme
  dossier ne melange pas templates, execution de pricing, controle
  d'architecture et donnees de tuning sans sous-frontiere explicite ;
- que chaque `ModelSpec`, `ProductSpec` et courbe possede un prefixe canonique
  derive du chemin `src`, reutilise sans seconde table par `catalog`,
  `datasets`, YAML, generateurs et documentation ;
- que l'audit compare la bijection des prefixes reellement suivis dans
  `catalog`, des chemins `datasets` references par les recettes/YAML et, lorsque
  les artefacts locaux sont disponibles, de leur arborescence effective ;
- que `validation/**`, lu seulement comme consommateur externe eventuel, ne
  devient pas une dependance du runtime ou des generateurs ordinaires ;
- que la structure des modeles markoviens, rough, fixed income, courbes et
  produits est symetrique pour les responsabilites communes ;
- que toute difference d'arborescence est justifiee par mathematique,
  dependance optionnelle ou frontiere de compilation, et non par l'historique ;
- que sampling modele, bindings `sample.cuh`/`sample.cu`, orchestration offline,
  recettes `samples_01`/`samples_02` et artefacts publies restent dans leurs
  domaines respectifs avec une symetrie explicite entre familles d'engine.

### File responsibilities and granularity

Verifier notamment :

- que chaque fichier porte une responsabilite identifiable et des
  consommateurs reels ;
- que **chaque** fichier d'implementation ou d'automatisation ecrit a la main
  sous `src`, `tools`, `tests` et `cmake`, hors bindings generes sous
  `src/model/**/product/**`, commence par une phrase courte, specifique et
  autonome nommant son contenu **et son utilite**, avant `#pragma once`, le
  premier include ou le code du module ; employer un commentaire C++/CMake ou
  une docstring Python selon le langage ;
- qu'un nom canonique comme `dynamics.cuh` reste stable mais que son en-tete
  distingue transition exacte, schema a pas, Volterra FFT ou lift N-facteurs;
  `dynamics.cuh` annonce le contrat public et `dynamics_impl.cuh` les
  definitions device, jamais deux descriptions interchangeables ;
- que l'en-tete n'est ni un marqueur generique (`generated`, `utilities`,
  `implementation`) ni la simple repetition du nom : il nomme au minimum
  modele ou domaine, responsabilite et consommateur/phase d'utilisation ;
- que les exceptions a cette regle sont limitees aux outputs marques comme
  generes, aux fichiers de donnees/manifeste declaratifs, aux recettes de
  catalogue volontairement minimales et aux fichiers dont le format impose un
  preambule ; chaque exception est classee, pas devinee ;
- que les headers publics exposent seulement les declarations necessaires,
  les definitions device incluses vivent dans `*_impl.cuh` et les `.cu`
  autonomes sont de vraies unites de traduction ;
- qu'aucun `.cu` n'est inclus textuellement et qu'aucun `.cuh` ne simule une
  unite autonome sans frontiere explicite ;
- que les rows device restent compacts et independants des loaders,
  validateurs et objets JSON host ;
- qu'un fichier ne melange pas mathematique runtime, generation de parametres,
  orchestration CUDA, assemblage de dataset et serialisation ;
- que les gros modules sont separes lorsque leurs parties evoluent et se
  testent independamment, sans micro-fichiers sans abstraction ;
- que les wrappers modele-produit generes ne masquent pas une logique manuelle,
  une exception ou une dependance non declaree ;
- que les bindings et recettes de sample ne melangent pas dynamics modele,
  generation de parametres, generation de calendriers, strategie CUDA,
  assemblage du dataset et serialisation ;
- que toute fusion ou scission CUDA mesure compilation, incrementalite, taille
  objets/cubins et ressources kernel.

### Naming and semantic conventions

Auditer dossiers, fichiers, namespaces, types, concepts, fonctions, arguments,
variables, constantes, tests, targets et identifiants de catalogue. Le nom
indique la responsabilite et le domaine reels, pas l'anciennete.

L'inventaire des noms de dossiers et fichiers est exhaustif. Il consigne pour
chaque chemin : role predit sans ouverture, role confirme par l'en-tete et les
consommateurs, convention appliquee, ambiguite et verdict. Les symboles publics
et targets sont eux aussi exhaustifs ; les symboles prives, arguments et
variables sont couverts par recherches systematiques des noms interdits/unites
manquantes puis revue des composants de chaque famille. Un simple echantillon
de fichiers bien nommes ne certifie jamais Naming.

Verifier notamment :

- `snake_case` pour chemins, fichiers, fonctions et variables, `PascalCase`
  pour types et concepts, et une convention unique pour les constantes ;
- que le chemin et le nom suffisent a repondre, sans ouvrir le fichier, a
  « quel domaine ? quel modele/produit ? quel role ? quelle strategie lorsque
  plusieurs existent ? »; si une de ces reponses manque, le nom est ambigu ;
- la correspondance arborescence/namespaces, sans namespace plat sous un
  dossier specialise ni niveau sans responsabilite ;
- singulier/pluriel coherent pour definition, famille, collection, dataset et
  variante de prix ;
- des noms de fichiers homologues stables : `parameters`, `dataset`,
  `dynamics`, `analytics`, `concepts`, `pricing_policy`, `schedule`, `state`,
  `sample`, `kernel`, `launch` et `workspace` ;
- que les tests nomment proprietaire, comportement et nature CPU/CUDA, et
  qu'un terme tel que `additional`, `misc`, `all`, `new` ou un numero d'ordre
  ne remplace jamais le contrat teste ;
- qu'un helper non canonique encode sa strategie reelle dans son nom : par
  exemple `volterra_fft_*` ou `markovian_n_factor_*`, jamais `hybrid`,
  `numerics` ou `pricing` seuls lorsque plusieurs engines coexistent ;
- l'absence de noms fourre-tout tels que `misc`, `utils`, `helpers`, `base`,
  `core`, `common`, `shared`, `generic`, `stuff` ou `tmp` lorsqu'ils ne sont
  pas qualifies par une responsabilite precise et une frontiere verifiable ;
- qu'un nom comme `state`, `kernel`, `workspace`, `preparation` ou `pricing`
  n'est accepte seul que s'il est canonique et sans ambiguite dans son dossier;
  des que plusieurs methodes ou phases coexistent, le nom porte leur
  qualificatif semantique ;
- que les symboles ne repetent pas modele, courbe, produit ou methode deja
  portes par leur namespace ;
- les noms canoniques `ModelParameters`, `ProcessParameters`, `PreparedModel`,
  `PreparedTransition`, `PreparedDynamics`, `State`, `DeviceInputs` et
  `PreparedRow`, avec exceptions documentees ;
- des verbes stables : `load`, `validate`, `prepare`, `compose`, `simulate`,
  `advance`, `observe`, `evaluate`, `price`, `launch`, `write`, `generate` ;
- les memes noms et ordre d'arguments entre declarations, definitions,
  wrappers, tests, templates et code genere ;
- les unites/domaines dans le nom : `_days`, `_years`, `_steps`, `_count`,
  `_index`, `_offset`, `_bytes`, paths, prix, taux et accruals ;
- que les booleens se lisent comme des predicats et qu'un enum remplace un
  booleen de mode ;
- que les symboles mathematiques courts restent dans les formules locales ;
- l'absence de `new`, `old`, `legacy`, `temporary` ou `historical` dans les
  noms permanents ;
- que tout renommage est atomique dans sources, tests, CMake, codegen,
  catalogue, diagnostics et documentation principale ; les consommateurs sous
  `validation/**` peuvent servir de preuve de compatibilite sans que leur
  nommage interne soit audite ici ;
- que les identifiants publies ne changent pas sans migration explicite.

Une exploration qui oblige a ouvrir plusieurs fichiers pour distinguer leurs
roles, ou a connaitre l'histoire du projet pour comprendre leur placement, est
un impact architectural et devient un constat. Seules les preferences
stylistiques sans effet sur navigation, ownership, extension ou maintenance
restent exclues.

### Dependency boundaries

La direction conceptuelle attendue est :

```text
common <- model / curve / product <- tools <- catalog
                               \-> tests
                               \-> validation (consommateur hors audit)
```

Verifier notamment :

- qu'aucun fichier ou target runtime de `src` n'inclut ou ne lie `tools`, et
  que les include roots rendent cette violation difficile ;
- qu'un produit concret ne remonte pas dans une primitive de modele ou un
  moteur generique ;
- qu'un modele concret ne remonte pas dans un schedule, une reduction, un
  regressor ou une infrastructure cross-asset ;
- que taux, courbe, swap et accrual ne contaminent pas equity, et que spot,
  dividende et payoff vanilla ne contaminent pas fixed income ;
- que types de mode, conventions temporelles, cardinalites et mappings runtime
  appartiennent a `src` ;
- qu'une primitive mathematique partagee runtime/tools a une definition
  canonique dans `src`, tandis que sampling/publication restent offline ;
- que mathDx/cuFFTDx, Premia, QuantLib et dependances optionnelles restent
  limites aux targets qui les consomment ;
- que tout deplacement donne dependances avant/apres, consommateurs et tests
  statiques de frontiere.

### Cleanup and extension locality

Rechercher systematiquement :

- sources et headers sans consommateur ;
- targets, options, facades, aliases et agregats inutilises ;
- placeholders, prototypes non integres et READMEs obsoletes ;
- compatibilites et implementations paralleles devenues sans usage ;
- fragments generes orphelins ou fichiers manuels dupliquant le codegen ;
- sources absentes du build et fichiers enregistres deux fois.

Mesurer les fichiers, manifestes et listes a modifier pour ajouter un modele,
un produit et un couple modele-produit. Un ajout ordinaire ne doit pas exiger
plusieurs listes CMake, bindings, recettes et registres de tests concurrents.
Cette mesure alimente Code generation and extension cost.

### Navigation exercises and completion gate

Executer les exercices suivants depuis la seule arborescence, sans recherche
globale initiale et sans connaissance historique du depot :

1. trouver la dynamics publique, ses definitions device, son sampling et ses
   produits pour un modele markovien, rough FFT, rough N-facteurs et fixed
   income ;
2. trouver, depuis un produit, son contrat, sa composition avec un modele, son
   launcher, sa recette catalogue, son artefact dataset, son target CMake et
   ses tests, en suivant le meme prefixe d'entite ;
3. trouver les templates pricing et sampling de chaque engine, leur renderer,
   leur manifeste, leurs outputs et leur checker de derive ;
4. distinguer immediatement formule mathematique, policy, moteur generique,
   workspace, launch unit, outil offline et recette ;
5. retrouver le proprietaire d'une primitive de `src/common`, d'un outil CUDA,
   d'un test et d'un module CMake sans ouvrir une serie de fichiers ambigus.

Chaque exercice consigne le chemin attendu, le chemin trouve, les fichiers
ambigus ouverts et toute connaissance externe necessaire. Un chemin finalement
trouve ne compense pas une navigation trompeuse.

La section n'est `complete` que si toutes les racines du tableau ont ete
classifiees exhaustivement, que les exceptions de profondeur/noms/en-tetes sont
inventoriees, que la bijection `src -> catalog -> datasets` a ete verifiee pour
chaque entite declaree, que les dependances interdites et orphelins ont ete
recherches, et que les cinq exercices ont ete executes. L'absence locale des
datasets est notee comme exclusion de l'arborescence physique, mais les chemins
declares restent auditables. Un checker existant n'est qu'une preuve parmi
d'autres : ses propres hypotheses et angles morts sont audites.

## Contract homogeneity

Auditer la forme des contrats entre composants jouant le meme role. Une API
similaire ne prouve pas qu'une implementation doit etre partagee ; la
factorisation est examinee separement. Les exceptions mathematiques doivent
rester visibles et testees.

**Perimetre :** dynamics, analytics, produits, concepts, policies, schedules,
droits d'exercice, types prepares et tests de contrat.

**Hors perimetre :** placement/nommage purs, factorisation des implementations
et cout detaille des kernels.

**Preuves attendues :** matrices de capacites, declarations/definitions,
concepts instancies, call graphs et tests homologues.

**Livrable :** differences d'ossature classees en incoherences ou exceptions,
avec contrat cible, preuve et critere de cloture.

Le passage construit une matrice de capacites exhaustive, et non une liste
d'exemples. Une ligne correspond a chaque modele, courbe, produit, engine et
composition declares ; les colonnes indiquent `required`, `provided`,
`optional`, `unsupported` ou `not_applicable` pour preparation, transition,
observables, analytics, schedule, exercise, side, sampling et pricing. Toute
case absente est `unknown` et interdit le statut `complet`.

Pour chaque contrat/concept, compiler au moins une composition positive par
famille et les compositions negatives correspondant aux capacites interdites
ou manquantes. Une erreur negative doit etre rejetee au concept ou au
`static_assert` proprietaire avec un diagnostic lisible, pas plus tard au link
ou dans un template sans rapport.

### Dynamics

Auditer `parameters`, `state`, `dynamics.cuh` et `dynamics_impl.cuh` de tous les
modeles actifs declares, en separant transitions exactes, pas fixe, rough FFT
et lifts rough N-facteurs dans la couverture.

Verifier notamment :

- la meme separation entre parametres bruts, modele prepare, transition
  preparee, dynamique preparee et etat mutable lorsque ces roles existent ;
- le meme ordre de declaration/definition des types, helpers et fonctions
  publiques homologues ;
- types de retour, qualificateurs CUDA, ordre des arguments, valeur/reference
  et conventions `const` coherents ;
- une surface canonique pour preparation, transition, etat initial, avance et
  extraction des observables ;
- des concepts specialises pour les capacites optionnelles plutot qu'un
  contrat de base toujours plus large ;
- la coherence entre transition exacte, pas fixe, wrappers et lifts, sans type
  artificiel ne portant aucun invariant ;
- qu'un etat markovien contient toute l'information necessaire a la transition
  suivante ;
- qu'un lift N-facteurs reste identifie comme approximation rough meme s'il
  utilise l'execution markovienne ;
- l'ordre et la stabilite Philox pour sauts, tirages conditionnels, rejet et
  transitions exactes ;
- l'usage coherent des types entiers, FP32/FP64 et structures d'etat ;
- la symetrie des tests compilation, moments, limites, convergence et RNG.
- la symetrie des surfaces de sampling terminal/calendrier entre modeles de
  meme engine, avec une absence explicite plutot qu'un binding manquant
  decouvert seulement au link.

### Analytics

Auditer Black-Scholes et fixed income, providers, concepts, compositions
modele-courbe, kernels closed form et tests.

Verifier notamment :

- une signature canonique pour coefficients affines, bond, taux, discount,
  option de taux, swap et primitives lognormales ;
- que les providers declarent leurs capacites minimales et possedees ;
- que les compositions ajustees reutilisent le processus de base sans changer
  sa semantique ;
- types, arguments, qualificateurs, temps, payer/receiver et signes coherents ;
- l'absence de formule produit dans une couche analytique inferieure ;
- l'utilisation des providers/concepts sans bypass ponctuel ;
- des tests symetriques d'API, compilation, identites, limites et invariance ;
- qu'une exception propre a un modele ne grossit pas le provider commun.

### Products, policies and exercise

Auditer les concepts et policies composant dynamics, schedules, handlers,
continuation states, regressors et kernels.

Verifier notamment :

- que chaque exigence de concept correspond a un appel reel ;
- que `Schedule` parcourt les dates sans calculer payoff ou prix ;
- que les handlers observent l'etat sans posseder dynamics ou launcher ;
- que `PricingPolicy` transporte seulement inputs, ligne preparee et
  evaluation propres au produit ;
- que `ContinuationState` contient uniquement stockage et variables de
  regression necessaires ;
- que le regressor ne connait ni produit, ni modele, ni courbe ;
- que les types device sont coherents, trivialement copiables et bornes par
  des budgets documentes ;
- que les compositions invalides echouent avec un concept ou `static_assert`
  lisible ;
- que les pricers europeens n'incluent aucun Longstaff-Schwartz ;
- que les briques American/Bermudan partagees se limitent a backward,
  regression, workspace, execution plan et continuation generiques ;
- que payoff, normalisation, calendrier et variables restent sous le produit ;
- que call/put et payer/receiver sont compile-time si seul payoff/signe change ;
- que `AmericanOption` sur grille est documente comme approximation
  bermudeenne avec resolution et limite continue ;
- que les tests separent moteur, policy et composition modele-produit.

Chaque difference observee est classee explicitement en `incoherence`,
`exception mathematique`, `capacite optionnelle`, `non applicable` ou
`unsupported`. Une exception cite son invariant et son test ; `different` ou
`historique` ne sont pas des justifications.

La section n'est `complete` que si la matrice couvre toutes les declarations du
manifeste et tous les composants suivis, si declarations et definitions ont ete
comparees, si les tests positifs/negatifs des concepts ont ete executes, et si
chaque difference a recu l'une des cinq classifications.

## Factorization pyramid

Auditer la reutilisation effective des implementations selon une pyramide
d'invariants. La base contient les algorithmes specialises ; chaque niveau
superieur ne conserve que ce qui est vrai pour tous ses consommateurs. Une
ressemblance d'API, l'emploi commun du Monte-Carlo ou une ambition future ne
suffisent pas a justifier une abstraction.

La pyramide cible est :

```text
transition exacte --+
schema a pas fixe ---+--> markovien general -------+
                                                   +--> commun rough-markovien
rough FFT -----------+                             |
rough N-facteurs ----+--> rough general -----------+

Black-Scholes closed form --+
fixed income closed form ----+--> execution closed form commune

equity -------+
fixed income -+--> infrastructure cross-asset strictement neutre
```

**Perimetre :** implementations communes, concepts, templates, policies,
engines, primitives mathematiques, composition, consommateurs et cout des
abstractions.

**Hors perimetre :** simple difference de nom/placement et opportunite sans
consommateur reel.

**Preuves attendues :** matrice producteurs/consommateurs/invariants,
duplications, call graph, instanciations, tests de parite, objets/cubins et
ressources des kernels affectes.

**Livrable :** pour chaque niveau, une table `invariants communs`,
`implementation partagee`, `partie specifique`, `consommateurs`, `exceptions`
et `cout CUDA/build`.

Chaque abstraction existante ou proposee recoit une ligne dans une matrice de
decision contenant au minimum :

| Abstraction | Producteurs | Consommateurs reels | Invariants partages | Duplication supprimee | Adaptateurs/exceptions | Branches runtime | Cout CUDA | Cout build | Decision |
|---|---|---|---|---|---|---|---|---|---|

`Cout CUDA` couvre taille des types prepares, valeurs vivantes, registres,
spills, shared, occupation et taille machine ; `Cout build` couvre nombre
d'instanciations, objets/cubins, clean et incremental. Une cellule inconnue
reste `a mesurer`. Une abstraction sans au moins deux consommateurs reels, ou
qui n'encode qu'un consommateur futur, est rejetee ou maintenue explicitement
locale ; la seule similarite syntaxique ne constitue pas un invariant.

### Markovian factorization

#### Exact transitions

Verifier notamment :

- une primitive commune pour preparation et avance sur intervalle arbitraire
  lorsque la loi exacte le permet ;
- des schedules terminal, regular et calendar reutilisant cette capacite sans
  sous-pas artificiels ;
- une consommation Philox stable entre terminal et dates contractuelles ;
- que increments de Levy, sauts composes et diffusions exactes gardent leurs
  lois specifiques derriere un contrat minimal ;
- qu'un produit dense emploie les pas requis par son observation et ne pretend
  pas beneficier d'une transition terminale exacte.

#### Fixed-step Markovian schemes

Verifier notamment :

- une ossature unique pour preparation de `dt`, boucle de transitions,
  observation et handlers ;
- des schedules terminal, dense, regular et calendar sans simulateur recopie ;
- que Euler, QE, full truncation ou autre schema restent dans la dynamics et
  non dans le moteur de trajectoires ;
- l'absence de branche runtime par modele ou schema dans la boucle chaude ;
- que les etats multi-facteurs et tirages variables n'elargissent pas le
  contrat de tous les modeles.

#### General Markovian layer

Verifier que transitions exactes et pas fixe partagent seulement :

- preparation des inputs et cardinalites ;
- creation de l'etat initial et observation ;
- composition dynamics/schedule/handler/product ;
- kernels Monte-Carlo et sampling quand les invariants coincident ;
- construction des resultats, reductions et diagnostics ;
- mapping deterministe resultat/trajectoire/sequence Philox.

Pour le sampling modele, verifier en plus que les layouts `un chemin par
parametre` et `plusieurs chemins conditionnels par parametre` reutilisent la
meme preparation et le meme mapping logique, tout en autorisant des strategies
de kernel distinctes lorsque leur cout mesure le justifie.

Le niveau general ne contient ni branche runtime `exact/fixed`, ni adaptateur
vide, ni membre prepare inutilise dans l'un des chemins.

### Rough factorization

#### FFT and Gaussian-Volterra paths

Verifier notamment :

- la factorisation des kernels de Volterra, coefficients, convolution,
  padding, plans FFT et reconstruction gaussienne ;
- que les mappings propres a chaque modele rough restent sous le modele ;
- que generation gaussienne, correlation, observation et reduction ne sont
  pas recopies par modele ;
- que cuFFTDx et specialisations de longueur sont isoles des autres familles.
- que pricing et sampling reutilisent convolution, padding, dispatch de
  longueur et preparation sans dupliquer une seconde table de tuning par
  modele, tout en gardant leurs layouts de sortie propres.

La composition Gaussian-Volterra est inspectee comme exemple obligatoire de
frontiere de policies :

- la `KernelPolicy` possede uniquement les parametres/preparations du noyau de
  covariance, sa variance Volterra et la reconstruction de sa valeur ;
- la `PathPolicy` possede l'etat, la correlation, les observables et la mise a
  jour propres au chemin du modele ;
- le moteur FFT possede `dt`, `sqrt(dt)`, padding, plans, dispatch, buffers et
  execution, et traite le noyau prepare comme opaque ;
- la policy produit ne fuit ni dans le noyau Volterra ni dans le plan FFT ;
- les concepts verifient la relation exacte entre types de parametres kernel et
  path, avec compositions positives et negatives ;
- pricing et sampling partagent ces invariants sans forcer le meme layout de
  sortie ni recopier une preparation du noyau.

#### Markovian N-factor rough lifts

Verifier notamment :

- une representation commune des facteurs exponentiels et de leur
  preparation ;
- une specialisation compile-time ou objet compact pour le nombre de facteurs,
  sans tableau local provoquant spills ;
- la reutilisation du moteur markovien seulement pour l'execution semantiquement
  compatible ;
- que parametres de lift, erreur d'approximation et convergence restent
  visibles dans modele et tests ;
- qu'un lift n'est jamais presente comme transition exacte du processus rough.

#### General rough layer

Verifier ce qui peut etre partage entre FFT et N-facteurs :

- contrat des parametres de rugosite et horizon ;
- correlation et observables communes ;
- policies produit, construction des prix, reductions et diagnostics ;
- tests de convergence vers la dynamique rough cible ;
- metadonnees necessaires au codegen et a la selection d'engine.

Ne pas imposer une fausse `DynamicsPolicy` markovienne au chemin Volterra ni
une dependance FFT au lift N-facteurs.

### Rough-Markovian factorization

Auditer l'intersection independante de la memoire du processus :

- generation et adressage Philox ;
- device inputs et validation des cardinalites ;
- products, schedules compatibles, handlers et payoffs ;
- construction aligned/cartesian ;
- reductions, erreurs standards et publication ;
- buffers RAII, diagnostics et propagation du stream ;
- moteurs d'exercice anticipe seulement si un etat fini fournit les variables
  de continuation necessaires.

Refuser tout moteur universel ajoutant branche runtime rough/markovian, appel
indirect, gros variant device, valeur vivante inutile ou dependance cuFFTDx aux
targets markoviennes.

### Closed-form factorization

Auditer separement factorisation mathematique et factorisation d'execution.

Verifier notamment :

- les primitives lognormales/Black-Scholes partageables : normal CDF/PDF,
  discount, forward, d1/d2 et symetries call/put ;
- les primitives fixed income partageables : coefficients affines, Gaussian
  bond options, Jamshidian, cashflows et compositions de courbe ;
- qu'une primitive devient commune seulement si domaine, unites et conventions
  de signe coincident ;
- que les formules propres a CIR, Gaussian one/two-factor, Black-Scholes ou un
  produit specialise restent dans leur proprietaire ;
- la reutilisation des concepts, inputs, mappings, kernels scalaires ou
  cooperatifs, launchers et diagnostics closed form ;
- que le choix scalaire/cooperatif est une strategie mesuree, pas une
  duplication de formule ;
- que la factorisation ne force pas un provider universel rempli de capacites
  optionnelles.

### Fixed-income and equity factorization

Le niveau cross-asset reste le sommet le plus strict. Verifier qu'il contient
seulement :

- checks CUDA, RAII, streams et evenements ;
- Philox, reductions, moments et construction des resultats ;
- moteurs Monte-Carlo, sampling et Longstaff-Schwartz reellement neutres ;
- generation de parametres/calendriers, observations de sample et streaming
  d'artefacts seulement lorsque leur contrat est independant de l'asset class ;
- indexation, batching, workspace planning et diagnostics ;
- infrastructure de loaders, datasets, codegen et publication independante de
  la semantique financiere.

Spot, dividende, taux court, courbe, accrual, swap, nominal, strike vanilla ou
payer/receiver restent hors de ce niveau.

### Factorization cost and limits

Toute factorisation proposee compare avant/apres :

- branches runtime, indirect calls et specialisations compile-time ;
- taille des `Prepared*`, valeurs vivantes et donnees par thread ;
- registres, spills, stack, shared memory et occupation ;
- taille objets/archives/cubins et instruction cache ;
- temps clean et incremental ;
- lisibilite des erreurs de compilation ;
- consommateurs et fichiers manuels supprimes ;
- sorties, tolerances et mapping aleatoire.

Refuser une abstraction fondee sur un consommateur futur, des adaptateurs
vides, des branches de capacites ou une baisse de duplication qui augmente
materiellement le cout runtime ou build.

La section n'est `complete` que si tous les moteurs et abstractions communs du
graphe ont une ligne de decision, si les duplications ont ete recherchees dans
tous leurs consommateurs reels, si les exceptions sont bornees, et si chaque
cout est mesure ou demontre immateriel. Une case `a mesurer` maintient la
section partielle et interdit une conclusion positive sur le cout, sans imposer
a elle seule une refactorisation.

## Code generation and extension cost

Auditer le chemin complet d'ajout d'un modele ou produit. L'auteur doit coder
le minimum mathematique et semantique, puis une source typee derive bindings,
recettes, inscriptions et controles sans masquer les exceptions.

**Perimetre :** manifests, templates, generateurs Python, sorties `.cuh/.cu`,
recettes catalogue, YAML, parametres, fragments CMake, tests et checkers.

**Hors perimetre :** choix mathematique d'un schema/payoff et generation des
references independantes traitee par l'audit de validation.

**Preuves attendues :** generation temporaire comparee au tree, matrice de
capacites/outputs, builds avec/sans dependances optionnelles, execution de
chaque engine et compte des interventions manuelles.

**Livrable :** contrat d'extension model/product, source de verite, outputs
derives, exceptions explicites et cout manuel restant.

Chaque combinaison `model x product x variant`, chaque binding sample et chaque
recette possede un etat d'ownership explicite parmi : `generated`,
`hand_written`, `explicit_exception`, `unsupported` ou `deferred`. L'existence
ou l'absence d'un fichier ne sert jamais a inferer cet etat. Une exception ou
un report nomme son proprietaire, sa justification et sa condition de sortie.

Pour chaque output, le passage fournit une trace exacte :

```text
declaration source -> resolver -> template -> renderer -> chemin output
                   -> target CMake -> recette/YAML -> checker
```

Un maillon sans source de verite ou recopie dans une seconde table est un
defaut de contrat, meme si la regeneration courante est zero-diff.

### Minimum hand-written model and product

Pour un nouveau modele, verifier que le minimum manuel est limite a :

- parametres compacts, loader et schema ;
- dynamics ou analytics mathematiques propres ;
- primitives de preparation et observables specifiques ;
- specification de generation des parametres core/stress ;
- tests mathematiques et numeriques ;
- declaration de capacites, observables de sample et dependances.

Pour un nouveau produit, verifier que le minimum manuel est limite a :

- parametres compacts et loader ;
- schedule, observation et payoff/pricing policy ;
- continuation state si l'exercice anticipe l'exige ;
- specification de generation des parametres ;
- tests semantiques ;
- declaration des besoins d'engine et variantes de side.

Un couple ordinaire ne doit pas exiger de recopier launcher, kernel, runner
CUDA, recette complete ou liste CMake.

### Canonical capability manifest

Auditer une source typee exprimant au minimum :

- `ModelSpec` : famille mathematique, transition, engine rough, analytics,
  etat, sampling, observables, dependances et architectures ;
- `ProductSpec` : schedule, observation, exercice, policy, sidedness,
  parametres et capacites requises ;
- `EngineSpec` : concepts, templates, launchers, runners, dependances et
  strategie d'instanciation ;
- `DatasetSpec` : sources de parametres, aligned/cartesian, chemins catalogue,
  profils numeriques, layouts de sample et metadonnees derivables.

Le resolver produit explicitement :

```text
(model, product, variant) -> engine -> bindings -> target -> catalogue recipe
```

Il derive aussi le prefixe canonique d'entite depuis `src`; ce meme prefixe est
injecte dans `catalog` et `datasets`, y compris le niveau de famille
`markovian`, `rough` ou toute future famille. Aucun template ou renderer ne
reconstruit manuellement un chemin plus court, pluriel ou historique.

Verifier notamment :

- qu'une incompatibilite est refusee par resolver ou concept lisible ;
- qu'aucune deuxieme liste manuelle ne recopie models, products, sides,
  schedules, engines ou dependances ;
- que transition exacte, pas fixe, rough FFT, rough N-facteurs, closed form et
  early exercise ainsi que sampling ordinaire, prepare et Volterra sont
  representables sans champ ambigu ;
- que les exceptions algorithmiques sont nommees, bornees et testees ;
- que le manifeste ne redeveloppe pas les algorithmes C++ dans Python.

### Generated bindings and catalogue recipes

Verifier que les champs derivables produisent :

- `<model>/product/<product>.cuh` et `<model>/product/<product>.cu` ;
- `<model>/sample.cuh` et `<model>/sample.cu` lorsque la capacite est publiee ;
- specialisations call/put, payer/receiver ou non-sided ;
- schedule, policy et engine ;
- instanciations et fragment CMake ;
- generateur executable de catalogue ;
- `dataset.yaml` ou ses champs structurels derivables ;
- tests de compilation et checkers d'architecture.

Pour chaque engine pricing genere — markovien, rough N-facteurs, Volterra FFT,
Black-Scholes closed form, fixed-income closed form et early exercise —
retrouver une paire de templates binding `.cuh/.cu` et le template de
generateur de catalogue correspondant, ou un etat d'ownership non genere
explicite. Une famille fixed-income closed form ne peut etre consideree couverte
par le seul template Black-Scholes ou par une chaine inline dans le renderer :
leurs inputs, providers, courbes et produits doivent partager des invariants
prouves ou conserver des templates distincts et nommes.

Les templates suivent la meme taxonomie que les engines et les artefacts. Un
lecteur doit trouver sans recherche globale, dans deux chemins differents, le
binding pricing et le binding sampling de `markovian`,
`rough/markovian_n_factor` ou `rough/volterra_fft`. Closed form equity,
closed form fixed income et `early_exercise` restent trois branches nommees.
Les fichiers generes complets ne vivent pas comme chaines C++ multilignes dans
le renderer Python; les fragments calcules y restent bornes a des valeurs ou
petites declarations injectees dans un template visible.

Le catalogue actuel utilise `generator.cpp`. Le referentiel emploie le terme
neutre **generateur de catalogue** : un passage a Python doit etre une decision
explicite, pas une coexistence accidentelle de deux formats.

Verifier que profils de chemins, trajectoires, time grids, seeds, batching et
timings restent explicites dans la source de verite ou une specification
versionnee. Aucun choix numerique important ne doit etre cache par default.

### Model-sample bindings and recipes

Auditer explicitement la chaine modele-seul, independamment de la matrice de
prix :

```text
ModelSpec -> sample engine -> sample.cuh/cu -> samples_01/samples_02
          -> generator.cpp -> dataset JSON + dataset.yaml
```

Verifier notamment :

- que chaque modele declare `available`, `deferred` ou `unsupported` pour son
  binding et pour chacune des deux recettes, sans inference par existence de
  fichier ;
- que les engines transition exacte, pas fixe, prepare N-facteurs et
  Gaussian-Volterra possedent chacun un template ou une exception bornee ;
- que les deux layouts contractuels, `12 000 x 250` et `3 000 000 x 1`,
  reutilisent un pipeline hote commun sans recopier allocation, warmup,
  launch, copie, streaming, YAML et controle de schema ;
- que la generation des parametres plausibles core, les bornes publiees et
  leur ordre de champs derivent d'une meme specification revue, sans tripler
  ces valeurs entre recette de parametres, code C++ et YAML ;
- que domaines et seeds Philox des parametres, calendriers et dynamics sont
  independants, versionnes et invariants au batching et a la geometrie ;
- que noms et layouts des observables sont declares par modele, valides par le
  binding et reproduits dans JSON/YAML ;
- que le memory planning couvre toutes les valeurs simultanement vivantes,
  refuse les overflows et permet un decoupage reproductible lorsqu'un dataset
  ne tient pas sur le GPU ou l'hote cible ;
- que `--smoke-test` execute exactement 1 000 lignes pour chaque layout,
  recharge JSON et YAML, verifie schema, jours, `T`, dimensions, finitude,
  repetitions conditionnelles et provenance ;
- que CMake et les checkers enregistrent exactement les bindings et recettes
  disponibles avec et sans dependances optionnelles ;
- que generation temporaire et zero-diff couvrent aussi `sample.cuh/.cu`,
  `generator.cpp` et les champs derivables de `dataset.yaml`.

Une premiere recette manuelle peut servir de prototype, mais elle ne satisfait
pas le contrat d'extension tant que sa logique derivable n'est pas representee
dans le manifeste et regenerable.

### Parameter dataset generation

Auditer la base de parametres et sa recette adjacente selon
`model-and-product-parameter-dataset-generation.md`.

Verifier notamment :

- specification complete des domaines, contraintes et transformations ;
- politique ordonnee 90/10 core/stress et provenance des lignes ;
- generation reproductible JSON/YAML ;
- coherence schema, loader, noms, unites et ordre des champs ;
- infrastructure de sampling, assemblage et serialisation non recopiee ;
- bornes, distributions, correlations et stress manuellement revus, meme si
  leur rendu code/YAML est genere ;
- aucun default silencieux produisant recette incomplete ou plage invalide.

### Regeneration, drift and exceptions

Verifier notamment :

- generation idempotente et comparaison zero-diff ;
- provenance/version du generator dans les outputs ;
- detection de toute edition manuelle d'un fichier genere ;
- detection des outputs manquants, excedentaires et orphelins ;
- tests negatifs supprimant temporairement un output, renommant une entite,
  ajoutant un output orphelin et cassant un mapping `src/catalog/datasets`, avec
  refus attribuable du checker ;
- builds avec/sans mathDx/cuFFTDx et architectures declarees ;
- builds et smoke tests des samples disponibles pour les deux layouts et pour
  chaque famille d'engine active ;
- tests de chaque template, resolver et engine, cas valides/invalides ;
- escape hatch explicite avec proprietaire, justification et checker ;
- suppression des outputs supersedes ;
- incrementalite, taille des unites CUDA et diagnostics lisibles preserves.

Mesurer separement, pour l'ajout d'un modele, d'un produit et d'une composition
representative : nombre de fichiers manuels, lignes semantiques, entrees de
manifeste, listes/targets touchees, commandes, outputs generes et modifications
residuelles. La mesure inclut un dry-run d'ajout puis de retrait afin de prouver
l'absence d'orphelin. Reduire le nombre de fichiers generes n'est pas un
objectif si le code manuel augmente ou si la matrice de capacites devient
opaque.

La section n'est `complete` que si tous les outputs attendus ont un etat
d'ownership, si leur trace source-vers-checker est fermee, si generation et
tests negatifs passent, si les chemins `src/catalog/datasets` sont bijectifs,
et si les trois couts d'extension ont ete chiffres. `validation/**` ne fait pas
partie des outputs ni du cout manuel de ce codegen.

## CMake and build graph

Auditer CMake comme un graphe de responsabilites et non une liste de sources.
L'objectif est un build lisible, minimal, sans targets/options inutiles, avec
frontieres CUDA explicites et bonne incrementalite.

**Perimetre :** CMake racine, modules, fragments generes, targets, sources,
dependances, options, architectures, instanciations et artefacts compiles.

**Hors perimetre :** ownership conceptuel deja traite par Structure et
performance runtime non causee par le build.

**Preuves attendues :** graphe de targets, depfiles, symboles, sources,
builds clean/no-op/incrementaux, configurations optionnelles et tailles
objets/archives/cubins.

**Livrable :** sources orphelines/dupliquees, targets/options inutiles,
dependances transitives et mesures de toute frontiere proposee.

Le passage commence par deux matrices fermees :

1. **Configuration** — build portable par defaut, chaque architecture annoncee,
   fatbin annonce, dependances optionnelles presentes/absentes, tests
   actives/desactives et profils Release/diagnostic necessaires ; chaque case
   vaut `configure`, `build`, `run`, `unsupported` ou `not_applicable` avec la
   commande et le resultat.
2. **Mutation/incrementalite** — no-op, modification d'un produit, d'une
   dynamics, d'une analytics, d'une courbe, d'une primitive `common`, d'un
   template et du manifeste ; chaque ligne donne les targets attendues puis les
   targets reellement regenerees/recompilees/relinkees.

Une configuration non executee reste `unknown`, jamais implicitement validee
par une configuration voisine.

Verifier notamment :

- que le CMake racine declare projet, options et orchestration de haut niveau,
  puis delegue les targets par domaine ;
- que chaque `.cpp` ou `.cu` autonome appartient exactement a une target et
  qu'aucune source n'est absente ou enregistree deux fois ;
- que les fragments generes viennent de la meme source de verite que bindings
  et recettes, sans liste parallele ;
- qu'aucun glob silencieux n'est la seule garantie de completude ; avec
  `CONFIGURE_DEPENDS`, un checker valide le contenu attendu ;
- que les targets sont assez fines sans micro-target sans responsabilite ;
- que les agregats ne sont jamais lies comme bibliotheques locales ;
- que chaque target publie seulement includes, definitions et bibliotheques
  necessaires, et que chaque dependance `PUBLIC`, `PRIVATE` ou `INTERFACE` est
  justifiee par l'usage de ses headers et symboles plutot que par commodite ;
- que mathDx/cuFFTDx et dependances optionnelles n'affectent pas les autres
  targets ;
- que chaque binding `sample.cu`, bibliotheque de publication et recette de
  sample appartient exactement a une cible, et que l'agregat
  `sample_generators` ne publie aucune recette differee ou non linkable ;
- que target, facade, alias, option, variable cache ou fonction sans
  consommateur est supprime ;
- qu'une fonction/specialisation non-inline a une definition unique et que les
  instanciations side/model/curve/N-facteurs ne dupliquent pas le code ;
- que les templates restent visibles au point d'instanciation sans exposer
  toutes les implementations ;
- que les architectures ne multiplient pas les cubins sans besoin explicite ;
- que ccache/depfiles sont actifs et que clean, no-op et modifications d'un
  produit, dynamics, analytics, curve ou manifeste recompilent la matrice
  attendue ;
- que temps par target, tailles objets/archives/cubins et erreurs de link sont
  conserves pour les specialisations principales ;
- que toute fusion/decomposition compare build clean/incremental, code genere,
  ressources kernel et lisibilite des erreurs.

Pour chaque source autonome, consigner son unique target proprietaire ; pour
chaque target, consigner type, responsabilite, sources, dependances directes,
usage requirements publics et consommateurs. Comparer ce graphe au manifeste
codegen afin qu'aucune ownership ne soit definie concurremment.

La section n'est `complete` que si toutes les cases applicables des deux
matrices ont un resultat, si chaque `.cpp/.cu` et target a un proprietaire
unique, si la visibilite des dependances a ete revue, et si les mesures clean,
no-op et incrementales ont ete conservees. Une impossibilite materielle ou de
dependance est une exclusion explicite et maintient l'axe partiel sur la case.

## Portability and hardware tuning

Auditer separement compatibilite fonctionnelle et performance portable. Une
RTX 4090 Laptop/SM89 peut etre le materiel de mesure et fournir les profils
livres par defaut ; elle n'est pas, a elle seule, la cible architecturale du
projet. Une valeur mesuree sur ce GPU reste un point de depart etiquete, jamais
un optimum universel.

Les invariants numeriques et reproductibles restent communs : pas de fast
math, mapping Philox, ordre de reduction, FP64 sensible, calendriers et
semantique des sorties. Les geometries, chunk sizes, longueurs/strategies FFT,
nombre de streams et seuils de dispatch sont des parametres de tuning dont la
validite performance depend du GPU, de la toolchain et du workload.

**Perimetre :** compute capabilities, PTX/SASS, dependances optionnelles,
presets, profils de lancement, ressources, limites memoire, baselines,
fallbacks et documentation de deploiement, pour pricing et sampling.

**Hors perimetre :** promesse de performance identique entre GPU et ajout
d'une auto-optimisation runtime sans preuve qu'elle est necessaire.

**Preuves attendues :** matrice declaree support/engine/architecture,
configurations offline, executions sur materiel disponible, diagnostics de
ressources, provenance des profils et procedure reproductible de retuning.

**Livrable :** compatibilites dures, profils de reference, valeurs a retuner,
fallbacks, exclusions et instructions utilisateur pour mesurer son propre GPU.

Pour chaque `engine x architecture`, publier une ligne contenant au minimum :

| Engine | Architecture | Configure | Compile | Runtime teste | Numerique teste | Fallback | Profil de tuning | Performance mesuree | Conclusion |
|---|---|---|---|---|---|---|---|---|---|

Les conclusions autorisees sont distinctes : `compilable`, `executable`,
`numerically_qualified`, `retunable` et `performance_profiled`. Elles se
cumulent seulement avec leurs preuves propres. Un build offline autorise
`compilable`, jamais `executable` ni une affirmation de performance. Un profil
SM89 peut etre propose comme fallback etiquete sur une autre architecture sans
etre presente comme retune ou optimal.

### Supported architecture matrix

Verifier notamment :

- que les architectures de compilation, les GPU reellement testes et les GPU
  seulement supposes compatibles sont trois ensembles distingues ;
- que chaque engine declare son minimum CUDA/compute capability et ses
  dependances, notamment cuFFTDx/mathDx ;
- qu'un preset local SM89 ne devient pas l'unique chemin de build public et
  qu'une configuration portable ou mono-architecture reste documentee ;
- qu'aucun garde `architecture == 89`, type ou specialisation SM89 ne bloque
  un GPU annonce sans justification issue de l'API utilisee ou d'un test ;
- que fatbins multi-architectures, PTX de repli ou builds separes sont choisis
  explicitement selon taille, temps de compilation et couverture voulue ;
- qu'une incompatibilite reelle echoue a la configuration avec le nom de
  l'engine, la capacite requise et une alternative, sans etre confondue avec
  une geometrie simplement sous-optimale ;
- que pricing, samples, tests et generateurs exposent la meme matrice de
  disponibilite sur une architecture donnee.

### Portable defaults and retuning

Pour chaque famille de kernel, classer chaque valeur en invariant,
dimension de dataset ou parametre de tuning. Auditer au minimum :

- `threads_per_block`, blocs persistants, `blocks_per_price`, work par thread,
  factor count, shared opt-in et seuils de strategie ;
- chunks de paths/samples, taille des batches, streams et marge VRAM/RAM ;
- dispatch direct/cooperatif, thread/warp/bloc, FFT length,
  elements-per-thread et FFTs-per-block ;
- seuils automatic/explicit et interaction avec registres, spills, shared,
  occupation, SM count, bandwidth et instruction cache.

Les valeurs livrees peuvent rester celles mesurees sur RTX 4090 Laptop/SM89,
mais leur provenance doit apparaitre dans le profil ou la documentation. Elles
doivent etre centralisees ou surchargeables sans modifier la mathematique ni
le mapping aleatoire. L'utilisateur est explicitement invite a executer les
benchmarks, diagnostics de kernels et tests numeriques sur son GPU avant de
publier sa propre baseline ou de remplacer un profil.

Refuser a la fois l'auto-tuning opaque qui change les sorties et la pretention
qu'une geometrie unique est optimale partout. Une selection par architecture
n'est necessaire que si des mesures montrent des profils durablement
differents ; sinon une valeur portable, sure et clairement etiquetee suffit.

### Cross-architecture evidence

Verifier notamment :

- compilation offline de chaque engine sur toutes les compute capabilities
  annoncees, avec ressources SASS/PTX des specialisations representatives ;
- tests runtime et sanitizers sur chaque architecture effectivement possedee,
  sans extrapoler leurs temps aux autres GPU ;
- baselines separees par GPU, architecture, toolchain, clocks/power et protocole,
  jamais comparaison d'un candidat a une reference SM89 incompatible ;
- tests de limites materielles : max threads, shared opt-in, grids, memoire,
  FFT supportee et residence minimale ;
- equivalence numerique et reproductibilite entre geometries sur un meme GPU,
  puis tolerance documentee entre architectures lorsque le bit-a-bit n'est pas
  contractuel ;
- conservation des resultats, ressources et commandes brutes permettant a un
  utilisateur de qualifier une nouvelle architecture.

L'absence d'un GPU physique reste une exclusion de runtime dans `status.md`.
Elle n'autorise ni une affirmation de performance, ni un blocage logiciel
artificiel d'une architecture que les dependances et les builds supportent.

La section n'est `complete` que si chaque engine et architecture annonces ont
une ligne, si toute conclusion est soutenue par son niveau de preuve, si les
fallbacks et parametres retunables sont explicites, et si la procedure de
qualification d'un nouveau GPU couvre build, tests numeriques, sanitizers,
ressources et benchmark. Les cases runtime impossibles faute de materiel sont
des exclusions honnetes ; elles n'invalident pas la couverture structurelle de
la matrice mais interdisent la qualification runtime correspondante.

## Numerical robustness and reproducibility

Auditer la robustesse numerique des dynamics, analytics, solveurs, reductions,
regressions et kernels. Une optimisation n'est acceptable que si elle preserve
le contrat mesure sur le domaine utile et les cas limites publies.

**Perimetre :** precision, domaines mathematiques, conditionnement,
convergence, propagation d'erreur et reproductibilite CPU/CUDA.

**Hors perimetre :** gain de temps ou ressources considere seul ; il releve de
Performance et reutilise les tolerances etablies ici.

**Preuves attendues :** domaines de parametres, budgets d'erreur absolue,
relative, ULP ou statistique, limites et references independantes.

**Livrable :** contrat numerique par famille, cas non couverts, risques prouves
et hypotheses de precision a mesurer.

Le passage construit une matrice par primitive/famille avec : domaine core et
stress, precision stockage/calcul/accumulation, reference locale, tolerance,
invariant ou ordre de convergence, mode d'echec et tests CPU/CUDA. Une case
sans reference ou budget reste `unknown` et ne peut etre certifiee par la seule
finitude d'une sortie.

Chaque tolerance numerique est definie avant d'observer le resultat candidat.
Elle provient d'un contrat mathematique, d'une reference de precision
superieure, d'une propagation d'erreur, d'un ordre de convergence ou d'un jeu
de qualification distinct. Une regle telle que `resultat observe * marge`, une
tolerance deduite du seul run que l'on cherche a accepter ou un ajustement
apres echec est interdit. Une calibration empirique utilise des donnees
separees des cas qui certifient ensuite le budget et conserve son protocole.

Dans ce referentiel, une **reference independante** est une identite
mathematique, une formule locale distincte, une implementation CPU simple, une
evaluation haute precision ou une limite analytique appartenant aux tests du
composant. L'inventaire Premia/QuantLib, les caches et la certification de prix
publies appartiennent exclusivement a `../validation/query.md`.

### Precision and mathematical domains

Verifier notamment :

- une politique explicite pour stockage, calcul device, accumulation et sortie :
  FP32 par defaut seulement lorsqu'il respecte le budget, FP64 pour reductions,
  moments, regressions, factorisations ou operations sensibles ;
- que chaque FP64 dans une boucle chaude est justifie et que chaque descente en
  FP32 est mesuree contre la reference et le domaine complet ;
- inventorier tout calcul FP64 atteignable depuis du code device, y compris
  preparation, kernels auxiliaires, reductions et fonctions
  `__host__ __device__`. Pour chacun : frequence d'appel, volume d'operations,
  justification numerique, alternative FP32/mixte/host, ressources compilees
  et cout mesure. Une justification mathematique sans comparaison numerique et
  performance ne suffit pas ;
- domaines de `log`, `log1p`, `sqrt`, `pow`, divisions, exponentielles, CDF/PDF,
  fonctions inverses et speciales, y compris zero, negatif, infini, sous-normal
  et valeurs proches des frontieres ;
- positivite, bornes de correlation, frontieres absorbantes/reflechissantes et
  contraintes, sans clamp ou projection silencieuse changeant la loi ;
- stabilite pour petits temps, faible volatilite ou mean reversion,
  correlations aux bornes, intensites extremes et limites d'admissibilite ;
- cancellation, overflow/underflow, echelles de prix/taux/accrual et formules
  alternatives stables (`log1p`, `expm1`, series ou limites) ;
- conventions temporelles, day counts, accruals, unites et calendriers aux
  frontieres host/device et dans les rows publiees.

### Solvers and linear algebra

Verifier notamment :

- encadrement, monotonie, critere d'arret, iterations maximales et precision
  effective de chaque recherche de racine ou inversion ;
- conditionnement, pivot/regularisation, normalisation et propagation d'erreur
  de Cholesky, moindres carres, equations normales et regressions ;
- conditionnement Longstaff-Schwartz selon bases, variables, paths et dates,
  ainsi que distinction entre erreur de solveur, bruit MC et biais d'exercice ;
- comportement sur systeme singulier, quasi-singulier, aucun candidat, faible
  variance et coefficients non finis ;
- reduction FP64, ordre des sommes, compensation eventuelle et budget d'erreur
  lorsque le nombre de paths ou de facteurs croit ;
- reference locale de petite taille permettant de distinguer formule, stockage,
  solveur et orchestration CUDA.

### Stochastic dynamics and convergence

Verifier notamment :

- moments, lois terminales, positivite et limites des transitions exactes ;
- convergence forte/faible des schemas a pas, sans sous-pas artificiel sur une
  transition exacte ou une date contractuelle ;
- biais et ordre selon `dt`, resolution FFT, approximation directe/hybride,
  nombre de facteurs rough et grille d'exercice ;
- stabilite des sauts, rejets, tirages conditionnels et correlations extremes ;
- pour les samples, domaines de parametres, calendriers, observables et deux
  layouts contractuels, sur lignes core et stress representatives ;
- separation mesuree entre erreur de discretisation, erreur statistique,
  approximation rough/FFT et erreur arithmetique ;
- invariance des lois et tolerances lorsque batching, chunking, geometrie ou
  strategie de kernel change.

### Determinism and random-number mapping

Verifier notamment :

- mapping Philox contractuel `(path_index: uint64, local_group_index: uint64)`
  sous la cle de ligne, sans reservation aplatie dependante du batching ;
- domaines/seeds independants pour parametres, calendriers, dynamics et
  variantes, avec absence de chevauchement prouvee ;
- ordre et cardinalite de consommation stables pour transitions exactes,
  fixed-step, sauts, rejet, rough FFT/N-facteurs et exercise ;
- reproductibilite contractuelle entre executions, lots, streams et geometries
  sur une meme architecture ;
- politique explicite entre egalite bit a bit, tolerance numerique et intervalle
  statistique, y compris entre CPU/CUDA ou architectures differentes ;
- ordre des reductions et toute source de non-determinisme atomique documentes.

Ne pas demander l'egalite bit a bit entre algorithmes equivalents si elle
n'est pas contractuelle. Ne pas relacher non plus un mapping aleatoire ou un
ordre de reduction contractuel sous pretexte que les distributions semblent
equivalentes.

### Failure propagation and completion gate

Verifier notamment :

- validation des inputs avant lancement et erreurs attribuables au modele,
  produit, ligne, parametre ou solveur concerne ;
- absence de NaN/Inf silencieux dans prix, erreurs standards, coefficients,
  etats, samples et datasets ;
- propagation distincte des echecs de domaine, convergence, allocation,
  lancement CUDA et sortie ;
- aucun default, clamp, fallback numerique ou changement de methode silencieux
  ne transformant un echec en valeur finie plausible ;
- tests negatifs des domaines invalides et tests des frontieres valides, avec
  meme semantique host/device lorsque les deux existent.

La section n'est `complete` que si toutes les familles de dynamics, analytics,
solveurs, reductions, pricing et sampling actifs ont une ligne de matrice, si
domaines core/stress et limites applicables sont couverts, si chaque tolerance
a une reference locale et si chaque mode d'echec est teste. Une reference
externe de validation peut completer la confiance mais ne remplace aucune de
ces preuves proprietaires. Toute occurrence FP64 atteignable depuis le device
qui reste non classee, non justifiee ou non mesuree maintient la section
partielle.

## CUDA execution and memory safety

Auditer la surete host/device independamment de la justesse des formules et de
la performance. L'absence d'un sanitizer est une exclusion, pas un constat ;
un finding exige un defaut ou risque precis prouve.

**Perimetre :** tailles/offsets, acces memoire, duree de vie et aliasing,
streams, synchronisations, erreurs asynchrones et outils host/CUDA.

**Hors perimetre :** ecart numerique fini et cout d'un layout ou d'une
synchronisation valide.

**Preuves attendues :** revue launchers/workspaces, overflow, ASan/UBSan host,
puis `compute-sanitizer` memcheck, racecheck, initcheck et synccheck sur une
matrice minimale fermee.

**Livrable :** erreurs de bornes, duree de vie, initialisation, concurrence ou
propagation prouvees, cas minimal et familles non executees.

La matrice sanitizer contient une ligne par cas et quatre colonnes obligatoires
`memcheck`, `racecheck`, `initcheck`, `synccheck`, chacune avec commande,
artefact, resultat et logs conserves. Elle couvre au minimum :

1. closed form scalaire et cooperatif lorsque les deux strategies existent ;
2. pricing markovien transition exacte, schedule regular et layout aligned ;
3. pricing markovien fixed-step, schedule explicit/ragged et layout cartesian ;
4. Longstaff-Schwartz equity multi-etats et fixed income deux facteurs, avec
   workspace reutilise ;
5. Volterra FFT pricing puis sampling dans les deux layouts contractuels ;
6. rough N-facteurs pricing puis sampling avec le plus grand facteur publie ;
7. lots minimaux, derniers blocs partiels, tailles non multiples et un lot
   traversant une frontiere de paths conditionnels ;
8. stream non-default, appels repetes, workspace/buffers reutilises et au moins
   un scenario concurrent supporte par l'API.

Un executable peut couvrir plusieurs lignes, mais `status.md` montre la
correspondance exacte entre cas, engine, layout, batch, stream et workspace.
ASan/UBSan couvrent en plus loaders, planners, arithmetique de tailles et
orchestration host representatifs de chaque famille.

Verifier notamment :

- multiplications/additions de cardinalites, strides, offsets, tailles et
  alignements avant allocation/lancement, avec overflow ;
- couverture des pointeurs et validation des vues ragged, offsets, longueurs,
  strides et pools ;
- duree de vie des buffers, plans, evenements et objets prepares jusqu'a la
  derniere operation asynchrone ;
- absence de data race entre blocs, streams/appels concurrents et portee des
  atomiques, fences et synchronisations ;
- initialisation de tous champs, coefficients, flags et reductions sur chaque
  branche, lot partiel ou absence de candidat ;
- capture des erreurs de lancement/asynchrones au point attribuable ;
- propagation du stream sans synchronisation implicite ni default stream
  accidentel ;
- securite des workspaces reutilises, pools generes et buffers RAII dans les
  exceptions ;
- geometries sanitizer regular/explicit, aligned/cartesian, exact/fixed-step,
  early exercise, samples `12 000 x 250`/`3 000 000 x 1`, rough FFT et rough
  N-facteurs ;
- ecritures sample-major/time-major, lots commencant ou finissant au milieu
  d'un paquet conditionnel, derniers blocs partiels et streaming des sorties.

Un sanitizer non execute reste une exclusion dans `status.md`. Son absence ne
devient pas un finding generique et ne masque pas les preuves statiques.

La section n'est `complete` que si la revue statique couvre tous les launchers
et workspaces actifs, si les huit lignes applicables sont mappees a des cas, si
ASan/UBSan et les quatre modes `compute-sanitizer` passent pour chaque cas, et
si les scenarii de lots/streams/reutilisation sont executes. Une ligne
inapplicable est justifiee par le contrat ; une ligne simplement indisponible
maintient la section partielle.

## Performance

Effectuer quatre audits de performance distincts avec protocole et gates de
ressources communs : CUDA generique, sampling modele, exercice anticipe et
rough. Pression registre, spills, strategie de kernel et cout des abstractions
sont des criteres de premier rang, pas de simples informations d'occupation.

**Perimetre :** runtime CUDA generique, sampling modele, early exercise et
rough FFT/N-facteurs sur architectures et workloads officiellement supportes.

**Hors perimetre :** optimisation sans baseline comparable ni contrat
numerique etabli par Numerical robustness.

**Preuves attendues :** binaires, PTX/cubins, ressources statiques, profils
Nsight, benchmarks repetes, environnement et validation numerique.

**Livrable :** quatre rapports avec baseline, manifeste de workloads,
ressources, mediane/p95/CV, hypothese, resultat et decision.

Un manifeste de workloads unique possede toutes les cles de performance. Pour
chaque cle, il declare engine, binding/specialisation, executable, inputs,
shape, layout, side, schedule, paths/steps/facteurs, profil de tuning,
dependances, metriques requises et budget numerique/ressources/temps. Harness,
baseline et checker consomment cette source ; aucune seconde liste manuelle ne
peut omettre silencieusement un workload. Le checker rejette toute cle
manquante, inconnue, dupliquee, stale ou sans baseline compatible.

Le manifeste separe deux niveaux. Le noyau independant de l'architecture porte
l'identite stable du workload, les inputs, shapes, variantes algorithmiques,
invariants et budgets numeriques contractuels. Un profil par environnement
porte GPU/architecture, toolchain, flags, hashes binaires, parametres
retunables, ressources compilees et baselines de temps. Ajouter un GPU ne
duplique ni ne renomme les cles communes ; comparer deux profils incompatibles
ou laisser un profil redefinir silencieusement le workload est interdit.

Chaque workload produit quatre temps distincts lorsqu'ils s'appliquent, avec
les frontieres exactes du chronometrage consignees :

- **kernel** : events CUDA autour de l'execution device mesuree ;
- **public API** : temps wall a la frontiere de l'appel public, incluant toute
  preparation, allocation, copie ou synchronisation effectuee par cet appel ;
- **pipeline** : job appelant complet, incluant preparation externe, creation
  ou reutilisation du workspace, batching, copies situees hors de l'API et
  recuperation des sorties, mais excluant leur serialisation ;
- **publication** : generation/streaming/serialisation/verification de
  l'artefact, sans l'attribuer au kernel.

Une optimisation recoit exactement une decision `accept`, `reject`,
`inconclusive` ou `unavailable`. `Inconclusive` couvre bruit ou signal trop
faible ; `unavailable` couvre compteur, GPU ou artefact absent. Ces deux etats
ne passent aucun gate et n'autorisent aucune conclusion positive.

### Common performance protocol and kernel strategy

Pour chaque kernel et specialisation representative, consigner :

- strategie thread/warp/bloc par resultat ou trajectoire ;
- decomposition/fusion des phases et dependances qui la justifient ;
- registres par thread, valeurs vivantes, spills, stack et local memory ;
- shared statique/dynamique, banques et synchronisations ;
- blocs/warps residents, occupation theorique et atteinte ;
- taille machine, archives/cubins et risque instruction cache ;
- geometrie grid/block et seuils de changement de strategie ;
- trafic global, coalescence, caches, divergence et melange FP32/FP64 ;
- allocations, copies, streams, evenements, validations et synchronisations ;
- impact de toute factorisation, concept, template ou codegen sur la
  specialisation exacte.

Pour la specialisation et le binaire exacts, distinguer explicitement
registres/thread, spill loads, spill stores, stack frame, tableaux ou memoire
locale, shared statique, shared dynamique, code machine et occupation a la
geometrie lancee. `cudaFuncAttributes.localSizeBytes` seul ne prouve ni
l'absence de spills ni leur cout ; la taille de l'executable ou de l'archive
entiere ne remplace pas la taille SASS/cubin de chaque specialisation de
kernel. Croiser selon le besoin rapports `ptxas`, attributs runtime,
`cuobjdump`/cubin et Nsight Compute pour occupation, stalls, trafic local,
caches, branches, debit memoire et instructions. Tout ecart entre les sources
est explique ; un compteur inaccessible est une limite explicite et fail-closed
s'il est requis par le manifeste.

Avant l'experience, fixer GPU, architecture, compilateur, flags, power limit,
frequences, etat thermique, workload, warmups, repetitions, chronometrage,
traitement des outliers, variabilite maximale et seuil de gain utile. Conserver
baseline et regle d'acceptation avant le resultat.

Executer un preflight de stabilite avant et apres la campagne : identite GPU et
driver, clocks/power, temperature/thermal throttling, processus concurrents,
toolchain, hash des binaires et inputs. Une campagne dont l'environnement sort
des bornes publiees est `inconclusive`, pas nettoyee a posteriori.

Une campagne est une execution complete du manifeste obligatoire dans un
environnement stable. Au moins une campagne complete doit satisfaire les
bornes de stabilite. La methode d'agregation entre campagnes est fixee avant
leur execution et reste non opportuniste : mediane des medianes de campagnes
eligibles, campagne complete centrale ou autre estimateur robuste documente.
Il est interdit de choisir le minimum, le meilleur essai, le meilleur sous-lot
ou de recomposer une baseline en prenant pour chaque cle la campagne la plus
favorable. Les runs rejetes et les valeurs brutes restent conserves avec leur
motif ; une exclusion d'outlier suit uniquement la regle predeclaree.

Les budgets sont fail-closed et portent au minimum sur : tolerance numerique,
registres/thread, spills, stack/local memory, shared, occupation/residence,
taille cubin/code, VRAM, mediane, p95 et CV. Chaque budget donne valeur absolue
et/ou delta autorise ainsi que sa raison. Une metrique requise manquante ou
`unknown` fait echouer le gate du workload ; elle n'est jamais interpretee
comme zero ou `unchanged`.

Le budget numerique reutilise le contrat independant etabli par Numerical
robustness et est fige avant la campagne ; il n'est jamais calcule depuis la
sortie observee ou depuis la candidate a accepter. Les budgets de ressources
et de temps peuvent etre exprimes relativement a une baseline compatible,
mais leur seuil et leur justification precedent eux aussi la mesure candidate.

La VRAM est ventilee au minimum entre inputs persistants, workspace possede par
l'appelant, allocations transitoires de l'appel, sorties et marge. Le pic
d'allocations possedees pendant l'appel est mesure, pas extrapole depuis le
seul etat final. Contexte/driver, cache de plans ou pools persistants et usage
d'autres processus sont releves separement afin de ne pas etre attribues au
kernel. Un simple delta `total-free` apres l'appel ne mesure ni le pic ni
l'ownership et ne peut servir que de controle de fuite complementaire.

Un rebaselining est une operation versionnee et revue, jamais une correction
a lui seul. Il conserve l'ancienne baseline, son hash et son environnement,
les sorties brutes candidates, puis produit un diff exhaustif ancien/nouveau.
Chaque budget depasse ou relache indique la cause, l'effet numerique et
ressources, la famille concernee, la justification et l'approbation explicite.
Executer le checker contre la baseline que l'on vient de remplacer ne prouve
pas l'absence de regression et ne clot aucun constat par construction. Une
premiere initialisation sans predecessor est nommee comme telle et ne pretend
pas mesurer un delta ; toute mise a jour ulterieure suit la procedure complete.

Les baselines et geometries de reference livrees depuis RTX 4090 Laptop/SM89
ne s'appliquent qu'a cet environnement. Pour tout autre GPU, executer le meme
manifeste, inspecter ressources et limites, retuner les parametres explicitement
classes comme tels, puis publier une baseline distincte. Aucun gate ne compare
des environnements incompatibles.

Interdire `--use_fast_math`. Ne pas ajouter `__launch_bounds__` sans design
architecture-aware et mesures sur chaque architecture cible. Une hausse de
registres, apparition de spills ou baisse d'occupation ne peut etre masquee par
une mediane unique.

### Generic CUDA performance

Auditer formules fermees, Monte-Carlo markovien, fonctions device, kernels
communs, launchers, layouts et echanges host/device. Le sampling modele possede
en plus son audit dedie ci-dessous.

Verifier notamment :

- inlining/noinline, taille code et instruction cache ;
- invariants dans les boucles et preparation host/device ;
- emploi de `fmaf`, `expm1f`, `log1pf`, `sincosf` et primitives stables sans
  approximation globale ;
- FP32 chaud et FP64 seulement selon contrat numerique ;
- index device compact, divisions/modulos 64 bits et promotions d'adresse ;
- geometries mesurees pour policies simples/riches et closed form
  scalaire/cooperatif ;
- AoS/SoA, alignement, dimensions, schedules ragged et coalescence ;
- divergence des branches, calendriers, rejet, sauts et arrets iteratifs ;
- reductions, tableaux locaux et packs de valeurs par thread ;
- consommation Philox utile et absence de chevauchement ;
- cout fixe des launchers, batching, reutilisation sure et synchronisations ;
- comparaison un-thread, un-warp ou un-bloc par resultat ;
- gates registres, spills, stack, shared, code size, mediane, p95 et numerique.

### Model-sample performance

Auditer chaque engine de sampling disponible sur les deux layouts contractuels
et separer cout kernel, transferts, generation host, streaming JSON/YAML et
validation de l'artefact.

Verifier notamment :

- thread grid-stride pour `3 000 000 x 1` contre parameter-block pour
  `12 000 x 250`, ainsi que les seuils ou une autre strategie devient meilleure ;
- amortissement de la preparation modele par parametre et cout des calendriers
  aleatoires par sample ;
- registres, spills, stack, shared, occupation, blocs residents et divergence
  pour transition exacte, pas fixe, prepare N-facteurs et Volterra FFT ;
- geometries `threads_per_block`/block count, saturation selon nombre de SM et
  absence de plafond derive uniquement des 76 SM du GPU de reference ;
- debit en samples/s, kernel median/p95/CV et cout wall de publication, sans
  confondre serialisation host et simulation device ;
- taille et reutilisation des buffers de parametres, maturites et observables,
  copies D2H, pinned memory eventuelle et marge VRAM/RAM ;
- batch/chunk sizes, invariance des sorties, effet sur reduction inexistante et
  possibilite de traiter un GPU avec moins de memoire ;
- pour Volterra, longueurs FFT, elements-per-thread, FFTs-per-block, shared
  opt-in, nombre de paths conditionnels et reutilisation du spectre ;
- pour N-facteurs, croissance des valeurs vivantes et ressources avec le
  nombre de facteurs ;
- profils de reference etiquetes par architecture et procedure de retuning
  identique a celle des pricers.

Le smoke test valide le pipeline mais n'est pas une baseline de debit. Une
campagne de performance doit conserver les deux shapes de production ou une
reduction explicitement justifiee qui preserve le regime de saturation et la
pression memoire.

### Early-exercise performance

Auditer Longstaff-Schwartz sur au moins un modele equity un facteur, un equity
multi-etats, un taux un facteur et un taux deux facteurs. Couvrir prix, paths,
dates, variables de continuation et blocs/prix.

Verifier notamment :

- arithmetique, offsets, alignements et overflow de `WorkspaceLayout` ;
- budget VRAM/marge de `ExecutionPlan`, y compris un prix qui ne tient pas ;
- tous champs SoA, cashflows, coefficients, reductions, etats et temporaires ;
- allocations, copies, evenements et synchronisations par batch, jamais par
  date sans preuve ;
- absence d'aller-retour CPU dans backward ;
- stockage SoA coalescent et cout des schedules heterogenes ;
- decomposition forward, accumulation, resolution, decision et reduction ;
- lancements et arbitrage fusion/separation selon registres/shared/sync ;
- evaluation unique des bases et absence de tableaux locaux avec spills ;
- FP64 limitee aux equations/regressions sensibles ;
- reductions inter-blocs, atomiques et shared sans serialiser les prix ;
- mesures `threads_per_block`/`blocks_per_price` selon base et paths ;
- batching des dates heterogenes seulement si le gain couvre la permutation ;
- invariance des prix/erreurs selon decoupage lorsque contractuelle ;
- exercice immediat/terminal, limites naturelles et convergence de grille.

Produire par specialisation workspace/prix, VRAM, registres, spills, stack,
shared, occupation, lancements et temps forward/backward/reduction. Distinguer
cout fixe, debit asymptotique et saturation.

### Rough performance

Auditer les deux strategies rough puis leurs composants communs.

Pour FFT/Volterra, verifier notamment :

- crossover direct/hybride/FFT a precision/discretisation equivalentes ;
- longueurs FFT, padding, batch, layouts complexes et zeros inutiles ;
- fusion/separation generation, convolution, reconstruction, integration,
  payoff et reduction ;
- registres, shared cuFFTDx, spills, occupation et blocs residents par
  architecture ;
- transpositions, strides, coalescence et trafic VRAM ;
- reutilisation des coefficients, plans et workspaces ;
- chunking, streams et budget memoire ;
- synchronisations, barrieres et copies intermediaires ;
- Philox, correlations, precision, aliasing et convergence directe ;
- isolation des specialisations et erreur explicite pour taille non supportee.

Pour les lifts rough N-facteurs, verifier notamment :

- cout de preparation/evaluation des facteurs exponentiels ;
- croissance registres, stack, instructions et temps avec les facteurs ;
- stockage compile-time, register/shared/global ;
- reutilisation markovienne sans branche ni objet surdimensionne ;
- geometrie selon facteur count, paths et produits ;
- compromis nombre de facteurs, erreur d'approximation et debit ;
- absence d'inflation des kernels markoviens sans rapport.

Produire courbes temps/debit selon pas ou facteurs, decomposition des phases et
table registres/shared/spills/occupation/VRAM/erreur. Mesurer aussi les
composants rough communs afin qu'une factorisation ne penalise ni FFT ni
N-facteurs.

Pour les quatre audits, une mediane seule ou un gain inferieur au bruit ne
suffit jamais. Toute optimisation retenue conserve contrat numerique, mapping
aleatoire et baselines des artefacts affectes.

La section n'est `complete` que si le manifeste couvre toutes les familles et
strategies officiellement actives, si chaque workload a une baseline de meme
environnement, si tous les budgets requis ont une mesure, si kernel/public API/
pipeline/publication sont separes, et si chaque hypothese a l'une des quatre
decisions. Une campagne partielle, un compteur indisponible ou un workload non
construit reste visible et empeche la certification du sous-audit concerne.
