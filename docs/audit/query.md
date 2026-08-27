# Referentiel des audits d'architecture C++/CUDA

Version du referentiel : 3, 2026-08-27.

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
performance, des artefacts compiles. Les noms de dossiers et commentaires ne
constituent pas seuls une preuve de responsabilite ou de factorisation. Il ne
propose aucune abstraction sans identifier ses consommateurs reels.

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

## Sommaire

- [Structure, conventions and ownership](#structure-conventions-and-ownership)
  - [Repository tree and domain ownership](#repository-tree-and-domain-ownership)
  - [File responsibilities and granularity](#file-responsibilities-and-granularity)
  - [Naming and semantic conventions](#naming-and-semantic-conventions)
  - [Dependency boundaries](#dependency-boundaries)
  - [Cleanup and extension locality](#cleanup-and-extension-locality)
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
  - [Parameter dataset generation](#parameter-dataset-generation)
  - [Regeneration, drift and exceptions](#regeneration-drift-and-exceptions)
- [CMake and build graph](#cmake-and-build-graph)
- [Numerical robustness and reproducibility](#numerical-robustness-and-reproducibility)
- [CUDA execution and memory safety](#cuda-execution-and-memory-safety)
- [Performance](#performance)
  - [Common performance protocol and kernel strategy](#common-performance-protocol-and-kernel-strategy)
  - [Generic CUDA performance](#generic-cuda-performance)
  - [Early-exercise performance](#early-exercise-performance)
  - [Rough performance](#rough-performance)

## Structure, conventions and ownership

Auditer l'arborescence, les responsabilites, les noms et les frontieres de
dependances comme un contrat de lisibilite architecturale. Cette section dit
ou un composant doit vivre et comment il doit se nommer ; elle ne conclut pas
qu'une implementation doit etre partagee, ce qui releve de Factorization.

**Perimetre :** dossiers, fichiers, namespaces, types, fonctions, variables,
targets, identifiants de catalogue, includes et liens entre domaines.

**Hors perimetre :** differences d'API entre composants homologues et
duplication d'implementation, sauf si elles prouvent un mauvais ownership.

**Preuves attendues :** arborescence complete, graphes d'includes et de liens,
inventaire des conventions, consommateurs, tailles des fichiers et nombre de
lieux modifies par une extension representative.

**Livrable :** carte des domaines et dependances, contradictions de nommage ou
placement, elements orphelins et localite mesuree d'un ajout modele/produit.

### Repository tree and domain ownership

Verifier notamment :

- que `src/common`, `src/model`, `src/curve`, `src/product`, `src/generative`,
  `tools`, `catalog`, `validation`, `tests`, `docs` et le site ne portent que
  leurs responsabilites declarees ;
- que `src/model/equity/markovian`, `src/model/equity/rough` et
  `src/model/fixed_income` classent les modeles par famille mathematique et non
  par commodite du moteur actuellement employe ;
- que les dossiers modele restent centres sur parametres, datasets, dynamics,
  analytics, sampling et primitives propres, sans devenir proprietaires de
  tous les produits ;
- que contrats produit, schedules, payoffs, pricing policies et continuation
  states restent sous `src/product` lorsqu'ils ne dependent pas d'un modele ;
- que les primitives mathematiques ou d'execution reellement neutres vivent
  sous `src/common`, sans type concret de modele, courbe ou produit ;
- que `tools` possede sampling offline, orchestration, serialisation,
  publication et codegen, tandis que `catalog` contient les recettes et
  metadonnees reproductibles ;
- que `validation` ne devient pas une dependance du runtime ou des generateurs
  ordinaires ;
- que la structure des modeles markoviens, rough, fixed income, courbes et
  produits est symetrique pour les responsabilites communes ;
- que toute difference d'arborescence est justifiee par mathematique,
  dependance optionnelle ou frontiere de compilation, et non par l'historique.

### File responsibilities and granularity

Verifier notamment :

- que chaque fichier porte une responsabilite identifiable et des
  consommateurs reels ;
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
- que toute fusion ou scission CUDA mesure compilation, incrementalite, taille
  objets/cubins et ressources kernel.

### Naming and semantic conventions

Auditer dossiers, fichiers, namespaces, types, concepts, fonctions, arguments,
variables, constantes, tests, targets et identifiants de catalogue. Le nom
indique la responsabilite et le domaine reels, pas l'anciennete.

Verifier notamment :

- `snake_case` pour chemins, fichiers, fonctions et variables, `PascalCase`
  pour types et concepts, et une convention unique pour les constantes ;
- la correspondance arborescence/namespaces, sans namespace plat sous un
  dossier specialise ni niveau sans responsabilite ;
- singulier/pluriel coherent pour definition, famille, collection, dataset et
  variante de prix ;
- des noms de fichiers homologues stables : `parameters`, `dataset`,
  `dynamics`, `analytics`, `concepts`, `pricing_policy`, `schedule`, `state`,
  `sample`, `kernel`, `launch` et `workspace` ;
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
  catalogue, validation, diagnostics et documentation ;
- que les identifiants publies ne changent pas sans migration explicite.

Seules les contradictions semantiques prouvees deviennent des constats. Les
preferences stylistiques sans impact sont exclues.

### Dependency boundaries

La direction conceptuelle attendue est :

```text
common <- model / curve / product <- tools <- catalog
                               \-> tests
                               \-> validation
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
plusieurs listes CMake, bindings, recettes, registres de tests et validations
concurrents. Cette mesure alimente Code generation and extension cost.

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

### Dynamics

Auditer `parameters`, `state`, `dynamics.cuh` et `dynamics_impl.cuh` de tous les
modeles selectionnes, en separant transitions exactes, pas fixe, rough FFT et
lifts rough N-facteurs dans la couverture.

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

### Minimum hand-written model and product

Pour un nouveau modele, verifier que le minimum manuel est limite a :

- parametres compacts, loader et schema ;
- dynamics ou analytics mathematiques propres ;
- primitives de preparation et observables specifiques ;
- specification de generation des parametres core/stress ;
- tests mathematiques et numeriques ;
- declaration de capacites et dependances.

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
  etat, dependances et architectures ;
- `ProductSpec` : schedule, observation, exercice, policy, sidedness,
  parametres et capacites requises ;
- `EngineSpec` : concepts, templates, launchers, runners, dependances et
  strategie d'instanciation ;
- `DatasetSpec` : sources de parametres, aligned/cartesian, chemins catalogue,
  profils numeriques et metadonnees derivables.

Le resolver produit explicitement :

```text
(model, product, variant) -> engine -> bindings -> target -> catalogue recipe
```

Verifier notamment :

- qu'une incompatibilite est refusee par resolver ou concept lisible ;
- qu'aucune deuxieme liste manuelle ne recopie models, products, sides,
  schedules, engines ou dependances ;
- que transition exacte, pas fixe, rough FFT, rough N-facteurs, closed form et
  early exercise sont representables sans champ ambigu ;
- que les exceptions algorithmiques sont nommees, bornees et testees ;
- que le manifeste ne redeveloppe pas les algorithmes C++ dans Python.

### Generated bindings and catalogue recipes

Verifier que les champs derivables produisent :

- `<model>/<product>.cuh` et `<model>/<product>.cu` ;
- specialisations call/put, payer/receiver ou non-sided ;
- schedule, policy et engine ;
- instanciations et fragment CMake ;
- generateur executable de catalogue ;
- `dataset.yaml` ou ses champs structurels derivables ;
- tests de compilation et checkers d'architecture ;
- squelette de validation seulement sans choix arbitraire d'une reference.

Le catalogue actuel utilise `generator.cpp`. Le referentiel emploie le terme
neutre **generateur de catalogue** : un passage a Python doit etre une decision
explicite, pas une coexistence accidentelle de deux formats.

Verifier que profils de chemins, trajectoires, time grids, seeds, batching et
timings restent explicites dans la source de verite ou une specification
versionnee. Aucun choix numerique important ne doit etre cache par default.

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
- builds avec/sans mathDx/cuFFTDx et architectures declarees ;
- tests de chaque template, resolver et engine, cas valides/invalides ;
- escape hatch explicite avec proprietaire, justification et checker ;
- suppression des outputs supersedes ;
- incrementalite, taille des unites CUDA et diagnostics lisibles preserves.

Mesurer pour une extension representative : fichiers manuels, lignes
semantiques, entrees de manifeste, commandes de generation et modifications
residuelles. Reduire le nombre de fichiers generes n'est pas un objectif si le
code manuel augmente ou si la matrice de capacites devient opaque.

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
  necessaires ;
- que mathDx/cuFFTDx et dependances optionnelles n'affectent pas les autres
  targets ;
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

Verifier notamment :

- une politique de precision explicite : etats/calculs device FP32 par defaut,
  FP64 pour reductions, moments, regressions, factorisations ou operations
  sensibles ;
- que chaque FP64 chaud est justifie et chaque remplacement FP32 mesure ;
- domaines de `log`, `log1p`, `sqrt`, `pow`, divisions, exponentielles,
  fonctions de repartition et inversions, y compris zero, negatif, infini et
  sous-normal ;
- positivite, frontieres absorbantes et contraintes sans projection
  silencieuse changeant la loi ;
- stabilite pour petits temps, faible volatilite/mean reversion, correlations
  aux bornes, intensites extremes et limites d'admissibilite ;
- solveurs : encadrement, monotonie, convergence, iterations max, arret et
  comportement d'echec ;
- conditionnement Longstaff-Schwartz, normalisation, regularisation, Cholesky
  et biais ;
- stabilite/ordre des reductions, cancellation et reproductibilite attendue ;
- conventions temporelles, accruals, unites de taux et calendriers aux
  frontieres host/device ;
- propagation explicite des erreurs et absence de NaN/Inf silencieux dans
  prix, erreurs standards, coefficients et datasets ;
- convergence de pas, nombre de facteurs rough et approximations FFT/directes ;
- tests de limites, identites, sensibilite et reference independante.

Ne pas demander l'egalite bit a bit entre algorithmes equivalents si elle
n'est pas contractuelle. Distinguer reproductibilite d'un meme chemin,
tolerance numerique et convergence statistique.

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
matrice representative.

**Livrable :** erreurs de bornes, duree de vie, initialisation, concurrence ou
propagation prouvees, cas minimal et familles non executees.

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
  early exercise, rough FFT et rough N-facteurs.

Un sanitizer non execute reste une exclusion dans `status.md`. Son absence ne
devient pas un finding generique et ne masque pas les preuves statiques.

## Performance

Effectuer trois audits de performance distincts avec protocole et gates de
ressources communs. Pression registre, spills, strategie de kernel et cout des
abstractions sont des criteres de premier rang, pas de simples informations
d'occupation.

**Perimetre :** runtime CUDA generique, early exercise et rough FFT/N-facteurs
sur architectures et workloads officiellement supportes.

**Hors perimetre :** optimisation sans baseline comparable ni contrat
numerique etabli par Numerical robustness.

**Preuves attendues :** binaires, PTX/cubins, ressources statiques, profils
Nsight, benchmarks repetes, environnement et validation numerique.

**Livrable :** trois rapports avec baseline, manifeste de workloads,
ressources, mediane/p95/CV, hypothese, resultat et decision.

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

Utiliser `ptxas` ou cubin pour les ressources statiques et Nsight Compute pour
occupation, stalls, local memory, caches, branches, debit memoire et
instructions. Un compteur inaccessible est une limite explicite.

Avant l'experience, fixer GPU, architecture, compilateur, flags, power limit,
frequences, etat thermique, workload, warmups, repetitions, chronometrage,
traitement des outliers, variabilite maximale et seuil de gain utile. Conserver
baseline et regle d'acceptation avant le resultat.

Interdire `--use_fast_math`. Ne pas ajouter `__launch_bounds__` sans design
architecture-aware et mesures sur chaque architecture cible. Une hausse de
registres, apparition de spills ou baisse d'occupation ne peut etre masquee par
une mediane unique.

### Generic CUDA performance

Auditer formules fermees, Monte-Carlo markovien, sampling, fonctions device,
kernels communs, launchers, layouts et echanges host/device.

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

Pour les trois audits, une mediane seule ou un gain inferieur au bruit ne
suffit jamais. Toute optimisation retenue conserve contrat numerique, mapping
aleatoire et baselines des artefacts affectes.
