# Liste des audits d'architecture CUDA

## Objet

Ce document contient la liste stable des audits a effectuer sur l'architecture
CUDA du projet. Il formule les questions a poser et les invariants a verifier ;
il ne contient aucun resultat d'audit.

Les constats encore ouverts sont inscrits dans `audit-response.md`, sous la
section de meme nom et dans le meme ordre. Chaque constat doit citer une preuve
dans le code, les mesures disponibles et un critere de cloture. Lorsqu'un
constat est corrige et verifie, il est retire de `audit-response.md`.

Avant de retirer un constat, transferer toute decision durable vers le contrat
d'implementation concerne. Une optimisation mesuree puis rejetee doit etre
documentee dans `abandoned-work.md`; un travail volontairement reporte doit
etre place dans `deferred-work.md`. `audit-response.md` reste ainsi une liste
de problemes ouverts, et non un historique.

Un audit doit distinguer explicitement :

- les incoherences prouvees par le code ou les artefacts compiles ;
- les risques qui necessitent une mesure pour etre confirmes ;
- les choix deja satisfaisants qu'une correction doit preserver ;
- les exceptions mathematiques necessaires, avec leur justification.

## Dynamics

Auditer les fichiers `parameters`, `state`, `dynamics.cuh` et `dynamics.cu` de
tous les modeles concernes. Les modeles rough, dont l'ossature est differente
par construction, doivent etre analyses separement lorsqu'ils entrent dans le
perimetre.

Verifier notamment :

- l'homogeneite des namespaces, avec la forme
  `workbench::model::<asset_class>::<model>` ;
- la presence et l'homogeneite des gardes d'inclusion, des includes et des
  dependances entre couches ;
- la meme separation entre declarations, definitions, parametres bruts,
  modele prepare, transition preparee, dynamique preparee et etat mutable ;
- l'emploi coherent des noms canoniques `ModelParameters` ou
  `ProcessParameters`, `PreparedModel`, `PreparedTransition`,
  `PreparedDynamics` et `State` ;
- le meme ordre de declaration et de definition des types, helpers et
  fonctions publiques dans chaque modele ;
- l'homogeneite des noms, types de retour, qualificateurs CUDA, ordre des
  arguments, passage par valeur ou reference et conventions `const` ;
- l'interface commune de preparation du modele, preparation des transitions,
  creation de l'etat initial, avance d'un pas et extraction des observables ;
- la coherence entre processus exacts, schemas a pas fixe et wrappers de
  processus, sans imposer une structure artificielle lorsque la mathematique
  exige une exception ;
- la reutilisation des briques communes et l'absence de duplication entre un
  processus de base et ses compositions ou extensions ;
- l'ordre et la stabilite de consommation des nombres aleatoires, notamment
  pour les modeles a sauts et les transitions exactes ;
- l'usage coherent de `std::uint32_t`, `std::size_t`, `float` et des structures
  d'etat suivant leur role ;
- la symetrie des tests de contrat, de compilation, de moments, de cas limites
  et de consommation aleatoire entre les modeles.

Signaler toute difference d'ossature. Pour chacune, determiner si elle est une
incoherence a corriger ou une exception mathematique a documenter.

## Analytics

Auditer les analytics Black-Scholes et fixed income, leurs providers, concepts,
wrappers, kernels closed form et tests.

Verifier notamment :

- que les formules partagees par les modeles de meme famille sont factorisees
  au niveau le plus bas qui possede reellement les invariants necessaires ;
- que les providers ne contiennent que les primitives partageables et que les
  formules propres a un modele ou a un produit restent dans leur couche ;
- l'homogeneite des namespaces, noms de fichiers, noms de fonctions, ordre des
  declarations et definitions, types de retour, ordre des arguments,
  qualificateurs CUDA et conventions `const` ;
- l'existence d'une signature canonique pour chaque primitive commune : bond,
  coefficient affine, taux, option de taux, valeur de swap payer/receiver et
  formule closed form applicable ;
- que les compositions, notamment les modeles ajustes a une courbe ou les
  extensions multi-facteurs, reutilisent les analytics du processus de base
  sans les redevelopper ;
- l'absence de duplication de formules entre les modeles, les produits, les
  launchers et les tests ;
- la stricte utilisation des providers et concepts par tous les chemins, sans
  bypass ponctuel de l'ossature commune ;
- la proprete des dependances et l'absence d'include d'un produit ou d'un
  dataset dans une couche analytique inferieure ;
- la coherence semantique des unites et conventions, notamment spot contre
  ratio normalise, temps absolu contre maturite residuelle, payer contre
  receiver et signe du nominal ;
- la symetrie des tests entre modeles et produits : API, compilation des
  concepts, identites mathematiques, cas limites, references independantes et
  invariance lorsque le contrat l'exige.

Toute factorisation proposee doit expliciter la responsabilite du provider et
eviter de creer une interface commune qui ne serait satisfaite que par des
adaptateurs artificiels.

## Naming

Auditer ensemble les noms de dossiers, fichiers, namespaces, types, concepts,
fonctions, arguments, variables, constantes, tests, targets et identifiants de
catalogue. Le nom doit indiquer la responsabilite reelle, pas l'historique de
creation de l'element.

Verifier notamment :

- l'emploi systematique de `snake_case` pour les chemins, fichiers, fonctions
  et variables, de `PascalCase` pour les types et concepts, et de la convention
  choisie pour les constantes ;
- la correspondance entre l'arborescence et les namespaces, en evitant a la
  fois les namespaces plats sous des dossiers specialises et les niveaux de
  namespace sans responsabilite propre ;
- l'usage coherent du singulier et du pluriel pour une definition, une famille,
  une collection, un dataset et une variante de prix ;
- que l'extension d'un fichier decrit son mode de compilation : header public,
  implementation incluse, unite de traduction hote ou unite CUDA compilee ;
- que les noms de fichiers homologues suivent une meme convention, notamment
  `concepts`, `kernel`, `pricing_policy`, `schedule`, `state`, `sample`,
  `analytics`, `dynamics`, `launch` et `workspace` ;
- que les symboles qualifies ne repetent pas inutilement leur modele, courbe,
  produit ou methode deja portes par le namespace ;
- que les types internes a un namespace de modele ne repetent pas le nom du
  modele et utilisent les noms canoniques etablis par les contrats ;
- que les fonctions emploient un verbe stable suivant leur responsabilite :
  `load`, `validate`, `prepare`, `compose`, `simulate`, `advance`, `evaluate`,
  `price`, `launch`, `write` ou `generate` ;
- que deux fonctions equivalentes utilisent le meme nom, le meme ordre
  d'arguments et les memes noms d'arguments dans les declarations,
  definitions, wrappers, tests et generateurs ;
- que les variables indiquent leur unite et leur domaine lorsque le type ne le
  fait pas : jours, annees, pas, indices, offsets, nombres de lignes, paths,
  prix, taux et fractions d'accrual ;
- que les booleens se lisent comme des predicats et qu'un enum remplace un
  booleen lorsque celui-ci selectionne un mode de construction ou d'execution ;
- que les symboles mathematiques d'une lettre restent limites aux formules
  locales evidentes ; les valeurs transportees entre plusieurs blocs utilisent
  des noms semantiques ;
- que les noms temporels tels que `new`, `old`, `legacy`, `temporary` ou
  `historical` ne survivent pas dans une API ou un test permanent ;
- que les renommages sont atomiques dans les headers, implementations, tests,
  CMake, generateurs, validations, diagnostics et documentation ;
- que les identifiants publies de datasets et URLs ne sont pas renommes sans
  une migration explicite et verifiee.

Le resultat doit separer les noms objectivement contradictoires des preferences
de style. Toute nouvelle convention retenue doit pouvoir etre verifiee par une
recherche ou un test statique.

## Project structure

Auditer l'arborescence du projet en meme temps que le nommage. Pour chaque
fichier ou dossier, identifier son proprietaire conceptuel, ses dependances et
la raison de sa frontiere de compilation. Rechercher les responsabilites mal
placees, les duplications, les fragments devenus inutiles et les fichiers trop
larges ou trop fins.

Verifier notamment :

- que `src/model`, `src/curve`, `src/product`, `src/common`, `src/generative`,
  `tools`, `catalog`, `validation` et `tests` ne portent que leurs
  responsabilites declarees ;
- qu'une politique de pricing dependant d'un contrat produit appartient a la
  couche produit, tandis que les primitives mathematiques ou d'execution
  reellement independantes restent dans `common` ;
- que les dossiers de modele restent centres sur parametres, dataset,
  dynamique, analytics, sampling et compositions necessaires, sans devenir une
  matrice manuelle de tous les produits ;
- que les wrappers modele-produit repetitifs sont remplaces par une
  factorisation typee ou une generation declarative lorsque cela conserve des
  erreurs de compilation lisibles et des unites CUDA de taille maitrisee ;
- que les moteurs American/Bermudan, sampling, closed form et Monte-Carlo ne
  sont pas recopies par modele lorsque seules une policy et la mathematique du
  modele changent ;
- que les compositions modele-courbe reutilisent une implementation parametree
  plutot que deux copies Nelson-Siegel/Svensson quasi identiques ;
- que les implementations textuellement incluses sont rangees et nommees comme
  telles, sans simuler une unite de traduction CUDA autonome ;
- que les headers de parametres device restent compacts et independants des
  loaders hote ; centraliser le parsing commun ne doit pas fusionner ces deux
  couches ;
- que les fichiers communs volumineux sont separes par responsabilite lorsque
  leurs parties evoluent independamment, sans creer des micro-fichiers ne
  portant aucune abstraction ;
- que les generateurs de catalogue partagent les allocations RAII, copies,
  warmups, evenements, batching, chronometrage et ecriture des artefacts, et ne
  repetent que leur configuration specifique ;
- que le CMake racine orchestre le projet mais delegue les manifestes de
  targets par domaine au lieu d'accumuler toutes les recettes ;
- que les scripts de validation repetitifs sont remplaces par un CLI ou un
  registre declaratif, tout en conservant les points d'entree publies qui sont
  encore contractuels ;
- que chaque chemin de notebook, rapport, cache et dataset pointe vers un
  artefact existant et conforme au workflow de validation actuel ;
- que les dossiers vides, prototypes non integres, couches de compatibilite,
  READMEs locaux non maintenus et code sans consommateur sont soit integres,
  soit supprimes apres verification des usages externes ;
- que toute fusion de fichiers CUDA compare le temps de compilation,
  l'incrementalite, la taille des objets/cubins, les registres et les temps
  kernel avant d'etre retenue.

Une proposition de nouvelle arborescence doit minimiser simultanement la
duplication, le nombre de lieux a modifier pour ajouter un modele-produit et la
taille des unites de traduction. Ne pas fusionner des fichiers uniquement pour
reduire leur nombre.

## Tools and src ownership

Auditer explicitement la frontiere entre `src` et `tools`. La direction de
dependance normale est `tools -> src` : `src` fournit les bibliotheques
numeriques et d'acces aux donnees reutilisables, tandis que `tools` fabrique,
valide et publie des artefacts offline. Une cible de `src` ne doit pas dependre
d'un header, d'une bibliotheque ou d'une convention de catalogue situes dans
`tools`.

Verifier notamment :

- qu'aucun fichier de `src` n'inclut `tools` et qu'aucune bibliotheque runtime
  ou publique ne lie transitivement une cible de generation ;
- que les include roots et les dependances CMake rendent cette direction
  impossible a violer accidentellement, au lieu de seulement la documenter ;
- que les parametres compacts, loaders publics, validations necessaires a la
  lecture, dynamics, analytics, kernels, launchers et primitives de layout
  reutilisables restent dans `src` ;
- que les recettes de sampling, seeds de publication, bornes core/stress,
  generation de lignes, ecriture JSON/YAML, URLs, chronometrage de generateur,
  CLI et codegen restent dans `tools` ou `catalog` ;
- qu'une formule mathematique deterministe utilisee par le runtime et par un
  generateur possede une seule definition dans le domaine correspondant de
  `src`, avec une surface host/device si necessaire ; le RNG, le rejet et les
  metadonnees de construction restent dans `tools` ;
- que les types de mode, conventions temporelles, calculs de cardinalite et
  mappings d'indices consommes par les APIs de pricing appartiennent a `src`,
  meme si leur premier consommateur est un generateur ;
- que la validation de schema necessaire aux loaders de `src` est separee des
  controles propres a la publication d'un artefact ou d'un catalogue ;
- que les helpers CUDA de diagnostic restent dans `src` lorsqu'ils font partie
  du comportement opt-in des launchers ; un outil d'analyse autonome ou un
  export de rapport reste dans `tools` ;
- qu'un faible nombre de consommateurs internes ne suffit pas a deplacer une
  API numerique publique vers `tools`, et qu'a l'inverse le mot `reusable` ne
  transforme pas une recette offline en bibliotheque runtime ;
- que les fichiers de `tools` ne melangent pas sampling de parametres,
  orchestration CUDA, assemblage des resultats, serialisation et publication
  lorsqu'ils peuvent evoluer ou etre testes independamment ;
- que les facades de compatibilite, targets sans consommateur, dossiers
  placeholders et prototypes non integres sont supprimes apres verification
  des usages externes ;
- que toute proposition de deplacement fournit la table source/destination,
  les dependances avant/apres, les consommateurs migres et les tests de
  frontiere necessaires.

Pour chaque fonction partagee, separer la mathematique pure de sa politique de
generation. Ne pas deplacer un generateur complet dans `dynamics` uniquement
parce qu'il reconstruit un parametre du modele ; seule la primitive
mathematique canonique appartient alors a `src`.

## Performance

Auditer les fonctions device, les kernels, les templates communs, les
launchers, les layouts memoire et les echanges hote-device. Effectuer l'audit
sur les architectures CUDA cibles et sur des jeux de donnees representatifs,
en separant closed form, Monte-Carlo, sampling, early exercise et modeles
numeriquement atypiques.

Verifier notamment :

- les flags de compilation, l'inlining, la taille du code machine et les
  risques de pression sur l'instruction cache ;
- l'emploi pertinent de `fmaf`, `expm1f`, `log1pf`, `sincosf` et des primitives
  numeriquement stables, sans appliquer globalement `--use_fast_math` ni des
  approximations rapides non validees ;
- la presence de calculs invariants dans les boucles de trajectoire et leur
  deplacement eventuel dans `PreparedModel`, `PreparedTransition` ou
  `PreparedRow` ;
- le choix des types : minimiser FP64 dans les chemins chauds tout en
  conservant la precision requise pour les reductions, moments, regressions et
  distributions sensibles ;
- l'emploi de types d'index compacts sur le device, les divisions et modulos
  entiers 64 bits, les promotions necessaires pour les adresses et le maintien
  de compteurs 64 bits lorsque l'espace global l'exige ;
- les registres par thread, spills, frames de stack locale, valeurs vivantes,
  occupation theorique et occupation atteinte pour chaque specialisation ;
- la geometrie des blocs et grilles, avec comparaison mesuree de plusieurs
  tailles de bloc plutot qu'une valeur globale supposee optimale ;
- la quantite et l'organisation de la shared memory, les synchronisations, les
  broadcasts, les reductions et les conflits de banques ;
- la contiguite, l'alignement et la coalescence des acces globaux, ainsi que le
  choix AoS/SoA, l'ordre des dimensions, les strides et les schedules ragged ;
- la divergence due aux branches, calendriers variables, algorithmes de rejet,
  sauts et criteres d'arret iteratifs ;
- le cout des reductions generiques et la pression registre induite par les
  tableaux ou packs de valeurs conserves par chaque thread ;
- la qualite de Philox, la reutilisation des tirages, l'absence de recalculs et
  la stabilite du mapping entre resultat, trajectoire et sequence aleatoire ;
- les copies, allocations, validations de pointeurs, requetes de proprietes du
  device, synchronisations, streams et possibilites de reutilisation des
  buffers ;
- la pertinence de kernels generiques un-thread-par-resultat ou
  un-bloc-par-resultat selon la quantite de travail et les reductions internes ;
- les seuils de regression portant sur les registres, spills, stack, shared,
  taille du code, temps median et p95, ainsi que la validation numerique de
  toute optimisation.

Pour chaque point mesure, consigner le GPU, l'architecture, le compilateur, les
flags, la geometrie, la taille du jeu de donnees, les warmups et la methode de
chronometrage. Utiliser PTXAS/cubin pour les ressources statiques et Nsight
Compute pour l'occupation atteinte, les stalls, les acces locaux, les caches,
les branches, le debit memoire et le melange d'instructions FP32/FP64.
