# Referentiel des audits d'architecture C++/CUDA

Version du referentiel : 2, 2026-08-27.

## Objet

Ce document contient la liste stable des audits a effectuer sur l'architecture
C++/CUDA du projet. Il formule les questions a poser et les invariants a
verifier ; il ne contient aucun resultat d'audit.

`status.md` conserve la date, la revision, le perimetre, les exclusions et
les preuves du dernier passage de chaque audit, qu'il ait produit ou non un
constat. Une section vide de `response.md` ne doit donc jamais servir a
deduire qu'un audit est recent ou complet.

La provenance doit identifier a la fois la version de ce referentiel et le
contenu reellement audite. Sur un worktree propre, la revision Git suffit. Sur
un worktree modifie, `status.md` conserve aussi l'etat porcelain, un manifeste
des fichiers non suivis et une empreinte du diff suivi; si le contenu non suivi
entre dans le perimetre, son manifeste doit egalement porter une empreinte de
contenu. Le passage initial et l'etat apres remediation sont deux snapshots
distincts. Lorsqu'un snapshot exact n'est pas reconstructible, le statut le dit
explicitement et aucune revision seule ne doit etre presentee comme sa source
complete.

Tous les constats non resolus sont inscrits dans `response.md`, y compris
ceux dont le traitement est explicitement reporte dans le cadre de l'audit.
Toute section qui contient un constat doit reprendre le meme nom et le meme
ordre que dans ce document. Chaque constat doit conserver un identifiant
stable, un etat courant, une preuve dans le code ou les mesures disponibles et
un critere de cloture.

`closed.md` est le registre compact de tous les constats corriges,
refutes par mesure, fusionnes ou devenus inapplicables. Avant de creer un
identifiant, l'auditeur doit rechercher dans `response.md` et `closed.md` une
signature equivalente. L'identite d'un constat est definie par sa cause et son
perimetre, pas par la correction envisagee ni par la formulation de son titre.

Lorsqu'un constat est clos, transferer toute decision durable vers le contrat
d'implementation concerne, puis le deplacer de `response.md` vers `closed.md`.
Son entree fermee doit conserver un titre explicite, la nature de la cloture,
la signature du probleme initial, la resolution ou la decision, la preuve et
la condition de reouverture. Les rapports volumineux ne sont pas recopies si
une preuve durable plus precise peut etre liee.

Si le meme probleme reapparait, reutiliser son identifiant et redeplacer
l'entree vers `response.md`, en conservant une ligne sur la cloture
precedente et la raison de la reouverture. Un nouvel identifiant n'est permis
que si la cause ou le perimetre differe reellement. Un identifiant possede
exactement un etat courant : non resolu dans `response.md` ou ferme dans
`closed.md`. `status.md` conserve la comptabilite et les preuves du
passage, sans devenir un troisieme registre de constats.

L'audit transversal des references independantes, caches, notebooks,
fingerprints et artefacts publies est volontairement separe dans
`../validation/query.md`. Il n'est pas execute pendant un passage ordinaire
de ce referentiel. Les tests cibles necessaires aux audits proprietaires
Dynamics, Analytics, Numerical robustness ou Build restent dans leur perimetre.

Un audit doit distinguer explicitement :

- les incoherences prouvees par le code ou les artefacts compiles ;
- les risques qui necessitent une mesure pour etre confirmes ;
- les choix deja satisfaisants qu'une correction doit preserver ;
- les exceptions mathematiques necessaires, avec leur justification.

Un constat transversal est inscrit une seule fois, sous la section qui porte sa
cause primaire. Les autres audits concernes sont references depuis ce constat.
Un choix satisfaisant n'est pas inscrit dans `response.md` : s'il exprime
une decision durable, il est documente dans le contrat d'implementation qui en
est proprietaire. Une exception necessaire suit la meme regle.

L'auditeur doit partir du code, des dependances CMake, des tests et, pour les
questions de performance, des artefacts compiles. Les noms de dossiers et les
commentaires ne constituent pas a eux seuls une preuve de responsabilite ou de
factorisation. Il ne doit pas supposer que l'architecture actuelle est bonne
ou mauvaise, ni proposer une abstraction sans identifier ses consommateurs
reels.

Pour chaque audit, renseigner d'abord dans `status.md` la matrice de
couverture effectivement inspectee. Chaque constat doit ensuite indiquer :

- son **etat** : ouvert ou reporte par decision explicite ;
- sa **severite**, c'est-a-dire l'impact technique si le probleme subsiste :
  critique, haute, moyenne ou faible ;
- sa **priorite**, c'est-a-dire l'ordre recommande de traitement : haute,
  moyenne ou basse ;
- sa **confiance** : prouvee, forte ou a mesurer ;
- les fichiers et symboles concernes, la preuve, l'impact concret, la correction
  minimale proposee et le test ou la mesure qui permettrait de le clore ;
- la reference aux preuves du passage, la date de leur derniere verification et
  un proprietaire explicite ou la mention `non attribue`.

Un constat reporte indique en plus la date et l'auteur de la decision, son
motif, ainsi qu'un evenement de reexamen. Un report sans condition de reprise
ne constitue pas un etat durable acceptable.

Une preuve issue d'un binaire, d'un profil ou d'une mesure est attachee au
snapshot, a la configuration de build et a l'artefact exacts. Elle est marquee
`a rafraichir` des qu'un fichier ou une option susceptible de modifier cet
artefact change. Une preuve fermee reste compacte, mais son origine durable
doit permettre de reproduire la decision : commande, environnement, manifeste
de workloads, sortie conservee ou lien vers un artefact versionne. Une
affirmation chiffree sans cette provenance peut orienter une nouvelle mesure,
mais ne constitue pas a elle seule une baseline courante.

Le statut `complet` signifie que toutes les preuves attendues de la section ont
ete couvertes sur le perimetre declare. Des tests cibles executes pendant une
remediation ne transforment pas retroactivement un audit partiel en audit
complet; `status.md` distingue couverture d'audit et verification des
corrections.

Lorsque deux constats partagent les memes fichiers, mesures ou une migration
qui ne doit etre faite qu'une fois, ajouter un champ **Coordination** dans
`response.md`. Il doit citer les identifiants concernes et preciser si le
traitement commun est obligatoire, recommande ou simplement ordonne. Cette
coordination ne fusionne pas deux causes distinctes et ne change pas leurs
criteres de cloture.

Une opportunite non prouvee reste une hypothese a mesurer. Sa confiance est
`a mesurer` et sa cloture consiste d'abord a accepter ou rejeter l'hypothese,
pas a imposer prematurement une implementation.

## Sommaire

- [Dynamics](#dynamics)
- [Analytics](#analytics)
- [Numerical robustness](#numerical-robustness)
- [CUDA execution and memory safety](#cuda-execution-and-memory-safety)
- [Naming](#naming)
- [Project structure](#project-structure)
- [Exercise and dynamics-family boundaries](#exercise-and-dynamics-family-boundaries)
- [Pricing policies and concepts](#pricing-policies-and-concepts)
- [Tools and src ownership](#tools-and-src-ownership)
- [Build and CUDA instantiations](#build-and-cuda-instantiations)
- [Performance](#performance)
  - [Generic CUDA performance](#generic-cuda-performance)
  - [Early-exercise performance](#early-exercise-performance)
  - [Rough FFT performance](#rough-fft-performance)

## Dynamics

Auditer les fichiers `parameters`, `state`, `dynamics.cuh` et `dynamics_impl.cuh` de
tous les modeles concernes. Les modeles rough, dont l'ossature est differente
par construction, doivent etre analyses separement lorsqu'ils entrent dans le
perimetre.

**Perimetre :** interfaces et implementations de dynamique, etats, preparation,
consommation aleatoire et tests de contrat des modeles selectionnes.

**Hors perimetre :** formules de pricing, architecture rough specifique et
performance des kernels, sauf lorsqu'elles prouvent une rupture du contrat de
dynamique.

**Preuves attendues :** matrice des modeles et transitions, declarations,
definitions, concepts, graphe d'includes, tests de compilation, lois et mapping
Philox.

**Livrable :** differences d'ossature classees en incoherences ou exceptions
mathematiques, avec un proprietaire et un critere de cloture.

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

**Perimetre :** primitives analytiques, providers, concepts, compositions
modele-courbe, surfaces publiques et tests Black-Scholes/fixed income.

**Hors perimetre :** simulation des trajectoires, orchestration generale des
kernels et policies produit sans formule analytique propre.

**Preuves attendues :** matrice modele/primitive, call graph, definitions de
formules, instanciations de concepts et references mathematiques independantes.

**Livrable :** primitives canoniques, duplications ou bypass prouves et
frontiere explicite de chaque provider.

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

## Numerical robustness

Auditer la robustesse numerique des dynamics, analytics, solveurs, reductions,
regressions et kernels de pricing. Cet audit est distinct d'une comparaison de
performance : une optimisation n'est acceptable que si elle preserve le
contrat numerique mesure sur le domaine utile et les cas limites publies.

**Perimetre :** precision, domaines mathematiques, conditionnement, convergence,
propagation d'erreur et reproductibilite numerique des chemins CPU/CUDA.

**Hors perimetre :** gain de temps ou de ressources considere seul ; il releve
de `Performance` et doit reutiliser les tolerances etablies ici.

**Preuves attendues :** domaines de parametres, budgets d'erreur absolue,
relative, ULP ou statistique, tests de limites et references independantes.

**Livrable :** contrat numerique par famille, cas non couverts, risques prouves
et hypotheses de precision a mesurer.

Verifier notamment :

- que la politique de precision est explicite et respectee : calculs device et
  etats en FP32 par defaut, FP64 reserve aux reductions, moments, regressions,
  factorisations ou operations dont la sensibilite le justifie ;
- que chaque emploi de FP64 dans un chemin GPU chaud possede une justification
  numerique et que chaque remplacement par FP32 est valide par une mesure
  d'erreur, pas seulement par un gain de temps ;
- les domaines de `log`, `log1p`, `sqrt`, `pow`, divisions, exponentielles,
  fonctions de repartition et inversions, ainsi que le comportement face aux
  valeurs nulles, negatives, infinies ou sous-normales ;
- la positivite ou les frontieres absorbantes imposees par les modeles, sans
  projection silencieuse qui changerait la loi sans etre documentee ;
- la stabilite des formules pour petits temps, faible volatilite, faible
  mean-reversion, correlation proche de sa borne, intensite de saut faible ou
  forte et parametres proches des limites d'admissibilite ;
- les solveurs scalaires, notamment frontieres de Jamshidian et inversions de
  lois : encadrement, convergence, nombre maximal d'iterations, critere
  d'arret, monotonie requise et comportement en cas d'echec ;
- le conditionnement des equations normales Longstaff-Schwartz, la
  normalisation des etats, la regularisation, la factorisation de Cholesky et
  le biais introduit par un terme de regularisation trop important ;
- la stabilite des reductions inter-thread et inter-bloc, leur ordre
  d'accumulation, les erreurs de cancellation et la reproductibilite attendue ;
- la coherence des conventions temporelles, des fractions d'accrual, des
  unites de taux et des transformations de calendrier a toutes les frontieres
  host/device ;
- la propagation des erreurs et l'absence de NaN ou Inf silencieux dans les
  prix, erreurs standards, coefficients de regression et datasets publies ;
- la presence de tests de limites, identites mathematiques, convergence de pas,
  sensibilite a la precision et comparaison a une reference independante.

Ne pas demander une egalite bit a bit entre deux algorithmes mathematiquement
equivalents lorsqu'elle n'est pas contractuelle. Distinguer reproductibilite
d'un meme chemin compile, tolerance numerique et convergence statistique.

## CUDA execution and memory safety

Auditer la surete d'execution host/device independamment de la justesse des
formules et de la performance. L'absence d'un passage de sanitizer est une
exclusion de couverture, pas un constat de code; un constat n'est cree qu'a
partir d'un defaut ou d'un risque precis prouve par le code, un sanitizer ou un
test cible.

**Perimetre :** arithmetique de tailles et d'offsets, acces memoire, duree de vie
et aliasing des buffers, streams, synchronisations, erreurs asynchrones et
outils de detection host/CUDA.

**Hors perimetre :** ecart numerique d'une valeur finie, traite par `Numerical
robustness`, et cout des synchronisations ou layouts valides, traite par
`Performance`.

**Preuves attendues :** revue des launchers et workspaces, tests d'overflow et
de cardinalites limites, AddressSanitizer/UndefinedBehaviorSanitizer pour les
chemins host pertinents, puis `compute-sanitizer` memcheck, racecheck, initcheck
et synccheck sur une matrice representative des familles CUDA supportees.

**Livrable :** erreurs de bornes, duree de vie, initialisation, concurrence ou
propagation d'erreur prouvees, avec le sanitizer et le cas minimal qui les
reproduisent; matrice explicite des familles non executees.

Verifier notamment :

- les multiplications et additions de cardinalites, tailles, strides, offsets
  et alignements avant allocation ou lancement, avec detection d'overflow ;
- que chaque pointeur device couvre la plage effectivement adressee et que les
  vues ragged valident offsets, longueurs et pools associes ;
- la duree de vie des buffers temporaires, plans, evenements et objets prepares
  jusqu'a la derniere operation asynchrone qui les consomme ;
- l'absence de data race entre blocs, streams ou appels concurrents, ainsi que
  la portee correcte des atomiques, fences et synchronisations ;
- l'initialisation de tous les champs, coefficients, flags et reductions avant
  lecture, y compris sur les branches sans candidat ou les lots partiels ;
- la capture des erreurs de lancement et des erreurs asynchrones au point ou
  elles peuvent encore etre attribuees au kernel et a la ligne concernes ;
- que le stream contractuel est propage sans synchronisation implicite ou usage
  accidentel du stream par defaut ;
- que les tests sanitizer utilisent des geometries et chemins assez riches pour
  exercer regular/explicit, aligned/Cartesian, exact/fixed-step, early exercise
  et rough lorsqu'ils sont disponibles.

Un sanitizer non execute reste une exclusion dans `status.md`. Son absence ne
doit pas etre transformee en finding generique ni masquer les preuves statiques
deja disponibles.

## Naming

Auditer ensemble les noms de dossiers, fichiers, namespaces, types, concepts,
fonctions, arguments, variables, constantes, tests, targets et identifiants de
catalogue. Le nom doit indiquer la responsabilite reelle, pas l'historique de
creation de l'element.

**Perimetre :** tous les identifiants internes et publies ainsi que leur
coherence avec les responsabilites, unites et niveaux de namespace.

**Hors perimetre :** preferences stylistiques sans contradiction semantique,
cout de migration ou changement d'identifiant publie sans plan explicite.

**Preuves attendues :** inventaire des conventions, recherches globales,
comparaison des familles homologues et references CMake/catalogue/tests.

**Livrable :** contradictions objectivement verifiables, convention cible et
plan de renommage atomique.

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

**Perimetre :** responsabilite des dossiers et fichiers, taille des
unites, duplication structurelle et chemins d'extension du projet.

**Hors perimetre :** details d'API deja portes par `Dynamics`, `Analytics` ou
`Pricing policies and concepts`, sauf lorsqu'ils expliquent un mauvais
placement.

**Preuves attendues :** arborescence, graphes d'includes et de liens,
consommateurs, tailles/diffs de fichiers et chemins necessaires pour ajouter une
combinaison.

**Livrable :** carte des responsabilites et propositions de deplacement, fusion,
scission ou suppression avec dependances avant/apres.

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

## Exercise and dynamics-family boundaries

Auditer les frontieres internes a `src` a partir de l'arborescence mais aussi du
graphe reel des includes, des liens CMake et des instanciations de templates.
Cet audit doit verifier deux separations orthogonales : droits d'exercice et
nature de la dynamique. Il ne doit pas confondre produit europeen avec formule
fermee, ni modele rough avec modele simplement simule en Monte-Carlo.

**Perimetre :** frontieres European/early exercise et Markovian/rough ainsi que
leurs composants reellement partageables.

**Hors perimetre :** formule de payoff particuliere, calibration et optimisation
detaillee des moteurs, traitees par leurs audits proprietaires.

**Preuves attendues :** graphe d'includes et de targets, instanciations de
templates, dependances optionnelles et matrices de tests par famille.

**Livrable :** dependances injustifiees entre familles, composants communs
legitimes et exceptions mathematiques documentees.

### European and early exercise

Un produit europeen possede une seule date d'exercice, quelle que soit sa
methode de valorisation : formule fermee, semi-analytique ou Monte-Carlo. Un
produit American/Bermudan possede une logique d'exercice anticipe et peut
reutiliser une simulation markovienne, mais ne doit pas imposer son workspace ou
sa regression aux pricers europeens.

Verifier notamment :

- que les pricers europeens n'incluent ni Longstaff-Schwartz, ni regressor, ni
  workspace ou execution plan propres a l'exercice anticipe ;
- que le moteur Longstaff-Schwartz commun ne depend d'aucun type concret de
  payoff equity, taux de swap, courbe, modele ou convention produit ;
- que les seules briques partagees entre options americaines et swaptions
  bermudeennes sont generiques : kernels de backward induction, regression,
  workspace, execution plan et contrats de continuation necessaires ;
- que le payoff immediat, l'etat de continuation, les normalisations et le
  calendrier contractuel restent sous le produit qui les definit ;
- que la composition `Dynamics + Schedule + ContinuationState + Regressor`
  reste une composition typee modele-produit et ne duplique pas le moteur ;
- que les notions payer/receiver, swap, courbe et accrual ne remontent pas dans
  le moteur American equity, et que spot, strike vanilla ou dividende ne
  remontent pas dans le moteur Bermudan fixed income ;
- que call/put et payer/receiver sont des specialisations compile-time lorsque
  leur choix modifie seulement le payoff ou le signe, sans kernels recopies ;
- que schedules d'exercice, simulation des etats, observation des dates,
  regression et reduction finale possedent chacun un proprietaire conceptuel
  clairement identifie ;
- que les includes ne creent aucun chemin de dependance d'un produit europeen
  vers un produit a exercice anticipe, meme indirectement ;
- que les tests distinguent le moteur generique, chaque policy produit et les
  compositions modele-produit.

Le produit equity nomme `AmericanOption` est numeriquement exerce sur une grille
discrete. Verifier que cette approximation bermudeenne est explicite dans les
parametres, la documentation, les datasets et les validations : resolution de
la grille, premier exercice, maturite alignee et limite vers l'exercice continu.
Ne pas presenter ce calcul comme une solution exacte a exercice continu.

### Markovian and rough dynamics

Verifier notamment :

- que les dynamiques markoviennes utilisent l'ossature commune seulement
  lorsqu'un etat fini contient effectivement l'information necessaire a la
  transition suivante ;
- que les modeles rough/Volterra ne sont pas forces dans une fausse
  `DynamicsPolicy` markovienne pour satisfaire un kernel existant ;
- qu'aucun path simulator ou schedule markovien ne contient de branche runtime
  `rough/non-rough` ou de dependance optionnelle a cuFFTDx ;
- que convolution, noyaux de Volterra, historique, hybrid scheme, FFT et
  plans associes restent dans des composants rough explicitement identifies ;
- que les briques rough partagees entre Rough Bergomi et Rough Heston, lorsque
  ces deux modeles appartiennent au perimetre, sont placees dans un niveau
  commun uniquement si leurs invariants mathematiques et leurs layouts sont
  reellement identiques ;
- que les modeles rough peuvent reutiliser les composants independants de la
  memoire du processus : Philox, device inputs, payoff terminal, reductions,
  diagnostics CUDA et primitives de lancement pertinentes ;
- que les dependances mathDx/cuFFTDx, architectures supportees et flags de
  compilation restent optionnels et limites aux targets qui les consomment ;
- qu'une approximation markovienne ou un lift eventuel est nomme comme tel et
  ne modifie pas silencieusement le contrat de la dynamique rough ;
- que l'ajout d'un modele markovien ne recompile pas les kernels FFT et que
  l'ajout d'un modele rough n'etend pas les concepts markoviens artificiellement ;
- que les tests de contrat, convergence et performance utilisent des matrices
  separees pour les familles markovienne et rough.

Pour chaque dependance commune, demander quel invariant justifie le partage.
Une ressemblance d'API ou l'emploi commun du Monte-Carlo ne suffit pas.

## Pricing policies and concepts

Auditer les concepts et policies qui composent dynamics, schedules, handlers,
continuation states, regressors et kernels. L'objectif est de verifier que les
concepts expriment les besoins minimaux des consommateurs reels et que les
policies transportent seulement les donnees necessaires a leur specialisation.

**Perimetre :** concepts C++, policies, types prepares et points de composition
entre dynamique, calendrier, observation, payoff, continuation et regression.

**Hors perimetre :** exactitude des formules et cout detaille des kernels, sauf
si l'interface impose une operation ou une ressource inutile.

**Preuves attendues :** exigences de concepts reliees a leurs expressions
consommatrices, instanciations valides/invalides, tailles de types et call graph.

**Livrable :** exigences inutiles ou manquantes, policies trop larges et
composition canonique minimale par famille.

Verifier notamment :

- que chaque exigence d'un concept correspond a un appel reel dans au moins un
  template consommateur et qu'aucune methode hypothetique n'est imposee ;
- que les capacites optionnelles, par exemple `spot`, `log_spot`, transition
  exacte ou etat joint, utilisent des concepts specialises plutot qu'un contrat
  de base toujours plus riche ;
- que fixed step et exact transition partagent leur contrat commun sans
  transporter `PreparedModel` ou `PreparedTransition` lorsqu'ils sont inutiles ;
- que `Schedule` construit ou parcourt la sequence des dates et fournit les
  transitions temporelles necessaires, mais ne calcule ni payoff, ni observable
  produit, ni prix ;
- que les handlers observent un etat aux dates decidees sans prendre possession
  de la dynamique, du calendrier ou du launcher ;
- que `PricingPolicy` definit les inputs, la ligne preparee et l'evaluation
  propres au produit sans reimplementer simulation, reduction ou allocation ;
- que `ContinuationState` definit uniquement le stockage et les variables de
  regression requises par l'exercice anticipe ;
- que le regressor definit base, accumulation, resolution et evaluation sans
  connaitre le produit ou le modele ;
- que les types `Parameters`, `PreparedDynamics`, `State`, `Calendar`,
  `TimeConfiguration`, `DeviceInputs` et `PreparedRow` sont coherents entre les
  policies composees et trivialement copiables lorsqu'ils traversent le device ;
- que les limites de taille de `PreparedRow` ou d'autres objets device sont
  justifiees par une ressource mesuree et non utilisees comme substitut a un
  audit de registres ;
- que les erreurs d'une composition invalide sont detectees au plus pres par un
  concept ou `static_assert` lisible ;
- que les tests de compilation couvrent les familles de policies et leurs
  capacites optionnelles, sans multiplier des concepts redondants.

Ne retenir un concept universel que si ses consommateurs partagent les memes
invariants sans adaptateurs vides, branches runtime ni donnees inutiles. Ne
retenir une specialisation par modele que si un contrat commun introduirait un
cout compile/runtime ou masquerait une difference semantique.

## Tools and src ownership

Auditer explicitement la frontiere entre `src` et `tools`. La direction de
dependance normale est `tools -> src` : `src` fournit les bibliotheques
numeriques et d'acces aux donnees reutilisables, tandis que `tools` fabrique,
valide et publie des artefacts offline. Une cible de `src` ne doit pas dependre
d'un header, d'une bibliotheque ou d'une convention de catalogue situes dans
`tools`.

**Perimetre :** fichiers, fonctions, targets et dependances traversant la
frontiere runtime/offline.

**Hors perimetre :** organisation interne detaillee d'une couche lorsque rien
ne traverse la frontiere ; elle releve de `Project structure`.

**Preuves attendues :** includes dans les deux sens, liens CMake, include roots,
consommateurs reels et nature runtime ou publication de chaque symbole partage.

**Livrable :** table source/destination, DAG cible et tests statiques empechant
le retour d'une dependance `src -> tools`.

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

## Build and CUDA instantiations

Auditer les frontieres de compilation C++/CUDA, les includes, les
instanciations explicites, le graphe CMake et l'incrementalite du build. Le but
n'est pas de minimiser le nombre de fichiers, mais de limiter les recompilations,
les definitions dupliquees et les unites CUDA inutilement volumineuses.

**Perimetre :** unites de traduction, templates, instanciations, targets,
dependances optionnelles, artefacts compiles et incrementalite.

**Hors perimetre :** organisation conceptuelle sans effet sur le build et
performance runtime non causee par les frontieres de compilation.

**Preuves attendues :** graphe CMake, depfiles, symboles, tailles
objets/archives/cubins et mesures clean, no-op et incrementales.

**Livrable :** source de verite des combinaisons, recompilations inattendues,
definitions dupliquees et comparaison mesuree de toute nouvelle frontiere.

Verifier notamment :

- que chaque `.cpp` ou `.cu` autonome correspond a une unite de traduction
  reelle et que chaque implementation textuellement incluse est identifiee et
  protegee en consequence ;
- que l'inclusion d'un `.cu` est necessaire a la visibilite de definitions
  device force-inline ou de templates, et ne masque pas une frontiere de lien
  mal concue ;
- qu'une fonction ou specialisation non-inline possede une seule definition et
  que les instanciations explicites call/put, payer/receiver et curve/model ne
  produisent ni symbole manquant ni code duplique inutilement ;
- que les templates generiques restent visibles au point d'instanciation sans
  exposer toutes les implementations concretes dans tous les consommateurs ;
- que chaque launcher compile dans une target fine avec uniquement ses loaders,
  courbes, produits et bibliotheques communes necessaires ;
- que les targets agreges ne deviennent pas des dependances transitives d'un
  test ou d'un generateur local ;
- que modifier un produit, une dynamics, un analytics ou une curve recompile
  seulement la matrice de consommateurs attendue ;
- qu'une source de verite explicite ou generee enumere completement les sources,
  dependances et combinaisons supportees, et qu'un controle detecte tout fichier
  orphelin ou entree manquante sans imposer le globbing comme solution ;
- que les dependances optionnelles, notamment mathDx/cuFFTDx, n'affectent ni la
  configuration ni la compilation des targets qui ne les utilisent pas ;
- que les variantes d'architecture CUDA ne multiplient pas les cubins sans
  besoin de distribution explicite ;
- que ccache, depfiles et compilation incrementale sont effectivement actifs et
  que les temps clean, no-op et incrementaux sont mesures separement ;
- que la taille des objets, archives et cubins ainsi que le temps de compilation
  par target sont suivis pour les principales specialisations.

Toute proposition de fusion ou de deplacement doit comparer le temps de build
clean et incremental, la taille du code genere, les ressources kernel et la
lisibilite des erreurs de compilation.

## Performance

Effectuer trois audits de performance distincts. Utiliser les memes regles de
mesure et de reproductibilite, mais ne pas diluer les contraintes specifiques
de Longstaff-Schwartz ou des modeles rough dans une liste CUDA generique.

**Perimetre :** performance runtime CUDA generique, early exercise et rough/FFT
sur les architectures et charges officiellement supportees.

**Hors perimetre :** toute optimisation sans baseline comparable ou dont le
contrat numerique n'est pas etabli par `Numerical robustness`.

**Preuves attendues :** binaires et ressources statiques, profils Nsight,
benchmarks repetes, configuration materielle/logicielle et validation numerique.

**Livrable :** trois rapports separes, chacun avec baseline, mesures de
ressources, mediane/p95, hypothese, resultat et decision.

### Generic CUDA performance

Auditer les fonctions device, kernels, templates communs, launchers, layouts
memoire et echanges hote-device des formules fermees, du Monte-Carlo markovien
et du sampling. Les chemins early exercise et rough FFT font l'objet des deux
sections suivantes.

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
- l'absence de chevauchement des sequences Philox, de reutilisation non
  contractuelle des tirages et de generation inutile, ainsi que la stabilite du
  mapping entre resultat, trajectoire et sequence aleatoire ;
- les copies, allocations, validations de pointeurs, requetes de proprietes du
  device, synchronisations, streams et possibilites de reutilisation des
  buffers ;
- la pertinence de kernels generiques un-thread-par-resultat ou
  un-bloc-par-resultat selon la quantite de travail et les reductions internes ;
- les seuils de regression portant sur les registres, spills, stack, shared,
  taille du code, temps median et p95, ainsi que la validation numerique de
  toute optimisation.

### Early-exercise performance

Auditer separement le moteur Longstaff-Schwartz utilise par les options
americaines et les swaptions bermudeennes. Couvrir au minimum un modele equity
a un facteur, un modele equity a plusieurs etats, un modele de taux a un facteur
et un modele de taux a deux facteurs. Utiliser un plan d'experience couvrant le
nombre de prix, de trajectoires, de dates d'exercice, de champs de continuation
et de blocs par prix, avec variations isolees et interactions representatives.

Verifier notamment :

- l'exactitude arithmetique du calcul de `WorkspaceLayout`, des offsets, des
  alignements et des tailles, y compris les protections contre overflow ;
- que `ExecutionPlan` utilise un budget VRAM avec marge de securite, construit
  des batches valides pour toutes les cardinalites et echoue proprement lorsqu'un
  prix unique ne tient pas ;
- que le plan prend en compte tous les champs SoA, cashflows, coefficients,
  reductions, etats de regression, erreurs standards et buffers temporaires ;
- que les allocations, copies host/device, evenements et synchronisations sont
  faites par batch ou par lancement, jamais a chaque date d'exercice sans
  necessite mesuree ;
- qu'aucun aller-retour CPU n'intervient dans la backward induction du petit
  regressor device ;
- que les etats observes sont stockes en SoA coalescent et que les dates ou
  schedules heterogenes ne provoquent pas de grands trous ou lectures inutiles ;
- que la simulation forward, l'accumulation des equations normales, la
  resolution, l'application de la decision d'exercice et la reduction finale
  ont une decomposition en kernels justifiee par les dependances et les couts ;
- que le nombre de lancements varie comme prevu avec les batches et dates, et
  que fusionner deux phases n'augmente pas les registres, la shared ou les
  synchronisations au-dela du gain mesure ;
- que la base de regression est evaluee une seule fois par etat lorsque
  possible, sans tableau local provoquant spills ou stack importante ;
- que les equations normales et Cholesky utilisent FP64 la ou le
  conditionnement le justifie, sans propager FP64 dans la simulation ou le
  stockage massif des trajectoires ;
- que shared memory, reductions inter-blocs, atomiques et synchronisations du
  regressor conservent une bonne occupation et ne serialisent pas les prix ;
- que `threads_per_block` et `blocks_per_price` sont mesures pour plusieurs
  tailles de regression et nombres de trajectoires, sans valeur universelle
  imposee par intuition ;
- que les produits avec nombres de dates differents sont batches ou groupes de
  facon mesuree, sans surcalcul excessif ni explosion du nombre de lancements ;
- que le contrat precise si prix et erreurs standards doivent rester identiques
  lorsque seul le decoupage en batches change, puis que cette invariance est
  testee lorsqu'elle est requise ;
- que les limites naturelles du prix, l'exercice immediat, l'exercice terminal
  et la convergence quand la grille se raffine sont preserves par toute
  optimisation.

Produire une table par specialisation avec workspace par prix, VRAM totale,
registres, spills, stack, shared, occupation, lancements, temps forward,
regression/backward et reduction finale. Distinguer cout fixe, debit asymptotique
et point de saturation GPU.

### Rough FFT performance

Auditer separement les chemins rough reposant sur convolution directe, hybrid
scheme ou FFT/cuFFTDx. Les comparer uniquement a precision et discretisation
equivalentes. Couvrir plusieurs maturites, nombres de pas, nombres de prix et
trajectoires ainsi que les architectures CUDA officiellement supportees.

Verifier notamment :

- la complexite et le seuil mesure de bascule entre convolution directe,
  implementation hybride et FFT, sans supposer que la FFT gagne pour les
  petites grilles ;
- le choix des longueurs FFT, padding, batch size et layouts complexes, ainsi
  que le cout memoire des zeros et frequences inutiles ;
- la fusion ou separation de generation gaussienne, convolution, reconstruction
  de variance, integration du spot, payoff et reduction ;
- la shared memory statique et dynamique requise par cuFFTDx, les registres,
  spills, occupation, nombre de blocs residents et limites par architecture ;
- les transpositions, strides, alignements et coalescence des lectures/ecritures
  entre l'espace temps et l'espace frequentiel ;
- la reutilisation des coefficients de noyau, twiddles, plans ou objets FFT
  entre trajectoires, prix et lancements ;
- le trafic VRAM et les buffers temporaires, notamment lorsque l'historique ne
  tient plus dans un bloc ou qu'un prix est traite par plusieurs blocs ;
- le nombre et le cout des synchronisations de bloc, lancements, barrieres et
  eventuelles copies intermediaires ;
- le mapping Philox et les correlations gaussiennes, afin qu'une optimisation
  FFT ne change pas silencieusement la loi simulee ;
- la precision de convolution, les erreurs de padding, l'aliasing et la
  convergence vers une implementation directe de reference ;
- l'isolation des specialisations et dependances cuFFTDx pour ne pas augmenter
  registres, taille du code ou temps de compilation des chemins markoviens ;
- le comportement lorsque la taille demandee n'est pas supportee par une
  specialisation compilee : erreur explicite, fallback documente et absence de
  chemin lent silencieux.

Produire une courbe temps et debit selon le nombre de pas, avec decomposition
simulation/convolution/payoff/reduction, ainsi qu'une table registres, shared,
occupation, VRAM temporaire et erreur numerique pour chaque variante comparee.

Pour les trois audits de performance, consigner le GPU, l'architecture, le
compilateur, les flags, la geometrie, la taille du jeu de donnees, les warmups,
le power limit, les frequences, l'etat thermique, le mode de performance du GPU
et la methode de chronometrage. Utiliser `ptxas` ou le cubin pour les ressources
statiques et Nsight Compute pour l'occupation atteinte, les stalls, les acces
locaux, les caches, les branches, le debit memoire et le melange
d'instructions FP32/FP64. Comparer mediane et p95 sur plusieurs repetitions,
conserver une baseline avant modification et valider numeriquement toute
optimisation retenue.

Avant chaque experience, fixer le nombre de repetitions, la methode de rejet
des valeurs aberrantes, l'incertitude ou la variabilite observee, le seuil
minimal de gain techniquement utile et la regle d'acceptation ou de rejet. Le
manifeste de workloads et cette regle sont conserves avec la baseline avant de
voir le resultat; une mediane seule ou un gain inferieur au bruit mesure ne
suffit pas a retenir une transformation.
