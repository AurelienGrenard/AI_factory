# Constats d'audit ouverts

## Objet

Ce document contient uniquement les constats encore ouverts produits par les
audits de `audit-query.md`. Les sections et leur ordre doivent rester
identiques. Un constat n'est retire qu'apres correction et verification de son
critere de cloture. Les decisions permanentes sont alors reportees dans le
contrat concerne ; les essais rejetes ou les travaux reportes vont
respectivement dans `abandoned-work.md` ou `deferred-work.md`.

Chaque constat doit conserver un identifiant stable, une priorite, une preuve,
le changement a evaluer et un critere de cloture. Une hypothese de performance
ne devient pas une exigence d'implementation avant d'avoir ete mesuree sur les
architectures cibles.

## Dynamics

## Analytics

## Naming

### NAME-002 — Faire correspondre les dossiers produit et leurs namespaces

**Priorite : haute.** Les 25 dossiers `src/product/<product>` declarent presque
tous leurs types et policies dans le namespace plat `workbench::product`.
L'arborescence distingue le produit, mais le namespace ne le fait pas ; les
types doivent alors repeter leur nom complet, par exemple
`product::EuropeanOptionParameters` et `EuropeanOptionPricingPolicy`.

Evaluer la convention symetrique
`workbench::product::<product>::Parameters`, `Dataset`, `PricingPolicy` et
helpers propres au produit. Les primitives partagees entre plusieurs produits
restent dans `common` ou dans un namespace explicitement commun.

**Critere de cloture :** une convention unique dossier/namespace/type est
documentee et appliquee a tous les produits, sans namespace `detail` global
partage accidentellement entre plusieurs dossiers. Les schemas et identifiants
de datasets publies restent stables ou disposent d'une migration explicite.

### NAME-003 — Supprimer la repetition des namespaces dans l'API publique

**Priorite : moyenne.** Les fonctions de lancement repetent le modele et parfois
la courbe deja presents dans leur namespace. Par exemple
`model::equity::heston::launch_heston_asian_option_cuda` et
`model::fixed_income::hull_white::nelson_siegel::launch_hull_white_nelson_siegel_rate_option_cuda`
encodent deux fois la meme hierarchie. Les launchers communs repetent aussi leur
methode : `closed_form::launch_closed_form_cuda` et
`monte_carlo::launch_monte_carlo_cuda`.

Retenir des noms qualifies courts et canoniques, par exemple
`heston::launch_asian_option_cuda` et `closed_form::launch_pricing_cuda`, en
conservant le nom complet uniquement dans les diagnostics et symboles externes
qui l'exigent.

**Critere de cloture :** une regle de redondance namespace/symbole est appliquee
aux launchers, kernels, policies et types, puis verifiee statiquement. Aucun
alias de compatibilite permanent ne doit conserver en parallele l'ancienne API.

### NAME-004 — Donner aux fichiers inclus une extension veridique

**Priorite : haute.** Au moins 26 fichiers `.cu` sont inclus textuellement.
Les 22 fichiers `dynamics.cu`/`analytics.cu`, les `term_structure.cu` et les
fragments LSM `linear_solver.cu`/`regression.cu` se comportent comme des headers
d'implementation, pas comme des unites de traduction CUDA autonomes. Le CMake
confirme explicitement cette exception.

Renommer ces fragments en `*_impl.cuh`/`*.inl.cuh`, ou placer les definitions
dans le `.cuh` lorsqu'une paire n'apporte aucune frontiere utile. Harmoniser en
meme temps `closed_form/closed_form_kernels.cuh` avec
`monte_carlo/monte_carlo_kernel.cuh`, ainsi que
`fixed_income/analytics_concepts.cuh` avec les autres `concepts.cuh`.

**Critere de cloture :** aucun `#include "*.cu"` ne subsiste ; chaque `.cu` est
compile comme une unite de traduction declaree par CMake. Les familles
homologues utilisent le meme schema de nom de fichier.

### NAME-005 — Remplacer les noms affines symboliques et les types redondants

**Priorite : moyenne.** Les analytics fixed income exposent les fonctions
`log_A`, `A` et `B`, ainsi que des champs `log_A` et `B`. Cette notation est
mathematiquement connue mais ne suit ni `snake_case` ni le vocabulaire des
autres fonctions. Les types `G2PlusPlusFittedParameters` et
`HullWhiteFittedParameters` repetent leur modele dans un namespace qui le porte
deja. Plusieurs implementations recopient aussi les parametres de mean
reversion dans des variables locales `a` et `b`.

Retenir des noms semantiques communs tels que `log_affine_scale`,
`affine_scale`, `affine_loadings`, `AffineBondCoefficients` et `FittedModel` ou
`FittedParameters`. Employer `mean_reversion_x/y` dans les blocs ou la valeur
reste vivante sur plusieurs expressions.

**Critere de cloture :** plus aucun callable ou champ public d'une lettre
majuscule ; les types ne repetent pas leur namespace ; les variables d'une
lettre sont limitees aux formules locales documentees. Mettre a jour le contrat
analytics et tous les tests dans le meme changement.

### NAME-006 — Rendre les unites, comptes et modes explicites

**Priorite : haute.** Les parametres produit stockent des jours en
`std::uint32_t` sous des noms tels que `maturity`, `exercise_time` et
`payment_time`, tandis que les lignes preparees reutilisent souvent les memes
noms pour des valeurs FP32 exprimees en annees. `cartesian_product` est un
booleen qui designe tantot un mode de construction, tantot une propriete des
inputs. `num_steps` subsiste a cote du nom canonique `step_count`.

Employer soit des types forts, soit des suffixes coherents tels que
`maturity_days`, `exercise_day`, `maturity_time` et `time_to_maturity`.
Remplacer le booleen de construction par un enum `PriceConstruction` commun au
host et au device, puis choisir un vocabulaire unique `*_count`, `*_index`,
`*_offset`, `*_stride` et `step_count`.

**Critere de cloture :** une meme variable ne change plus implicitement d'unite
entre les couches ; tous les modes se lisent comme des enums ou predicats ; une
recherche statique ne trouve plus les variantes de compte non retenues hors des
exceptions rough documentees.

### NAME-007 — Retirer les noms de tests relatifs a leur date de creation

**Priorite : basse.** `tests/new_equity_dynamics_cuda_test.cu` contient encore
des validations numeriques utiles, mais `new` n'a plus de sens permanent et le
test recoupe maintenant le contrat generique
`equity_dynamics_policy_cuda_test.cu`. Les tests fixed income alternent aussi
entre des noms `*_caplet_*` et `*_rate_options_*` pour des surfaces similaires.

Renommer le premier test en fonction de sa responsabilite numerique et repartir
les assertions entre contrat generique et tests mathematiques propres aux
modeles. Definir une matrice de noms de tests coherente par famille et type de
validation.

**Critere de cloture :** aucun fichier ou target permanent ne contient `new`,
`old` ou `legacy`; chaque test a une responsabilite non recouverte par un autre
et les noms CMake/CTest correspondent au fichier.

## Project structure

### STRUCT-001 — Replacer les policies fixed income dans leurs produits

**Priorite : haute.** Les produits equity possedent leur
`src/product/<product>/pricing_policy.cuh`. A l'inverse,
`rate_option`, `zero_coupon_bond_option` et `european_swaption` n'en ont pas :
leurs policies sont rangees dans
[`common/fixed_income/bond_option_pricing_policies.cuh`](../src/common/fixed_income/bond_option_pricing_policies.cuh)
et
[`common/fixed_income/european_swaption.cuh`](../src/common/fixed_income/european_swaption.cuh),
qui dependent directement des parametres produit.

Deplacer chaque policy dans son dossier produit. Conserver dans
`common/fixed_income` uniquement les analytics, cashflows, decomposition de
Jamshidian et helpers ne dependant d'aucun contrat produit concret.

**Critere de cloture :** chaque dossier produit applicable possede son point de
composition/pricing ; aucune policy de `common` n'inclut un
`product/<concrete_product>/parameters.hpp`.

### STRUCT-002 — Factoriser le moteur American modele-independant

**Priorite : haute.** Bates, Heston, NIG et Variance-Gamma contiennent quatre
implementations `american_option.cu` de 878 a 903 lignes, soit 3 562 lignes.
Elles declarent les memes sept kernels. Bates et Heston deviennent identiques
apres remplacement du nom du modele ; les deux modeles Levy ne different que
sur une petite partie de preparation/simulation.

Deplacer kernels, orchestration et workspace dans un moteur commun type par une
policy de dynamique/etat. Les fichiers de modele doivent seulement fournir le
binding et les instanciations publiques necessaires.

**Critere de cloture :** une seule implementation des sept etapes LSM, quatre
bindings minces, parite numerique et de consommation Philox, et comparaison des
temps de compilation, registres, stack, shared et temps kernel.

### STRUCT-003 — Remplacer la matrice manuelle de bindings modele-produit

**Priorite : haute.** Les wrappers equity produit dans `src/model/equity`
representent environ 21 186 lignes et des centaines de paires `.cu/.cuh`. La
majorite ne fait que choisir `DynamicsPolicy`, `Schedule`, `PricingPolicy`, la
configuration de temps et le nom de diagnostic. Le prototype
[`tools/codegen/pricing_bindings`](../tools/codegen/pricing_bindings) n'est
integre nulle part et ne couvre que quatre bindings.

Choisir une seule solution : traits/configurations C++ declaratifs ou generation
de code complete depuis un manifeste controle. Organiser eventuellement les
bindings sous `src/pricing/<asset_class>` ou un sous-dossier `pricing` des
modeles pour que leur coeur reste lisible. Le generateur prototype doit etre
industrialise ou supprime.

**Critere de cloture :** ajouter un couple modele-produit ne demande plus de
copier deux fichiers entiers ; le manifeste couvre toutes les instanciations et
est verifie en CI. Les unites CUDA restent suffisamment fines pour conserver
la compilation incrementale et une taille de cubin maitrisee.

### STRUCT-005 — Unifier le sampling et clarifier `generative`

**Priorite : moyenne.** Huit samples equity et six samples fixed income
repetent les memes kernels terminal/calendar, le decodage des paths, la
validation et les launchers. Bates et Heston sont identiques apres renommage ;
Kou et Merton le sont egalement. Black-Scholes concentre 697 lignes et une
seconde responsabilite : generation des parametres et calendriers. Dans le meme
temps, `src/generative` est vide et ne contient qu'un `.gitkeep`.

Creer un moteur commun de sampling pilote par `DynamicsPolicy` et des handlers
d'observation. Deplacer la generation de parametres/calendriers dans une vraie
couche `generative`, ou supprimer ce dossier et l'API Black-Scholes specialisee
si elle n'appartient plus au produit. Rough reste une implementation separee.

**Critere de cloture :** une implementation commune pour les modeles Markoviens,
bindings minces par modele, responsabilite explicite de `generative`, tests
symetriques terminal/calendar et absence de dossier placeholder.

### STRUCT-006 — Centraliser le squelette des loaders sans polluer les types device

**Priorite : moyenne.** Les 43 paires `dataset.hpp/.cpp` representent environ
4 059 lignes. Chaque `.cpp` repete ouverture du fichier, parsing JSON, gestion
d'erreur, validation du document, extraction des rows, reservation du vecteur
et construction du prefixe d'erreur. Les validations de champs restent
legitimement propres a chaque schema.

Ajouter un helper host-only dans une couche d'acces aux donnees sous `src`, par
exemple `src/io/json_dataset`, pour le cycle commun et laisser a chaque loader
seulement la conversion/validation d'une row. `src` ne doit pas dependre de
`tools`. Ne pas fusionner `parameters.hpp` et `dataset.hpp`, car les parametres
doivent rester CUDA-friendly sans `filesystem`, `vector` ni JSON.

**Critere de cloture :** un seul chemin teste d'ouverture/parsing/validation de
document, loaders locaux declaratifs, messages d'erreur preserves et aucun
surcout/dependance hote dans les headers de parametres.

### STRUCT-007 — Rendre les generateurs de catalogue declaratifs

**Priorite : haute.** Le catalogue contient 357 `generator.cpp` totalisant
environ 89 155 lignes ; 303 depassent 200 lignes. Ils recopient allocations
CUDA, copies, warmup, evenements, batching, nettoyage manuel, chronometrage et
ecriture. Le `CMakeLists.txt` racine atteint 1 769 lignes pour enregistrer une
grande partie de cette matrice.

Introduire des runners RAII communs pour closed form, Monte-Carlo, sampling et
LSM, puis exprimer chaque dataset par une configuration typee/declarative.
Deplacer les manifestes de catalogues et tests dans des fichiers CMake par
domaine, ou les produire depuis une source de verite verifiee.

**Critere de cloture :** un generateur normal ne contient plus de gestion CUDA
manuelle ; les ressources sont liberees par RAII ; la configuration specifique
reste lisible ; le CMake racine redevient un orchestrateur court. Les artefacts
et timings produits restent identiques dans leur schema.

### STRUCT-008 — Nettoyer les points d'entree et artefacts de validation

**Priorite : haute.** La validation contient 434 fichiers Python, dont 212 font
dix lignes ou moins et servent principalement de wrappers CLI par
modele-produit. Par ailleurs, 313 YAML de catalogue declarent un champ
`notebook`, mais un seul fichier existe a l'emplacement indique. Deux notebooks
sous `validation/notebook` ne sont references nulle part.

Remplacer les wrappers non contractuels par un CLI unique prenant modele,
produit et side, ou generer les points d'entree qui doivent rester stables.
Supprimer les references de notebook du workflow cache-only migre et les
artefacts orphelins ; conserver uniquement les notebooks explicitement encore
utilises comme outils de recherche.

**Critere de cloture :** aucun chemin de notebook/rapport/cache inexistant dans
les YAML, aucun wrapper manuel sans logique propre, imports et CLI couverts par
tests, et arborescence de validation conforme au workflow documente.

### STRUCT-009 — Normaliser les petits modules et les gros headers de `common`

**Priorite : moyenne.** `src/model/fixed_income/common` ne contient qu'un helper
`mean_reverting_gaussian.cuh`, mais son namespace est
`model::fixed_income::mean_reverting_gaussian`, sans niveau `common`.
`common/time_configuration.cuh` utilise le namespace `time` tandis que
`time_grid.cuh` utilise `time_grid`. Enfin, `common/sample.cuh` regroupe sur
411 lignes types de rows, generation aleatoire, indexation et validations hote.

Deplacer le helper mean-reverting au niveau correspondant a son namespace,
regrouper les abstractions temporelles sous une convention commune et scinder
le sampling seulement selon des responsabilites reutilisables, par exemple
types, generation et validation.

**Critere de cloture :** chemins et namespaces isomorphes pour ces modules,
dependances acycliques et chaque fichier commun de taille importante justifie
une responsabilite coherente plutot qu'un regroupement historique.

### STRUCT-010 — Supprimer ou centraliser les READMEs locaux non maintenus

**Priorite : moyenne.** Vingt READMEs sous `src/common`, `src/curve` et
`src/model` n'ont aucune reference entrante. Ils recopient des arbres de
fichiers et des surfaces d'API qui evoluent avec la factorisation, tout en
contenant parfois des formules uniques.

Transferer les seules equations ou decisions durables vers les contrats
centraux pertinents, puis supprimer les inventaires d'implementation locaux.
Un README local ne doit rester que s'il constitue le point d'entree maintenu
d'une abstraction autonome et est reference par l'index de documentation.

**Critere de cloture :** aucun arbre de fichiers copie manuellement, aucun
README orphelin, liens verifies et documentation centrale alignee avec le code.

## Tools and src ownership

### BOUNDARY-001 — Supprimer la dependance inverse de `src` vers `tools`

**Priorite : haute.** Les 43 implementations de loaders sous `src/curve`,
`src/model` et `src/product` incluent
[`tools/datasets/dataset_validation.hpp`](../tools/datasets/dataset_validation.hpp).
Le CMake les lie publiquement a `ai_factory_dataset_core`, cible qui compile a
la fois cette validation et les 1 467 lignes de serialisation/generation de
`tools/datasets/dataset.cpp`. La facade publique `cuda_workbench` lie elle aussi
ce `dataset_core`. La dependance reelle est donc actuellement
`src -> tools`, contrairement a la frontiere declaree.

Scinder le composant en deux. Un target host-only de `src`, par exemple
`ai_factory_dataset_io`, doit porter lecture JSON, validation des enveloppes
model/curve/product necessaires a leur lecture et helpers communs aux loaders.
`tools` doit conserver l'ecriture des artefacts, YAML, URLs, metadonnees de
catalogue et validation propre a la publication, en dependant du nouveau
target de `src`. Restreindre ensuite les include roots pour que les targets de
`src` ne voient pas `tools`.

**Critere de cloture :** aucun include `tools/` dans `src`, aucune cible de
`src` ou facade runtime liee a une cible de `tools`, et un test CMake/CI echoue
si cette direction est reintroduite. Les 43 loaders et les contrats de
catalogue doivent continuer a passer apres la scission.

### BOUNDARY-002 — Deplacer le layout de resultats de `tools` vers `src`

**Priorite : haute.** `PriceConstruction` et `price_row_count` sont declares
dans [`tools/datasets/dataset.hpp`](../tools/datasets/dataset.hpp), alors qu'ils
decrivent le layout accepte par les APIs et kernels de pricing. `src` expose en
parallele un booleen `cartesian_product` et implemente son decodage dans
[`common/result_index.cuh`](../src/common/result_index.cuh). Le type canonique
est ainsi dans la couche consommatrice offline, tandis que la couche numerique
ne recoit qu'un booleen non type.

Deplacer l'enum, les calculs de cardinalite verifies et les fonctions de
decodage dans un module de `src/common`, par exemple `result_layout.cuh`.
Faire consommer ce type par les launchers et par `tools`; les kernels peuvent
etre specialises par mode lorsque l'audit de performance le justifie. La
traduction vers les champs JSON reste dans `tools`.

**Critere de cloture :** une seule definition du mode aligned/Cartesian et de
sa cardinalite, aucune conversion manuelle enum-vers-bool dans les generateurs,
et tests host/device symetriques du mapping pour deux et trois axes, y compris
overflow et dimensions incompatibles.

### BOUNDARY-003 — Remonter dans `src` les primitives mathematiques recopiees

**Priorite : moyenne.** Les generateurs Nelson-Siegel et Svensson redefinissent
chacun `instantaneous_forward`, alors que la meme primitive existe dans leurs
`term_structure` sous `src/curve`. Les generateurs Ornstein-Uhlenbeck, Vasicek
et G2 recopient egalement la reconstruction
`volatility = stationary_standard_deviation * sqrt(2 * mean_reversion)`, qui
est une identite du processus et non une politique de catalogue.

Exposer dans `src/curve/<curve>` les formules de courbe pures utilisables sur
host et device, et ajouter la conversion de dispersion stationnaire dans le
helper mean-reverting commun de `src/model/fixed_income`. Les implementer de
facon generique sur le type scalaire si la generation doit conserver FP64
alors que le device utilise FP32. Les bornes, RNG, rejet de candidats et
metadonnees JSON restent dans `tools`; aucun generateur complet ne doit migrer
dans `dynamics`.

**Critere de cloture :** une seule definition algebrique par primitive, tools
appelle les helpers de `src`, parite FP32/FP64 testee et ensembles de lignes
acceptes inchanges pour les seeds publies.

### BOUNDARY-004 — Reorganiser `tools` par etape offline

**Priorite : moyenne.** [`tools/datasets/dataset.cpp`](../tools/datasets/dataset.cpp)
melange sur 1 467 lignes sampling uniforme, grilles, cardinalites, assemblage
des resultats, ecriture JSON et rendu YAML. Le header de 224 lignes expose ces
responsabilites dans une seule API. En parallele,
[`european_swaption_price_generation.hpp`](../tools/datasets/european_swaption_price_generation.hpp)
est un runner CUDA de 387 lignes range comme un simple helper de dataset.

Conserver ces responsabilites offline dans `tools`, mais les repartir par role,
par exemple `tools/generation/{common,model,curve,product,pricing}` pour la
construction et les runners, puis `tools/catalog` pour l'assemblage, la
validation de publication et les writers JSON/YAML. Cette reorganisation doit
etre coordonnee avec `STRUCT-007`; elle ne justifie pas de remonter les writers
ou les recettes dans `src`.

**Critere de cloture :** sampling, execution CUDA, assemblage et serialisation
ont des APIs et tests independants ; un generateur depend seulement des etapes
qu'il emploie ; aucun nouveau monolithe ne remplace `dataset.cpp`.

### BOUNDARY-005 — Nettoyer les facades et placeholders ambigus

**Priorite : basse.** `ai_factory_dataset_tools` est une facade CMake de
compatibilite sans consommateur dans le depot et sans regle d'installation ou
d'export. `cuda_workbench`, facade des launchers de `src`, expose en revanche
le `dataset_core` situe dans `tools`. Enfin, `src/generative` ne contient qu'un
`.gitkeep`, comme deja releve par `STRUCT-005`.

Supprimer la facade tools si aucun consommateur externe n'est confirme,
retirer toute dependance tools de `cuda_workbench` et supprimer les dossiers
placeholders. Si une vraie API generative method-neutral apparait, definir
d'abord son contrat runtime sous `src/generative`; les recettes et writers de
training datasets resteront sous `tools/generation`.

**Critere de cloture :** aucune target de compatibilite sans consommateur,
facades alignees sur leur couche, aucun dossier versionne vide et une seule
destination documentee pour le runtime generatif et pour sa generation
offline.

## Performance

Contexte de l'audit initial : analyse statique des sources et des objets CUDA
SM89 disponibles, sans execution Nsight sur GPU. Les chiffres de ressources
doivent etre regeneres avec le meme compilateur et le meme etat des sources une
fois la migration de namespaces stabilisee.

### PERF-001 — Reduire l'indexation 64 bits dans les chemins chauds

**Priorite : haute.** Le decodage des resultats emploie `std::size_t`, y compris
pour les divisions et modulos cartesiens dans
[`result_index.cuh`](../src/common/result_index.cuh). Le SASS SM89 disponible
montre une longue sequence d'instructions de division 64 bits avant le calcul
Black-Scholes.

Evaluer une representation device en `std::uint32_t` apres validation hote,
avec promotion 64 bits seulement pour les adresses qui l'exigent. Conserver les
compteurs Philox et les dimensions globales susceptibles de depasser cette
plage en 64 bits. Separer egalement les modes aligned et Cartesian par template
ou par kernel pour supprimer le branchement et le decodage inutiles du chemin
aligned.

**Critere de cloture :** absence de division/modulo 64 bits dans le chemin
aligned et reduction mesuree du cout du decodage Cartesian, sans changement du
mapping des resultats ni des sequences aleatoires.

### PERF-002 — Adapter la geometrie aux specialisations riches en registres

**Priorite : haute.** Les objets SM89 disponibles indiquent environ 92 a
96 registres par thread pour plusieurs kernels Bates Phoenix/cliquet. Avec
512 threads, cela limite theoriquement ces specialisations a un bloc resident,
soit environ 33 % des warps maximaux sur Ada. L'appel d'occupation dans
[`monte_carlo_kernel.cuh`](../src/common/monte_carlo/monte_carlo_kernel.cuh)
verifie seulement qu'un bloc peut resider ; il ne choisit pas la geometrie.

Comparer au minimum 128, 256 et 512 threads pour chaque famille representative.
Comparer aussi 64, 128 et 256 threads pour les closed forms lourdes, ou
1 000 resultats et 256 threads ne produisent que quatre blocs.

**Critere de cloture :** geometrie choisie a partir de mesures mediane/p95 et
des compteurs Nsight, avec absence de nouveaux spills. Ne pas introduire de
plafond global de registres ni de `__launch_bounds__` sans preuve sur toutes les
architectures cibles.

### PERF-003 — Reduire la pression registre des reductions LSM

**Priorite : haute.** La base LSM de taille 6 produit 21 termes de matrice de
Gram, 6 termes de second membre et un compteur. La reduction generique de
[`reductions.cuh`](../src/common/reductions.cuh) conserve ainsi 28 valeurs FP64
par thread. Les kernels de regression observes atteignent 86 a 88 registres par
thread.

Evaluer une reduction specialisee distribuant les statistiques entre lanes ou
warps, ou separant Gram et second membre. La shared memory disponible n'est pas
le facteur limitant. Conserver la resolution de Cholesky et les operations
sensibles en FP64. Ne pas reintroduire le tri des lignes d'exercice deja mesure
et rejete dans `abandoned-work.md` sans preuve nouvelle.

**Critere de cloture :** baisse des registres et amelioration du temps LSM sur
les quatre familles American, sans degradation des coefficients, prix, erreurs
standards ni ordre deterministe de reduction au-dela des tolerances acceptees.

### PERF-004 — Reduire la taille du code CIR non-central-chi-square

**Priorite : haute.** Les sections machine des kernels CIR swaption SM89
atteignent environ 192 Ko par specialisation, contre environ 8 Ko pour une
European Black-Scholes et 24 Ko pour une swaption Vasicek/OU. Les grandes
routines iteratives de
[`noncentral_chi_square.cuh`](../src/common/noncentral_chi_square.cuh) sont
force-inlinees et dupliquees dans plusieurs chemins.

Evaluer `__noinline__` uniquement sur les grandes routines gamma,
Poisson-mixture et saddlepoint, tout en conservant les petites primitives
force-inlinees.

**Critere de cloture :** reduction documentee de la taille du code et des
instruction-cache stalls, ou preuve mesuree que l'inlining actuel est plus
rapide. Les probabilites et prix CIR doivent conserver leurs tolerances sur les
cas centraux et de stress.

### PERF-006 — Optimiser les schedules explicites ragged

**Priorite : moyenne.** Les pools de
[`schedule.cuh`](../src/product/european_swaption/schedule.cuh) sont contigus
par schedule, mais deux threads adjacents peuvent lire des offsets eloignes et
traiter des nombres de paiements differents. Cela expose les warps a des acces
non coalesces et a de la divergence.

Mesurer les transactions globales et la branch efficiency. Si le cout est
confirme, evaluer soit un groupement par `payment_count` avec layout
payment-major/ELLPACK, soit le kernel un-bloc-par-swaption.

**Critere de cloture :** coalescence et divergence mesurees ; optimisation
retenue seulement si elle ameliore les schedules explicites sans penaliser le
chemin regulier.

### PERF-007 — Supprimer les grandes frames locales des kernels de sample

**Priorite : moyenne.** Le sample Black-Scholes avec calendrier de taille 16
atteint jusqu'a 80 registres et une frame de stack de 224 octets. Plusieurs
tableaux locaux restent simultanement vivants dans
[`sample.cu`](../src/model/equity/black_scholes/sample.cu) : jours, temps,
intervalles, transitions et spots observes.

Evaluer la preparation hote ou par bloc des calendriers, le partage des donnees
communes et une production progressive evitant de conserver tous les tableaux.

**Critere de cloture :** diminution de la stack et des acces locaux confirmee
par PTXAS et Nsight, avec conservation exacte du layout time-major et du contenu
des samples.

### PERF-008 — Evaluer une accumulation mixte dans les boucles chaudes

**Priorite : moyenne.** Les parametres, etats et dynamiques sont majoritairement
FP32, ce qui est adapte au GPU. Le FP64 apparait toutefois a chaque observation
pour certaines moyennes dans
[`handlers.cuh`](../src/common/equity/handlers.cuh), dans les moments par thread
du kernel Monte-Carlo et dans les reductions LSM. Son cout relatif peut etre
important pour les produits exacts ou courts comportant beaucoup de paths.

Evaluer, famille par famille, une accumulation locale FP32 compensee ou par
chunks, suivie d'une aggregation finale FP64. La regression et Cholesky LSM
restent FP64 tant qu'une validation numerique complete ne demontre pas qu'un
autre schema est acceptable. Ne pas activer globalement `--use_fast_math` ni
remplacer systematiquement `expf`/`logf` par leurs variantes approximatives.

**Critere de cloture :** decision mesuree par famille avec compte des
instructions FP32/FP64, temps kernel, biais, erreur standard et cas de stress.
Une absence de gain ou une erreur accrue doit etre documentee dans
`abandoned-work.md` avant retrait du constat.

### PERF-009 — Reduire les couts fixes des lancements courts

**Priorite : moyenne.** Les launchers repetent les validations de pointeurs,
requetes de proprietes du device et parfois les requetes d'occupation. Les
generateurs utilisent des allocations et copies synchrones sur le stream par
defaut. Ces couts sont secondaires pour le Monte-Carlo lourd, mais peuvent
dominer les petits batches closed form.

Evaluer un cache par device des proprietes et donnees d'occupation, des buffers
GPU reutilisables et, pour un usage streaming, des buffers hote pinned avec
copies asynchrones et double buffering. Conserver une frontiere explicite entre
la validation initiale et l'execution repetee plutot que de supprimer les
controles de securite.

**Critere de cloture :** profil hote/device separant validation, allocation,
copies, kernel et I/O ; optimisation retenue uniquement pour les workflows ou
le cout fixe est significatif.

### PERF-010 — Installer une baseline de regression reproductible

**Priorite : haute.** Les diagnostics existants exposent les ressources et
l'occupation theorique, et les catalogues enregistrent certains temps, mais il
n'existe pas encore de matrice de performance controlee couvrant les principales
classes de kernels.

Creer une baseline incluant au minimum Black-Scholes closed form et range
accrual, Heston Asian, Bates Phoenix/cliquet, un produit Levy exact a grand
nombre de paths, CIR ZCB et swaption, Vasicek swaption, les quatre modeles
American et les samples N=1/8/16. Rough doit etre mesure separement. Enregistrer
GPU, SM, compilateur, flags, geometrie, donnees, warmups, mediane et p95, ainsi
que registres, spills, stack, shared et taille du code.

**Critere de cloture :** benchmark reproductible et seuils par architecture,
associes a une validation numerique. Les temps wall des catalogues ne doivent
pas etre utilises seuls, car ils incluent allocation, copies et I/O.
