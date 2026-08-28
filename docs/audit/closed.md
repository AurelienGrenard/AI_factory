# Constats d'audit fermes

## Objet

Ce document est le registre compact des constats issus de `query.md` qui sont
effectivement clos. Avant de creer un constat, rechercher ici une cause et un
perimetre equivalents : une autre formulation ou une autre correction proposee
ne justifie pas un nouvel identifiant.

Chaque entree conserve la signature du probleme initial, la nature de la
cloture, la decision prise, sa preuve et la condition minimale de reouverture.
Les regles durables restent dans les contrats d'implementation. Un constat qui
regresse reprend son identifiant et retourne dans `response.md` avec son
historique de cloture.

## Analytics

### ANALYTICS-001 — Ne pas faire dependre les analytics modele du schedule ou des formules d'un produit concret

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** des analytics fixed income situes sous `src/model` incluaient le schedule et des types de formule appartenant a un produit concret.
- **Cloture :** `BusinessDayFixedLegScheduleView` porte la vue partagee sous `src/common`; les produits adaptent leurs parametres sans devenir une dependance des providers modele.
- **Preuve :** recherche finale sur les 18 fichiers analytics modele sans aucun include `src/product`; tests du contrat analytics fixed income et matrice CUDA passes.
- **Reouvrir seulement si :** un provider ou une formule analytique sous `src/model` inclut de nouveau un produit concret.

## Numerical robustness

### NUM-001 — Interdire a Jamshidian de publier une racine apres stagnation ou epuisement des iterations

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** les solveurs Jamshidian scalaire et cooperatif pouvaient rendre leur dernier milieu de bracket sans certifier le residu apres stagnation FP32 ou nombre maximal d'iterations.
- **Cloture :** le residu final est certifie; un resultat non convergent devient `NaN` au lieu d'un prix silencieusement plausible.
- **Preuve :** `numerical_robustness_cuda` couvre iteration maximale nulle et bracket FP32 effondre pour les deux solveurs; 48/48 tests CUDA passent.
- **Reouvrir seulement si :** un chemin peut de nouveau produire une racine finie sans satisfaire la tolerance documentee.

### NUM-002 — Distinguer les causes d'echec des regressions Longstaff-Schwartz

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** un booleen unique confondait absence de candidats, echantillon sous-determine et echec numerique, puis conservait silencieusement la continuation.
- **Cloture :** `RegressionStatus` distingue succes, zero candidat, `candidate_count <= basis_size`, statistiques non finies, echec de Cholesky et coefficients non finis. Les deux premiers cas non resolubles conservent explicitement la continuation et restent comptes; les trois echecs numeriques invalident prix et erreur standard, remontent dans `LaunchResult` et bloquent les generateurs avant publication.
- **Preuve :** cinq causes synthetiques classees sur GPU, test d'invalidation `NaN`, nouveau test OU/G2 payer/receiver reproductible bit a bit, sept tests LSM cibles passes et generateurs OU/G2 executes avec diagnostics non fatals explicites. Les anciens echecs Heston ligne 2 et Levy ligne 3 sont identifies comme `insufficient_candidates`, pas comme echecs de factorisation.
- **Reouvrir seulement si :** une cause de regression redevient indifferenciee, un echec numerique fatal peut produire un prix fini publiable, ou un cas vide/sous-determine est traite sans politique ni diagnostic explicites.

### NUM-003 — Preserver les coefficients et la decision Longstaff-Schwartz en FP64

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** les coefficients de regression FP64 et la prediction de continuation etaient retrecis en FP32 avant la comparaison exercice/continuation.
- **Cloture :** coefficients, prediction et comparaison restent en `double`; `exercise_decision.cuh` centralise la selection sans cast intermediaire.
- **Preuve :** test d'une frontiere sensible ou le chemin FP64 choisit `1.0`, contre `0.25` pour l'ancien retrecissement; tests LSM et matrice CUDA complets passent.
- **Reouvrir seulement si :** un cast FP32 reapparait entre regression, prediction et decision d'exercice.

### NUM-004 — Stabiliser les endpoints Schobel-Zhu quand la mean reversion est petite

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** les expressions `1 - exp(-x)` des variances et correlations d'endpoint perdaient leur precision lorsque `mean_reversion * dt` etait petit.
- **Cloture :** les differences exponentielles utilisent `expm1f` et conservent les limites analytiques attendues.
- **Preuve :** grille de 10 points comparee a une reference host FP64, incluant mean reversion 0.03 et 10 ainsi que de petits pas; `numerical_robustness_cuda` passe.
- **Reouvrir seulement si :** une nouvelle formule d'endpoint reutilise une soustraction exponentielle instable ou echoue sur cette grille.

### NUM-005 — Refuser les moments Monte Carlo invalides au lieu de masquer leur variance

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** la reduction acceptait moins de deux echantillons ou des moments non finis et rabattait toute variance negative vers zero.
- **Cloture :** cardinalite et finitude sont validees; une variance materiellement negative produit `NaN`, seul l'arrondi negatif borne est ramene a zero.
- **Preuve :** cas valides, 0/1 echantillon, NaN, Inf, variance negative et erreur d'arrondi couverts dans `numerical_robustness_cuda`.
- **Reouvrir seulement si :** un prix ou une erreur standard peut etre publie depuis des moments non finis, insuffisants ou incoherents.

## Naming

### NAME-004 — Ne plus inclure textuellement des fichiers `.cu` comme headers d'implementation

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** 27 fragments `dynamics.cu`, `analytics.cu` ou `term_structure.cu` etaient inclus par `#include` au lieu d'etre des unites de traduction CUDA.
- **Cloture :** les fragments inclus portent `*_impl.cuh`; les `.cu` restants sont des unites autonomes enregistrees dans CMake.
- **Preuve :** build `all_models` en 315 etapes et recherche statique sans aucun `#include "*.cu"`.
- **Reouvrir seulement si :** un fichier `.cu` est de nouveau inclus textuellement ou echappe au graphe CMake.

### NAME-005 — Remplacer les noms affines symboliques et les types redondants

- **Nature :** devenu inapplicable apres meta-revue le 2026-08-27; aucun changement de code.
- **Signature :** les analytics fixed income exposent `log_A`, `A` et `B`, certains types ajustes repetent le modele dans leur nom et plusieurs implementations emploient `a`/`b` localement.
- **Cloture :** [`cuda/model-analytics-contract.md`](../cuda/model-analytics-contract.md) definit explicitement `log_A`, `A` et `B` comme la surface canonique des coefficients affines et distingue le loading `B` a un ou deux facteurs. Cette notation mathematique publique est donc une exception documentee, pas une incoherence. La seule repetition d'un nom de modele dans un type ne prouve ni ambiguite, ni collision, ni impact technique et reste une preference de style exclue de l'audit.
- **Preuve :** section `Signatures fixed income`, contrat des providers et section `Nommage` du meme contrat; meta-revue E13.
- **Reouvrir seulement si :** une ambiguite, collision, erreur d'usage ou divergence entre modeles est attribuee a ces noms, ou si le contrat canonique abandonne explicitement cette notation.

### NAME-007 — Remplacer les noms permanents de tests fondes sur `new`, `old` ou `legacy`

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** `new_equity_dynamics_cuda_test.cu` et sa cible decrivaient l'age historique du test plutot que les modeles et le contrat verifies.
- **Cloture :** le test et la cible se nomment `merton_kou_cev_schobel_zhu_dynamics_cuda`; le type LSM `LegacyTwoFactorLaguerreBasis` a aussi recu un nom semantique.
- **Preuve :** recherche statique sans les anciens symboles; build et 48/48 tests CUDA passent.
- **Reouvrir seulement si :** un nom permanent de source, cible ou type repose de nouveau sur son anciennete plutot que sa responsabilite.

## Project structure

### STRUCT-001 — Garder les pricing policies produit hors de `src/common`

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** des policies fixed income et equity concretes vivaient dans `common` et incluaient directement les parametres d'un produit.
- **Cloture :** les policies sont dans `src/product/<product>/pricing_policy.cuh`; `common` ne conserve que les primitives independantes des produits.
- **Preuve :** recherche statique sans include produit dans les analytics modele; builds host/CUDA complets passes.
- **Reouvrir seulement si :** une policy liee a un produit concret ou son header de parametres retourne sous `src/common`.

### STRUCT-009 — Aligner chemins, namespaces et responsabilites des modules communs

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** le helper mean-reverting etait sous `src/model/fixed_income/common` sans namespace `common`, deux abstractions de grille temporelle divergeaient et le sampling restait regroupe historiquement.
- **Cloture :** helper deplace sous `src/common/fixed_income`, `time_grid.cuh` inutilise supprime et sampling/LSM repartis en modules nommes par responsabilite.
- **Preuve :** anciens chemins absents, documentation `src/common` alignee et builds agreges passes.
- **Reouvrir seulement si :** chemins et namespaces divergent de nouveau ou un module commun redevient un regroupement sans responsabilite coherente.

## Tools and src ownership

### BOUNDARY-001 — Supprimer toute dependance de `src` vers les composants de `tools`

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** les loaders de `src` incluaient `tools/datasets/dataset_validation.hpp` et leurs cibles liaient la serialisation/generation offline de `tools`.
- **Cloture :** validation de lecture deplacee dans `src/common/dataset_validation.*`; `tools` depend de cette cible et non l'inverse.
- **Preuve :** aucun include `tools` sous `src`; 3/3 tests host et build de 1 193 etapes des `price_generators` sans mathDx passent.
- **Reouvrir seulement si :** un fichier ou une cible runtime de `src` depend de nouveau de `tools`.

### BOUNDARY-003 — Centraliser dans `src` les primitives mathematiques reutilisees par `tools`

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** les generateurs recopiaient les forwards instantanes Nelson-Siegel/Svensson et la conversion mean-reversion/dispersion stationnaire appartenant au modele.
- **Cloture :** primitives pures host/device exposees sous `src/curve` et `src/common/fixed_income`, puis consommees par les generateurs.
- **Preuve :** une seule definition par primitive, tests de parite FP32/FP64 et builds generateurs passes.
- **Reouvrir seulement si :** un generateur reimplemente une identite mathematique deja possedee par `src`.

### BOUNDARY-005 — Supprimer les facades sans consommateur et les dossiers placeholders

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** la facade CMake `ai_factory_dataset_tools`, un `.gitkeep` sous `src/generative` et des dependances tools de la facade runtime n'avaient pas de consommateur ou de contrat reel.
- **Cloture :** facade et placeholder supprimes; la facade runtime ne publie plus de composant offline.
- **Preuve :** anciens noms absents des sources et du graphe CMake; builds agreges passes.
- **Reouvrir seulement si :** une facade de compatibilite ou un dossier vide est ajoute sans consommateur, export ou contrat documente.

## Build and CUDA instantiations

### BUILD-001 — Ne pas publier sans mathDx une cible rough Bergomi qui ne peut pas etre liee

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** CMake enregistrait un generateur de prix rough Bergomi sans `AI_FACTORY_MATHDX_ROOT`, puis le link echouait faute de CUDA runtime et d'instanciation du launcher FFT.
- **Cloture :** l'enregistrement des generateurs dependants de mathDx est garde par `AI_FACTORY_MATHDX_ROOT`; les cibles publiees sans mathDx sont toutes constructibles.
- **Preuve :** configuration fraiche sans mathDx et construction reelle des 1 193 etapes de `price_generators`.
- **Reouvrir seulement si :** une configuration supportee publie de nouveau une cible dont une dependance ou instanciation requise est absente.

## Performance

### PERF-003 — Reduire la pression registre des reductions LSM

- **Nature :** hypothese de performance refutee par mesure le 2026-08-27; aucun prototype conserve.
- **Signature :** remplacer l'accumulation combinee de 28 statistiques FP64 par thread soit par deux passages Gram/second membre, soit par une distribution des statistiques entre lanes d'un warp.
- **Cloture :** le kernel combine d'origine est conserve. La separation Gram/second membre reduit les registres G2 SM89 de 118 a 103 et 78, mais ne change pas l'occupation du passage Gram (33,3 %), ajoute un kernel et une lecture des chemins par date, et augmente le temps profile de 435 ms a 534 ms (+22,6 %). La distribution par warp conserve elle aussi les prix G2 bit a bit, mais reduit de 32 fois le parallelisme entre trajectoires et fait passer le temps kernel du generateur de 0,415 s a 4,965 s, soit environ x12.
- **Preuve :** profils Nsight Systems du generateur G2 1 000 lignes, diagnostics exacts SM89 sur RTX 4090 Laptop, comparaison des JSON hors metadonnees de timing bit a bit et sept tests LSM GPU passes pour chaque prototype fonctionnel. La baseline finale G2 retrouve 118 registres, aucune memoire locale, 33,3 % d'occupation theorique et 0,413 s kernel.
- **Reouvrir seulement si :** une autre reduction conserve un chemin par thread et demontre un gain reproductible sur les familles American et Bermudan representatives sans modifier prix, erreurs standards, FP64 ou diagnostic type.

### PERF-011 — Trier les lignes LSM American/Bermudan par nombre de dates d'exercice pour supprimer les blocs backward inactifs

- **Nature :** hypothese de performance refutee par mesure le 2026-08-27.
- **Signature :** tri intra-batch ou global par `exercise_count`, reduction de `grid.y` au prefixe encore actif, puis restauration de l'ordre original des resultats.
- **Cloture :** 354 executions CUDA et 42 configurations donnent environ 1,3 % de gain median global; le tri global peut regresser, notamment Heston de 2,1 %, et la complexite de permutation n'est pas justifiee. Une cle secondaire par nombre de pas ne supprime aucune divergence intra-warp et aucun profil ne lui attribue un cout significatif.
- **Preuve :** resultats bit-identiques; sur le catalogue 1 000 lignes a 1 048 576 paths, le tri intra-batch gagne 5,9 % en Variance Gamma, 4,4 % en NIG, 2,0 % en Heston et 3,6 % en Bates, sans gain robuste sur les distributions homogenes ou etroites.
- **Limite de provenance :** le rapport brut, le manifeste des 42 configurations et l'empreinte des binaires ne sont pas versionnes dans ce quadriptyque. Cette entree conserve la decision historique, mais ses chiffres ne constituent pas une baseline courante sous le referentiel v2 tant que cette provenance n'est pas rattachee a un artefact durable.
- **Reouvrir seulement si :** un nouveau materiel ou une distribution d'exercices differente montre un gain reproductible suffisamment important pour justifier indices, permutation et scatter, ou si la provenance historique ne peut pas etre produite au moment ou cette decision doit etre reutilisee.

## Naming — second passage de remediation

### NAME-002 — Clarifier le namespace public des produits

- **Nature :** clarifie et verifie le 2026-08-27.
- **Signature :** le namespace public plat `ai_factory::workbench::product` pouvait etre lu comme une taxonomie incomplete ou entrer en contradiction avec les sous-repertoires du catalogue.
- **Cloture :** le namespace plat est confirme comme surface publique volontaire; les types portent le nom descriptif du produit et les helpers internes restent dans `detail`. Cette regle et l'absence de namespaces miroir par dossier sont documentees dans le README racine.
- **Preuve :** inventaire des declarations produit, compilation agregee de tous les modeles et tests host/CUDA; aucun symbole public ambigu n'a ete introduit.
- **Reouvrir seulement si :** deux produits exposent des symboles publics en collision, si une API exige une taxonomie imbriquee stable, ou si le contrat de namespace est abandonne.

### NAME-003 — Distinguer les launchers de modele des primitives generiques

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** les launchers rough publics `launch_european_option_cuda` et `launch_hybrid_fft_cuda` ne nommaient pas le modele alors que leurs signatures et implementations lui etaient propres.
- **Cloture :** les launchers externes rough Bergomi et rough SABR portent des noms qualifies par modele; seuls les composants reellement communs sous `src/common/volterra` conservent des noms methodes-neutres.
- **Preuve :** sources, tests, catalogues, probe de validation et templates de bindings migrent atomiquement; `all_models`, la matrice CUDA et la compilation manuelle du probe rough SABR passent.
- **Reouvrir seulement si :** un launcher public propre a un modele reprend un nom generique, ou si deux implementations distinctes revendiquent le meme symbole non qualifie.

### NAME-006 — Encoder les unites dans tous les noms temporels

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** des membres, pointeurs et variables entiers nommes `maturity`, `payment_times` ou equivalents ne distinguaient pas jours ouvrables, pas de grille et annees preparees.
- **Cloture :** les quantites calendaires entieres portent le suffixe `_days`, les valeurs preparees en annees `_years`, et les nombres de transitions `_steps`; les cles JSON externes restent stables. `time::year_fraction` est utilisable host et device.
- **Preuve :** recherche statique sur les identifiants temporels entiers sans reste ambigu; tests du contrat catalogue pour les bases 252, 360 et 365, builds host/CUDA complets.
- **Reouvrir seulement si :** une quantite temporelle publique ou transmise au device perd son unite dans son nom, ou si une conversion implicite remplace `year_fraction`.

## Project structure — second passage de remediation

### STRUCT-006 — Unifier les 46 loaders de datasets de parametres

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** chaque produit, modele et courbe repetait ouverture JSON, verification de famille, cardinalite, enveloppe `parameters` et parcours des lignes, avec des diagnostics susceptibles de diverger.
- **Cloture :** `src/common/dataset_validation.*` possede la lecture et la validation communes via `ParameterDatasetFamily`, `read_parameter_dataset` et `load_parameter_rows`; les 46 loaders les reutilisent. Le loader de swaptions garde seulement son assemblage specialise du pool de schedules.
- **Preuve :** aucune ouverture directe `std::ifstream stream(dataset_path)` ne subsiste dans les loaders; cas d'erreurs communs, familles et schedules multi-lignes sont testes; 3/3 tests host passent.
- **Reouvrir seulement si :** un loader reimplemente l'enveloppe commune, produit un diagnostic incompatible, ou si une nouvelle famille ne peut pas exprimer sa validation specialisee apres la lecture commune.

## Pricing policies and concepts — second passage de remediation

### POLICY-001 — Rendre explicites et testables les budgets de stockage des policies

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** les concepts imposaient des caps de taille a 128, 256 ou 2 048 octets sans contrat central, message actionnable ni preuve de ressources sur les architectures ciblees.
- **Cloture :** le contrat publie les cinq budgets, les raisons et les alternatives `view`/pool device; les `static_assert` indiquent le cap exact et la remediation. Un test CUDA compile chaque limite et rejette conceptuellement la premiere taille hors budget.
- **Preuve :** le test `policy_size_budgets_cuda` passe sur SM89; les probes offline SM75/86/89 conservent les tailles demandees sans spill observe dans les objets inspectes. Les cinq geometries atteignent 100 % d'occupation theorique sur le SM89 mesure.
- **Reouvrir seulement si :** un cap change sans mesure, une policy valide depasse le budget, ou un materiel cible montre une regression de ressources qui justifie un budget architecture-dependant.

## Tools and src ownership — second passage de remediation

### BOUNDARY-002 — Remplacer le booleen public de construction des prix par un type metier

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** le booleen `cartesian_product` traversait launchers, outils et recettes sans rendre lisibles les deux modes d'indexation.
- **Cloture :** `PriceConstruction::{aligned,cartesian}` est la representation canonique dans `src/common/price_construction.cuh`; la migration est atomique jusque dans les templates de bindings et aucun wrapper booleen transitoire n'est conserve.
- **Preuve :** recherche statique sans ancien parametre public; tests aligned/cartesian, cardinalites et debordements; compilation des generateurs, de tous les modeles et des tests CUDA.
- **Reouvrir seulement si :** un booleen ou un entier brut encode de nouveau le mode de construction a une frontiere publique.

## Performance — second passage de remediation

### PERF-001 — Reduire le cout du decodage d'indices de resultats

- **Nature :** corrige et mesure le 2026-08-27.
- **Signature :** chaque thread recalculait avec des divisions `size_t` les indices parametre/produit alors que les grilles CUDA publiees sont bornees.
- **Cloture :** `result_index.cuh` centralise les decodeurs et fournit un chemin device `uint32_t` apres validation host de la cardinalite; toute cardinalite superieure a `UINT32_MAX` est refusee avec instruction de decoupage. La specialisation compile-time du mode, sans le passage 32 bits, est rejetee faute de gain.
- **Preuve :** mappings et checksums identiques sur les modes 1D, aligned, cartesian et trois dimensions; sur SM89, la mediane passe de 0,270 ms a 0,124 ms, soit environ 54 %, sans changer les identifiants. Les CV kernel de 0,7 % et 1,3 % restent sous le seuil de 5 % du protocole.
- **Reouvrir seulement si :** une cardinalite valide exige plus de 32 bits dans un lancement unique, si un nouveau mapping contourne le decodeur canonique, ou si une architecture cible invalide le gain.

### PERF-002 — Mesurer la geometrie des kernels aux policies riches

- **Nature :** hypothese tranchee par mesure le 2026-08-27; geometrie de production conservee.
- **Signature :** les policies Phoenix et Phoenix-memory Bates riches en registres pouvaient sembler favoriser 128 ou 256 threads plutot que les 512 utilises par les generateurs.
- **Cloture :** 512 threads restent la geometrie par defaut : une occupation theorique plus faible ne predit pas le debit de ce workload et les variantes plus petites sont plus lentes.
- **Preuve :** a 32 prix et 4 096 trajectoires, Phoenix mesure 11,10/6,28/4,04 ms et Phoenix-memory 11,14/6,30/4,05 ms pour 128/256/512 threads. Prix identiques, 101 registres, 112 octets de shared statique et aucun spill local pour les trois variantes sur SM89.
- **Reouvrir seulement si :** la policy, la distribution des calendriers, l'architecture ou le nombre de trajectoires change assez pour inverser reproductiblement ce classement.

### PERF-004 — Borner l'inlining des grandes primitives noncentral-chi-square

- **Nature :** corrige et mesure le 2026-08-27.
- **Signature :** plusieurs grandes routines numeriques etaient force-inlinees dans les kernels CIR, augmentant registres, taille de code et temps de compilation sans preuve de debit.
- **Cloture :** les series gamma, fraction continue, paire gamma, melange de Poisson et saddlepoint sont des helpers internes `static __noinline__`; la variante force-inline ne vit que dans une cible experimentale separee.
- **Preuve :** sur 16 384 resultats, le chemin de production mesure 1,063 ms contre 1,158 ms force-inline (-8,2 %), avec des CV kernel de 0,5 % et 1,2 %. Il utilise 56 contre 64 registres, 75 % contre 66,7 % d'occupation theorique, aucun spill; l'archive passe d'environ 3,40 Mo a 1,64 Mo. Les compteurs de stalls Nsight Compute n'etaient pas accessibles (`ERR_NVGPUCTRPERM`) et ne sont pas revendiques.
- **Reouvrir seulement si :** un compilateur ou GPU cible mesure un avantage reproductible de l'inlining sans inflation disproportionnee des registres, cubins ou temps de build.

### PERF-006 — Coalescer et cooperer sur les schedules explicites de swaptions

- **Nature :** corrige et mesure le 2026-08-27.
- **Signature :** le pool row-major des schedules explicites donnait des lectures stridees et un seul thread effectuait l'evaluation Jamshidian d'une ligne heterogene.
- **Cloture :** le loader produit un ELLPACK payment-major, les vues transportent le stride de produits et les surcharges explicites CIR, OU, Vasicek et Hull-White evaluent les paiements cooperativement. Le chemin regulier scalaire reste intact.
- **Preuve :** le cas heterogene 2–30 paiements passe d'environ 6,99 ms a 0,292 ms (environ x24); l'homogene explicite reste comparable au regulier, 0,296 contre 0,286 ms. Le kernel explicite descend a 46 registres, 83,3 % d'occupation theorique et aucun spill; un test deux lignes verifie exactement la transposition du pool.
- **Reouvrir seulement si :** une nouvelle representation de calendrier recree des lectures stridees, si le chemin explicite diverge numeriquement du contrat, ou si une autre architecture inverse le gain.

### PERF-008 — Conserver FP64 pour les accumulations numeriquement sensibles

- **Nature :** hypothese de precision mixte refutee pour le contrat generique le 2026-08-27.
- **Signature :** remplacer les sommes FP64 de moments et statistiques par FP32 compense ou par des chunks FP32 pouvait accelerer les boucles chaudes.
- **Cloture :** FP64 reste le contrat commun; aucune approximation globale n'est introduite sans budget d'erreur propre a une famille de prix et a son erreur standard.
- **Preuve :** sur un flux non negatif a echelles mixtes, FP64 mesure 11,77 ms avec reference exacte, FP32 compense 3,10 ms mais `1,47e-2` d'erreur absolue, et chunks FP32 5,04 ms mais `1,43e-1`. Les erreurs relatives sont petites mais non nulles et ne satisfont pas un contrat universel non specifie.
- **Reouvrir seulement si :** une famille documente une tolerance prix/erreur standard, une distribution representative et une validation independante qui autorisent explicitement une accumulation mixte specialisee.

### PERF-009 — Quantifier et amortir le cout fixe des launchers courts

- **Nature :** mesure et decision documentee le 2026-08-27; validation fail-fast conservee.
- **Signature :** allocations, inspections de pointeurs et synchronisations des petits launchers closed-form pouvaient dominer les prix tres courts et suggerer un cache de metadata device.
- **Cloture :** les appels doivent etre groupes par les lanceurs grid-stride existants; aucun cache de pointeur n'est ajoute car `cudaFree` et la reutilisation d'adresse rendraient sa duree de vie incorrecte. La mutualisation ulterieure des runners a ete traitee independamment sous `STRUCT-007`/`BOUNDARY-004`, sans introduire ce cache non possede.
- **Preuve :** 256 appels par echantillon mesurent environ 5–7 microsecondes par lancement pour 1, 32 et 1 024 resultats. La variance dynamique est elevee (CV 22–28 %), donc le protocole classe les comparaisons fines comme inconclusives, mais confirme l'ordre de grandeur; les generateurs catalogue amortissent deja sur 1 000 resultats.
- **Reouvrir seulement si :** un appel public unitaire devient un workload catalogue significatif, ou si une API de contexte possede explicitement et surement la duree de vie des allocations.

### PERF-012 — Mesurer le crossover convolution directe/FFT pour Volterra

- **Nature :** hypothese de chemin direct refutee par mesure le 2026-08-27.
- **Signature :** une convolution directe bornee pouvait eviter le cout fixe cuFFTDx aux tres petites grilles Volterra.
- **Cloture :** le chemin de production reste exclusivement FFT. L'implementation directe reproductible est exclue par preprocesseur de la bibliotheque normale et ne vit que dans une archive/benchmark experimental borne a 32 pas.
- **Preuve :** a 8 pas, FFT mesure 0,050 ms contre 0,068 ms direct (+34,7 %); a 16 pas 0,071 contre 0,216 ms; a 32 pas 0,099 contre 0,803 ms. Prix identiques a 8/16 pas et ecart de `1,86e-9` a 32 pas avec le meme mapping Philox.
- **Reouvrir seulement si :** un nouveau GPU, compilateur ou primitive de convolution directe bat reproductiblement FFT a une longueur supportee tout en conservant mapping aleatoire, prix et taille de production.

### PERF-013 — Retuner les instanciations FFT longues

- **Nature :** corrige et mesure le 2026-08-27.
- **Signature :** l'instanciation FFT 8 192 points a 32 elements par thread utilisait 139 registres, limitait l'occupation et gonflait les archives rough.
- **Cloture :** cette longueur utilise 16 elements par thread; le reste de la courbe 16–4 096 conserve ses specialisations mesurees et le dispatch deterministe.
- **Preuve :** la mediane 8 192 tombe de 22,45 a 17,87 ms (-20,4 %), les registres de 139 a 72 et l'occupation theorique de 16,7 % a 33,3 %, sans spill. Les archives Bergomi/SABR diminuent d'environ 10,3 % et la compilation mesuree d'environ 6,6 %; prix et erreur standard restent dans le contrat et tous les tests rough passent.
- **Reouvrir seulement si :** une architecture ou version cuFFTDx change le classement, si une longueur devient dominante sans baseline, ou si registres/spills regressent.

### PERF-014 — Borner le workspace Volterra et evaluer le recouvrement multi-stream

- **Nature :** mesure et decision documentee le 2026-08-27; strategie de production conservee.
- **Signature :** le chunking sequentiel et un seul stream pouvaient sous-utiliser le GPU ou retenir un workspace excessif sur les jeux Volterra de production.
- **Cloture :** un workspace reutilisable de 65 536 trajectoires et un stream restent la strategie canonique. Il borne le cas 252 pas/1 048 576 trajectoires a environ 63,14 MiB, sature mieux le GPU que les petits chunks et preserve l'ordre des reductions et les seeds.
- **Preuve :** medianes SM89 de 37,68, 14,07 et 11,17 ms pour des chunks 4 096, 16 384 et 65 536, sorties strictement identiques; huit prix sequentiels mesurent 91,83 ms, soit 11,48 ms/prix, sans gain de batching concurrent visible. Les quatre configurations sont conservees dans la baseline v1.
- **Reouvrir seulement si :** le budget memoire cible devient inferieur a 63,14 MiB, si un nouveau GPU montre un recouvrement reproductible superieur a 5 %, ou si le workspace peut etre partage sans modifier ordre deterministe ni duree de vie.

## Project structure — troisieme passage de remediation

### STRUCT-003 — Deriver la matrice de bindings depuis une source de verite unique

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** les couples modele-produit equity etaient materialises par des paires `.cu/.cuh` et des listes CMake manuelles susceptibles de diverger du prototype de generation.
- **Cloture :** `tools/codegen/pricing_bindings/manifest.py` est la source typee des 18 modeles, de leurs familles et des 21 produits non American. Le generateur produit toute la matrice `.cu/.cuh` ainsi que `cmake/generated/EquityPricingBindings.cmake`; `AIFactoryTargets.cmake` consomme ce fragment pour enregistrer les unites ordinaires, Volterra/mathDx et N-factor. Les recettes catalogue sont decouvertes depuis leur arborescence avec `CONFIGURE_DEPENDS`, sans liste de couples recopiee.
- **Preuve :** la generation de 757 artefacts dans `/tmp` compare bit a bit avec le tree; le CTest `pricing_binding_codegen` passe; `all_models` avec mathDx termine 784/784 et la cible sans mathDx `price_generators` 1 156/1 156. Preuve consolidee E21.
- **Reouvrir seulement si :** un binding equity non American ou son inscription CMake doit etre ajoute manuellement hors manifeste, si un fichier genere diverge sans echec CI, ou si une recette catalogue exige une nouvelle inscription explicite.

### STRUCT-007 — Centraliser l'orchestration CUDA des generateurs de catalogue

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** les recettes ordinaires recopiaient allocations, transferts, warmup, evenements, chronometrage, copie des resultats et nettoyage CUDA.
- **Cloture :** `tools/cuda/pricing_runner.cuh` fournit buffers et evenements RAII ainsi que les chemins types analytique et Monte Carlo. Les 293 recettes ordinaires et les helpers de swaptions reutilisent cette execution; huit pipelines Longstaff-Schwartz et deux anciens pipelines cuFFTDx conservent leur workspace algorithmique comme echappatoires explicites et bornees.
- **Preuve :** le checker `catalog_generator_boundaries` inspecte 382 recettes, accepte exactement les 10 echappatoires revues et refuse toute nouvelle gestion CUDA brute; les 1 156 etapes de `price_generators` passent. Le test CUDA autonome du runner verifie sur GPU allocations, transferts, resultats analytique/Monte Carlo, erreurs standards, evenements et chronometrage. Les tests de schema et de catalogue passent. Preuve consolidee E21.
- **Reouvrir seulement si :** une recette ordinaire gere de nouveau une ressource CUDA, si une echappatoire apparait sans revue explicite, si le runner perd sa propriete RAII, ou si la publication change le schema des artefacts.

## Tools and src ownership — troisieme passage de remediation

### BOUNDARY-004 — Separer les responsabilites offline de `tools/datasets`

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** `tools/datasets/dataset.*` melangeait sampling, grilles, assemblage des datasets, JSON, YAML et orchestration de pricing.
- **Cloture :** le monolithe est supprime. `sampling.*`, `artifact_io.*`, `parameter_dataset.*` et `price_dataset.*` possedent des API et bibliotheques distinctes; l'execution CUDA vit sous `tools/cuda`, les orchestrations produit reutilisables sous `tools/pricing`, et les generateurs lient seulement les etapes qu'ils emploient. La facade `ai_factory_dataset_core` restante est une cible `INTERFACE` de compatibilite et ne reintroduit aucune implementation agregee.
- **Preuve :** quatre tests de stage independants, les tests `dataset_catalog`, `dataset_loaders` et `rough_sabr_dataset_loader`, le test GPU autonome du runner et les deux matrices de generateurs passent. Le checker interdit le retour du monolithe et des anciens helpers. Preuve consolidee E21.
- **Reouvrir seulement si :** sampling, serialisation, assemblage ou execution sont de nouveau fusionnes dans une implementation commune, si un generateur depend d'une etape inutilisee, ou si une etape perd son test independant.

## Project structure — passage de remediation du 2026-08-28

### STRUCT-010 — Supprimer les inventaires README locaux recopies et non maintenus

- **Nature :** corrige une seconde fois et verifie le 2026-08-28.
- **Signature :** un README local non reference recopiait l'arbre et l'API
  d'implementation, tandis que le README racine decrivait encore des `.cu`
  inclus textuellement.
- **Cloture :** l'inventaire `src/common/README.md` de 985 lignes est retire;
  le README racine decrit les definitions incluses `*_impl.cuh` et les `.cu`
  autonomes sans recreer une seconde liste exhaustive.
- **Preuve :** ancien fichier absent, aucun include textuel de `.cu` et aucune
  occurrence documentaire restante de `dynamics.cu`/`analytics.cu` comme
  headers d'implementation; preuve ciblee E23.
- **Reouvrir seulement si :** un README local non reference recommence a
  dupliquer arbre/API ou si la documentation de build cite une frontiere
  source qui n'existe plus.

### STRUCT-012 — Declarer et generer les recettes American/LSM

- **Nature :** corrige et verifie le 2026-08-28.
- **Signature :** huit recettes American recopiaient manuellement le workspace,
  l'execution, les diagnostics et la publication Longstaff-Schwartz derriere
  une escape hatch du checker.
- **Cloture :** `AmericanRecipeSpec`, un template commun et
  `american_option_price_generation.cuh` generent les huit corps minces; le
  checker ne conserve aucune escape hatch CUDA American.
- **Preuve :** les huit recettes sont zero-diff dans la generation complete,
  les huit executables compilent, les quatre tests CUDA American/LSM passent et
  le checker refuse toute recette possedant de nouveau ses ressources CUDA;
  preuve ciblee E23.
- **Reouvrir seulement si :** une recette American redevient manuelle, sort du
  manifeste, gere directement CUDA ou peut publier apres un diagnostic de
  regression fatal.

### STRUCT-011 — Etendre la source de verite typee a toute la matrice de capacites

- **Nature :** corrige et verifie le 2026-08-28.
- **Signature :** prix equity, prix fixed income, LSM, parametres et samples
  etaient decrits par des listes partielles ou concurrentes qui ne resolvaient
  pas une matrice de capacites canonique et publiable.
- **Cloture :** le manifeste type couvre 12 engines, 24 modeles, 26 produits et
  689 datasets disponibles sans report sample; la specification samples derive
  les 24 bindings, 24 helpers et 48 recettes, et la provenance SHA-256 accompagne
  les sorties generees.
- **Preuve :** le checker controle les 689 recettes, les quatre tests du
  manifeste passent et la generation `--family all --compare-root .` est
  zero-diff sur bindings, recettes, CMake et manifeste de capacites; preuves
  E20 et E23.
- **Reouvrir seulement si :** une capacite ordinaire disponible doit etre
  declaree dans une liste concurrente, si une recette/binding/CMake derive
  sans etre couvert par le zero-diff, ou si provenance et snapshot divergent.

### STRUCT-013 — Publier les deux recettes de samples contractuelles par modele

- **Nature :** corrige structurellement et verifie le 2026-08-28; le defaut
  numerique isole est transfere sous `NUM-007`.
- **Signature :** les 24 modeles ne possedaient pas tous leurs bindings
  `sample.cuh/.cu` et leurs deux recettes contractuelles `samples_01` et
  `samples_02`, imposant une extension manuelle incomplete.
- **Cloture :** chaque modele possede ses deux corps minces generes, son helper
  type et son binding; l'agregat CMake `sample_generators` est derive de
  l'arborescence et les dependances Volterra restent conditionnees par mathDx.
- **Preuve :** les 48 executables compilent sur SM89 avec mathDx; un build frais
  sans mathDx compile les 40 recettes independantes en 168 etapes; 47/48 smoke
  tests produisent et rechargent leur JSON/YAML. Le seul echec n'est ni une
  absence ni un drift de codegen : Quadratic rough-Heston `samples_02` produit
  une valeur non finie et reste ouvert sous `NUM-007`; preuves E20 et E22.
- **Reouvrir seulement si :** un modele perd un des deux layouts, si un binding
  ou une recette doit etre ajoute manuellement hors specification, si le
  zero-diff ne couvre plus cette surface ou si un target disponible ne linke
  plus avec ses dependances declarees.

## Numerical robustness — passage de remediation du 2026-08-28

### NUM-006 — Definir la frontiere CEV des dynamiques SABR au lieu de la projeter silencieusement

- **Nature :** corrige et verifie le 2026-08-28.
- **Signature :** SABR et rough SABR projetaient la coordonnee de Lamperti sur
  `1e-12` apres un franchissement de zero, permettant a une trajectoire CEV de
  repartir sans condition de frontiere documentee.
- **Cloture :** une primitive commune impose la frontiere absorbante : un spot
  traverse devient `log_spot = -inf`, ne ressuscite pas, tandis que la
  volatilite et la consommation Philox poursuivent leur contrat.
- **Preuve :** contrat dynamics mis a jour; tests forces `beta=0` et `beta=0.5`;
  comparaison SABR beta-zero a la loi brownienne tuee, raffinement de pas
  rough-SABR, reproductibilite et trois tests CUDA cibles passes sur SM89;
  preuve E23.
- **Reouvrir seulement si :** une projection positive reapparait, une
  trajectoire absorbee peut repartir, le mapping Philox change ou un domaine
  publie montre un biais hors de la borne/convergence documentee.

## CUDA safety — passage de remediation du 2026-08-28

### CUDA-001 — Eliminer les lectures non initialisees du `PreparedRow` Bermudan

- **Nature :** corrige et verifie le 2026-08-28.
- **Signature :** les copies structurelles du `PreparedRow` Bermudan lisaient
  le padding non initialise du workspace, produisant 1 792 erreurs initcheck.
- **Cloture :** le workspace `PreparedRow` est initialise deterministement
  avant le kernel de preparation, sans changer sa representation ni les copies
  device.
- **Preuve :** `compute-sanitizer --tool initcheck` sur le test Bermudan OU/G2
  passe de 1 792 a zero erreur; memcheck, racecheck et synccheck restent propres
  et les sorties fonctionnelles restent reproductibles; preuve E23.
- **Reouvrir seulement si :** initcheck signale de nouveau une lecture de row,
  si l'initialisation ajoute un cout materiel mesure ou si un champ semantique
  peut rester non initialise.

## Remediation portabilite et samples du 2026-08-28

### STRUCT-014 — Retirer la reference morte au registre volontairement supprime

- **Nature :** corrige et verifie le 2026-08-28.
- **Signature originale :** `docs/deferred-work.md` avait ete volontairement
  supprime, mais `AGENTS.md` exigeait encore sa lecture avant une extension et
  rendait les instructions impossibles a suivre.
- **Cloture :** l'instruction morte est retiree sans restaurer le registre ni
  inventer de chemin de remplacement. Les mentions restantes sont uniquement
  l'historique de cette fermeture dans les registres d'audit.
- **Preuve :** aucun fichier d'instruction, index documentaire ou workflow ne
  reference le chemin supprime; la suppression reste presente dans le diff.
- **Reouvrir seulement si :** une instruction executable ou un index suivi
  exige de nouveau un document absent, ou si le registre supprime est restaure
  sans nouvelle responsabilite explicite.

### BUILD-002 — Ne pas confondre tuning SM89 et compatibilite cuFFTDx

- **Nature :** corrige par matrice explicite et builds representatifs le
  2026-08-28.
- **Signature originale :** activer mathDx imposait exactement `sm_89`, alors
  que le projet annoncait `75;86;89`; une provenance de tuning bloquait ainsi
  fonctionnellement les engines Gaussian-Volterra sur les autres GPU.
- **Cloture :** cuFFTDx 26.06 accepte les descripteurs
  `75,80,86,87,89,90,100,103,110,120,121`. Un build mono-architecture choisit
  son descripteur exact; un fatbin choisit le plus ancien descripteur demande
  comme profil d'implementation et emet chaque architecture nvcc. Includes et
  definitions mathDx sont possedes par l'interface CMake unique
  `ai_factory_cufftdx`.
- **Preuve :** configurations et compilations fraiches pricing plus sample
  passent avec mathDx pour SM75, SM86, SM89 et le fatbin `75;86;89`; le fatbin
  sample s'execute sur le SM89 disponible. Le preset local SM89 et le build
  sans mathDx restent utilisables; une architecture sans descripteur est
  refusee avec la matrice supportee. README et messages CMake distinguent
  compatibilite offline et runtime mesure.
- **Reouvrir seulement si :** une version mathDx change sa matrice, si un
  descripteur annonce ne compile plus un binding pricing ou sample, si un
  fatbin selectionne une implementation non executable sur une cible, ou si
  un profil de performance redevient une garde fonctionnelle.

### PERF-015 — Etiqueter et rendre retunables les profils livres depuis SM89

- **Nature :** corrige structurellement le 2026-08-28; aucune performance hors
  SM89 n'est inferee.
- **Signature originale :** threads, blocs, chunks Volterra et geometries
  samples provenaient de la RTX 4090 Laptop/SM89 mais etaient disperses dans
  manifests, templates et helpers, sans identifiant de profil ni workflow
  complet de retuning.
- **Cloture :** `tools/cuda/tuning_profile.hpp` et les variables cache
  `AI_FACTORY_CUDA_*` centralisent les valeurs par famille. Le profil par
  defaut `sm89_reference_v1` est compile dans les recettes et publie dans les
  metadonnees pricing, LSM et samples; un utilisateur peut fournir un nouvel
  identifiant et de nouvelles valeurs sans modifier les algorithmes ou le
  codegen. Les contrats documentent diagnostics, invariants, workloads et
  baseline separee par GPU/toolchain.
- **Preuve :** generation complete zero-diff sur 1 407 sorties, checker des
  689 recettes, builds representatifs Markovian, N-facteurs, Volterra,
  analytique, American, Bermudan et samples. Le benchmark samples couvre les
  quatre engines et deux layouts; le checker refuse un environnement
  incompatible. Les valeurs par defaut sont inchangees et restent etiquetees
  comme reference SM89, jamais comme optimum universel.
- **Reouvrir seulement si :** une geometrie de production redevient codee hors
  profil sans justification, si les artefacts perdent la provenance, si une
  surcharge requiert d'editer les fichiers generes, ou si une baseline compare
  des GPU/toolchains incompatibles.

## Naming et frontieres de policies — passage de remediation du 2026-08-28

### NAME-100 — Nommer une policy d'apres la responsabilite qu'elle implemente

- **Nature :** corrige et verifie le 2026-08-28.
- **Signature :** les types `*HybridDriverPolicy`, `PreparedDriver` et les
  methodes `driver_parameters`, `variance` et `value` etaient presentes comme
  un driver de chemin, alors que leur responsabilite etait exclusivement la
  discretisation d'un noyau Volterra, ses poids, sa variance et la
  reconstruction de la cellule singuliere. Le terme masquait la separation
  entre convolution et transformation propre au modele.
- **Cloture :** les trois implementations sont desormais des
  `*HybridKernelPolicy` avec `PreparedKernel`, `kernel_parameters`,
  `volterra_variance` et `reconstruct_volterra_value`. Le vocabulaire est
  propage dans le pricer FFT, le sampling, les quatre modeles Volterra, les
  tests, CMake, le manifeste et le template codegen. Le contrat et le schema de
  composition sont documentes dans
  [`cuda/pricing-policy-composition.md`](../cuda/pricing-policy-composition.md).
- **Preuve :** recherche statique sans ancien symbole ou ancien header sous
  `src`, `tools` et `tests`, hors signatures historiques de ce registre;
  generation samples `--compare-root .`
  zero-diff; compilation des quatre pricers europeens et des quatre samplers
  Volterra; cinq tests CUDA cibles passes sur SM89; preuve E27.
- **Reouvrir seulement si :** une policy est nommee comme une source aleatoire,
  un modele ou un produit alors qu'elle ne possede que le noyau mathematique,
  ou si des noms generiques comme `value`/`variance` rendent de nouveau
  ambigu le passage convolution -> valeur Volterra -> etat du modele.

### POLICY-002 — Ne pas faire dependre un moteur generique des champs internes d'une policy

- **Nature :** corrige et verifie le 2026-08-28.
- **Signature :** le moteur FFT lisait directement
  `PreparedDriver::sqrt_time_step`, bien que ce champ ne fasse pas partie du
  contrat annonce. Une nouvelle implementation conforme aux methodes
  documentees pouvait donc echouer seulement a l'instanciation profonde du
  kernel CUDA.
- **Cloture :** `volterra::HybridKernelPolicy` impose exactement `prepare`,
  `far_cell_weight`, `volterra_variance` et
  `reconstruct_volterra_value`; `HybridPathPolicyFor` impose le contrat de
  chemin et l'egalite du type retourne par `kernel_parameters`. Le moteur
  conserve lui-meme `sqrt_time_step` dans son `PreparedRow` et traite
  `PreparedKernel` comme opaque. Chaque noyau concret et chaque composition
  modele/noyau sont controles par `static_assert`.
- **Preuve :** les trois kernels satisfont le concept commun; les quatre
  compositions pricing et sampling compilent; `volterra_kernel_policy_cuda`,
  `rough_bergomi_dynamics_cuda`, `rough_bergomi_european_option_cuda`,
  `rough_volterra_product_policy_cuda` et `rough_volterra_samples_cuda`
  passent sur GPU; preuve E27.
- **Reouvrir seulement si :** un moteur generique accede a un membre concret
  d'une policy hors types explicitement contractuels, si une relation de types
  entre deux policies n'est verifiee qu'au fond d'un kernel, ou si une nouvelle
  policy exige de modifier le moteur malgre un contrat semantiquement
  identique.

## Structure et naming — remediation du 2026-08-28

### STRUCT-016 — Isoler les compositions modele-produit de l'infrastructure modele

- **Nature :** corrige et verifie le 2026-08-28.
- **Signature originale :** 832 fichiers de bindings produit partageaient la
  racine de chaque modele avec dynamics, analytics, parametres, datasets,
  sampling et helpers; une exploration ne permettait pas de separer le
  catalogue de produits de l'infrastructure du modele.
- **Cloture :** chaque paire modele-produit vit sous
  `src/model/equity/<family>/<model>/product/` ou
  `src/model/fixed_income/<model>/product/[<curve>/]`. Le dossier `product/`
  ne contient reciproquement aucune infrastructure, et les noms de targets
  CMake publics restent stables malgre le deplacement physique.
- **Preuve :** inventaire exhaustif de 832 fichiers et 416 paires; checker de
  profondeur/ownership/references, codegen zero-diff, configuration CMake,
  CTests architecture et builds representatifs passes; preuve E26.
- **Reouvrir seulement si :** un binding produit revient a la racine d'un
  modele, une infrastructure entre sous `product/`, un niveau non semantique
  apparait ou un target public derive a cause du chemin physique.

### STRUCT-017 — Classer les templates codegen par artefact et engine

- **Nature :** corrige et verifie le 2026-08-28.
- **Signature originale :** les templates pricing, samples et recettes etaient
  entasses sous des noms plats; `header.tpl`/`source.tpl` ne revelaient ni
  l'artefact ni l'engine et plusieurs fichiers C++ complets restaient encodes
  en chaines Python dans `generate.py`.
- **Cloture :** les 35 templates vivent sous `pricing/`, `sampling/` ou
  `catalog/`, puis sous `markovian`, `rough/markovian_n_factor`,
  `rough/volterra_fft`, `closed_form/black_scholes` ou la branche de recette
  explicite. Le renderer assemble ces templates sans cacher un artefact C++
  complet inline. La lacune fixed-income closed form reste separee sous
  `STRUCT-015` et n'est pas masquee par cette cloture.
- **Preuve :** aucun template plat, aucun ancien chemin reference, checker
  bloquant, regeneration bit a bit des 1 407 sorties et CTests codegen passes;
  preuve E26.
- **Reouvrir seulement si :** un template generique plat reapparait, une
  methode ne peut plus etre localisee depuis son chemin, un artefact complet
  retourne dans le renderer ou la generation diverge du tree suivi.

### NAME-011 — Rendre le role des fichiers d'infrastructure modele immediatement lisible

- **Nature :** corrige et verifie le 2026-08-28.
- **Signature originale :** des helpers comme `hybrid_pricing.cuh`,
  `pricing_workspace.cuh`, `markovian_pricing.cuh` et `numerics.hpp` ne
  nommaient pas leur engine; plusieurs fichiers canoniques hors produits,
  notamment des couples `dynamics.cuh`/`dynamics_impl.cuh`, n'expliquaient pas
  immediatement la difference entre contrat, preparation host et definitions
  device.
- **Cloture :** les helpers portent les qualificatifs
  `volterra_fft_*`/`markovian_n_factor_*`. Les 199 fichiers C++/CUDA
  d'infrastructure hors `product/` commencent par une phrase courte de contenu
  et d'utilite; les headers publics et leurs `*_impl.cuh` ont des roles
  explicitement distincts. Le checker refuse nom non revu, nom ambigu,
  en-tete generique, profondeur inattendue et paire publique/impl mal decrite.
- **Preuve :** inventaire exhaustif des 199 fichiers, zero ancien basename ou
  reference, checker `model_source_layout`, regeneration et builds
  representatifs passes; preuve E26.
- **Reouvrir seulement si :** le role d'un fichier ne peut plus etre deduit de
  son chemin et de son nom, si son en-tete n'en precise pas contenu et utilite,
  si deux engines partagent un helper non qualifie ou si le checker est
  contourne par une nouvelle exception non documentee.
