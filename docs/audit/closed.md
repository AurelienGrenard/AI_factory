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

## Remediation locale du 2026-08-30

### NAME-012 — Ajouter un en-tete de responsabilite aux fichiers handwritten restants

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** sept fichiers handwritten commencaient par
  pragma, include ou code, et le checker ne couvrait exhaustivement que
  l'infrastructure modele.
- **Cloture :** les sept fichiers portent une phrase specifique de contenu et
  d'utilite. `check_model_layout.py` inventorie tout fichier C++/CUDA, Python,
  CMake ou template sous `src`, `tools`, `tests` et `cmake`, plus le
  `CMakeLists.txt` racine; les outputs codegen, manifests et preambules de
  format sont classes explicitement.
- **Preuve :** `model_source_layout` accepte 829 fichiers generes et 572
  handwritten, dont quatre manifests et trois shebangs; les 572/572 ont une
  phrase valide et une fixture sans phrase est refusee.
- **Reouvrir seulement si :** un fichier handwritten du perimetre perd son
  en-tete, si une nouvelle famille echappe a l'inventaire ou si une exception
  generated/manifest/format est deduite implicitement de son contenu.

### NUM-009 — Valider integralement la preparation N-factor QRH explicite

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** la surcharge QRH explicite ne validait ni tous les
  parametres modele, ni horizon, nodes et weights; un node nul et des taux ou
  feedback non finis produisaient silencieusement un `PreparedDynamics` non
  fini.
- **Cloture :** tous les champs modele sont valides selon leur domaine, les
  nodes/weights explicites sont finis et strictement positifs, `dt` et
  l'horizon d'ajustement sont controles, puis les coefficients prepares sont
  certifies finis. L'horizon inutilise est retire de la surcharge recevant un
  noyau deja ajuste.
- **Preuve :** `quadratic_rough_heston_preparation_cuda` couvre NaN/Inf pour
  chaque champ, zero/negatif pour les domaines positifs, `H=0.5`, horizon et
  `dt` invalides, chaque node/weight invalide, overflow prepare, frontieres
  valides et le contrat de noyau rough-Heston; build et CTest passent.
- **Reouvrir seulement si :** un consommateur direct peut construire un
  prepare non fini, si un noyau non positif est accepte, ou si un argument
  public de preparation redevient inutilise et non valide.

### BUILD-003 — Donner un owner CMake aux quatre bindings American additionnels

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** les quatre unites American CEV, Kou, Merton et
  Schobel-Zhu existaient et leurs huit launchers etaient appeles, mais elles
  etaient absentes du manifeste early, du fragment CMake et de la compile DB;
  l'agregat CUDA echouait au link avec huit references indefinies.
- **Cloture :** les quatre modeles sont declares dans la source typee des
  recettes American; le codegen produit leur owner CMake unique et leurs huit
  recettes call/put. Les archives, generateurs et test LSM portent tous la
  meme capacite declaree.
- **Preuve :** les quatre TU figurent dans la compile DB avec un owner unique;
  les huit generateurs et l'agregat `ai_factory_cuda_tests` compilent et
  lient; `black_scholes_cev_kou_merton_schobel_zhu_american_lsm_cuda` passe
  sur RTX 4090 Laptop.
- **Reouvrir seulement si :** une TU modele-produit n'a plus exactement un
  owner CMake, si une capacite declaree diverge des recettes generees, ou si
  l'agregat/test early ne linke ou ne s'execute plus.

### CUDA-002 — Verifier les plafonds avant de dimensionner le workspace Volterra

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** la division plafond des moments partiels formait
  `path_count + threads - 1`; `SIZE_MAX` wrappait a zero, sous-dimensionnait le
  workspace puis exposait une ecriture device hors bornes.
- **Cloture :** les divisions plafond n'additionnent plus avant division;
  produits, sommes et offsets sont checked, les limites de grille sont
  refusees avant allocation et la progression par chunks utilise la taille du
  dernier chunk reel. Le test adresse sur device le dernier `double2` du
  dernier bloc partiel calcule par le planner.
- **Preuve :** `volterra_fft_workspace_bounds_cuda` couvre `SIZE_MAX`, le plus
  grand nombre de blocs representable, les overflows de convolution/somme,
  le dernier bloc partiel et sa frontiere device; il passe sur RTX 4090
  Laptop. Compute Sanitizer `memcheck`, `racecheck`, `initcheck` et `synccheck`
  rapportent zero erreur ou hazard.
- **Reouvrir seulement si :** une taille de workspace, un offset ou un nombre
  de blocs est de nouveau calcule sans arithmetique checked, si une conversion
  vers `dim3` precede sa validation, ou si un sanitizer detecte une erreur sur
  les frontieres Volterra.

### CUDA-003 — Refuser le débordement des nombres de transitions fixed-step

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** les calendriers et schedules d'exercice
  multipliaient `steps_per_day` par un nombre de jours en `uint32_t` sans
  controle host; un overflow pouvait donc lancer un kernel avec zero ou un
  nombre tronque de transitions et publier un prix fini mais faux.
- **Cloture :** `checked_fixed_step_transition_count` centralise le produit en
  `uint64_t`, refuse le depassement de `UINT32_MAX` et les fractions d'annee
  FP32 non finies. Tous les calendriers terminal, regular, stubbed, static et
  d'exercice sont valides avant lancement. Les launchers Monte Carlo recoivent
  le miroir host des produits; les sources de calendriers sample et la formule
  geometric Asian suivent le meme contrat. L'arithmetique device sous les
  bornes et le mapping Philox sont inchanges.
- **Preuve :** `simulation_schedule_validation` couvre la borne exacte et la
  valeur suivante pour chaque forme de calendrier, les schedules Regular et
  MaturityAligned, les sources sample UniformMaturity et RandomIncreasing et
  les fractions fixed/exact non finies. Le codegen complet est zero-diff,
  l'agregat de 121 cibles CUDA compile et 63/64 CTests CUDA passent; l'unique
  timeout QRH passe directement en environ 61 secondes sans erreur. Les modes
  Compute Sanitizer memcheck, racecheck, initcheck et synccheck sur les 21
  compositions du test `path_product_factorization_cuda` rapportent zero
  erreur ou hazard.
- **Reouvrir seulement si :** une duree fixed-step atteignable par un launcher
  CUDA est de nouveau multipliee sans controle host, si un rejet peut survenir
  apres un lancement, si une fraction de maturite non finie est acceptee, ou
  si l'arithmetique device sous la borne change sans preuve de reproductibilite.

### STRUCT-003 — Deriver et borner tous les artefacts depuis la source de verite

- **Nature :** corrige une seconde fois et verifie contradictoirement le
  2026-08-30.
- **Signature originale :** le zero-diff ne parcourait que les sorties
  attendues; il acceptait fichiers orphelins, retraits incomplets et mauvais
  mappings, et levait un `FileNotFoundError` brut pour une sortie absente.
- **Cloture :** l'inventaire type enumere les 832 fichiers de bindings, les 697
  recettes, les 48 bindings samples et les 24 helpers samples. Le comparateur
  confronte ensembles reels et declares dans les deux sens et emet
  `CODEGEN_MISSING`, `CODEGEN_EXTRA`, `CODEGEN_RENDERER_*` ou
  `CODEGEN_MISMATCH` sans exception implicite.
- **Preuve :** generation complete de 1 500 fichiers zero-diff; fixtures
  negatives pour absence, contenu modifie, orphelin, rename, famille retiree,
  produit pluralise et renderer incomplet; CTests `pricing_binding_codegen`,
  `pricing_capability_manifest`, `model_source_layout` et
  `catalog_generator_boundaries` passes.
- **Reouvrir seulement si :** un artefact peut exister hors inventaire, un
  retrait laisse une sortie acceptee, un mapping incorrect passe le checker ou
  une absence produit de nouveau une exception non diagnostiquee.

### STRUCT-015 — Generer les bindings et recettes fixed-income closed form

- **Nature :** corrige apres report explicite et verifie le 2026-08-30.
- **Signature originale :** 21 paires de bindings et 42 recettes fixed-income
  closed form etaient inventoriees mais toutes handwritten, sans branche de
  templates possedant leurs signatures et corps.
- **Cloture :** les branches `affine_one_factor/{cir,gaussian}`,
  `affine_two_factor`, `curve_fitted_one_factor` et
  `curve_fitted_two_factor` generent les 21 paires et 42 recettes. CIR et les
  modeles gaussiens restent separes lorsque leur API differe; les huit paires
  et seize recettes Bermudan sont explicitement `hand_written`.
- **Preuve :** zero-diff complet; 74 templates nommes inventories; les 21 TU
  closed form, les 42 executables de recettes et l'agregat fixed-income
  compilent. Les 24 CTests fixed-income passent hors sandbox sur RTX 4090
  Laptop sans benchmark de performance.
- **Reouvrir seulement si :** une composition closed form compatible redevient
  manuelle, si un nouveau modele/courbe exige de copier un corps hors branche
  semantique, ou si bindings et recettes divergent du manifeste.

### STRUCT-018 — Aligner la taxonomie canonique de `src`, `catalog` et `datasets`

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** les 18 modeles equity perdaient la famille
  `markovian`/`rough` dans `catalog` et `datasets`, et les 26 produits etaient
  ranges sous asset class avec un nom de dossier pluralise distinct de `src`.
- **Cloture :** `catalog` et `datasets` reutilisent exactement les prefixes
  `model/equity/<markovian|rough>/<model>`,
  `model/fixed_income/<model>`, `curve/<curve>` et `product/<product>` de
  `src`. Recettes, YAML, JSON, URLs, CMake, tests et documentation ont ete
  migres; toute transformation concurrente est refusee par les specs.
- **Preuve :** identite physique 24/24 modeles, 26/26 produits et 2/2 courbes;
  aucune ancienne taxonomie hors fixtures historiques; configuration CMake
  fraiche, test loader, 697 recettes, zero-diff et fixtures negatives passes.
- **Reouvrir seulement si :** une couche ajoute asset class, pluralisation ou
  retire la famille par rapport a `src`, si un chemin est reconstruit hors
  `source_prefix`, ou si catalog/dataset/URL cessent d'etre identiques.

### BUILD-004 — Reduire les dependances transitives publiques des unites CUDA

- **Nature :** corrige et mesure le 2026-08-30.
- **Signature originale :** chaque archive de launcher CUDA exportait les
  loaders modele, produit et courbe en `PUBLIC`; le test Heston tirait ainsi
  quinze archives produit, son loader modele et cinq facades offline alors
  qu'il n'appelait aucun loader.
- **Cloture :** les launchers lient `runtime`, Longstaff--Schwartz et cuFFTDx
  en `PRIVATE`. Les executables resolvent directement les seules archives
  correspondant aux headers dataset qu'ils incluent via
  `ai_factory_collect_source_dependencies`; les tests CUDA ne lient plus
  `ai_factory_dataset_core`. Les recettes Bermudan declarent explicitement le
  loader produit qu'elles utilisent. La configuration refuse toute archive
  CUDA qui reexporte un target `*_dataset`.
- **Preuve :** le lien Heston passe de 39 a 17 archives et conserve ses quinze
  launchers. Le generateur Heston European lie sept archives, dont exactement
  launcher, loaders modele/produit et pipeline prix; le generateur
  Hull-White/Nelson-Siegel lie aussi son loader courbe. Apres toucher
  `european_option/dataset.cpp`, Heston est no-op tandis que son generateur
  reconstruit seulement loader, archive et executable. Configuration, builds,
  CTests Heston et American LSM passent; un generateur CIR Bermudan relie son
  loader produit apres configuration fraiche.
- **Reouvrir seulement si :** un launcher exporte un loader ou une facade
  offline, si un consommateur obtient un loader sans l'inclure explicitement,
  ou si une modification de dataset invalide de nouveau un test de kernel pur.

### NUM-008 — Allouer des domaines Philox disjoints aux recettes et variantes

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** les recettes samples espacaient leurs bases de 10
  ou 1 000 et les prix stochastic reutilisaient deux bases globales; Bates
  `samples_01` ligne 10 et `samples_02` ligne 0 partageaient ainsi la meme
  cle et le meme compteur Philox.
- **Cloture :** `RngDomainSpec` version 1 alloue a chaque chemin de recette
  stochastique canonique un domaine de `2^32` cles et des flux nommes de
  `2^30` cles. Samples separe `parameters`, `schedule` et `dynamics`; pricing
  reserve `dynamics`; l'allowlist CRN est explicitement vide. Codegen et les
  seize recettes Bermudan consomment ces seules graines.
- **Preuve :** les 14 tests du manifeste prouvent couverture exhaustive,
  intervalles disjoints, Bates distinct et rejet d'une fixture collision. Le
  checker fail-closed valide les 697 recettes et chaque literal declare; le
  codegen complet de 1 500 sorties est zero-diff. Le test CUDA
  `black_scholes_samples_cuda` rejoue exactement entre 128/512 threads et deux
  decoupages de batch. Les generateurs Black-Scholes samples, Heston European
  et CIR Bermudan compilent.
- **Reouvrir seulement si :** une recette Philox obtient une graine hors du
  manifeste, si deux intervalles se recouvrent hors allowlist CRN versionnee,
  si le volume d'une recette atteint `2^30` lignes, ou si batch/geometrie entre
  de nouveau dans la derivation de cle ou de compteur.

### NUM-007 — Stabiliser les samples Quadratic rough-Heston sur le domaine publie

- **Nature :** corrige, qualifie et verifie le 2026-08-30.
- **Signature originale :** `samples_02` produisait deterministement un spot
  non fini sur les bornes core; apres equilibrage de la recurrence, seuls la
  ligne historique et un smoke 1 000 lignes avaient ete qualifies. En outre,
  la recette ajustait un noyau L2 par `H`, soit environ 77 heures extrapolees
  de preparation hote pour trois millions de lignes.
- **Cloture :** la recurrence device equilibree est conservee. Le helper sample
  construit 257 fits L2 positifs exacts sur `H in [0.01, 0.20]`, puis
  interpole lineairement nodes et weights; preparation scalaire et pricers
  gardent leurs fits exacts. Le mode generique `--preflight` execute toute la
  forme production, rejoue sous une seconde geometrie et refuse maturites ou
  sorties non finies/differentes sans publier d'artefact.
- **Preuve numerique :** les 256 milieux de cellule restent sous 0,11 %
  d'erreur L2 relative contre le noyau fractionnaire analytique; sur sept fits
  exacts hors grille, la penalite maximale vaut `1.67e-6`. Sur
  `H={0.005,0.01,0.10,0.20,0.45}`, l'erreur contracte strictement de N=2 a
  N=3 puis N=7; N=7 varie de `3.23e-5` a `1.031e-3`. Les coins core/stress,
  1 024 trajectoires chacun et N=2/3/7 sont finis.
- **Preuve production et prix :** les deux layouts de 3 000 000 lignes sont
  finis et bitwise identiques entre 256 et 128 threads. `samples_01` mesure
  33,45 ms kernel / 19,53 s total et `samples_02` 28,04 ms / 20,66 s. Les
  48/48 recettes smoke passent. Le pricer European passe replay, parite,
  comparaison grille/fit exact et raffinements N=2/3/7 ainsi que
  `dt=1/252,1/504,1/1008` dans l'incertitude Monte Carlo.
- **Ressources et surete SM89 :** random-terminal N=7 utilise 82 registres,
  32 octets de stack, zero local; la variante parameter-block utilise 91
  registres, 160 octets shared, 32 octets stack et zero local. Le pricer N=7
  utilise 91 registres, 192 octets shared, zero stack/local. Memcheck,
  racecheck, initcheck et synccheck rapportent zero erreur/hazard; quatre
  CTests CUDA cibles et cinq checks architecture/codegen passent.
- **Reouvrir seulement si :** un output QRH redevient non fini, si un layout
  3M n'est plus reproductible entre geometries, si l'erreur grille depasse ses
  bornes, si le domaine sample sort de `[0.01,0.20]` sans nouvelle grille, ou
  si un raffinement temps/facteurs ou un pricer echoue a sa qualification.

### NUM-010 — Qualifier le FP64 de finalisation des statistiques Monte Carlo

- **Nature :** usage FP64 conserve, qualifie et verifie le 2026-08-30.
- **Signature originale :** moyenne, variance centree, seuil d'annulation,
  divisions et racine de `compute_statistics` restaient en FP64 sans
  comparaison propre FP32/mixte/host pour Monte Carlo, Volterra et LSM.
- **Cloture :** `monte_carlo_statistics_precision_cuda` compare finalisation
  FP64, racine FP32 mixte, FP32 complete et host contre une reference
  `long double` sur quatre distributions core/stress/faible variance. FP64
  reste le contrat commun : une seule finalisation est executee par prix, hors
  boucle de trajectoires, et preserve la semantique de `NUM-005`.
- **Preuve numerique et cout SM89 :** l'erreur relative maximale mesuree vaut
  `4.10e-12` sur l'erreur standard FP64. La finalisation FP32 l'annule sur les
  deux cas de faible variance. Pour 1 048 576 resultats, FP64 mesure
  `0.1847 ms`, mixte `0.1199 ms`, FP32 `0.0241 ms`; la seule copie device-host
  des moments mesure `10.217 ms`, puis le calcul host `3.859 ms`. Le gain mixte
  isole est donc sans effet end-to-end demontre aux volumes reels et changerait
  l'arrondi publie.
- **Ressources et consommateurs :** le kernel isole FP64 utilise 35 registres,
  contre 31 mixte et 19 FP32, sans stack/shared/local. Heston MC utilise 66
  registres et 80 octets shared; Heston LSM finalise avec 35 registres, 128
  octets shared dynamiques, zero spill et 100 % d'occupation theorique;
  Volterra conserve son finalizer 31 registres sans spill. Les trois CTests
  consommateurs et les quatre outils Compute Sanitizer passent.
- **Reouvrir seulement si :** une famille demontre un gain end-to-end de la
  finalisation mixte sur chaque architecture cible avec budget prix/erreur
  standard, ou si le contrat de sortie cesse d'etre FP32.

### NUM-020 — Qualifier la formation FP64 des seconds moments Monte Carlo

- **Nature :** usage FP64 chaud conserve, qualifie et verifie le 2026-08-30.
- **Signature originale :** chaque payoff FP32 etait promu puis carre en FP64
  dans Monte Carlo, Volterra et LSM sans mesure separee de la multiplication ni
  comparaison des alternatives.
- **Cloture :** le test commun compare produit FP64 direct, FMA FP64, carre
  FP32 promu, carre FP32 mis a l'echelle et somme FP32 compensee contre une
  accumulation `long double`. Le produit de deux valeurs FP32 est exact en
  FP64 avant addition; ce contrat commun est conserve dans les trois callsites.
- **Preuve numerique :** produit et FMA FP64 restent sous `4.10e-12` d'erreur
  relative d'erreur standard. Sur les cas de faible variance aux echelles 100
  et 2 048, le carre FP32 produit respectivement 100 % et 15,48 % d'erreur; la
  compensation atteint 45,87 % et 15,45 %. Prix et mapping stochastic restent
  inchanges.
- **Cout et ressources SM89 :** pour 256 lignes de 32 768 payoffs, produit
  FP64 `0.1558 ms`, FMA FP64 `0.1549 ms`, FP32 promu `0.1908 ms`, FP32 mis a
  l'echelle `0.1920 ms`, FP32 compense `0.1174 ms`. Les cinq variantes
  utilisent 21--22 registres, zero stack/local; l'alternative plus rapide ne
  respecte pas le budget numerique. Les kernels reels MC, Volterra et LSM
  passent leurs tests; `moment_partials` LSM utilise 23 registres, 128 octets
  shared dynamiques, zero spill et 100 % d'occupation theorique.
- **Reouvrir seulement si :** une famille bornee prouve sur son domaine complet
  qu'une formation mixte respecte prix et erreur standard et gagne end-to-end
  sur toutes ses architectures cibles.

### NUM-011 — Qualifier l'accumulation FP64 des moyennes arithmetiques de chemin

- **Nature :** FP64 chaud elimine, qualifie et verifie le 2026-08-30.
- **Signature originale :** les deux facades Asian additionnaient chaque spot
  FP32 et divisaient en FP64 dans les 18 compositions Markov, N-factor et
  Volterra, sans budget ni cout marginal mesure.
- **Cloture :** `CompensatedFloatSum` implemente une somme de Kahan FP32 compacte
  et devient l'unique accumulateur des deux facades. Division et valeur publiee
  restent en FP32. `asian_mean_precision_cuda` compare FP64, FP32 simple, Kahan
  FP32 et chunks FP32 contre une reference `long double` sur 17, 253, 1 765 et
  4 097 observations, faible variance, forte dispersion et echelles extremes.
- **Preuve numerique et prix :** l'erreur relative maximale de coordonnee Kahan
  vaut `5.29e-8` et les quatre moyennes arithmetiques publiees sont identiques a
  la reference arrondie FP32. Le payoff vanille est 1-Lipschitz, donc cette
  borne, multipliee par le discount borne, se propage au payoff et au prix. La
  somme FP32 simple atteint `3.39e-6` sur la valeur publiee.
- **Cout et ressources SM89 :** 8 192 trajectoires x 1 765 observations mesurent
  FP64 `0.4169 ms`, FP32 simple `0.3108 ms`, Kahan FP32 `0.2817 ms`, chunks
  `0.4188 ms`. Heston reste a 70 registres/80 octets shared; QRH N=7 passe de
  108 a 93 registres; le kernel de payoff Volterra passe de 71 a 69 registres.
  Aucun stack/local/spill n'apparait. Markov, QRH N-factor et Volterra passent,
  ainsi que memcheck sur le microbenchmark et QRH.
- **Reouvrir seulement si :** le domaine depasse 4 097 observations ou les
  echelles balayees sans nouveau budget, si un payoff non 1-Lipschitz reutilise
  la moyenne, ou si une architecture cible regresse en ressources/end-to-end.

### NUM-012 — Qualifier l'accumulation FP64 des moyennes geometriques de chemin

- **Nature :** FP64 chaud elimine, qualifie et verifie le 2026-08-30.
- **Signature originale :** les deux facades geometric Asian additionnaient
  chaque log-spot et divisaient en FP64 avant `expf`, sans distinguer erreur de
  somme, exponentielle et cout sur les trois moteurs.
- **Cloture :** les deux variantes `GeometricMeanObservationHandler` et la
  facade `GeometricAsianOptionPathPolicy` reutilisent `CompensatedFloatSum`;
  la politique explicite de spot non positif est preservee. Le meme sweep
  compare la coordonnee log et la moyenne publiee a `long double`.
- **Preuve numerique et prix :** l'erreur relative Kahan maximale vaut
  `3.58e-8` sur la coordonnee et `1.04e-7` apres exponentielle. Le test
  Black--Scholes conserve sa comparaison analytique FP64; les tests Heston,
  QRH et Volterra sont finis, QRH rejoue bitwise et sa call geometrique reste
  sous la call arithmetique dans l'incertitude Monte Carlo.
- **Cout et ressources SM89 :** pour 14 458 880 log-observations, FP64 mesure
  `0.3747 ms`, FP32 simple `0.1907 ms`, Kahan FP32 `0.1149 ms`, chunks
  `0.3812 ms`. Heston reste a 68 registres; QRH N=7 passe de 108 a 91 et
  Volterra de 71 a 69, sans stack/local/spill.
- **Reouvrir seulement si :** domaine de log-spots, nombre d'observations ou
  fonction aval sort du sweep/budget, ou si une architecture cible invalide le
  gain end-to-end ou les ressources.

### NUM-013 — Qualifier la somme FP64 du range accrual analytique Black-Scholes

- **Nature :** FP64 chaud elimine, qualifie et verifie le 2026-08-30.
- **Signature originale :** le closed form additionnait jusqu'a 1 764
  probabilites FP32 en FP64 sans budget de prix ni comparaison de cout propre.
- **Cloture :** `RangeAccrualClosedFormPricingPolicy` reutilise
  `CompensatedFloatSum`. `range_accrual_sum_precision_cuda` compare FP64, FP32
  simple, Kahan FP32, chunks FP32 et reference analytique host sur quatre coins
  core/stress : 12, 252 et 1 764 observations, barrieres etroites/larges,
  volatilites `0.01`--`1.0`, taux `-0.03`--`0.12` et coupons jusqu'a `0.25`.
- **Preuve numerique :** les quatre strategies device publient les memes prix
  FP32 sur le sweep; Kahan reste sous `9.63e-8` d'erreur relative contre la
  reference host. Le test Black--Scholes existant conserve ses controles
  analytiques sur trois spots.
- **Cout et ressources SM89 :** 1 024 prix x 1 764 observations mesurent FP64
  `0.4605 ms`, FP32 simple `0.4238 ms`, Kahan FP32 `0.4282 ms`, chunks
  `0.4905 ms`, contre `39.07 ms` host. Le kernel de production passe de 34 a
  35/36 registres selon sa specialisation, sans stack/shared/local/spill.
  Les deux CTests et memcheck passent.
- **Reouvrir seulement si :** plus de 1 764 observations, probabilites hors
  `[0,1]`, nouveaux extremes de barrieres/taux/volatilite/coupon, ou regression
  ressources/end-to-end sur une architecture cible.

### NUM-014 — Qualifier le noyau de resolvante fractionnaire FP64 sur device

- **Nature :** FP64 chaud elimine, qualifie et verifie le 2026-08-30.
- **Signature originale :** Rough Stein--Stein evaluait noyau, fonction de
  Mittag--Leffler et neuf points de chaque poids lointain en FP64, jusqu'a 96
  termes ou points et avec fonctions speciales, sans comparaison FP32/mixte ni
  cout device propre.
- **Cloture :** le chemin device est FP32 compense. La serie stable reste sous
  `x <= 2`; la representation positive par densite de Laplace prend le relais
  avant la zone de cancellation FP32. Series, quadratures et Simpson utilisent
  `CompensatedFloatSum`. Une variante FP32 gardant l'ancien crossover a ete
  rejetee : jusqu'a 11,94 % d'erreur noyau, 22,23 % sur les integrales et
  70,16 % sur les poids.
- **Preuve numerique et prix :** sur 100 coins couvrant `H=0.01--0.45`, mean
  reversion `0--8`, temps `1/504--7` ans et lags `2--1008`, les erreurs
  relatives maximales contre quadrature `long double` valent `8.23e-5` pour le
  noyau et `7.14e-5` pour les poids. Sur 1 000 sorties sample a parametres,
  calendriers et seeds identiques, l'ecart terminal relatif maximal vaut
  `1.41e-6`, le p99 `7.85e-7` et le decalage de moyenne `1.17e-8` ecart-type.
  Le payoff European est 1-Lipschitz, donc son ecart de prix couple est borne
  par l'ecart terminal absolu maximal `2.39e-6` avant discount.
- **Cout et ressources SM89 :** le microbenchmark apparie de 1 024 lignes x 16
  poids passe de `46.44 ms` FP64 a `5.64 ms` FP32. Le sampler `samples_02`
  passe de `3.2472 s` a `0.4145 s` kernel et de `9.2128 s` a `1.6393 s` wall.
  Son kernel terminal passe de 148 a 128 registres/thread et de 96 a 88 octets
  locaux/thread; l'occupation theorique reste 16,7 %, donc le kernel FFT reste
  contraint par d'autres ressources. Les 21 bibliotheques produit et les deux
  generateurs European compilent; cinq CTests cibles et memcheck passent.
- **Reouvrir seulement si :** le domaine `H`/mean reversion/temps/lag est
  etendu, le crossover ou la quadrature change, une sortie depasse le budget
  `5e-4`, ou une architecture cible invalide gain end-to-end ou ressources.

### NUM-015 — Qualifier les integrales de puissance FP64 de la resolvante

- **Nature :** FP64 chaud elimine et frequence originale rectifiee le
  2026-08-30.
- **Signature originale :** `power_integral` accumulait 96 contributions et
  appelait le noyau en FP64; le constat estimait `step_count + 2` quadratures
  par resultat prepare sans preuve propre de precision ou de cout.
- **Cloture :** les integrandes, fonctions speciales et sommes passent en FP32
  compense avec le noyau de `NUM-014`. L'inventaire du chemin compile montre
  surtout que Rough Stein--Stein declare `kUsesVolterraVariance = false` : la
  branche `volterra_variance` est eliminee a la compilation et seules les deux
  integrales de `prepare` sont executees, pas une integrale par pas.
- **Preuve numerique et cout SM89 :** le meme sweep compare les puissances 1 et
  2 a une quadrature `long double`; l'erreur relative maximale vaut
  `4.03e-5`. Le microbenchmark de 1 024 integrales passe de `8.379 ms` FP64 a
  `0.854 ms` FP32. La comparaison des 1 000 trajectoires de `NUM-014` couvre la
  propagation des deux loadings singuliers; les tests policy, samples,
  workspace et sanitizer passent.
- **Reouvrir seulement si :** une path policy active
  `kUsesVolterraVariance`, si la variance devient atteignable par pas, si les
  loadings sortent du domaine qualifie ou si le budget `5e-4` n'est plus tenu.

### NUM-016 — Qualifier les produits des statistiques de regression LSM en FP64

- **Nature :** usage FP64 chaud conserve, qualifie et verifie le 2026-08-30.
- **Signature originale :** cible actualisee, produits feature-feature du Gram
  et feature-cible du second membre etaient formes en FP64 a chaque candidat et
  date backward, sans comparaison propre des produits FP32/mixtes ni cout.
- **Cloture :** les operandes FP32 restent promus avant multiplication : leur
  produit est exact avant l'accumulation FP64 commune. Le test
  `longstaff_schwartz_precision_cuda` compare ces termes a des produits FP32
  ensuite promus sur 32 768 observations Laguerre deux facteurs, core et
  presque colineaires/stress, contre une reference `long double`.
- **Preuve numerique et cout SM89 :** l'erreur relative maximale des
  statistiques FP64 vaut `3.92e-13`, contre `2.22e-9` pour les produits FP32.
  Pour 16 384 lignes x 32 observations et 27 statistiques, le chemin FP64
  mesure `0.0853 ms`, contre `0.1258 ms` pour FP32 puis promotion :
  l'alternative est a la fois moins precise et plus lente avec les reductions
  FP64 contractuelles. Les kernels isoles utilisent 80 et 72 registres, sans
  stack/local; les kernels reels utilisent 88 registres Heston et 118 G2,
  sans local, aux occupations theoriques respectives 33,3 %.
- **Preuve consommateurs :** tests du regresseur, Heston American et Bermudan
  OU/G2 payer/receiver passent, ainsi que memcheck du nouveau sweep.
- **Reouvrir seulement si :** accumulation/reduction cesse d'etre FP64, la
  base ou sa taille change, un domaine de features/cashflows sort du sweep, ou
  une architecture cible demontre une alternative plus precise et plus rapide
  end-to-end.

### NUM-017 — Qualifier la resolution FP64 des equations normales LSM

- **Nature :** usage FP64 conserve, qualifie et verifie le 2026-08-30.
- **Signature originale :** assemblage, ridge, Cholesky, substitutions et
  coefficients utilisaient FP64 sans grille de conditionnement, reference
  haute precision ni cout isole compare a FP32.
- **Cloture :** le solveur FP64 reste l'invariant. Le test compare Cholesky
  FP64 et FP32 a une elimination `long double` sur un systeme bien conditionne
  et un SPD de conditionnement voisin de `1e8`, apres le ridge `1e-10` relatif.
- **Preuve numerique :** l'erreur relative FP64 maximale des coefficients vaut
  `7.47e-10`; la matrice stress reste resoluble en FP64 mais devient singuliere
  apres arrondi FP32 et sa Cholesky echoue. Ce cas couvre la colinearite que les
  equations normales amplifient et interdit une descente globale en FP32.
- **Cout et ressources SM89 :** 65 536 resolutions variables mesurent
  `0.0870 ms` FP64 contre `0.0403 ms` FP32. Ce facteur isole est accepte car la
  resolution n'a lieu qu'une fois par prix/date et l'alternative echoue. Les
  microkernels utilisent 80 contre 48 registres, sans stack/local; les kernels
  reels Heston/G2 utilisent 82 registres, zero local, avec 33,3 %/41,7 %
  d'occupation theorique.
- **Reouvrir seulement si :** taille/base, ridge ou equations normales
  changent, si un conditionnement superieur est publie, ou si une strategie
  mixte/QR respecte les budgets coefficients/decisions/prix et gagne
  end-to-end sur chaque architecture cible.

### NUM-018 — Qualifier la prediction et la decision d'exercice LSM en FP64

- **Nature :** usage FP64 chaud conserve; variante selective mesuree et rejetee
  le 2026-08-30.
- **Signature originale :** les FMA de prediction et la comparaison
  exercice/continuation restaient en FP64 pour chaque candidat; `NUM-003`
  prouvait une frontiere sensible mais ni cout ni strategie selective.
- **Cloture :** coefficients, prediction et comparaison restent FP64. Le sweep
  de `2^20` cas contient 16 384 marges sous `1e-7` et compare FP32 a une
  variante selective qui calcule une borne conservative d'arrondi puis
  reevalue FP64 pres de la frontiere.
- **Preuve decisions et prix :** FP32 diverge sur 7 040 decisions et deplace la
  moyenne des cashflows de `0.59406694` a `0.59094747` (`-0.00311947`, environ
  `-0.525 %`). La variante selective declenche 16 384 fallbacks, ne diverge
  jamais et retrouve exactement la moyenne FP64.
- **Cout et ressources SM89 :** sur `2^20` predictions, FP64 mesure
  `0.0581 ms`, FP32 `0.0239 ms`, mais la variante selective `0.2324 ms`. Les
  kernels isoles utilisent 33, 30 et 40 registres respectivement, sans
  stack/local. En production, `update_cashflows` utilise 40 registres Heston
  et 64 G2, sans local, avec 100 % et 66,7 % d'occupation theorique. Heston
  call/put et OU/G2 payer/receiver passent avec replay bitwise.
- **Reouvrir seulement si :** une nouvelle base ou plage de coefficients/marges
  est introduite, ou si une selection bornee conserve zero divergence et prix
  tout en gagnant end-to-end et en ressources sur American et Bermudan de
  chaque architecture cible.

## Concepts, structure et naming — remediation du 2026-08-30

### POLICY-003 — Faire representer au concept produit le call graph reellement instancie

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** le concept produit exigeait `log_spot()` meme pour
  une observation spot et ne verifiait pas les deux callbacks du handler.
- **Cloture :** `StatePolicyForObservationCoordinate` selectionne
  `SpotStatePolicy` ou `LogSpotStatePolicy` selon la coordonnee declaree;
  `PathProductObservationHandler` exige les callbacks directs et leur retour
  booleen avant toute instanciation profonde du moteur.
- **Preuve :** probes compile-time spot-only/spot positif, spot-only/log
  negatif et handler incomplet negatif; les 21 identites de factorisation sont
  conservees. Les tests Markov, QRH N-facteurs et rough Heston Volterra passent
  sur GPU; le codegen complet reste zero-diff.
- **Reouvrir seulement si :** le concept exige une observable non appelee,
  accepte un handler incomplet ou reporte de nouveau l'erreur dans un corps de
  kernel plutot qu'a la frontiere de composition.

### STRUCT-020 — Supprimer ou integrer les headers runtime sans consommateur

- **Nature :** surface morte supprimee et verifiee le 2026-08-30.
- **Signature originale :** `equity/observables.cuh` et
  `simulation/barrier_handlers.cuh` exposaient quatre types sans include ni
  consommateur, en parallele des policies barriere actives.
- **Cloture :** les deux headers orphelins sont supprimes; la responsabilite
  barriere active reste uniquement dans `equity/barrier_pricing_policy.cuh`.
- **Preuve :** recherche sans ancien include ou symbole; le checker de layout,
  les builds Markov/N-facteurs/Volterra et les tests GPU representatifs passent.
- **Reouvrir seulement si :** un header runtime sans consommateur reapparait ou
  si deux implementations independantes revendiquent la meme responsabilite
  barriere sans contrat de composition.

### NAME-007 — Nommer les diagnostics du test d'apres ses modeles et son contrat

- **Nature :** reouverture corrigee et verifiee le 2026-08-30.
- **Signature originale :** cinq diagnostics employaient encore le repere
  historique « New equity dynamics test ».
- **Cloture :** chaque diagnostic nomme desormais le contrat dynamics
  Merton/Kou/CEV/Schobel-Zhu et l'operation CUDA concernee.
- **Preuve :** recherche sans `new`, `additional` ou `remaining` dans les noms
  et diagnostics de tests; le CTest dynamics correspondant passe sur GPU.
- **Reouvrir seulement si :** un nom de fichier, target, CTest ou diagnostic
  decrit l'anciennete plutot que le modele et le contrat testes.

### NAME-013 — Nommer et placer explicitement la policy Phoenix partagee

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** `pricing_policy_core.cuh`, possede par Phoenix sans
  memoire, cachait une policy commune a deux produits et l'invariant de memoire
  coupon selectionne a la compilation.
- **Cloture :** le proprietaire neutre
  `src/product/phoenix_coupon_memory_path_policy.cuh` expose des types nommes
  `PhoenixCouponMemory*` et le parametre explicite `RememberMissedCoupons`; les
  deux facades produit choisissent respectivement `false` et `true`.
- **Preuve :** ancien chemin et anciens symboles absents; commentaire d'en-tete
  explicite; checker de layout, codegen zero-diff et test des 21 produits
  factorises passent.
- **Reouvrir seulement si :** le partage Phoenix redevient possede par un seul
  produit concret, si le nom masque l'invariant de memoire coupon ou si les
  deux facades cessent de composer la meme implementation.

### NAME-006 — Encoder les unites dans les coordonnees temporelles publiques et device

- **Nature :** reouverture corrigee et verifiee le 2026-08-30.
- **Signature originale :** les deux courbes exposaient encore les coordonnees
  annuelles de `forward_rate` sous les noms ambigus `start` et `end`, hors du
  checker ayant motive la premiere cloture.
- **Cloture :** declarations, definitions, commentaires et references de
  courbes utilisent `start_years` et `end_years`. Le checker couvre maintenant
  `start` et `end` en plus des anciens noms, avec trois fixtures negatives, et
  inspecte aussi les references README locales.
- **Preuve :** inventaire sans coordonnee flottante publique/device ambigue;
  `model_source_layout` passe; Hull--White et G2++ compilent et leurs CTests
  passent sur GPU. Aucune cle de serialisation ou valeur dataset n'est modifiee.
- **Reouvrir seulement si :** une coordonnee temporelle flottante publique,
  device ou documentee perd de nouveau son unite, ou si le checker cesse de
  rejeter l'un des noms ambigus couverts.

### STRUCT-010 — Supprimer les inventaires README locaux recopies et non maintenus

- **Nature :** reouverture corrigee et verifiee le 2026-08-30.
- **Signature originale :** 22 references locales n'avaient aucun index; deux
  pages de famille niaient la taxonomie physique active et les courbes
  recopiaient des signatures deja obsoletes.
- **Cloture :** `docs/model-and-curve-reference-index.md` indexe exhaustivement
  les 22 references mathematiques conservees pres du source et explicite
  qu'elles ne possedent ni arbre, ni signatures, ni matrice de capacites. Les
  taxonomies markovian/rough sont synchronisees; les inventaires de fichiers et
  blocs de signatures des courbes sont remplaces par des liens vers les headers
  proprietaires.
- **Preuve :** le checker impose la bijection entre index et README physiques,
  rejette lien stale/duplique, inventaire Files/signatures, ancienne taxonomie
  et coordonnee temporelle sans unite. Les trois CTests codegen/layout/catalogue
  passent.
- **Reouvrir seulement si :** une reference locale redevient non indexee,
  recopie une API/arborescence, publie une taxonomie fausse ou si le controle
  bidirectionnel des liens est retire.

### STRUCT-021 — Synchroniser les contrats CUDA normatifs avec l'architecture active

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** le guide de composition annonçait N-facteurs et
  closed form comme futurs, decrivait deux corps produit et le contrat dynamics
  recommandait encore le FP64 des handlers remplaces par somme compensee FP32.
- **Cloture :** le guide couvre les douze engines actifs du manifeste et les
  compositions markovienne, Volterra FFT, N-facteurs, closed form, LSM et
  sampling. Les alias publics sont documentes comme identiques a l'unique
  `ProductPathPolicy`. Le contrat dynamics reporte les domaines mesures de
  `CompensatedFloatSum` et borne les usages FP64 encore contractuels.
- **Preuve :** le checker impose la presence des douze identifiants engine et
  rejette les deux formulations architecturales obsoletes; les symboles et
  branches cites correspondent aux templates requis et au codegen zero-diff;
  les CTests architecture passent.
- **Reouvrir seulement si :** un engine actif manque au guide, une branche
  supprimee y reste presentee comme active, un second corps produit est decrit
  ou une recommandation de precision contredit les decisions numeriques
  mesurees du registre.

### NUM-019 — Borner les surcharges device FP64 des courbes Nelson--Siegel et Svensson

- **Nature :** surface device FP64 supprimee et contrat host/device separe le
  2026-08-30.
- **Signature originale :** les deux headers d'instantaneous forward exposaient
  un template `__host__ __device__` et `forward_exponential(double)`, alors que
  seul le test les instanciait sur GPU et qu'aucune composition de production
  n'appelait FP64.
- **Cloture :** chaque courbe expose maintenant une surcharge FP32 explicite
  `__host__ __device__` et une surcharge FP64 explicite host-only. Le template
  et les helpers exponentiels surcharges disparaissent : un appel CUDA ne peut
  plus instancier `double` par inference. Les generateurs conservent le FP64
  pour leurs scans d'extrema et de domaine.
- **Preuve :** `numerical_robustness_cuda` compare device et host en FP32, puis
  les deux surcharges host FP64 a des expressions `long double` sous `1e-15`.
  Les bibliotheques de generation Nelson--Siegel/Svensson, les compositions
  fitted Svensson Hull--White/G2++ et leurs trois CTests representatifs
  compilent et passent. La recherche statique ne trouve plus aucun overload
  `double` marque `__device__` dans ces headers.
- **Reouvrir seulement si :** un calcul de courbe FP64 redevient atteignable
  depuis device, si une composition CUDA exige reellement FP64, ou si le
  contrat host-only n'est plus suffisant aux generateurs de domaine.

### FACTOR-001 — Eliminer ou justifier les deux facades produits equity

- **Nature :** duplication semantique eliminee et cout qualifie le 2026-08-30.
- **Signature originale :** 13 headers produit, les deux policies barriere et
  la policy Phoenix maintenaient un corps markovien direct en parallele de la
  `PathProductPolicy` reutilisee par N-facteurs et Volterra; les 21 produits
  ordinaires pouvaient diverger entre engines.
- **Cloture :** les 16 corps `*PricingPolicy` directs disparaissent. Chaque
  surface publique est un alias de
  `equity::PathProductMonteCarloPricingPolicy<Schedule, PathPolicy>` et le
  produit ne possede plus qu'un calendrier, une preparation, un handler et une
  finalisation. Le contrat interdit une nouvelle facade directe sans exception
  bornee, parite pathwise et gain end-to-end mesure.
- **Preuve fonctionnelle :** le temoin compile avant la bascule compare les
  anciennes facades et la composition canonique sur 21 produits Heston x 4 096
  trajectoires avec memes parametres, schedules, seeds et Philox; prix et
  erreur standard sont bitwise identiques et la taille `PreparedRow` maximale
  reste 84 octets. Apres bascule, `path_product_factorization_cuda` impose par
  `std::same_as` l'identite des 21 surfaces publiques avec leur composition et
  rejoue les 86 016 trajectoires. Les tests Heston terminal/path, Volterra FFT
  et QRH N=7 passent sur GPU.
- **Preuve matrice et build :** le codegen reproduit sans diff 12 engines,
  24 modeles, 416 compositions et 1 500 sorties. `all_models` compile les 872
  etapes de la matrice puis un second build est sans travail. Les archives
  Heston representative augmentent de 2,08 % a 3,25 %, effet accepte du nom de
  type generique; aucun second corps ni objet par engine n'est introduit.
- **Ressources SM89 :** European passe de 66 a 68 registres, Asian reste a 70,
  Athena de 75 a 78, range accrual de 72 a 75, down-and-out reste a 70 et
  Phoenix memory de 77 a 80. Les six restent a zero stack/local/spill, shared
  80 ou 96 octets et meme nombre de blocs residents; aucune classe
  d'occupation ne regresse. La faible hausse est acceptee contre la suppression
  de 16 implementations divergentes et reste couverte par la baseline globale.
- **Reouvrir seulement si :** une surface publique cesse d'etre identique a la
  composition canonique, un calendrier/payoff est duplique par engine, une
  architecture cible franchit un seuil d'occupation ou de spill, ou la
  campagne end-to-end montre une regression au-dela de son budget.

### PERF-010 — Rendre le controle de baseline exhaustif, budgete et bloquant

- **Nature :** protocole performance global remplace et rebaseline explicite
  le 2026-08-30; severite originale moyenne, priorite haute.
- **Signature originale :** la liste de workloads etait dupliquee dans le
  runner, cinq mesures effectivement emises etaient ignorees et plusieurs
  experiences compilees n'etaient pas lancees. Le checker bloquait mediane et
  CV, mais ni p95, wall public uniforme, numerique complet, registres,
  local/stack/spills, shared, occupation, taille code ou VRAM. La campagne v1
  de 30 cles montrait en outre des regressions samples et un cas inconclusif.
- **Cloture :** `baseline_sm89_v2.json` est l'unique manifeste autoritatif de
  commandes, cles, budgets et decisions. `run_baseline.py` ne contient plus de
  liste parallele : il execute les 22 commandes du manifeste avec diagnostics
  actives, exige exactement 41 mesures, rejette toute cle inconnue/manquante ou
  dupliquee ainsi que tout diagnostic orphelin, et attache une identite stable
  a chaque mesure. Le build Release, l'architecture 89 et l'absence de fast
  math sont verifies avant campagne.
- **Budgets fail-closed :** kernel et appel public complet bloquent mediane,
  p95 et CV; la publication sample ajoute son wall a 10 % de CV. Chaque champ
  numerique est exact, relatif, maximum ou explicitement derive. Chaque mesure
  porte les attributs runtime de toutes ses specialisations : registres,
  local/stack, shared statique/dynamique, residence/occupation et versions de
  code. VRAM live/residente et taille executable ont des plafonds. Ces budgets
  numeriques et ressources restent bloquants pour la latence closed-form dont
  seul le timing est informatif.
- **Preuve campagne SM89 :** trois campagnes completes produisent 41/41 cles,
  zero manque, doublon ou inconclusif bloquant; le checker retourne `PASS` avec
  deux messages de bruit attaches a l'unique timing informatif. Les tests
  fail-closed couvrent manifeste partiel/duplique/inconnu, environnement,
  protocole, mediane, p95, CV, publication, champs numeriques non budgetes,
  VRAM, taille binaire, registres, local et occupation. Les 41 mesures portent
  65 diagnostics de lancement; SHA-256 du manifeste
  `2e93b4dc7d44db7174b7e168d2de1194c2bfe4ef636af9bfc5e01d501ebd0697`
  et du candidat NDJSON
  `2735720804e9ef7b3e8cc64fd244fb20c3eabdd338bebd2b1fa5f5b7093a42c4`.
- **Decisions explicites :** six selections/rejets sont versionnees. Le chemin
  `uint32` valide mesure 1,69 ms contre 3,76 ms; Phoenix 512 threads environ
  4,07 ms contre 6,32/11,15; CIR noinline utilise 56 contre 64 registres et
  7,38 contre 8,14 ms; les accumulations mixtes plus rapides restent rejetees
  pour erreur numerique; FFT huit pas mesure 4,12 contre 8,74 ms/prix direct;
  le chunk Volterra 65 536 mesure 24,16 ms contre 27,85/75,98 ms.
- **Rebaseline :** la reference v1 incomplete est supprimee. La v2 est publiee
  avec la raison explicite `PERF-010 protocol v2 exhaustive manifest
  initialization after completed audit fixes`; l'outil refuse d'ecrire une
  nouvelle reference sans raison ou tant qu'une cle bloquante est bruyante ou
  hors budget.
- **Portabilite :** les valeurs runtime ne valent que pour la RTX 4090 Laptop
  SM89 et le toolchain declares. SM75, SM86 et tout GPU futur doivent publier
  leur propre manifeste natif; aucune geometrie SM89 n'est presentee comme
  optimale universelle.
- **Reouvrir seulement si :** une commande redevient hardcodee hors manifeste,
  une emission peut etre ignoree, une cle/champ/ressource echappe au budget, le
  p95 ou le wall public cesse de bloquer, une reference peut etre reecrite sans
  raison ni campagne stable, ou une architecture est acceptee avec des seuils
  infers d'un autre GPU.

### STRUCT-011 — Etendre la source de verite typee a toute la matrice de capacites

- **Nature :** reouverture corrigee et verifiee le 2026-08-30.
- **Signature originale :** les identites modeles etaient repetees entre les
  manifests pricing et sampling; les contrats modeles, produits, engines et
  datasets etaient incomplets, et le binding CIR LSM etait faussement declare
  exact par un contrat uniforme fixed-income.
- **Cloture :** les 24 contrats modeles canoniques possedent transition, etat,
  observables, analytics et architectures; les vues pricing sont derivees de
  cette table. Produits, engines et datasets portent leurs policies,
  calendriers, exercice, concepts, launchers, runners, instanciation,
  construction, profil numerique et layout. Chaque recette de prix se resout
  maintenant en un unique chemin engine, binding, target et recette. Les huit
  bindings FI LSM portent leur transition propre : CIR fixed-step, sept exacts.
- **Preuve :** 19 tests de contrat couvrent les champs, resolution complete,
  ajout/retrait de modele et produit, divergence de table, suppression et
  duplication de composition. Le codegen regenere 1 500 sorties sans diff et
  publie le schema de provenance version 2.
- **Reouvrir seulement si :** une identite modele est recopiee dans une table
  concurrente, un contrat minimal redevient infere hors manifeste, une recette
  publiee ne se resout pas exactement une fois, ou CIR joint est de nouveau
  classe comme transition exacte.

### BUILD-005 — Déléguer les targets CMake par domaine

- **Nature :** structure CMake corrigée et vérifiée le 2026-08-30; sévérité
  originale moyenne, priorité moyenne.
- **Signature originale :** le `CMakeLists.txt` racine enregistrait directement
  les targets runtime, catalogue, tests, performance et validation dans plus de
  1 800 lignes, sans propriétaire local par domaine.
- **Clôture :** la racine est réduite à 127 lignes de configuration globale,
  options et orchestration. Les enregistrements appartiennent désormais à
  `cmake/AIFactoryRuntime.cmake`, `AIFactoryCatalog.cmake`,
  `AIFactoryPerformance.cmake`, `AIFactoryTests.cmake` et
  `AIFactoryValidation.cmake`; les helpers partagés restent dans
  `AIFactoryTargets.cmake`. Le README rend ce découpage navigable.
- **Preuve :** l'inventaire des 573 targets exposés avant/après extraction est
  strictement identique. La configuration SM75 sans mathDx compile les 125
  étapes host puis les 39 étapes performance et passe ses 12 tests; la
  configuration fatbin SM75/86/89 avec mathDx compile un pricer rough
  représentatif. Le build principal compile host, CUDA et performance; le
  second build est sans travail en 0,12 s. Les cinq contrôles CTest
  codegen/layout/catalogue/QRH passent, le timeout QRH étant porté à 120 s pour
  sa durée observée de 36,54 s.
- **Propagation :** une régénération effective des sorties codegen a provoqué
  229 recompilations/relinks ciblés; aucune cible ni dépendance de domaine n'a
  disparu lors du déplacement. Le dry-run fondé sur le seul timestamp a été
  conservativement jugé non probant et n'est pas présenté comme preuve.
- **Reouvrir seulement si :** la racine réenregistre un target de domaine, si
  une configuration avec/sans mathDx diverge, si l'inventaire public change
  sans décision explicite, ou si une mutation réelle ne propage plus vers ses
  consommateurs attendus.

### STRUCT-019 — Donner au gate Performance un propriétaire dans le périmètre principal

- **Nature :** corrigé, qualifié et vérifié le 2026-09-03; sévérité originale
  moyenne, priorité haute, confiance prouvée.
- **Signature originale :** benchmarks, fixtures, manifeste, baseline, runners
  et checker de décision vivaient exclusivement sous `validation/**`; aucun
  propriétaire principal n'existait sous `tests/performance` et
  `tools/performance`, et le protocole complémentaire ne satisfaisait pas les
  quatre sous-audits Performance version 7.
- **Clôture :** `tests/performance` possède les benchmarks, fixtures, baseline
  et preuves par architecture; `tools/performance` possède exécution, checker,
  rebaseline et profilage; `docs/performance-regression-protocol.md` porte le
  contrat durable et `cmake/AIFactoryPerformance.cmake` le graphe de targets.
  Le protocole couvre 41 mesures, quatre frontières de temps et quatre rapports,
  interdit best-of-N et recomposition par clé, conserve chaque campagne brute,
  applique les budgets numériques, ressources compilées, VRAM et binaires, et
  distingue 5 % de bruit kernel de 10 % pour l'enveloppe host sans relâcher le
  seuil de régression médiane/p95 à 5 %.
- **Preuve :** trois campagnes SM89 sur trois sont admissibles sous
  `build-dev/performance_candidate_sm89_v3.ndjson.campaigns/20260903T184737.789921Z`;
  le checker passe 41 mesures avec zéro inconclusive bloquant et deux messages
  attachés à l'unique timing informatif. La baseline qualifiée porte le SHA-256
  `94b7370a2bf1ebed350a04a1037c42115d7a6946127399d7c4358f329786429f`;
  son prédécesseur conservé porte
  `26c4af09f90ffeb812d7ea906619981476d94f2c4e6b18dd07e5470c54381604`
  et le diff exhaustif d'initialisation
  `205cee367c9b0b152b37a02b1477252949fa9cf9753de41d429524cbd2323717`.
  Les quatre rapports contiennent 18/8/4/11 mesures. Les huit artefacts Nsight
  Compute 2026.2.1 sous `tests/performance/profiles/sm89` lient le symbole,
  l'exécutable, le candidat, la baseline, l'environnement et le CSV brut pour
  CIR, sample rough N-factor, LSM Heston et rough SABR FFT. Les 27 tests
  fail-closed, le CTest `performance_baseline_checker`, le build performance
  sans travail et `git diff --check` passent.
- **Portabilité :** cette baseline et ces profils qualifient uniquement le
  profil RTX 4090 Laptop SM89 et son toolchain déclarés. Chaque autre GPU ou
  toolchain doit publier son propre manifeste natif et ses propres profils;
  aucune géométrie SM89 n'est une valeur universelle.
- **Réouvrir seulement si :** un composant décisionnel principal retourne sous
  `validation/**`, une clé ou ressource peut échapper au manifeste, une
  campagne est sélectionnée ou recomposée opportunément, une rebaseline peut
  s'auto-valider sans prédécesseur/diff/raison/approbation, un scope de temps
  cesse de bloquer selon son budget, un profil ne correspond plus au binaire
  mesuré, ou une architecture réutilise les seuils observés d'un autre GPU.
