# Tableau de bord de l'audit principal

Référentiel actif : **v10.1 — 2026-10-02**. Dernier passage examiné : **2026-10-07, suivi des quatre constats encore ouverts** ; auparavant, reprise des dix constats hors rough, axe IV — performance, axe VI — hygiène et reprise de PRODUCT-001 et NUM-028 à NUM-030. Les preuves détaillées des passages précédents sont dans [véracité](history/truth-before-hygiene-2026-10-06.md) et [architecture](history/architecture-complete-before-truth-2026-10-06.md). Le passage hygiène s'est déroulé pendant ce chantier ; ses observations de génération correspondent au worktree au moment de leur exécution.

## Verdict et portée

| Axe | Couverture courante | Verdict | Motif |
|---|---|---|---|
| I. Véracité | produit fixed income à barrière qualifié ; signaux rough rejoués ; portée de l’erreur LSM explicitée | non conforme | `PRODUCT-001`, `NUM-029`, `NUM-036` réglés ; `NUM-037` clos administrativement mais non résolu ; `NUM-028`, `NUM-030` et validation externe ouverts |
| II. Factorisation | reprise ciblée des frontières LSM et du concept de régression | indéterminé | cinq constats architecturaux corrigés ; nouvelle revue complète nécessaire pour certifier l’axe |
| III. Arborescence et lisibilité | reprise des graphes rough et de la source de vérité equity | indéterminé | aucun des cinq constats ciblés encore ouvert ; couverture globale à réévaluer |
| IV. Performance | comparaisons A/B SM89/CUDA 12.9 complètes ; compromis PERF-024 accepté hors gate | non conforme | `PERF-025` et `PERF-026` ouverts ; aucun gate pour le profil courant |
| V. CMake | dry-run de fraîcheur seulement | indéterminé | pas de matrice d'incrémentalité |
| VI. Hygiène et artefacts | dépôt suivi, codegen, builds et racines locales vérifiés ; rétention des preuves classée | conforme | `DOC-004` clos : Quick start autonome et scripts PPTI privés ignorés |

Deux constats clos administrativement gardent un problème numérique non
résolu dans [unresolved-closures.md](unresolved-closures.md) : `NUM-037`
exclut 52 prix et SE SABR markoviens des références ; `PERF-027` retire la
Hessienne CIR–Jamshidian. `PERF-024`, `NUM-033`/`NUM-036` et `PRODUCT-001`
ont un compromis accepté ou un contrat borné documenté dans
[closed.md](closed.md).

Le verdict VI conserve les contrôles réussis : aucune bibliothèque, aucun exécutable, objet, cache Python ou checkpoint n'est suivi par Git dans les 5 846 entrées inspectées. Trois notebooks suivis ont une fonction de validation ou de rapport. Huit rapports de performance suivis contiennent des chemins absolus historiques ; [la politique locale](../local-artifacts.md) les classe comme références anciennes, sans en faire des commandes actuelles.

## Sélection du catalogue source — 2026-10-07 (étape fixed income)

Après réorganisation, le catalogue livrable contient 1 195 couples
`generator.cpp`/`recipe.yaml` : 25 modèles, 28 produits, 3 courbes, 655 prix
alignés et 484 prix avec gradients alignés. Les recettes cartésiennes et les
samples restent locales. Les 24 prix fixed income sur courbe flat ont été
réintégrés après un test d'exhaustivité ; le catalogue fixed income contient
117 prix et 104 recettes prix + gradients. Les 26 gradients de swaption
bermudéenne sélectionnés utilisent `frozen_regression_policy`, tandis que CIR
standalone/Jamshidian reste limité à l'ordre un. Le call up-and-out a 13
recettes de prix mais pas de binding de gradients. Le codegen comparé au
dépôt, 49 tests dataset, le plan de 221 jobs sur un paquet public sans `work/`,
et la compilation de cinq générateurs représentatifs passent. Le générateur
de paramètres du produit barrière a été exécuté en staging (1 000 lignes et
reçu authentique). Une première campagne de 221 jobs a été lancée : quatre
prix analytiques ont terminé, puis le premier prix barrière a été rejeté par
le contrôleur, car sa recette demandait 65 536 chemins alors que l'inspecteur utilisait
2²⁰ par défaut. Aucun dataset de cette campagne n'a été publié. Les
13 recettes barrières et leurs générateurs utilisent désormais 2²⁰ chemins,
comme tous les Monte Carlo publics, rough inclus. Le codegen complet passe
sans divergence ; un générateur CIR/barrière reconstruit a produit 1 000 prix
à 2²⁰ chemins avec reçu SM89 cohérent et contrôles d'artefact valides.
L'ancienne campagne reste figée ; une nouvelle campagne est requise. Les
618 anciens prix physiquement présents ne deviennent pas des références des
nouveaux paramètres par ce déplacement.
`NUM-028` et `NUM-030` restent ouverts ; aucune clôture numérique nouvelle
n'est revendiquée.

## Contrôle de la campagne fixed income — URL Bermudan, 2026-10-07

La campagne figée `work/generation/fixed-income-aligned-prices-gradients-02`
compte 7 jobs complets, 1 échec Bermudan CIR et 213 jobs non exécutés.
La recette du prix Bermudan demandait `/v2/`, le générateur natif écrivait
`/v1/` ; le calcul avait terminé mais le JSON a échoué au contrôle d'URL.
Les deux fabriques Bermudan écrivent maintenant `/v2/`. Les 26 gradients
Bermudans restent sous `frozen_regression_policy`. Un contrôle statique des
117 recettes de prix fixed income, 23 tests et la comparaison du codegen
passent. Deux exécutions natives reconstruites, CIR et CIR++/flat, passent
`check_outputs` sur 1 000 lignes à 2²⁰ chemins chacune. Aucun résultat de
`-02` n'est publié ; une nouvelle campagne est requise. La qualité numérique
des 221 jobs et la parité des prix centraux restent à contrôler.

## Comparaison des prix centraux fixed income — 2026-10-08

La campagne `-03` a 127 jobs terminés, 1 gradient CIR++/Nelson–Siegel payer
rejeté par un contrôle bit à bit et 93 gradients encore en attente. Le prix
seul scalaire et le central coopératif diffèrent au plus de 4,4517×10⁻⁷ sur
les 1 000 lignes du job ; 0 dépasse le budget inter-mode préexistant. Le
contrôle numérique est désormais une mesure enregistrée, sans arrêt de
génération. Un amendement explicite du contrôleur permet de reprendre `-03`
sans refaire les 127 jobs ; la reprise complète et la qualification des 221
sorties restent à vérifier. Les gardes d'intégrité des artefacts restent
bloquants.

## Implantation du catalogue — 2026-10-07

Les recettes et données sélectionnées occupent directement `catalog/` et `datasets/`. Les recettes locales, les données hors production et les métadonnées historiques (archives, qualifications, releases) sont sous `work/catalog/` et `work/datasets/`. Les manifestes historiques signés restent inchangés ; leur résolution de chemins est assurée par `catalog_layout.physical_path`. Les 696 sorties sélectionnées, les 24 sorties locales matérialisées et les payloads historiques sous `work/datasets/` sont présents après déplacement ; NUM-029, NUM-030, NUM-036, NUM-037, rough Bergomi et les gates martingale ont été revérifiés. Un générateur de prix aligné Black–Scholes/european_calls a été exécuté en staging sur les nouvelles racines ; il a émis JSON et reçu authentiques, sans publication ni validation indépendante du prix.

## Paramètres modèles et produits canoniques — 2026-10-07

Les [25 modèles et 27 produits](parameter-catalog-refresh-2026-10-07.md) ont été régénérés par leurs binaires natifs puis remplacés dans `datasets`, avec recettes et reçus vérifiés. Les nouvelles seeds et les bornes révisées donnent 52 nouveaux JSON de 1 000 lignes, toujours 900 cœur/100 stress ; aucune corrélation positive ne subsiste. Les essais européens des quatre familles rough, du SABR affiné et la parité gaussienne G2 passent sur les nouveaux couples. **Les anciens prix canoniques restent liés aux anciennes entrées** : la commande de repricing aligné est prête, mais les 655 prix doivent encore être reconstruits, exécutés et qualifiés avant usage comme références des nouveaux paramètres. `NUM-028` et `NUM-030` restent ouverts. Les entrées des anciennes qualifications sont [archivées à l’octet près](../../work/catalog/archives/parameter-catalog-20261007/manifest.json) ; leurs vérificateurs NUM-029, NUM-030, rough Bergomi et le gate de domaine passent après remplacement du catalogue courant.

## Domaines MC alignés — 2026-10-07

La [campagne candidate antérieure](aligned-mc-model-domain-2026-10-07.md)
conservait tous les produits à cette date. Une
[release SABR qualifiée](../../work/catalog/releases/sabr-aligned-mc-domain-20261007-v1/manifest.json)
de 30 000 lignes (1 modèle + 29 prix) est exécutée et vérifiée : zéro signal
de parité sur 1 000 paires et zéro prix significatif avec `SE/prix >25 %`
sur les 29 jeux. Les deux Bergomi et le QRH ont des domaines candidats
validés sur leurs paires européennes ; les replays QRH en grille/facteurs et
les queues du lookback 770 sont stables. Les anciens prix NUM-028 et NUM-030
restent non qualifiés et les deux constats restent ouverts en attendant une
qualification complète des autres payoffs et de la portée de l’estimand.

## Reprise des dix constats hors rough — 2026-10-06

Les constats `BOUNDARY-001`, `STRUCT-001`, `FACTOR-004`, `FACTOR-005` et
`NAME-014` ont été corrigés et transférés dans [closed.md](closed.md).
Le callback hôte de progression LSM appartient désormais au commun neutre,
les limites de grille des gradients sont fournies par leurs adaptateurs,
le concept de régression teste conditionnellement le raffinement, le graphe
concret européen est sous son produit et la table de capacités porte le nom
equity. La génération complète du codegen a été comparée au worktree sans
différence ; les tests C++/CUDA ciblés des graphes rough, LSM American et
Bermudan, de la progression et de la matrice de capacités passent.

`NUM-036` est clos dans [closed.md](closed.md) avec un contrat de SE
conditionnelle explicite. Les 36 jeux de prix
LSM (36 000 lignes) ont une [qualification versionnée](../../work/catalog/qualifications/num036-lsm-error-scope-20261006-v1/manifest.json)
qui ajoute la portée de `standard_error` sans modifier aucun résultat. Les
hashes des originaux, des copies et de leurs reçus ont été revérifiés ; le
test multi-graines donne 12 erreurs nulles à `t0` et quatre positives, avec
0,004241 d'écart de prix entre fits. Cette SE reste conditionnelle aux chemins
du fit et ne couvre pas l'incertitude de politique.

`PERF-024` est **clos par acceptation explicite du compromis mémoire/temps**,
avec les mesures défavorables conservées dans [closed.md](closed.md). La comparaison
oppose le même `node_graph` en stride actif et maximal ; `mono` contrôle
seulement la parité numérique. Le stride actif économise jusqu'à 100 663 320
octets de pic attribué aux allocations Heston à un million de chemins, avec
sorties bit à bit égales sur les 12 charges. À 512 opérations par échantillon
court, 11/12 charges passent et CIR 4 axes/4 096 chemins dépasse le budget
p95 API de +5 % avec **+5,51 %** ; l'utilisateur accepte cet écart. L'essai
complet à 1 024 opérations reste archivé : cinq p95 dépassent +5 %, jusqu'à
+23,24 % kernel, et un CV kernel est non admissible. Les préflights des six
campagnes passent sur secteur à 175 W. La clôture ne signifie donc **pas** que
le gate de performance soit passé ni que tous les p95 soient qualifiés ; le
protocole général et le baseline restent inchangés.
[Preuves A/B](../../artifacts/audit/perf024-2026-10-07/README.md) ;
[décision et limites acceptées](closed.md).
`PERF-025` reste ouvert : la forte mémoire locale de l'instanciation Bates
mixte n'a pas encore de variante profilée et validée aux capacités courante
et maximale. `PERF-026` reste ouvert : le protocole nomme le preset courant
et distingue le baseline CUDA 13.3 ; il manque un baseline CUDA 12.9 et trois
campagnes admissibles. `PERF-027` est clos administrativement et suivi dans
[unresolved-closures.md](unresolved-closures.md) : la Hessienne du binding
CIR–Jamshidian est retirée, mais son calcul reste à résoudre ; le contrat
se limite au prix et à l’ordre un. Les demandes d'ordre deux sont
rejetées, le manifeste et les recettes ne publient que l'ordre un, et le
benchmark diagonal sort de la matrice active. Les 1 000 lignes core et huit
stress passent le gate prix/gradient ; la référence FP64 contrôle des lignes
alignées. Quatre datasets prix + gradient et leurs reçus ont été exécutés
dans un build Ninja isolé, mais restent en staging : la publication exige un
worktree Git propre. [Campagne ordre un](../../artifacts/audit/perf027-2026-10-06/first_order_scope/result.json).

## Reprise ciblée véracité et produit de taux — 2026-10-06

- PRODUCT-001 est transféré dans [closed.md](closed.md) : les 13 compositions
  fixed income sont branchées au call up-and-out sur zéro-coupon, avec
  surveillance aux points de grille et actualisation stochastique. Les
  13 générateurs de prix et celui des paramètres ont été exécutés dans un
  build isolé ; les 13 000 prix, 1 000 produits et leurs reçus ont été
  contrôlés. La référence OU indépendante, les limites
  européenne et déterministe, la barrière déjà franchie et le raffinement
  temporel passent. Le prix de barrière continue reste une approximation
  sur grille. Les 14 recettes sont livrées dans `catalog/` ; leurs anciens
  reçus locaux ne qualifient pas les paramètres rafraîchis et aucun nouveau
  `generation.yaml` n'est livré pour ces recettes en attente d'exécution.
  [Preuve produit](../../artifacts/audit/product001-2026-10-06/result.json).
- NUM-029 est transféré dans [closed.md](closed.md) : le pas Lamperti FP32
  est corrigé ; les 15 écarts rough SABR signalés disparaissent sur trois
  replays par ligne. Les 29 prix rough, 29 prix markoviens et quatre jeux de
  samples affectés ont été exécutés et publiés dans une
  [release versionnée](../../work/catalog/releases/num029-lamperti-20261006-v1/manifest.json)
  avec 62/62 reçus authentiques et 12 058 000 lignes vérifiées. Le scan des
  1 000 paires européennes rough SABR publiées a zéro signal de parité.
  Les treize lignes rough SABR à `ρ>0` ont toutes `β<1` ; leur plus grand
  résidu vaut 2,109 SE combinées et le modèle continu reste martingale
  ([analyse du domaine](sabr-martingale-domain.md)).
  Les 26 signaux persistants du SABR markovien sont suivis séparément sous
  `NUM-037`, suivi dans
  [les clôtures non résolues](unresolved-closures.md) ; leurs 52 prix restent non certifiés comme
  références.
  [Validation de release](../../artifacts/audit/rough-price-quality-2026-10-06/num029_release_validation.json).
- `NUM-037` est clos administrativement par exclusion des références, mais
  son problème numérique reste dans [unresolved-closures.md](unresolved-closures.md) :
  les 26 paires markoviennes
  signalées sont explicitement exclues des références de prix et de SE par la
  [qualification versionnée](../../work/catalog/qualifications/num037-sabr-parity-20261007-v1/manifest.json).
  Les 52 sorties originales restent traçables. Les replays à 65 536,
  262 144 et 2²⁰ chemins, trois graines, premier moment, absorption, queues,
  schéma Euler spot indépendant et raffinement temporel documentent la limite ;
  le vérificateur contrôle les 26 masques et les empreintes. Aucun prix de
  remplacement n'est revendiqué. [Preuves](../../artifacts/audit/num037-2026-10-06/README.md).

- NUM-028 : 507 replays historiques confirment le déficit extrême de
  premier moment sur la ligne 984. Le nouveau domaine QRH borné passe les
  1 000 paires européennes, 18 replays temps/facteurs et neuf replays de
  queue lookback. Les 29 payoffs et l’estimand continu restent non qualifiés :
  constat ouvert.
- NUM-030 : le domaine est désormais établi pour le modèle continu :
  le noyau log modulé satisfait les hypothèses du théorème de Gassiat,
  `ρ>0` implique une martingale locale stricte, et les huit résidus stress
  sont négatifs dans ce domaine. Sur la grille finie, l'espérance exacte
  du schéma reste le forward ; les replays ne quantifient pas le défaut
  continu. Build courant isolé : 20 replays ciblés, contrôles `ρ=0`,
  FFT/direct et deux outliers à un million de chemins. Puis 96 prix stress
  supplémentaires à 2²⁰ chemins (six payoffs × 16 graines) montrent des
  queues très instables, jusqu'à 0,07027–3,74733 pour le lookback 960. Une
  [qualification versionnée](../../work/catalog/qualifications/num030-lmb-stress-20261007-v1/manifest.json)
  exclut 1 219 lignes de prix/SE sur 29 datasets sans changer les originaux.
  La queue et les prix stress manquent encore de référence indépendante :
  **constat ouvert**. [Analyse](num030-martingale-domain.md).
- Le [gate martingale des datasets](martingale-reference-datasets.md) passe :
  les 15 lignes `ρ>0` du rough Bergomi standard sont maintenant exclues des
  références dans 29 jeux (435 prix/SE), les 42 du log-modulated rough
  Bergomi restent masquées, et les deux SABR publiés ont tous `β<1`. Les
  contrôles CTest `martingale_reference_domain` et
  `published_martingale_reference_domains` passent dans le build isolé.
  Ce verdict porte sur le domaine, pas sur les prix/queues encore ouverts.

Les inventaires chiffrés de la suite de ce document sont ceux du passage
hygiène, antérieur à cette reprise ciblée ; ils ne dénombrent donc pas les
nouveaux jeux fixed income ni les campagnes rough de cette reprise.

## Audit performance — 2026-10-06

Le passage IV couvre le code des stratégies closed form scalaire/coopérative,
MC terminal exact/à pas fixes et produits de chemin, LSM equity/fixed income,
Volterra FFT et causal, samples à un ou plusieurs chemins et graphes diagonal
et mixte. Les preuves sont sous `artifacts/audit/axis-performance-2026-10-06/`
(snapshot et hashes dans `receipt.json`). Le build temporaire de 1,1 Gio a été supprimé après archivage et vérification des six exécutables profilés, de la configuration et des rapports Nsight. Les quatre tests CUDA de graphes
terminal, path, mixte terminal et mixte path passent dans un build isolé
Release SM89/CUDA 12.9.41. Les mesures restent diagnostiques : ni trois
campagnes homogènes ni qualification d’un million de chemins par prix sur
chaque famille.

| Famille | Observation courante et limite |
|---|---|
| MC prix et produits de chemin | Revue de la topologie un bloc/prix, row préparée en shared, chemins indépendants et réduction FP64 ; géométrie Bates Phoenix 512 threads plus rapide que 128/256 sur 32 prix × 4 096 chemins, sorties identiques. Pas de sweep production 100–10 000 prix × 2²⁰ chemins. |
| Closed form | Black–Scholes scalaire six dérivées, 65 536 lignes : 0,151 ms kernel médian. CIR–Jamshidian ne publie plus de Hessienne ; son benchmark ordre 1 passe le gate prix/gradient core/stress. Les anciens temps de diagonale ne sont pas qualifiés. |
| LSM equity/fixed income | Heston American et CIR Bermudan, replay `frozen_policy`, 4 096 chemins : graphes 1,589/0,382 ms contre mono 3,479/0,788 ms kernel ; hashes de sortie égaux. Le replay `frozen_exercise` graphe passe aussi sur les deux modèles (1,498/0,384 ms kernel), avec son propre estimand. Workspaces internes et graphe mesurés séparément ; batching LSM consulte la VRAM libre après l’allocation du graphe et garde au moins 1 Gio ou 10 % de marge. Stride diagonal actif corrigé ; A/B sur 12 charges Heston/CIR, parité bit à bit et gains mémoire jusqu'à 100,7 Mo. Les comparaisons complètes à 512 et 1 024 opérations par échantillon court conservent des dépassements p95 ou un CV non admissible : `PERF-024` est clos par acceptation, sans passage du gate ; mesures défavorables dans `closed.md`. |
| Graphes diagonal/mixte | Workspace MC plafonné à 512 Mio ; graphe Gaussian FFT plafonné à 512 Mio en incluant FFT/cache. Le diagonal et le mixte allouent maintenant leurs valeurs selon la capacité effective ; le pic VRAM diagonal doit être mesuré. Nsight trouve 5 680 octets locaux/thread et 84,85 % de secteurs L2 locaux sur Bates range accrual mixte (`PERF-025`). Les lectures de reconstruction terminale sont stridées, mais le profil réduit n’estime que 2,46 % de gain ; pas de constat séparé sans A/B. |
| Rough FFT et causal | Rough Bergomi hybride : quatre phases lancées avec 0 octet local/thread sur la charge diagnostique 64 pas/8 192 chemins ; causal FFT à 256 pas : tile 32 à 0,578/0,936 ms contre tile 64 à 0,657/1,082 ms pour 8 192/32 768 chemins ; le noyau d’avance occupe 84,6 % du temps GPU à 8 192 chemins. Pas de campagne comparative admissible ni de qualification du nouveau moteur (`PERF-026`). |
| Samples | Huit configurations exact Markovian, fixed step, rough N-facteurs et Volterra FFT mesurées, dont 3 M × 1 et 12 000 × 250 pour les trois premières familles ; Volterra utilise ses réductions déclarées. Les kernels lancés ont 23–166 registres et 32 octets locaux/thread ; la mémoire suivie totale du cas rough N-facteurs atteint 1 007 Mio pour 3 M paramètres indépendants. Ce coût est distingué des sorties et reste à qualifier sur d’autres GPU. |
| Précision et inlining | Aucune des 3 915 unités CUDA/C++ du build n’utilise `--use_fast_math` ou `--ftz=true`. Moments et régressions LSM restent FP64. Variante CIR `noinline` 11,179 ms contre 12,262 ms `inline` sur le même batch de 131 072 swaptions, ce qui confirme le choix actuel sur ce profil ; aucun forçage global d’inlining n’est justifié. |

Le baseline SM89 v3 vient de CUDA 13.3 et ne sert pas de référence de
régression pour CUDA 12.9. Le profil configuré `sm89_cuda12_unqualified_v1`
n’active pas `performance_regression_gate`. Les quatre constats et leurs
conditions de clôture sont dans [response.md](response.md).

## Sources, codegen et orphelins

- Régénération `python3 tools/codegen/pricing_bindings/generate.py --family all --output /tmp/ai_factory_hygiene_codegen_20261006 --compare-root .` : **7 036 sorties comparées, zéro différence, zéro chemin manquant ou en trop**. Le contrôle d'inventaire de la famille `all` confronte aussi bindings, recettes, helpers de samples et chemins attendus du renderer. Le journal compressé et son empreinte sont dans la [preuve locale](../../artifacts/audit/axis-hygiene-2026-10-06/receipt.json).
- Croisement des sources suivies présentes avec `compile_commands.json` du build local : **1 222/1 222** unités de `src`, **17/17** unités de `tools`, **696/696** générateurs de `catalog` et **142/142** unités de tests présentes y figurent. Un ancien test rough FFT suivi était déjà supprimé dans le worktree et ne fut pas compté comme fichier physique orphelin. Le croisement avec `CTestTestfile.cmake` ne laisse hors CTest que vingt exécutables explicitement nommés benchmark, probe ou study ; ce ne sont pas des tests oubliés. Ce croisement de graphe n'est pas une compilation réussie du snapshot courant.
- Le balayage des imports/includes de `src`, `learning`, `tools` et `catalog` ne trouve pas de dépendance obligatoire vers `work/` ou `artifacts/`. `work/catalog` est enregistré seulement si ses recettes physiques existent, et `work/experiments` exige un opt-in CMake explicite.
- Balayage des includes et des templates : treize headers de compatibilité `frozen_exercise` ne contenaient qu'un include vers `frozen_replay` et n'avaient aucun consommateur interne. Ils ont été retirés ; aucun install/export de ces headers n'existe. Les unités CUDA American et Bermudan correspondantes **recompilent** avec CUDA 12.9/SM89. `BOUNDARY-005` est transféré dans [closed.md](closed.md). Aucun autre header existant sans consommateur n'a été démontré par ce balayage ; les deux chemins Volterra trouvés étaient déjà des suppressions du worktree.

## Datasets, campagnes et rétention

| Racine | Inventaire et preuve | Décision |
|---|---|---|
| `datasets/` | 696 sorties sélectionnées, toutes présentes ; **696/696** SHA-256 et comptes de lignes correspondent aux `generation.yaml` | conserver les données publiées |
| `work/datasets/` | 1 713 recettes locales dont 24 sorties matérialisées, plus les payloads historiques de qualification et release ; les 24 sorties locales gardent leurs hashes | conserver ces jeux et les références historiques |
| `work/generation/` | 18 campagnes gelées : cinq publiées complètes, quatre publiées inachevées, cinq études privées complètes et quatre privées inachevées ; certaines portent `running` sans processus actif | conserver les checkpoints, tentatives et preuves de publication ; reprendre explicitement ou classer avant suppression |
| `artifacts/` | 6,9 Gio : audit, performance, validation, learning, Premia et dataset historique relocalisé | conserver les preuves citées et leurs hashes ; ne pas confondre avec un cache |
| `builds/local-cuda12-9-0-sm89/` | 14 Gio, CUDA 12.9/SM89 ; le dry-run `ai_factory_cuda_tests` annonce **64 compilations CUDA** | garder comme build actif, sans employer ses anciens binaires comme preuve courante |

Pour les feuilles historiques non régénérées, les recettes actuelles ne doivent pas être présentées comme leurs recettes d'origine : les hashes diffèrent pour **644/696** feuilles de production. Les 52 paramètres modèles/produits régénérés correspondent à leurs reçus. Les hashes diffèrent encore pour **23/24** feuilles locales matérialisées. [Le guide de déploiement](../deployment-catalog.md) annonce déjà cette limite et le manifeste la marque avec `recipeMatchesReceipt: false`. Les hashes des artefacts prouvent la conservation des octets historiques, pas la reproduction par le code actuel.

## Nettoyage effectué

- Suppression de `build/` (8,9 Gio, cache CUDA 13.3/SM75), `builds/ppti/` (173 Mio, cache CUDA 13 obsolète) et `work/builds/heston-samples-3m-current/` (247 Mio, sous-build). Les campagnes qui y pointent historiquement copient leurs propres binaires, inputs, sources et configurations ; aucun processus ne les utilisait.
- Suppression de 77 répertoires initiaux de cache Python/pytest (environ 5 Mio), puis de trois caches recréés pendant les contrôles. `.gitignore` exclut désormais aussi les caches pytest, Ruff, mypy, checkpoints de notebooks, couvertures et rapports binaires de profiler ; `git check-ignore` a vérifié ces motifs. Un cache NumPy de 62 Mio a été supprimé après vérification SHA-256 de son dataset relocalisé et de ses deux jeux de paramètres.
- Le build isolé de l'audit véracité (3,5 Gio) a été remplacé par 2,8 Mio de snapshot, configuration et journal CTest hashés sous [axis-truth-2026-10-06](../../artifacts/audit/axis-truth-2026-10-06/receipt.json). Les sorties temporaires du codegen, les objets CUDA de vérification et les logs non retenus ont aussi été supprimés.
- `docs/local-artifacts.md` indique maintenant les propriétaires et la règle de conservation des principales racines locales. Les treize alias supprimés ont leur clôture dans l'archive.

Ce nettoyage libère **environ 13 Gio** tout en gardant les datasets, checkpoints et preuves nécessaires. `DOC-004` est fermé : les trois scripts génériques du Quick start sont suivis, les deux scripts PPTI restent ignorés et leurs commandes ont quitté la documentation publique. Les campagnes marquées `running` sans processus sont classées comme interrompues ou reprenables, pas supprimées. Les rapports de performance à chemins absolus restent des preuves historiques dont les binaires référencés ne sont plus le build actif.
