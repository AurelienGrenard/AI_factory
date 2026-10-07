# Tableau de bord de l'audit principal

Référentiel actif : **v10.1 — 2026-10-02**. Passage du **2026-10-06** : **axe I, véracité**. Les verdicts des axes II et III proviennent du [passage architectural complet archivé](architecture-complete-before-truth-2026-10-06.md) ; les axes IV à VI n'ont pas été certifiés par les lectures faites ici.

## Verdict et portée

| Axe | Couverture du dernier passage | Verdict | Motif |
|---|---|---|---|
| I. Véracité | partielle : matrice et cœurs exécutables couverts ; références indépendantes et lignes de production incomplètes | non conforme | `NUM-028` à `NUM-030` non élucidés, `NUM-036` sur la portée de l'erreur LSM ; validation persistante indisponible (`VALID-001`) |
| II. Factorisation | passage précédent, unités architecturales déclarées | non conforme | sept constats architecturaux ouverts |
| III. Arborescence et lisibilité | passage précédent, graphe et parcours | non conforme | mêmes constats architecturaux |
| IV. Performance | non examinée | indéterminé | pas de campagne admissible |
| V. CMake | aperçu structurel seulement | indéterminé | pas de matrice d'incrémentalité |
| VI. Hygiène et artefacts | aperçu structurel seulement | indéterminé | pas de décision de rétention |

Ce verdict ne signifie pas que les prix ou sensibilités ordinaires des autres familles sont faux. Les tests et les chemins audités étayent des conformités **locales** ; ils ne remplacent pas une référence indépendante pour chaque cœur distinct ni le rejeu des lignes rough signalées. Les constats ouverts sont dans [response.md](../response.md), et les problèmes propres au pipeline de références sont suivis dans [l'audit de validation](../../validation/response.md).

## Snapshot et matrice de capacités

- Branche `main`, `HEAD` `03f72d81045330cd8a234b83c877292379cd0623` ; worktree déjà modifié. Snapshot de reconstruction pris avant le passage précédent sous `/tmp/ai_factory_full_audit_20261006/` : patch suivi SHA-256 `178c378252267a936bd9ded46f8fac093550aab9acf5f1a2ab9ef6b02102c099`, archive non suivie SHA-256 `2e861b3aa3a499a5c673f0dead870805b696d2bd7cb74b892abb591e1f72aebf`. Ce passage audite le worktree courant, y compris les corrections du passage précédent.
- Build isolé historique `/tmp/ai_factory_full_audit_20261006/build` : GCC 14.3.0, CUDA 12.9.41, cuFFTDx/mathDx 25.12.1, SM89, RTX 4090 Laptop. Les binaires du `build/` du dépôt n'ont pas servi de preuve, car leur fraîcheur n'était pas garantie.
- Le manifeste déclare **13 moteurs**, **25 modèles**, **26 produits**, **3 courbes**, **439 bindings prix**, **387 bindings delta**, **313 bindings gradients**. Les **2 409 recettes disponibles** ont toutes un générateur physique ; les **7 031 sorties** de codegen restent identiques au dépôt après recalcul de la provenance. La recette identifie moteur, modèle, produit, construction, ordre et particularité ; le regroupement ci-dessous suit les cœurs mathématiques plutôt que chaque wrapper généré.

| Cœur annoncé | Recettes prix / gradients | Chemin de calcul et preuve inspectée | Couverture numérique |
|---|---:|---|---|
| Equity closed form | 28 / 28 | analytics Black–Scholes, huit produits analytiques, stencils et Greeks ; prix/Greeks analytiques dans `black_scholes_cuda` et `price_gradients_european_cuda` | partielle pour les huit payoffs et frontières ; référence indépendante forte sur vanilla |
| Equity MC markovien | 668 / 668 | 12 dynamiques exactes ou à pas fixe, schedules terminal/chemin, payoff, réduction FP64 ; Heston comparé à une référence QuantLib intégrée, premiers moments et lois de sauts dans les tests modèle | partielle : lois et cas limites couverts par tests ciblés, pas de référence indépendante pour chaque couple modèle-produit |
| Equity LSM exact / à pas fixe | 32 / 64 | simulation forward, régression FP64, mise à jour backward, décision `t0`, replays d'exercice et de politique ; oracle CPU indépendant sur chemins américains construits | partielle : pas d'oracle indépendant produit-spécifique pour tous les replays, ni d'estimation de l'incertitude du fit |
| Fixed income closed form | 140 / 140 | analytics des sept modèles, taux/caplet/bond option, Jamshidian un facteur ; référence CPU FP64 et lignes CIR SciPy, racine/branches de stress | partielle : couverture forte sur un facteur, références persistantes externes indisponibles |
| Fixed income MC terminal | 16 / 16 | G2/G2++ sous Q, transition jointe facteurs–intégrale, actualisation pathwise, swaption européenne | partielle : identités covariance et prix représentatifs, pas chaque courbe/produit indépendamment |
| Fixed income LSM Bermudan | 52 / 104 | CIR/CIR++ sous mesure terminal-forward, gaussiens sous Q, cashflows co-terminaux, régression et deux replays | partielle : tests de parité/finitude ; référence externe de prix et oracle de replay non rejoués sur tout le périmètre |
| Rough FFT / lift N-facteurs | 348 / 0 recettes gradients | hybride Volterra, convolution causale FFT, réemploi du spectre seulement pour noyau/grille identiques ; lifts 2/3/7 facteurs, prix et graphes de sensibilités API | partielle : approximation et parités locales contrôlées, trois signaux de production ouverts |
| Samples | 50 recettes | même dynamique de modèle, grille d'échantillonnage et Philox ; tests de replay et de moments | partielle : pas de qualification statistique exhaustive par modèle |

Les bindings rough de gradients et delta existent comme API, même si le catalogue permanent n'émet pas de recettes de gradients rough ; leur absence de la colonne « recettes gradients » n'est pas une absence de code.

## Chaînes mathématiques vérifiées

- **Modèles et mesures.** Le code des diffusions Black–Scholes, CEV, Heston QE-M, Heston 3/2, SABR, Schöbel–Zhu et Stein–Stein a été relié à sa préparation et à son schedule. Les compensateurs Merton/Kou/Bates, les lois de subordination VG/NIG et l'isolation des consommations variables Philox ont été relus avec les tests de premiers moments et de relecture. Les modèles ajustés de taux composent OU/CIR/G2 avec leur courbe ; CIR Bermudan emploie la mesure terminal-forward et les gaussiens la transition jointe facteur–intégrale sous Q. Cela vérifie le chemin de méthode, pas une dérivation indépendante de toutes les lois.
- **Produits et temps.** Les 26 descriptions du manifeste ont été croisées avec leur policy : produits terminaux, surveillance dense sur la grille, observation régulière, calendrier à deux dates, option américaine ancrée à maturité, swaption européenne à jambe fixe, et Bermudan co-terminal. Le moteur MC prépare le nombre de pas validé, le payoff et l'actualisation puis réduit les deux moments en FP64. Les prix rough FFT et N-facteurs sont les prix des **approximations déclarées**, pas une simulation exacte du modèle continu.
- **Exercice anticipé.** LSM simule les chemins, ajuste sur candidats en monnaie, résout en FP64, actualise et décide en remontant les dates ; les échecs fatals de régression invalident la sortie. `frozen_exercise_time` conserve l'indice central mais recalcule état/payoff/discount du nœud ; `frozen_regression_policy` conserve coefficients, statuts et normalisations centraux mais recalcule l'exercice sur chaque chemin bumpé. La décision américaine `t0` et le terminal ont des branches explicites. La dispersion finale des cashflows LSM ne couvre pas le fit (`NUM-036`).
- **Bumps.** La chaîne `SensitivityRequest → plan → paramètre modèle/courbe/produit → stencil représenté → scénarios → valeurs pathwise couplées → moments → reconstruction` a été lue sur les voies `mono`, `node_graph`, Hessienne mixte, LSM et rough. Les stencils utilisent les endpoints FP32 réellement représentés, basculent vers l'unilatéral d'ordre deux aux frontières et refusent la maturité non alignée sur la grille. Les corners mixtes appliquent les deux changements avant l'évaluation. Les tests analytiques Black–Scholes contrôlent six dérivées et le gamma ; les autres familles ont surtout des parités de voies, des tests de domaines et des prix bumpés.
- **RNG et stabilité.** Le compteur Philox sépare chemin 64 bits, groupe 32 bits, domaine source/pas ; les sources à consommation variable ont un pas borné à 24 bits. Les tests vérifient la relecture et l'isolation des sources. Les réductions MC utilisent des moments FP64 ; les solveurs et les conditions de non-finitude sont testés. Cela ne chiffre pas le biais de pas, de facteur, de politique gelée ou de queues extrêmes.

## Exécutions et limites

- `cmake --build /tmp/ai_factory_full_audit_20261006/build --target ai_factory_cuda_tests` : réussi. Suite CUDA complète `ctest --test-dir /tmp/ai_factory_full_audit_20261006/build -L cuda --output-on-failure -j 2` : **122/122 passent**, zéro échec, 219,98 s réelles.
- Sous-sélection reconstruite de 34 tests CUDA représentatifs, dont Jamshidian, Black–Scholes, LSM American/Bermudan, rough, produits de chemin, Philox et gradients : **34/34 passent**.
- `python3 -m unittest discover -s validation/volterra/tests -q` : **35/35 passent** ; `python3 -m unittest discover -s tests/price_gradients -p 'test_*.py' -q` : **24/24 passent**.
- `python3 -m unittest discover -s validation/tests -q` : **40 tests, 2 échecs et 10 erreurs**, notamment chemins de datasets absents de l'arborescence attendue. Le pipeline fail-closed ne fournit donc pas une validation indépendante du snapshot (`VALID-001`) ; ses anciens caches et les binaires du `build/` ne sont pas des références courantes.
- Les lignes rough de `NUM-028` à `NUM-030` n'ont pas été repricées avec plusieurs seeds sur ce binaire. Les prix LSM n'ont pas été comparés à une valorisation indépendante de la politique ni à une étude multi-bumps/multi-seeds complète. Ces exclusions empêchent un verdict « conforme » global pour l'axe I malgré les tests réussis.

**Évolution du registre :** `NUM-036` ouvert pendant ce passage ; `NUM-028` à `NUM-030` conservés. Aucun écart de formule ou de branchement prix/gradient supplémentaire n'a été démontré dans les représentants testés. Les sept constats des axes II/III restent ouverts sans modification de leur portée.

**Rétention :** le build isolé de 3,5 Gio a été supprimé pendant le passage hygiène du même jour. Le snapshot, les empreintes, la configuration et le journal CTest sont conservés sous `artifacts/audit/axis-truth-2026-10-06/`, avec un `receipt.json` hashé.
