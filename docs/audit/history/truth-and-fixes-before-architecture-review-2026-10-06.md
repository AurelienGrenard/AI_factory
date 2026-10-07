# Tableau de bord de l'audit principal

Référentiel actif : **v10.1 — 2026-10-02**.

## Corrections du 2026-10-06

Les trois constats du passage ciblé, `NUM-035`, `FACTOR-003` et
`BOUNDARY-008`, sont corrigés et décrits dans [closed.md](../closed.md).
Leur signature initiale n'est plus présente sur les chemins testés.
La couverture des axes I à III demeure **partielle** : cette intervention
ne certifie pas toutes les combinaisons modèle × produit × méthode × dérivée.

- **Véracité rough :** les observations antérieures gardent les pas centraux
  sous un bump de maturité ; le seul terme final suit le pas bumpé. Un
  oracle déterministe de callbacks et de payoff couvre les calendriers
  régulier, régulier à premier intervalle différent et statique, à 503/504/505
  pas. Les graphes FFT et lift exécutent la sensibilité de maturité du
  cliquet et du forward-start.
- **Factorisation :** la policy terminale partagée est extraite du header
  d'exécution lift ; le graphe FFT inclut directement cette policy.
- **Arborescence :** le commun fixed income délègue la ligne de maturité ;
  l'adapter swaption est sous `product/european_swaption/price_gradients/`.
  Le template codegen analytique et ses unités générées sont alignés.
- **Vérification :** sept tests CUDA ciblés passent dans
  `/tmp/ai_factory_audit_20261005/build12` :
  `rough_path_schedule_node_graph_cuda`,
  `rough_lift_path_node_graph_cuda`,
  `rough_gaussian_path_node_graph_cuda`,
  `rough_lift_node_graph_cuda`,
  `rough_gaussian_node_graph_cuda`,
  `rough_heston_digital_node_graph_cuda` et
  `price_gradients_cir_european_swaption_cuda`.
  Les cibles de sensibilités swaption Hull–White/Flat et G2 compilent ;
  `python3 maintainer/tools/cuda/check_model_layout.py` passe.

## Passage ciblé du 2026-10-05 — état avant corrections

- **Mandat :** véracité des prix et sensibilités, avec priorité aux bumps,
  calendriers rough, replays LSM et Jamshidian ; factorisation et emplacement
  des briques rencontrées sur ces chemins. Aucun passage complet des six axes.
- **État global :** non conforme sur les portions examinées, car
  `NUM-035` prouve que le bump de maturité rough déplace des observations
  contractuelles antérieures. Les autres familles gardent leur propre
  couverture et ne reçoivent pas ce verdict par extrapolation.
- **Résultat à cette date :** trois constats alors ouverts :
  `NUM-035`, `FACTOR-003`, `BOUNDARY-008`, corrigés le 2026-10-06 et
  transférés dans [closed.md](../closed.md). Les constats antérieurs
  `PRODUCT-001` et `NUM-028` à `NUM-030` restent ouverts sans
  requalification par ce passage.
- **Corrections du code :** aucune ; ce passage documente les écarts.

## Snapshot et moyens

- **Source observée :** branche `main`, `HEAD`
  `03f72d81045330cd8a234b83c877292379cd0623`.
- **Worktree au départ :** 176 fichiers suivis modifiés, trois supprimés et
  184 fichiers non suivis. La migration rough non commitée fait partie du
  périmètre observé. Snapshot reconstructible sous
  `/tmp/ai_factory_audit_20261005/` : `tracked.patch` (SHA-256
  `5615eac4fab266280da794e28959474f8b5966be03b72c3336286c7a1e9f56f4`)
  et `untracked.tar.gz` (SHA-256
  `1374813b2fcc8edb566fb1f9d192a750b8442af7d01440190706a8317563013b`).
  La capture précède les changements des deux registres d'audit.
- **Matériel et build :** RTX 4090 Laptop SM89 ; CMake isolé sous
  `/tmp/ai_factory_audit_20261005/build12`, `Release`, CUDA 12.9.41,
  g++ 14.3.0, cuFFTDx 25.12.1, profil
  `sm89_cuda12_unqualified_v1`. L'essai initial avec CUDA 13.3 a été rejeté
  à la configuration par le contrat CMake courant (`CUDA Toolkit 12.9.x`).
- **Expériences :** lecture des contrats et du code, calcul déterministe des
  dates rough ; `python3 maintainer/tools/cuda/check_model_layout.py` réussi
  (2 404 fichiers modèle-produit classés). Cinq tests CUDA recompilés puis
  réussis via `ctest --test-dir /tmp/ai_factory_audit_20261005/build12` :
  `price_gradients_cir_european_swaption_cuda`,
  `price_gradients_bermudan_swaption_cuda`,
  `price_gradients_exact_transition_american_cuda`,
  `rough_lift_node_graph_cuda`,
  `rough_gaussian_path_node_graph_cuda`.
- **Portée des tests :** CIR/Jamshidian et Bermudan passent leurs fixtures ;
  les deux tests rough couvrent le prix central et surtout les bumps spot.
  Aucun de ces cinq tests ne vérifie les dates internes sous bump de
  maturité rough. Le test américain contient un oracle CPU indépendant des
  dates de la politique gelée et une référence QuantLib du prix central ;
  le test bermudéen vérifie surtout les parités des voies de replay. Ces
  preuves ne certifient pas tous les modèles ni tous les cashflows gelés.

## Couverture et verdict du passage initial

| Axe | Couverture | Verdict | Preuve et limite |
|---|---|---|---|
| I. Véracité | partielle | non conforme | `NUM-035` sur T rough régulier/statique ; stencils, cache, replay LSM, Jamshidian et tests ciblés examinés ; pas d'inventaire exhaustif ni de qualification multi-seeds |
| II. Factorisation | partielle | non conforme | `FACTOR-003` sur la frontière policy terminale / exécution lift-FFT ; autres moteurs non parcourus exhaustivement |
| III. Arborescence et lisibilité | partielle | non conforme | `BOUNDARY-008` sur la spécialisation swaption sous `common` ; parcours de découverte complets non exécutés |
| IV. Performance | non examinée | indéterminé | pas de profil ou de comparaison de coûts |
| V. CMake | non examinée | indéterminé | configuration et compilation utilisées comme moyens de preuve ; pas d'audit du graphe |
| VI. Hygiène et artefacts | non examinée | indéterminé | snapshot pris ; pas d'inventaire des propriétaires et de la rétention |

## Registres et suite

[response.md](../response.md) contient quatre constats ouverts :
`PRODUCT-001` et `NUM-028` à `NUM-030`. [closed.md](../closed.md)
conserve les clôtures historiques et les trois corrections du 2026-10-06.

Une référence indépendante de chaque dérivée rough, avec mêmes innovations
et calendrier contractuel fixe, reste utile pour une qualification numérique
plus large. Les replays américains et bermudéens méritent aussi des oracles
de décisions et de cashflows sur davantage de familles que les fixtures
actuelles. Cette intervention ne certifie pas toutes les combinaisons
moteur × modèle × produit × dérivée.
