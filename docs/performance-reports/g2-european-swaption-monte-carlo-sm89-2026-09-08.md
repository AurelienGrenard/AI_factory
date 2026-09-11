# Swaptions européennes G2/G2++ — 8 septembre 2026

Le pricing de production utilise le Monte Carlo terminal commun : une
transition exacte de `(x_T, y_T, I_0T)` sous Q, des ZCB conditionnels pour
valoriser le swap, puis l'actualisation du payoff. Les compositions G2++
Nelson–Siegel et Svensson ajoutent le shift déterministe aux analytics.
Il n'y a ni quadrature de production, ni `dt`, ni simulation aux paiements.
Les bindings et les six recettes payer/receiver sont générés depuis le
manifeste ; les formules fermées existantes ne changent pas.

## Contrôles numériques et structurels

Le test `g2_european_swaption_cuda` couvre les trois compositions, les deux
côtés, quatre lignes core et quatre stress par composition/côté, dont une
variante à un coupon : 48 prix à `2^18` chemins.

- Replay bit à bit des prix et SE entre 128 et 512 threads, batch groupé et
  appels unitaires ; calendriers explicites équivalents contrôlés à 256 threads.
- Calendriers invalides rejetés côté host. La construction cartésienne est
  comparée à une expansion alignée indépendante pour chaque composition.
- Contrôles memcheck et racecheck à 4 096 chemins, sans erreur ni hazard.
- Tests des swaptions fermées un facteur inchangés ; codegen zéro diff,
  layout des sources et frontières des générateurs contrôlés.

Les [48 diagnostics indépendants](../../tests/performance/reports/g2-european-swaption-sm89-2026-09-08/diagnostics.json)
conservent chaque prix, SE, méthode de référence, budget et exception technique :

- 34 références `QuantLib.G2SwaptionEngine`, comparées à 128 et 256 points ;
  l'écart de quadrature doit rester inférieur à `1e-7 * max(1, abs(reference))`.
- 14 échecs techniques d'encadrement de racine sur des lignes stress : contrôle
  par Monte Carlo CPU indépendant sous la mesure forward de l'exercice,
  à `2^18` chemins, avec PCG64, FP64, moments SciPy et ZCB QuantLib.
  Seule une exception technique autorise ce second diagnostic ; un écart fini
  ou une quadrature non convergée reste un échec, sans changement de référence.
- Budget : `5e-7 + 5e-5*abs(reference) + 6*hypot(SE_GPU, SE_reference)`.
  Les 48 contrôles passent. Trois tests CPU vérifient aussi l'identité à un
  coupon, la parité et la référence MC contre le moteur natif.

Ce diagnostic ne remplace pas la hiérarchie de certification Premia/QuantLib
sur 900 core + 100 stress. L'inventaire Premia contient notamment
`CF_EuropeanSwaption_HW2D` et `TR_SWAPTIONHW2D` dans HullWhite2D/STDi : leur
compatibilité complète doit être qualifiée avant sélection de la référence
persistante. Aucune indisponibilité de Premia n'est déduite de ce test.
Un prix nul avec SE nulle ne prouve pas la précision relative d'un événement
rare. Les datasets générés restent `pending` et `verified: false`.

## Générateurs réels : 1 000 prix × 2²⁰ chemins

Un passage séquentiel par recette, warmup séparé du timer GPU, 256 threads par
bloc, un bloc par prix, batches de 256/256/256/232 prix. Les JSON/YAML sont
écrits ensemble et les JSON sont rechargés par le validateur de production.

| Modèle/courbe | Côté | GPU | Runner hôte | Processus complet |
|---|---|---:|---:|---:|
| G2 | Payer | 2,024 s | 2,586 s | 2,67 s |
| G2 | Receiver | 2,214 s | 2,675 s | 2,78 s |
| G2++ / Nelson–Siegel | Payer | 5,813 s | 6,123 s | 6,23 s |
| G2++ / Nelson–Siegel | Receiver | 5,931 s | 6,304 s | 6,41 s |
| G2++ / Svensson | Payer | 6,881 s | 7,296 s | 7,38 s |
| G2++ / Svensson | Receiver | 6,976 s | 7,322 s | 7,42 s |

Le runner hôte inclut allocation, warmup et copies ; il exclut chargement JSON
initial et publication. Le processus complet inclut ces opérations.
Les coupons sont valorisés avec les analytics canoniques, sans duplication
des équations dans le kernel. La mise en cache des coefficients de chaque
coupon n'a pas été optimisée dans cette qualification fonctionnelle.

### Ressources compilées SM89

| Composition | Registres régulier / explicite | Local par thread | Shared statique régulier / explicite |
|---|---:|---:|---:|
| G2 | 72 / 80 | 32 octets | 144 / 160 octets |
| G2++ / Nelson–Siegel | 109 / 110 | 32 octets | 144 / 160 octets |
| G2++ / Svensson | 111 / 116 | 32 octets | 160 / 176 octets |

La réduction ajoute 128 octets de shared dynamique à 256 threads. Les trois
géométries 128/256/512 sont exécutables ; 1 024 n'est pas admissible pour ces
spécialisations. Le champ local compilé ne suffit pas à déduire un nombre de
spills. La pression registre des compositions fitted reste à suivre dans la
campagne de tuning. Ce passage ne désigne pas une géométrie optimale.

## Reproduction et limites

RTX 4090 Laptop, SM89, CUDA/NVCC 13.3, GCC 14, Release, sans fast math.
HEAD `872a986b1f0947a1a832af0615ffc6d80dbedb81`, worktree préalablement modifié :
HEAD seul ne décrit donc pas les sources testées. Les résultats sont locaux,
sans certification multi-GPU ni extrapolation validée vers un million de prix.
La température reste une télémétrie, sans seuil d'arrêt applicatif.

```sh
cmake --build build-dev --target test_g2_european_swaption_cuda -j2
AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS=1 timeout 180 \
  build-dev/test_g2_european_swaption_cuda > build-dev/g2-rows.ndjson \
  2> build-dev/g2-resources.log
OPENBLAS_NUM_THREADS=1 python3 -m tools.performance.check_g2_swaption \
  build-dev/g2-rows.ndjson --output build-dev/g2-checked.json
OPENBLAS_NUM_THREADS=1 python3 -m unittest tests.performance.test_g2_swaption_reference
```

Les contrôles mémoire emploient le même exécutable avec `--smoke-test`, sous
`compute-sanitizer --tool memcheck` puis `--tool racecheck`, séquentiellement.
Les logs des recettes sont conservés sous
`build-dev/generate_<composition>_european_<side>_swaptions_01-20260908.log`,
avec ressources exactes et temps du processus. Les graines de catalogue
proviennent des six nouveaux domaines Philox v2 ; les 588 domaines v1 sont
inchangés. Aucune donnée n'est téléversée et aucune baseline n'est remplacée.
