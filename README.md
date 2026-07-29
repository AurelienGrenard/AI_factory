# CUDA Workbench

`cuda_workbench` est un projet CUDA autonome qui construit des bases de donnees
de pricing reproductibles. Ses cas de reference sont le call europeen et le put
americain discretise sous Heston, simules avec le schema QE-M d'Andersen.

Le projet separe volontairement trois responsabilites :

- `src` contient le code financier et numerique qui calcule les prix ;
- `tools` contient les fonctions reutilisables qui construisent et ecrivent les
  bases ;
- `registry` contient les JSON, les YAML et leurs generators reproductibles.

Il n'existe pas de dossier `examples`. Les generators du registry sont les
programmes executables de reference.

## Architecture Globale

```text

  CMakeLists.txt
  README.md
  src/
    common/
      check_cuda.cuh
      least_squares.cuh/.cu
      philox.cuh
      reductions.cuh
    heston/
      parameters.hpp/.cpp
      dynamics.cuh/.cu
      european_call.cuh/.cu
      american_put.cuh/.cu
    products/
      european_call.hpp/.cpp
      american_put.hpp/.cpp
  tools/
    registry/
      dataset.hpp/.cpp
  registry/
    production/
      model_parameters/<model>/{data,specifications,generators}/
      product_parameters/<product>/{data,specifications,generators}/
      prices/<model>/<product>/{data,specifications,generators}/
```

### `src` : simulation et pricing

`src` possede toute la logique financiere et numerique :

- les structures FP32 chargees depuis les JSON ;
- la dynamique Heston et le schema QE-M ;
- le generateur Philox ;
- les payoffs europeen et americain ;
- les kernels CUDA specialises ;
- la reduction Monte Carlo et l'erreur standard.

`src/common` contient uniquement les mecanismes partages. `src/heston`
appartient au modele Heston. `src/products` decrit les parametres contractuels
independants du modele. Le fichier `src/heston/european_call.cu` reunit le
modele et le produit dans un kernel specialise.

Dans chaque dossier modele, les responsabilites suivent la meme convention :

- `parameters.hpp/.cpp` definit les parametres CPU et leur chargement JSON ;
- `dynamics.cuh/.cu` contient la preparation et la simulation reutilisables par
  tous les produits du modele ;
- `<product>.cuh/.cu` contient le payoff, le kernel et le launcher specialises.

Le fichier CUDA de chaque produit inclut l'implementation `dynamics.cu`. NVCC
voit ainsi la dynamique et le kernel dans la meme unite de compilation et peut
inliner la boucle de simulation. `dynamics.cu` n'est donc pas compile comme une
bibliotheque CUDA separee.

Le code de pricing ne depend jamais du registry ni de ses outils d'ecriture.

Le call europeen fusionne simulation, payoff et reduction dans un bloc par
prix. Le put americain stocke les etats aux dates d'exercice et utilise plusieurs
blocs par prix pour la simulation et les regressions Longstaff-Schwartz.

Philox est contrefactuel : une key et un indice de groupe suffisent pour
retrouver directement quatre uniformes. Le mapping canonique utilise un stream
zero, precalcule une key par ligne, aligne chaque trajectoire sur un groupe de
quatre valeurs et consomme trois uniformes par pas. Il ne gaspille donc qu'entre
zero et trois uniformes en fin de trajectoire.

### `tools` : construction et serialisation

`tools/registry` fournit le code reutilisable pour :

- tirer des parametres uniformement ;
- construire des grilles alignees ou cartesiennes ;
- lire les metadonnees des bases sources ;
- ecrire les JSON de donnees ;
- ecrire les YAML de specification ;
- construire les references entre modele, produit et resultat.

L'API publique est regroupee dans `dataset.hpp`; `dataset.cpp` contient les
constructions et toute la serialisation JSON/YAML.

Ces outils ne contiennent aucune seconde implementation de Heston ou du
payoff. Ils preparent et serialisent les donnees ; `src` calcule les prix.

### `registry` : donnees reproductibles

Le registry contient trois familles de bases aux responsabilites volontairement
distinctes :

```text
model_parameters/    parametres purs des modeles, sans produit ni prix
product_parameters/  parametres contractuels purs, sans modele ni prix
prices/              prix produits par un modele et une methode numerique
```

Ainsi, ouvrir `model_parameters/heston` ne laisse pas penser que des produits
Heston y sont deja prices. Les deux premieres branches definissent les entrees ;
seule la branche `prices/<model>/<product>` les associe et contient les sorties
de pricing.

Chaque base possede trois fichiers de meme nom :

```text
data/<database_id>.json              lignes machine-readable
specifications/<database_id>.yaml   description et construction
generators/<database_id>.cpp        programme qui regenere la base
```

Le JSON contient les lignes. Le YAML explique leur signification et leur
provenance. Le generator reconstruit les deux fichiers.

## Exemple Heston / Call Europeen

### 1. Base de parametres modele

Le JSON modele contient 1 000 jeux de parametres Heston. Chaque ligne possede
un identifiant local et un objet `parameters` :

```json
{
  "database_id": "heston_01",
  "model_family": "Heston",
  "specification": "registry/production/model_parameters/heston/specifications/heston_01.yaml",
  "generation_script": "registry/production/model_parameters/heston/generators/heston_01.cpp",
  "row_count": 1000,
  "models": [
    {
      "id": "000001",
      "parameters": {
        "spot": 1.0,
        "risk_free_rate": 0.02982048,
        "dividend_yield": 0.04603526,
        "initial_variance": 0.07711430,
        "kappa": 1.59042335,
        "theta": 0.14153057,
        "rho": -0.46044695,
        "gamma": 0.76693642
      }
    }
  ]
}
```

Le YAML documente les champs, la dynamique et la construction sans recopier
les 1 000 lignes :

```yaml
title: "Heston parameter database heston_01"
database_id: "heston_01"
model_family: "Heston"
json_path: "registry/production/model_parameters/heston/data/heston_01.json"
generation_script: "registry/production/model_parameters/heston/generators/heston_01.cpp"

parameters:
  spot: "Initial spot."
  risk_free_rate: "Continuously compounded risk-free rate."
  dividend_yield: "Continuously compounded dividend yield."
  initial_variance: "Initial variance v0."
  kappa: "Variance mean-reversion speed."
  theta: "Long-run variance."
  gamma: "Volatility of variance."
  rho: "Spot/variance Brownian correlation."

dynamics:
  spot: "dS_t / S_t = (r - q) dt + sqrt(V_t) dW_t^S"
  variance: "dV_t = kappa (theta - V_t) dt + gamma sqrt(V_t) dW_t^V"
  correlation: "d<W^S, W^V>_t = rho dt"

construction:
  row_count: 1000
  method: "conditional uniform sample"
  conditional_bounds:
    gamma:
      minimum: "max(sqrt(kappa * theta / 5), 0.1)"
      maximum: "min(sqrt(12 * kappa * theta), 0.8)"
```

`kappa`, `theta` et les autres parametres sont d'abord tires dans leurs bornes.
`gamma` est ensuite tire conditionnellement afin de controler le ratio de
Feller `2 * kappa * theta / gamma^2`.

### 2. Base de parametres produit

Le produit est stocke independamment du modele :

```json
{
  "database_id": "european_calls_01",
  "product_family": "European Calls",
  "specification": "registry/production/product_parameters/european_calls/specifications/european_calls_01.yaml",
  "generation_script": "registry/production/product_parameters/european_calls/generators/european_calls_01.cpp",
  "row_count": 1000,
  "products": [
    {
      "id": "000001",
      "parameters": {
        "strike": 0.98347145,
        "maturity": 0.08333334
      }
    }
  ]
}
```

Le YAML precise le payoff et la grille utilisee :

```yaml
title: "European Calls parameter database european_calls_01"
database_id: "european_calls_01"
product_family: "European Calls"
json_path: "registry/production/product_parameters/european_calls/data/european_calls_01.json"
generation_script: "registry/production/product_parameters/european_calls/generators/european_calls_01.cpp"

parameters:
  strike: "Strike in normalized spot units."
  maturity: "Maturity in years."

payoff:
  expression: "max(S_T - K, 0)"

construction:
  row_count: 1000
  method: "maturity-dependent exponential grid"
  rule: "For each T, x is linearly spaced on [-aT, aT] and K = exp(x)."
  grid:
    maturity: {minimum: "1 / 12", maximum: 3.0, count: 50, spacing: "linear"}
    strike:
      count_per_maturity: 20
      spacing: "linear in log-strike"
      conditional_bounds: "[exp(-aT), exp(aT)]"
      a: 0.2
```

Pour chacune des 50 maturites, 20 strikes couvrent une plage de
log-moneyness qui s'elargit avec `T`. La base contient donc 1 000 lignes.

### 3. Base de prix

Le YAML de prix indique quelles bases sont pricees et comment :

```yaml
title: "heston_01 x european_calls_01 cpp_gpu_philox"
database_id: heston_01__european_calls_01__01
json_path: "registry/production/prices/heston/european_calls/data/heston_01__european_calls_01__01.json"
generation_script: "registry/production/prices/heston/european_calls/generators/heston_01__european_calls_01__01.cpp"

summary:
  row_count: 1000
  monte_carlo_paths_per_price: 16384
  model: "Heston"
  numerical_method: "Andersen QE-M"
  payoff: "European Calls"
  implementation: CUDA
  device: gpu
  block_count: 1000
  threads_per_block: 512
  kernel_launch_count: 1
  maximum_prices_per_block: 1
  random_generator: "Philox"
  source_files:
    - "src/common/philox.cuh"
    - "src/heston/dynamics.cu"
    - "src/heston/european_call.cu"

time_grid:
  rule: nearest integer step count to target dt
  target_dt: "1 / 252"
  step_count: round(maturity / target_dt)
  effective_dt: maturity / step_count

outputs:
  price:
    estimator: Monte Carlo discounted payoff mean
  standard_error:
    estimator: Monte Carlo standard error of discounted payoff

model_database:
  id: heston_01
  json_path: "registry/production/model_parameters/heston/data/heston_01.json"

product_database:
  id: european_calls_01
  json_path: "registry/production/product_parameters/european_calls/data/european_calls_01.json"

result_construction:
  rule: "aligned row pairing"

timing:
  wall_seconds: "2.204893417 s"
  kernel_seconds: 0.295830536
```

Le JSON de prix reference les bases et les lignes utilisees. Il ne recopie pas
tous les parametres modele et produit :

```json
{
  "database_id": "heston_01__european_calls_01__01",
  "specification": "registry/production/prices/heston/european_calls/specifications/heston_01__european_calls_01__01.yaml",
  "generation_script": "registry/production/prices/heston/european_calls/generators/heston_01__european_calls_01__01.cpp",
  "row_count": 1000,
  "model_database": {
    "id": "heston_01",
    "json_path": "registry/production/model_parameters/heston/data/heston_01.json"
  },
  "product_database": {
    "id": "european_calls_01",
    "json_path": "registry/production/product_parameters/european_calls/data/european_calls_01.json"
  },
  "timing": {
    "wall_seconds": 2.204893417,
    "kernel_seconds": 0.295830536
  },
  "results": [
    {
      "id": "000001",
      "model_id": "000001",
      "product_id": "000001",
      "seed": 900000001,
      "outputs": {
        "price": 0.29776946,
        "standard_error": 0.00063856
      }
    }
  ]
}
```

Le JSON conserve les timings comme nombres en secondes. Le YAML affiche
`wall_seconds` avec son unite et ajoute les minutes au-dessus de 60 secondes,
puis les heures au-dessus de 3 600 secondes.

La ligne de prix `000001` signifie donc :

```text
modele  = ligne 000001 de heston_01
produit = ligne 000001 de european_calls_01
sorties = prix et erreur standard de ce couple
```

Cette representation normalisee evite la duplication, reduit la taille des
resultats et garantit que les parametres ont une source unique. Une exportation
denormalisee peut naturellement recopier les parametres modele et produit si
un fichier autonome ou tabulaire est necessaire.

Un identifiant de prix suit exclusivement la convention
`<model_database_id>__<product_database_id>__<price_database_id>`.
Deux bases coexistent ici :

- le generator historique construit 1 000 couples alignes avec 16 384 chemins ;
- le generator de prix `02` construit les 1 000 000 couples avec 131 072
  chemins par prix.

## Constructions Des Prix

Deux constructions sont actuellement disponibles :

- `Aligned` exige le meme nombre de modeles et de produits et associe `(i, i)` ;
- `CartesianProduct` price tous les couples sans materialiser un tableau de
  parametres duplique.

Pour un produit cartesien :

```text
model_index   = result_index / product_count
product_index = result_index % product_count
```

Le JSON de prix conserve dans les deux cas les `model_id` et `product_id`
effectivement utilises.

## Pipeline Executable

Les generators sont executes dans cet ordre :

```text
generate_heston_01
  -> heston_01.json + heston_01.yaml

generate_european_calls_01
  -> european_calls_01.json + european_calls_01.yaml

generate_heston_european_calls_01
  -> JSON + YAML de 1 000 prix alignes

generate_heston_european_calls_02
  -> JSON + YAML de 1 000 000 de prix cartesiens
```

Le generator resultat :

1. charge directement les JSON dans deux `std::vector` FP32 contigus ;
2. valide la construction `Aligned` ou `CartesianProduct` ;
3. alloue explicitement les tableaux GPU ;
4. copie les modeles et produits sur le GPU ;
5. lance le kernel Heston/call europeen ;
6. recupere prix et erreurs standards ;
7. ecrit automatiquement le JSON et le YAML resultat.

## Contrat CUDA

Le kernel europeen suit la convention logique :

```text
une ligne resultat -> un bloc worker -> un prix et une erreur standard
```

Le nombre de blocs est borne par la recette. Un grand calcul est decoupe en
lancements courts lorsque le watchdog graphique est actif, sans copie
intermediaire CPU/GPU. Philox recalcule directement chaque couple
`(ligne, trajectoire)` et ne stocke aucun etat RNG global.

Chaque thread traite plusieurs trajectoires. Simulation et payoff sont fusionnes
et restent dans le thread. Les etats Heston sont en FP32 ; les sommes et sommes
de carres sont accumulees en FP64 ; le prix et l'erreur standard sont finalement
stockes en FP32. `reduce_block` termine la reduction dans le bloc sans atomique
globale ni second kernel.

Les deux recettes utilisent 512 threads par bloc. La petite base utilise
16 384 chemins et un lancement ; la grande utilise 131 072 chemins et 1 954
lancements de 512 prix. Un warmup de 64 lignes charge le module et stabilise le
GPU hors du timer kernel.

Le workbench compile par defaut avec les operations FP32 precises et sans
`launch_bounds` propre a un GPU. `-DCUDA_WORKBENCH_FAST_MATH=ON` reste disponible
pour une experience locale, mais ne constitue ni le mode public ni la reference
de reproductibilite. Son activation doit toujours etre suivie d'une comparaison
numerique et d'un nouveau profilage sur le GPU cible.

### Put americain multibloc

L'exercice anticipe exige de conserver les spots et variances aux dates
d'exercice precedant la maturite. Le spot terminal est retourne directement
au payoff. Le put americain utilise donc une topologie differente :

```text
grid.x = blocs de trajectoires par prix
grid.y = prix traites simultanement
```

Chaque thread parcourt les trajectoires avec un pas
`gridDim.x * blockDim.x`. Les tableaux d'etats sont ranges par date puis par
trajectoire : les 32 lanes d'un warp accedent ainsi a 32 valeurs consecutives.

Le launcher interroge la memoire GPU libre, reserve une marge de securite, puis
construit des lots consecutifs avec le nombre exact de dates de chaque produit.
Un workspace contigu est alloue une seule fois et reutilise pour tous les lots.
Si une seule ligne ne tient pas, le lancement echoue avant le premier kernel
avec la memoire requise et invite a reduire le nombre de trajectoires.

La sequence Longstaff-Schwartz est entierement executee sur le GPU :

```text
simulation des etats et payoff terminal
sommes partielles de G et b sur plusieurs blocs
reduction deterministe et Cholesky par prix
mise a jour backward des cashflows actualises
reduction finale du prix et de l'erreur standard
```

Les etats et cashflows sont en FP32. Les matrices normales, reductions et
moments sont en FP64. Aucune matrice de base par trajectoire n'est materialisee :
chaque thread calcule ses six fonctions de base et accumule directement `G` et
`b`. Un warmup reduit precede le wall timer et les evenements CUDA ne mesurent
que les kernels des lots de production.

La base materialisee actuelle demande `256` threads et au plus `2 048` blocs
par prix. Un benchmark stratifie de 50 lignes permet de reprofiler cette
topologie sans regenerer la base. Sur la RTX 4090 Laptop de reference,
`256` threads et `152` blocs reduisent nettement le temps kernel; la recette de
production ne doit etre modifiee qu'avec une regeneration complete du resultat.
Le launcher borne toujours le nombre demande aux blocs de trajectoires utiles.

## Compilation Et Generation

Dependances : un compilateur C++17, CUDA et `nlohmann-json3-dev`.

Depuis la racine du depot :

```bash
cmake -S cuda_workbench -B /tmp/ai_factory_cuda_workbench \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_WORKBENCH_ARCHITECTURES=89 \
  -DCUDA_WORKBENCH_FAST_MATH=OFF
cmake --build /tmp/ai_factory_cuda_workbench -j
ctest --test-dir /tmp/ai_factory_cuda_workbench --output-on-failure

/tmp/ai_factory_generate_heston_01
/tmp/ai_factory_generate_european_calls_01
/tmp/ai_factory_generate_heston_european_calls_01
/tmp/ai_factory_generate_heston_european_calls_02
/tmp/ai_factory_generate_heston_american_puts_01
```

Le benchmark American put est volontairement exclu de la build normale :

```bash
cmake --build /tmp/ai_factory_cuda_workbench \
  --target heston_american_put_benchmark -j
/tmp/ai_factory_heston_american_put_benchmark \
  256 152 /tmp/heston_american_put_256_152.csv
```

Il price 50 lignes espacees regulierement dans les 1 000 lignes sources et
ecrit tous les prix et erreurs standards pour une comparaison ligne par ligne.

La build precise utilise naturellement 59 registres par thread pour le kernel
Philox, sans spill. Les campagnes de profilage ont retenu 512 threads par bloc
sur la RTX 4090 Laptop de developpement. Cette valeur est un reglage initial
portable, pas une constante materielle : elle doit etre reprofilee sur un autre
GPU ou apres une modification significative de la dynamique ou du compilateur.

La valeur `89` produit ici du code natif pour la RTX 4090 Laptop. La liste des
architectures reste configurable pour un binaire destine a d'autres GPU.

## Ajouter Un Nouveau Couple

1. definir la dynamique, le payoff et la construction des parametres ;
2. ajouter le type FP32 et le loader du modele dans `src/<model>` ;
3. ajouter le type FP32 et le loader produit dans `src/products` ;
4. ecrire le kernel specialise dans `src/<model>/<product>.cu` ;
5. ajouter les generators modele et produit au registry ;
6. ajouter le generator de prix et ses references de bases ;
7. enregistrer les executables dans CMake ;
8. regenerer JSON et YAML depuis un build propre ;
9. verifier IDs, nombres de lignes, valeurs finies et timings ;
10. profiler threads par bloc, registres, spills et temps kernel.

Pour un produit europeen, partir du call Heston et remplacer la dynamique ou
`evaluate_path`. Pour un produit a exercice anticipe, partir du put americain et
adapter la grille, les etats de regression et la politique d'exercice.
