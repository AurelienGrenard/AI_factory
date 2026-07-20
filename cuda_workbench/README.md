# CUDA Workbench

`cuda_workbench` est un projet CUDA autonome qui construit des bases de donnees
de pricing reproductibles. Son cas de reference est le call europeen sous
Heston, simule avec le schema QE-M d'Andersen.

Le projet separe volontairement trois responsabilites :

- `src` contient le code financier et numerique qui calcule les prix ;
- `tools` contient les fonctions reutilisables qui construisent et ecrivent les
  bases ;
- `registry` contient les JSON, les YAML et leurs generators reproductibles.

Il n'existe pas de dossier `examples`. Les generators du registry sont les
programmes executables de reference.

## Architecture Globale

```text
cuda_workbench/
  CMakeLists.txt
  README.md
  src/
    common/
      check_cuda.cuh
      philox.cuh
      reductions.cuh
    heston/
      parameters.hpp/.cpp
      dynamics.cuh/.cu
      european_call.cuh/.cu
    products/
      european_call.hpp/.cpp
  tools/
    registry/
      io.hpp/.cpp
      parameters.hpp/.cpp
      results.hpp/.cpp
  registry/
    production/
      models/<model>/{data,specifications,generators}/
      products/<product>/{data,specifications,generators}/
      results/<model>/<product>/{data,specifications,generators}/
```

### `src` : simulation et pricing

`src` possede toute la logique financiere et numerique :

- les structures FP32 chargees depuis les JSON ;
- la dynamique Heston et le schema QE-M ;
- le generateur Philox ;
- le payoff du call europeen ;
- le kernel CUDA specialise ;
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

### `tools` : construction et serialisation

`tools/registry` fournit le code reutilisable pour :

- tirer des parametres uniformement ;
- construire des grilles alignees ou cartesiennes ;
- lire les metadonnees des bases sources ;
- ecrire les JSON de donnees ;
- ecrire les YAML de specification ;
- construire les references entre modele, produit et resultat.

Ces outils ne contiennent aucune seconde implementation de Heston ou du
payoff. Ils preparent et serialisent les donnees ; `src` calcule les prix.

### `registry` : donnees reproductibles

Le registry contient trois familles de bases :

```text
models/    parametres des modeles
products/  parametres contractuels des produits
results/   prix produits par un modele et une methode numerique
```

Chaque base possede trois fichiers de meme nom :

```text
data/<database_id>.json              lignes machine-readable
specifications/<database_id>.yaml   description et construction
generators/<database_id>.cpp        programme qui regenere la base
```

Le JSON contient les lignes. Le YAML explique leur signification et leur
provenance. Le generator reconstruit les deux fichiers.

## Exemple Heston / Call Europeen

### 1. Base modele

Le JSON modele contient 1 000 jeux de parametres Heston. Chaque ligne possede
un identifiant local et un objet `parameters` :

```json
{
  "database_id": "heston_01",
  "model_family": "Heston",
  "specification": "cuda_workbench/registry/production/models/heston/specifications/heston_01.yaml",
  "generation_script": "cuda_workbench/registry/production/models/heston/generators/heston_01.cpp",
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
json_path: "cuda_workbench/registry/production/models/heston/data/heston_01.json"
generation_script: "cuda_workbench/registry/production/models/heston/generators/heston_01.cpp"

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

### 2. Base produit

Le produit est stocke independamment du modele :

```json
{
  "database_id": "european_calls_01",
  "product_family": "European Calls",
  "specification": "cuda_workbench/registry/production/products/european_calls/specifications/european_calls_01.yaml",
  "generation_script": "cuda_workbench/registry/production/products/european_calls/generators/european_calls_01.cpp",
  "row_count": 1000,
  "products": [
    {
      "id": "000001",
      "parameters": {
        "strike": 0.7,
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
json_path: "cuda_workbench/registry/production/products/european_calls/data/european_calls_01.json"
generation_script: "cuda_workbench/registry/production/products/european_calls/generators/european_calls_01.cpp"

parameters:
  strike: "Strike in normalized spot units."
  maturity: "Maturity in years."

payoff:
  expression: "max(S_T - K, 0)"

construction:
  row_count: 1000
  method: "Cartesian grid"
  grid:
    strike: {minimum: 0.7, maximum: 1.3, count: 20, spacing: "linear"}
    maturity: {minimum: 0.0833333, maximum: 3.0, count: 50, spacing: "linear"}
```

Les 20 strikes et 50 maturites forment ici 1 000 lignes produit.

### 3. Base resultat

Le YAML resultat indique quelles bases sont pricees et comment :

```yaml
title: "heston_01 x european_calls_01 cpp_gpu_philox"
database_id: heston_01__european_calls_01__cpp_gpu_philox_01
json_path: "cuda_workbench/registry/production/results/heston/european_calls/data/heston_01__european_calls_01__cpp_gpu_philox_01.json"
generation_script: "cuda_workbench/registry/production/results/heston/european_calls/generators/heston_01__european_calls_01__cpp_gpu_philox_01.cpp"

summary:
  row_count: 1000
  monte_carlo_paths_per_price: 16384
  model: "Heston"
  numerical_method: "Andersen QE-M"
  payoff: "European Calls"
  implementation: CUDA
  device: gpu
  threads_per_block: 512
  source_files:
    - "cuda_workbench/src/heston/european_call.cu"

time_grid:
  rule: nearest integer step count to target dt
  target_dt: 0.003968254197
  step_count: round(maturity / target_dt)
  effective_dt: maturity / step_count

outputs:
  price:
    estimator: Monte Carlo discounted payoff mean
  standard_error:
    estimator: Monte Carlo standard error of discounted payoff

model_database:
  id: heston_01
  json_path: "cuda_workbench/registry/production/models/heston/data/heston_01.json"

product_database:
  id: european_calls_01
  json_path: "cuda_workbench/registry/production/products/european_calls/data/european_calls_01.json"

result_construction:
  rule: "aligned row pairing"

timing:
  wall_seconds: 2.204893417
  kernel_seconds: 0.295830536
```

Le JSON resultat reference les bases et les lignes utilisees. Il ne recopie pas
tous les parametres modele et produit :

```json
{
  "database_id": "heston_01__european_calls_01__cpp_gpu_philox_01",
  "specification": "cuda_workbench/registry/production/results/heston/european_calls/specifications/heston_01__european_calls_01__cpp_gpu_philox_01.yaml",
  "generation_script": "cuda_workbench/registry/production/results/heston/european_calls/generators/heston_01__european_calls_01__cpp_gpu_philox_01.cpp",
  "row_count": 1000,
  "model_database": {
    "id": "heston_01",
    "json_path": "cuda_workbench/registry/production/models/heston/data/heston_01.json"
  },
  "product_database": {
    "id": "european_calls_01",
    "json_path": "cuda_workbench/registry/production/products/european_calls/data/european_calls_01.json"
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

La ligne resultat `000001` signifie donc :

```text
modele  = ligne 000001 de heston_01
produit = ligne 000001 de european_calls_01
sorties = prix et erreur standard de ce couple
```

Cette representation normalisee evite la duplication, reduit la taille des
resultats et garantit que les parametres ont une source unique. Une exportation
denormalisee peut naturellement recopier les parametres modele et produit si
un fichier autonome ou tabulaire est necessaire.

## Constructions De Resultat

Deux constructions sont actuellement disponibles :

- `Aligned` exige le meme nombre de modeles et de produits et associe `(i, i)` ;
- `CartesianProduct` price tous les couples sans materialiser un tableau de
  parametres duplique.

Pour un produit cartesien :

```text
model_index   = result_index / product_count
product_index = result_index % product_count
```

Le JSON resultat conserve dans les deux cas les `model_id` et `product_id`
effectivement utilises.

## Pipeline Executable

Les trois generators sont executes dans cet ordre :

```text
generate_heston_01
  -> heston_01.json + heston_01.yaml

generate_european_calls_01
  -> european_calls_01.json + european_calls_01.yaml

generate_heston_european_calls_01
  -> result JSON + result YAML
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

Le kernel europeen suit la convention :

```text
une ligne resultat -> un bloc CUDA -> un prix et une erreur standard
```

Chaque thread traite plusieurs trajectoires. Simulation et payoff sont fusionnes
et restent dans le thread. Les etats Heston sont en FP32 ; les sommes et sommes
de carres sont accumulees en FP64 ; le prix et l'erreur standard sont finalement
stockes en FP32. `reduce_block` termine la reduction dans le bloc sans atomique
globale ni second kernel.

Le cas actuel utilise 512 threads par bloc et 16 384 chemins Monte Carlo par
prix. Toute nouvelle configuration doit etre profilee sur le volume cible.

## Compilation Et Generation

Dependances : un compilateur C++17, CUDA et `nlohmann-json3-dev`.

Depuis la racine du depot :

```bash
cmake -S cuda_workbench -B /tmp/ai_factory_cuda_workbench \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_WORKBENCH_ARCHITECTURES=89
cmake --build /tmp/ai_factory_cuda_workbench -j

/tmp/ai_factory_cuda_workbench/generate_heston_01
/tmp/ai_factory_cuda_workbench/generate_european_calls_01
/tmp/ai_factory_cuda_workbench/generate_heston_european_calls_01
```

La valeur `89` produit ici du code natif pour la RTX 4090 Laptop. La liste des
architectures reste configurable pour un binaire destine a d'autres GPU.

## Ajouter Un Nouveau Couple

1. definir la dynamique, le payoff et la construction des parametres ;
2. ajouter le type FP32 et le loader du modele dans `src/<model>` ;
3. ajouter le type FP32 et le loader produit dans `src/products` ;
4. ecrire le kernel specialise dans `src/<model>/<product>.cu` ;
5. ajouter les generators modele et produit au registry ;
6. ajouter le generator resultat et ses references de bases ;
7. enregistrer les executables dans CMake ;
8. regenerer JSON et YAML depuis un build propre ;
9. verifier IDs, nombres de lignes, valeurs finies et timings ;
10. profiler threads par bloc, registres, spills et temps kernel.

Les options americaines et bermudeennes demandent une architecture de
regression distincte. Elles ne sont pas couvertes par ce contrat europeen.
