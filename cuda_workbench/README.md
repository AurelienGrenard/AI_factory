# CUDA Workbench

`cuda_workbench` est un projet CUDA autonome et pédagogique. Son code doit
rester assez direct pour qu'un lecteur puisse suivre toute la chaîne, depuis
les paramètres financiers jusqu'au JSON de prix, sans dépendre de l'ancien
`src_cpp` ni de la documentation générale du dépôt.

L'exemple de référence price un call européen sous Heston avec le schéma QE-M
d'Andersen :

```text
dS_t / S_t = (r - q) dt + sqrt(V_t) dW_t^S
dV_t       = kappa (theta - V_t) dt + gamma sqrt(V_t) dW_t^V
d<W^S,W^V>_t = rho dt
```

`gamma` désigne partout la volatility of variance.

## Structure

```text
cuda_workbench/
  CMakeLists.txt
  README.md
  src/
    common/
      check_cuda.cuh       erreurs et validations CUDA partagées
      philox.cuh           générateur aléatoire counter-based
      reductions.cuh       réduction de bloc et statistiques Monte Carlo
    heston/
      common.hpp/.cpp      type Heston FP32 et loader JSON
      dynamics.cuh         dynamique Heston QE-M réutilisable
      european_call.hpp    déclaration du launcher spécialisé
      european_call.cu     simulation, payoff, réduction et lancement
    products/
      european_call.hpp/.cpp
                           type produit FP32 et loader JSON
  tools/registry/
    common.hpp             lecture JSON, écriture texte/YAML, constructions
    parameter_database.*  génération uniforme, aligned et Cartesian grid
    result_output.*        écriture des résultats JSON/YAML
  registry/production/
    models/<model>/        générateur, JSON et YAML du modèle
    products/<product>/    générateur, JSON et YAML du produit
    results/<model>/<product>/
                           générateur CUDA, JSON et YAML des prix
```

Les responsabilités sont strictes :

- `src/common` contient uniquement du code CUDA partagé par plusieurs couples ;
- `src/<model>` possède les paramètres, le loader et la dynamique du modèle ;
- `src/products` possède les paramètres et loaders indépendants du modèle ;
- le `.cu` d'un couple modèle-produit contient son kernel spécialisé ;
- `tools/registry` construit et sérialise les bases, sans pricing financier ;
- les generators constituent les programmes exécutables complets.

Il n'existe pas de fichier d'exemple parallèle aux generators. Le generator de
résultat est l'exemple exécutable canonique, ce qui évite deux pipelines qui
finissent par diverger.

## Générer Les Paramètres

Un generator modèle ou produit définit les bornes, la construction et les
métadonnées, puis écrit simultanément le JSON et le YAML.

### Uniforme

```cpp
GeneratedRows rows = uniform_rows(1'000U, seed, {
    {"risk_free_rate", 0.0f, 0.08f},
    {"rho", -1.0f, -0.30f},
});
```

Chaque paramètre est tiré indépendamment. Les paramètres conditionnels sont
ajoutés ensuite. Dans Heston, `kappa` et `theta` sont d'abord tirés, puis :

```text
gamma_min = max(sqrt(kappa * theta / 5), 0.1)
gamma_max = min(sqrt(12 * kappa * theta), 0.8)
gamma ~ Uniform(gamma_min, gamma_max)
```

Cela contrôle ligne par ligne le ratio de Feller
`2 * kappa * theta / gamma^2` dans `[1/6, 10]`. La condition de Feller peut donc
être violée volontairement, mais seulement dans cette plage contrôlée.

### Grille aligned

```cpp
GeneratedRows rows = aligned_grid({
    {"strike", {0.8f, 1.0f, 1.2f}},
    {"maturity", {0.5f, 1.0f, 2.0f}},
});
```

Les valeurs de même indice forment une ligne. Toutes les listes doivent avoir
la même taille.

### Produit cartésien

```cpp
GeneratedRows rows = cartesian_grid({
    {"strike", linear_grid(0.70f, 1.30f, 20U)},
    {"maturity", linear_grid(1.0f / 12.0f, 3.0f, 50U)},
});
```

Les 20 strikes et 50 maturités produisent 1 000 lignes. Le YAML résume chaque
axe par `minimum`, `maximum`, `count` et `spacing: linear`; le JSON contient les
valeurs exactes.

## Charger Les JSON

Chaque famille possède un loader typé :

```cpp
const std::vector<heston::HestonModelInput> models =
    heston::load_heston(model_json_path);
const std::vector<products::EuropeanCallInput> products =
    products::load_european_calls(product_json_path);
```

Le parser JSON est temporaire. Les loaders retournent directement deux
`std::vector` contigus de structures FP32, prêts pour `cudaMemcpy`. Il n'existe
pas de table générique intermédiaire ni de seconde conversion.

Deux constructions de résultat sont disponibles :

- `Aligned` exige le même nombre de modèles et produits et price `(i, i)` ;
- `CartesianProduct` expose tous les couples sans dupliquer leurs structures.

Dans le second cas :

```text
model_index   = result_index / product_count
product_index = result_index % product_count
```

## Generator De Résultat

Le fichier

```text
registry/production/results/heston/european_calls/generators/
  heston_01__european_calls_01__cpp_gpu_philox_01.cpp
```

contient en tête toutes les décisions : chemins JSON, construction, nombre de
chemins Monte Carlo par prix, `target_dt`, threads par bloc, seed, méthode
numérique et chemins de sortie.

Son `main` suit toujours cet ordre :

1. charger les vectors modèle et produit ;
2. calculer le nombre de résultats et créer les sorties CPU ;
3. allouer explicitement les quatre tableaux GPU avec `cudaMalloc` ;
4. copier modèles et produits avec `cudaMemcpyHostToDevice` ;
5. créer les événements et lancer le kernel spécialisé ;
6. copier prix et erreurs standards vers le CPU ;
7. détruire événements et allocations ;
8. écrire le JSON et le YAML avec `write_monte_carlo_result`.

Les allocations restent visibles dans le generator. Il n'y a ni workspace
caché ni réutilisation implicite de buffers dans ce projet pédagogique.

## Architecture Du Kernel

La convention pour un pricing Monte Carlo européen est :

```text
une ligne de résultat -> un bloc CUDA -> un prix et une erreur standard
```

Dans le bloc :

```text
result_index = blockIdx.x
path = threadIdx.x, threadIdx.x + blockDim.x, ...
```

Le thread simule chaque trajectoire qui lui est attribuée et calcule son payoff
immédiatement. La dynamique, le payoff et l'accumulation sont fusionnés : aucun
tableau de trajectoires ou de payoffs n'est écrit en mémoire globale.

Les constantes communes à la ligne sont préparées une seule fois par le thread
0 dans `PreparedRow`, placé en shared memory. Cela inclut `log(S0)`, calculé une
fois au lieu d'une fois par trajectoire. Chaque état `(log_spot, variance)` reste
ensuite privé au thread, normalement dans ses registres.

Après sa boucle, chaque thread possède `sum` et `sumsq`. `reduce_block` :

1. réduit ces deux valeurs dans chaque warp avec `__shfl_down_sync` ;
2. écrit une paire FP64 par warp en shared memory ;
3. fait réduire ces paires par le premier warp ;
4. laisse le thread 0 calculer prix et erreur standard.

Il n'y a ni atomique globale, ni tableau de moments partiels, ni second kernel
de finalisation.

## Taille Des Blocs

Le generator Heston utilise actuellement `512` threads par bloc. C'est un
multiple de warp et une puissance de deux facile à expliquer et à réutiliser.
Sur la RTX 4090 Laptop testée, il est nettement meilleur que 128 threads pour
1 000 lignes et 16 384 trajectoires par prix.

Cette valeur n'est pas une constante universelle. Le nombre de registres par
thread, le nombre de pas, le coût du payoff et le nombre de lignes modifient
l'occupation et le compromis entre parallélisme intra-ligne et nombre de blocs
résidents. Pour tout nouveau kernel, comparer au minimum `128`, `256` et `512`
sur le volume cible, puis conserver une puissance de deux sauf gain mesuré et
justifié d'une autre taille.

## Précision

La frontière numérique est explicite :

```text
JSON, modèle et produit              float (FP32)
coefficients et états de trajectoire float (FP32)
transformations Philox               float (FP32)
payoff                               float (FP32)
sum et sumsq                         double (FP64)
statistiques intermédiaires          double (FP64)
prix et erreur standard stockés      float (FP32)
```

Les calculs répétés d'une trajectoire bénéficient du débit FP32. Les sommes
Monte Carlo utilisent FP64 car additionner des milliers de valeurs en FP32
peut perdre les petites contributions et dégrader fortement la variance. Les
sorties reviennent en FP32 après la réduction.

`--use_fast_math` n'est pas activé. Les fonctions `expf`, `logf` et `sqrtf`
rendent la précision choisie visible. Toute activation future doit être
benchmarkée et validée financièrement.

## QE-M Et Fallback

Le schéma QE choisit une loi quadratique gaussienne lorsque `psi <= 1.5`, puis
une masse en zéro avec queue exponentielle sinon. QE-M ajoute une correction
de martingale au log-spot.

Cette correction requiert l'existence d'un moment exponentiel :

- branche quadratique : `1 - 2 * A * a > 0` ;
- branche exponentielle : `A < beta` et moment strictement positif.

Si cette condition échoue, le code emploie le pas QE sans correction de
martingale pour éviter `NaN` ou infini. Ce fallback est défensif, mais des
activations fréquentes peuvent introduire un biais de martingale. Un nouveau
domaine de paramètres doit donc tester sa fréquence et la convergence des prix.

## RNG

Philox est counter-based : un tirage dépend de la seed, du numéro de stream et
de son index, sans état global partagé. Une ligne utilise `base_seed + row`, et
chaque trajectoire réserve une plage de `num_steps` indices. Trois streams
séparent normale de variance, normale de spot et uniforme de variance.

L'implémentation actuelle conserve trois objets de séquence distincts. Une
fusion future pourrait partager seed, index et gestion du cache entre ces trois
streams, réduire les registres et charger trois `RandomQuad` ensemble. Cette
optimisation doit préserver exactement le mapping Philox et faire l'objet d'un
benchmark dédié ; elle n'est pas cachée dans le code actuel.

## Compilation Et Exécution

Depuis la racine du dépôt :

```bash
cmake -S cuda_workbench -B /tmp/ai_factory_cuda_workbench \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ai_factory_cuda_workbench -j

/tmp/ai_factory_cuda_workbench/generate_heston_01
/tmp/ai_factory_cuda_workbench/generate_european_calls_01
/tmp/ai_factory_cuda_workbench/generate_heston_european_calls_01
```

CUDA 11.8 ou plus récent active par défaut `70;86;89`, dont le code natif
`sm_89` de la RTX 4090 Laptop. CUDA 11.5 ne sait pas compiler `sm_89` et utilise
`70;86`. La liste reste configurable :

```bash
cmake -S cuda_workbench -B /tmp/ai_factory_cuda_workbench \
  -DCUDA_WORKBENCH_ARCHITECTURES=89
```

Pour cette machine, préférer un toolkit CUDA >= 11.8 et `89` lors d'un build
spécialisé. Une liste plus large est utile pour distribuer le binaire.

## Ajouter Un Nouveau Couple

Pour reproduire correctement l'architecture :

1. définir la dynamique, le payoff, les bornes et la construction financière ;
2. ajouter le generator modèle sous `registry/production/models/<model>` ;
3. ajouter le generator produit sous `registry/production/products/<product>` ;
4. ajouter le type FP32 et le loader du modèle dans `src/<model>/common.*` ;
5. ajouter le type FP32 et le loader produit dans `src/products/<product>.*` ;
6. placer la dynamique réutilisable dans `src/<model>/dynamics.cuh` ;
7. écrire le `.hpp/.cu` spécialisé du couple modèle-produit ;
8. conserver `prepare_row`, `evaluate_path`, un bloc par prix et la réduction ;
9. écrire le generator résultat sous le registry et l'ajouter à CMake ;
10. générer modèle, produit, puis résultat depuis un build propre ;
11. vérifier schémas JSON/YAML, bornes, valeurs finies et correspondance des IDs ;
12. profiler registres, spills et temps pour 128/256/512 threads.

Un payoff path-dependent change surtout `evaluate_path`, qui maintient un petit
accumulateur au fil des appels à la transition. Un modèle différent remplace
les paramètres préparés et sa fonction de transition. Les options américaines
et bermudéennes demandent une architecture de régression distincte et ne sont
pas couvertes par le contrat européen ci-dessus.

## Validation Minimale

Avant de considérer un nouveau generator comme propre :

- compilation Release sans warning nouveau ;
- aucune allocation ou copie cachée dans le launcher ;
- aucune sortie non finie ou erreur standard négative ;
- même nombre de lignes dans la construction et les résultats ;
- IDs modèle/produit conformes à la construction ;
- kernel sans spill de registres selon `ptxas` ;
- prix stables lorsque le nombre de trajectoires augmente ;
- timing kernel mesuré uniquement avec des événements CUDA autour du launch ;
- timing wall couvrant allocation, transferts, kernel, retours et libération.

La validation CPU, les tests pathwise, les gradients, le C API et les notebooks
ne sont pas encore fournis par ce workbench. Ils devront être ajoutés avant de
remplacer un kernel certifié du projet principal.
