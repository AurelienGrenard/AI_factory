# Sensibilités Heston américaines et CIR/Jamshidian sur SM89

## Objet

Cette mesure valide le coût de la préparation compacte sur les deux pipelines
qui ne suivent pas la géométrie MC terminale : replay Longstaff--Schwartz gelé
et formule de Jamshidian coopérative. Elle compare l'ordre un à l'ordre un plus
Hessienne diagonale. Elle ne qualifie pas les tailles de bump.

Environnement : NVIDIA GeForce RTX 4090 Laptop, SM89, CUDA runtime 13.3,
compilateur CUDA 13.3.73. Les benchmarks suivent le protocole v3, avec 20
préchauffages pour Heston américain et 5 pour CIR, puis 21 répétitions.

## Heston américain

Commande :

```bash
./build/benchmark_price_gradients_american 262144 4 256 128 first
./build/benchmark_price_gradients_american 262144 4 256 128 diagonal
```

Fixture : quatre prix, neuf sensibilités, 262 144 chemins par prix, 256 threads,
128 shards par prix et seed 41871.

| Ordres | Médiane API | p95 API | CV API | Pic transitoire |
|---|---:|---:|---:|---:|
| premier | 65,914 ms | 69,164 ms | 2,10 % | 155 379 076 octets |
| premier + diagonal second | 74,425 ms | 83,951 ms | 4,75 % | 155 452 804 octets |

La diagonale ajoute 12,9 % à la médiane API. Le nombre de lancements reste 57 :
la spécialisation évalue plus de nœuds dans les deux kernels de sensibilité,
sans ajouter une pipeline LSM.

Le kernel `frozen_sensitivity_moments` utilise 72 registres et 32 octets de
frame local à l'ordre un. La spécialisation diagonale utilise 120 registres et
528 octets de frame local. L'occupation théorique passe de 50 % à 33,3 %.
Cette pression reste acceptable sur cette fixture au vu du temps complet ; ce
résultat ne dispense pas de mesurer un modèle dont l'état ou le payoff est plus
lourd.

## CIR/Jamshidian

Commandes :

```bash
./build/benchmark_price_gradients_jamshidian 1000 256 128 first
./build/benchmark_price_gradients_jamshidian 1000 256 128 diagonal
```

Fixture : 1 000 lignes, sept sensibilités, au plus douze paiements, 256 threads
pour le scalaire et 128 pour le coopératif.

| Ordres | Scalaire, médiane API | Coopératif, médiane API | Gain coopératif |
|---|---:|---:|---:|
| premier | 5,366 ms | 1,114 ms | 79,2 % |
| premier + diagonal second | 5,342 ms | 1,841 ms | 65,5 % |

Le kernel coopératif utilise 59 registres, aucun frame local et 304 octets
partagés statiques à l'ordre un. Il utilise 104 registres, 160 octets de frame
local et 368 octets partagés statiques avec la diagonale. Les variantes
scalaires utilisent respectivement 91/184 et 116/416 registres/octets locaux.

La voie coopérative reste la stratégie par défaut. Son ordre un a présenté un
CV de 7,6 % pendant cette exécution très courte ; la médiane est informative,
mais une publication de baseline demandera une campagne plus longue ou des
opérations groupées.

## Validation numérique et CUDA

- les 56 générateurs `price_gradients`, dont les quatre recettes CIR
  diagonales, compilent ;
- prix et gradients CIR scalaire/coopératif satisfont leur tolérance fondée sur
  l'écart des prix de nœuds ;
- la requête `second` et la requête `first_and_second` donnent les mêmes bits
  pour la diagonale coopérative ;
- la génération Heston européenne/américaine reprend exactement son
  checkpoint ; la génération analytique CIR est déterministe à la
  régénération ;
- memcheck, racecheck, initcheck et synccheck ne signalent aucune erreur sur
  Heston américain et CIR/Jamshidian.

La Hessienne CIR scalaire/coopérative n'est pas une comparaison de référence.
L'écart absolu maximal observé vaut 76,38 : les petits écarts FP32 des prix de
nœuds sont amplifiés par `1/h²`. Les sorties restent finies et cohérentes entre
requêtes d'ordre, mais une campagne de bumps et une référence plus précise sont
requises avant de publier une base de Hessiennes CIR comme qualifiée.
