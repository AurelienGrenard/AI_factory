# SABR européen : premier relevé prix, gradient et diagonale (SM89)

Ce relevé accompagne l'ajout du binding SABR européen. Il contrôle le coût du
launcher CUDA ; il ne qualifie ni un dataset complet, ni le biais des bumps.

## Travail mesuré

- RTX 4090 Laptop, SM89, build Release CUDA 13.3, sans fast math ;
- 64 lignes identiques, 8 192 chemins par ligne, call à 126 jours, grille
  `dt=1/504`, une sensibilité par bloc (`B=1`) ;
- `K=0` : prix seul ; `K=1` : spot ; `K=8` : sept paramètres SABR et strike ;
- 21 répétitions après warmup par configuration ; médiane du launcher public,
  préparations et allocations hôtes exclues.

Commande : `build/benchmark_price_gradients_sabr` ; ajouter
`AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS=1` pour les ressources. Le code de la charge et la
méthode de mesure sont dans
`maintainer/tests/performance/price_gradients/sabr.cu` et
`maintainer/tests/performance/price_gradients/benchmark_support.cuh`.

| Threads | K | Prix + ordre 1 (ms) | Prix + ordres 1 et 2 diagonal (ms) |
|---:|---:|---:|---:|
| 128 | 0 | 9,41 | — |
| 128 | 1 | 22,04 | 22,56 |
| 128 | 8 | 44,04 | 55,95 |
| 256 | 0 | 4,66 | — |
| 256 | 1 | 11,89 | 12,12 |
| 256 | 8 | 41,21 | 51,34 |

Sur cette compilation, 256 threads gagnent pour les trois cardinalités.
Le kernel diagonal utilise 128 registres et 608 octets locaux par thread,
avec une occupation théorique de 33,3 % à 128 comme à 256 threads.
À titre de comparaison, le prix seul utilise 52 registres, aucun octet local,
et 75 % ou 66,7 % d'occupation théorique. Le kernel ordre 1 utilise 108
registres et 152 octets locaux par thread. Le Heston diagonal existant utilise
également environ 600 octets locaux par thread : cette pression provient de la
voie terminale à plusieurs nœuds, pas du seul adaptateur SABR.

Ces valeurs ont été reprises après la compilation de la diagonale en maturité
à pas fixe : une mesure effectuée avant cette modification avait une autre
allocation de registres et ne sert pas de référence. Le test CUDA SABR et
Heston passe sur le même GPU ; `compute-sanitizer`
memcheck, racecheck, initcheck et synccheck ne signalent aucune erreur sur le
nouveau test SABR. Les mesures suivantes devront inclure préparation, copies,
batching, reprise et écriture du dataset, puis des charges à plusieurs tailles
de lignes et de chemins avant de fixer une géométrie de production pour
l'ordre 2.
