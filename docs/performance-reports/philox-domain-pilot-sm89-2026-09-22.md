# Pilote Philox par domaines — SM89, 2026-09-22

Ce rapport mesure la première étape sélective. La migration unifiée effectuée
ensuite applique le même compteur V2 à tous les modèles stochastiques ; seuls
les modèles à consommation variable ouvrent plusieurs flux. Les résultats
Merton ci-dessous restent des mesures de la voie multi-flux. Les mesures de
la voie mono-flux commune sont consignées séparément après qualification.

Dans cette première étape, `philox_source_step_v2` était choisi uniquement
par les modèles dont une source peut consommer un nombre variable de tirages :
Merton, Kou, Bates, Variance-Gamma et CIR/CIR++. Heston QE, Vasicek et les
autres consommateurs à tirage fixe conservaient provisoirement le compteur
historique. La migration unifiée remplace cette distinction de mapping par
une distinction du nombre de flux.

## Coût mesuré

GPU : NVIDIA GeForce RTX 4090 Laptop GPU (SM89) ; CUDA 13.3. Les tests ont été
compilés dans `build/` et exécutés sur le GPU local.

| Charge | Mono-flux historique | Domaines V2 | Portée |
|---|---:|---:|---|
| Merton, 2²⁰ chemins, 20 itérations × 10 tours | 0,07630 ms | 0,07358 ms | microbenchmark des innovations et de la transition ; 54 vs 48 registres/thread, aucun spill local |
| Merton gradients, 64 lignes, 7 sensibilités, 8 192 chemins/prix, B=1, 256 threads | 0,1734 ms | 0,1731 ms | médiane kernel V1 publiée le 2026-09-19 et mesure V2 non appariée ; équivalence indicative seulement |
| Même charge, appel API public V2 | — | 0,1776 ms | allocations et transferts d'entrée placés hors mesure par le benchmark |

La comparaison du kernel V1/V2 repose sur deux campagnes distinctes : elle
ne démontre pas un gain. Le temps d'appel V2 mesure la voie publique, mais ne
possède pas encore de témoin V1 apparié. La campagne de coût complet reste à
faire avant une revendication de performance globale.

## Contrôles

- 13 tests CUDA ciblés passent, dont Merton prix/gradients, Bates, Kou LSM,
  Variance-Gamma, CIR/CIR++ et Philox ; Kou garde une comparaison bit à bit
  entre géométries CUDA et une comparaison statistique à sa référence V1.
- 72 tests Python et 16 sous-tests passent pour le codegen, le catalogue et la
  provenance. Régénération codegen complète identique au dépôt.
- Compute Sanitizer : memcheck sur Philox, Bates, Variance-Gamma et CIR ;
  racecheck, initcheck et synccheck sur Philox ; aucune erreur ou alerte.
- Les 142 reçus `generation.yaml` locaux déjà complets pour les recettes V2
  de cette première étape ont une empreinte de recette différente : ils
  décrivent l'ancienne base V1. Après unification, les 598 reçus complets
  liés aux recettes stochastiques sont reconnus comme anciens.
  `check_dataset_compatibility.py` retourne effectivement `regenerate` pour
  l'ancien Merton call européen ; aucune base n'a été régénérée ici.

Les marques centrales événementielles de Merton/Bates et les gradients
d'intensité ou de maturité sont traités par le chantier sauts, pas par cette
migration de l'adressage Philox.
