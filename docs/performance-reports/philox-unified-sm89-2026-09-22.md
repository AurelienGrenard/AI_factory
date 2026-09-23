# Compteur Philox V2 commun — contrôle SM89 du 2026-09-22

Tous les consommateurs Philox utilisent désormais le compteur
`(path_bas, path_haut, groupe, domaine)`. Le domaine zéro porte le flux continu
unique ; Merton, Kou, Bates, Variance-Gamma et CIR/CIR++ ouvrent des sources
supplémentaires au besoin. L'accès aléatoire rough FFT et `Uint32Sequence`
passent par la même primitive de compteur.

Pour `groupe < 2^32`, le compteur du flux unique est bit à bit égal à l'ancien
compteur 64 bits de groupe, dont le mot haut valait zéro. Sous la même graine,
la fixture gradients 64 lignes a conservé exactement les 896 valeurs de sortie
Black–Scholes et les 1 408 puis 1 920 valeurs Heston. Les 1 024 sorties Merton
étaient également identiques à la première étape multi-flux. Cette parité ne
prétend pas couvrir les groupes hors de la borne V2 ni les anciennes bases des
modèles désormais multi-flux.

## Mesures pilotes

GPU NVIDIA GeForce RTX 4090 Laptop GPU (SM89), CUDA 13.3 ; 64 lignes,
8 192 chemins par prix, `B=1`, 256 threads. Les nombres sont des médianes
kernel en millisecondes. Les appels publics mesurés placent allocations et
transferts d'entrée hors chronométrage.

| Modèle et sensibilités | Avant unification | V2, passage 1 | V2, passage 2 |
|---|---:|---:|---:|
| Black–Scholes, 6 | 0,1157 | 0,1178 | 0,1198 |
| Heston, 10 | 45,0406 | 47,1009 | 44,9311 |
| Heston, 7 avec Hessienne diagonale | 38,8188 | 41,4607 | 39,3513 |
| Merton, 7 | 0,1731 | 0,1884 | 0,1782 |

La variation entre les deux passages V2, sans modification de code, interdit
de conclure à une régression stable ou à un gain à partir de ces mesures.
Le compteur mono-flux n'ajoute aucun contexte par domaine aux modèles
concernés. Une campagne appariée sur temps de bout en bout, avec contrôle des
fréquences et ressources du kernel, reste nécessaire avant qualification de
performance.

## Vérifications de comportement et de provenance

- 22 tests CUDA ciblés passent pour Philox, prix/gradients, LSM, Heston,
  Vasicek, G2, Bates, CIR, Variance-Gamma, rough FFT et samples. Les 72 tests
  Python et 16 sous-tests de codegen, catalogue et provenance passent.
- La régénération codegen complète correspond au dépôt. Les recettes
  stochastiques déclarent toutes `philox_source_step_v2` et publient sous
  `/v2/` ; les formules fermées déterministes restent hors du contrat RNG.
- Les 598 reçus locaux `generation.yaml` complets issus des anciennes
  recettes ont une empreinte différente. Le contrôle de compatibilité retourne
  `regenerate` sur les exemples Heston et Merton vérifiés ; aucune base n'a
  été régénérée ni publiée ici.
- Compute Sanitizer ne signale aucune erreur : memcheck, racecheck,
  initcheck et synccheck sur Philox ; memcheck sur l'oracle rough FFT.
