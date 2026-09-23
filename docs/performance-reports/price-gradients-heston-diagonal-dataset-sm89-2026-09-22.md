# Heston européen : génération gradient et diagonale (SM89)

Ce relevé accompagne le branchement de l'ordre deux diagonal dans la chaîne
de génération des datasets. Il s'agit d'une vérification bornée de la
géométrie et des ressources, pas d'une qualification des bumps.

## Pipeline vérifiée

- préparation compacte sur GPU, sans tableau hôte de scénarios bumpés ;
- grille `(row, sensitivity)`, `B=1`, chemins répartis entre les threads ;
- gradient et Hessienne diagonale reconstruits chemin par chemin puis réduits
  en FP64 ;
- écriture des stencils à trois ou quatre nœuds ;
- checkpoint comprenant prix, erreurs, gradients, Hessiennes et leurs erreurs ;
- reprise bit à bit des résultats déjà confirmés.

Le test `price_gradients_diagonal_dataset_cuda` exécute deux fois la même
petite campagne Heston. La seconde exécution reprend le préfixe complet depuis
le checkpoint. Les cinq tests CUDA terminaux Black--Scholes/Heston, CEV, SABR,
Merton et dataset diagonal passent sur la même compilation.

## Mesure bornée

RTX 4090 Laptop, SM89, build Release CUDA 13.3, sans fast math. Le benchmark
public utilise 64 lignes, sept sensibilités, 8 192 chemins par ligne, une
maturité de 126 jours et 21 répétitions après warmup.

| Threads | Temps médian public/kernel |
|---:|---:|
| 128 | 52,39 ms |
| 256 | 42,44 ms |

Le candidat à 256 threads reste donc préférable sur cette charge. Le kernel
final compilé utilise 128 registres et 624 octets locaux par thread, 624 octets
partagés statiques et 128 octets partagés dynamiques. L'occupation théorique
est de 33,3 %. Le kernel de préparation des stencils utilise également 128
registres mais n'est lancé qu'une fois par plage à matérialiser ou à restaurer.

La mémoire locale confirme que quatre états de modèle restent une limite à
surveiller pour chaque nouveau couple modèle-produit. Une extension doit
relever les ressources de sa spécialisation compilée et mesurer 128/256
threads avant de reprendre ce candidat.

Les tests finaux ont également relevé les spécialisations terminales qui
possèdent désormais une recette diagonale. Les nombres ci-dessous concernent
le kernel `first_and_second` compilé ; ils ne comparent pas des charges de
pricing équivalentes.

| Modèle/moteur | Registres/thread | Local/thread | Shared statique/bloc |
|---|---:|---:|---:|
| Black--Scholes, formule fermée | 116 | 496 octets | 0 |
| CEV, MC à pas | 126 | 560 octets | 448 octets |
| Heston, MC à pas | 128 | 624 octets | 624 octets |
| SABR, MC à pas | 128 | 608 octets | 576 octets |
| Merton, MC terminal exact | 128 | 560 octets | 560 octets |

Toutes restent sous la limite de 255 registres par thread, mais toutes
matérialisent une partie de leur état dans la mémoire locale. Heston est la
spécialisation la plus lourde de ce groupe sur SM89.
