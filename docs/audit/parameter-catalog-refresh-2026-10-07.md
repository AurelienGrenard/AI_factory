# Reprise complète des paramètres modèles et produits — 2026-10-07

Objectif : entraîner des fast pricers sur un domaine varié avec un cœur de 90 % et une queue de stress de 10 %, sans faire de la variance Monte Carlo ou des trajectoires explosives la cible principale. Chaque jeu de paramètres reste produit par son `generator.cpp` natif ; aucun JSON n'a été filtré ou retouché ligne par ligne.

## Périmètre et décision

Les **25 jeux modèles et 27 jeux produits** ont été reconstruits et exécutés dans le build Ninja isolé `/tmp/ai_factory_aligned_ninja_20261007`. Les 52 JSON canoniques de `datasets`, leurs 52 `recipe.yaml` et leurs 52 `generation.yaml` ont été remplacés par les sorties de cette exécution. Les générateurs aléatoires ont chacun une nouvelle seed ; les générateurs à grille conservent leur méthode déterministe et utilisent les nouvelles coordonnées de stress. Chaque nouveau JSON diffère de son prédécesseur. Le [manifeste de publication](../../artifacts/audit/parameter-catalog-refresh-20261007.json) fige les hashes des sources, binaires, recettes et sorties ; les reçus individuels passent le schéma de provenance.

Les méthodes de tirage ont été conservées : uniformes indépendantes pour les familles qui les utilisaient, tirages conditionnels Feller, reconstruction G2, propositions/rejets NIG et VG, grilles pour les produits standards et tirages contraints pour les produits structurés. Les bornes cœur restent en place sauf quelques choix motivés : `ρ` positif est retiré du cœur Schöbel–Zhu, rough Stein–Stein et G2/G2++ ; SABR markovien passe à `β≥0,50` et `ν≤1,50` en cœur après le signal de bruit MC décrit ci-dessous ; la première date d'exercice cœur Bermudan est bornée à quinze ans au lieu de trente.

| Familles | Changement des stress |
|---|---|
| Modèles corrélés | Corrélations strictement négatives, y compris les deux facteurs G2 ; Stein–Stein garde son `ρ=0` constitutif. Les bornes des volatilités, intensités de saut, feedbacks rough et états de taux sont réduites selon la famille. Cette règle est une politique de données d'entraînement, pas une interdiction du moteur pour `ρ>0`. |
| Modèles sans corrélation | Black–Scholes, CEV, sauts, NIG, VG, CIR et taux gaussiens conservent leurs cœurs ; les stress qui favorisaient les moments proches de leur domaine ou des taux/volatilités disproportionnés sont modérés. Le QRH conserve le domaine de feedback borné testé auparavant. |
| 18 produits equity à grille | Maturité stress maximale ramenée de sept à cinq ans. La grille logarithmique strike–maturité, les barrières dérivées et les calendriers propres à chaque produit restent générés par la méthode existante. |
| 5 produits structurés | Maturité stress maximale de six ans ; coupons, barrières et caps/floors stress resserrés dans leurs contraintes d'ordre. Les cœurs conservent leurs plages. |
| 4 produits fixed income | Retrait des combinaisons de premières dates d'exercice jusqu'à trente/cinquante ans, swaps de cinquante/cent ans et strikes 35 %. Les options de taux, swaptions et options sur zéro-coupon gardent des queues de maturité et de strike utilisables. |

Les [supports observés des 52 jeux](../../artifacts/audit/parameter-catalog-refresh-20261007-quality.json) donnent les minima/maxima réels de chaque paramètre en cœur et stress. Il y a **zéro corrélation positive** dans les 25 jeux modèles. Les 52 sorties ont 1 000 lignes chacune, ordonnées 900/100. En **aligné**, les lignes 901–1000 joignent deux stress ; pour le futur produit cartésien, la proportion théorique avec au moins un stress vaut `1 − 0,9² = 19 %`, dont `1 %` à double stress.

## Tests de prix sur les nouveaux domaines

Cinq couples européens call/put, chacun de 1 000 lignes et `2²⁰` chemins par prix, ont été exécutés depuis les binaires du worktree avec les nouveaux produits. Rough Bergomi, rough SABR, log-modulated rough Bergomi et quadratic rough Heston ont chacun **zéro écart de parité >5 SE** et **zéro prix >0,01 avec SE/prix >25 %** ; les maxima `|z|` sont respectivement 3,734 ; 3,618 ; 3,752 ; 2,969. Le premier tirage SABR markovien avait une ligne de parité à 5,067 SE et une ligne cœur à 36 % de SE relative. Trois replays de moment actualisé sur deux lignes stress n'ont montré ni non-fini ni défaut systématique, mais l'une absorbait environ 12,6 % des chemins. La borne cœur `ν≤1,50`, la borne stress `ν≤1,80` et les planchers `β` révisés éliminent ces deux signaux : sur le second tirage, maximum `|z|=3,676`, zéro dépassement de 5 SE et zéro prix matériel à SE relative >25 %. Les [résultats MC](../../artifacts/audit/parameter-catalog-refresh-20261007-quality.json) distinguent explicitement le premier essai du tirage canonique final.

Sur le nouveau G2 et les nouveaux produits fixed income, quatre générateurs de prix de 1 000 lignes ont été exécutés. La parité call/put des options sur zéro-coupon a un écart absolu maximal de `1,92×10⁻⁷` face à une formule gaussienne indépendante de `P(0,t)` ; les swaptions européennes payer/receiver ont zéro sortie non finie et zéro prix >0,01 à SE relative >25 %. [Détails et hashes](../../artifacts/audit/parameter-catalog-refresh-20261007-g2-check.json). La suite `tests/datasets` passe : **57 tests**. Les 52 reçus ont été relus, leur `record_sha256` recalculé et leurs recettes comparées aux SHA-256 annoncés après publication ; `git diff --check` passe.

## Implantation des données

Le catalogue de production occupe directement `catalog/` et `datasets/` ; les recettes locales et leurs sorties occupent `work/catalog/` et `work/datasets/`. Les archives, qualifications et releases de métadonnées sont sous `work/catalog/`, sans dossier `prod` ou `other`. Les anciens manifestes de qualification et de release conservent leurs chaînes de chemin et leurs hashes ; `catalog_layout.physical_path` traduit ces anciens chemins vers les nouveaux emplacements lors de la vérification. Les 696 sorties sélectionnées sont présentes ; les 24 sorties locales matérialisées et les payloads historiques de qualification et release restent sous `work/datasets/`, et les vérificateurs historiques passent après le déplacement. Le générateur aligné Black–Scholes/european_calls a également été exécuté en staging avec les nouveaux chemins et a produit un JSON et un reçu ; ce smoke test ne vaut pas validation indépendante du prix.

## Prix encore à recalculer

Les anciens prix alignés de `datasets` restent calculés sur les **anciens** paramètres : leurs reçus ne sont pas une qualification des nouveaux couples. Leurs entrées exactes sont conservées dans l'[archive versionnée](../../work/catalog/archives/parameter-catalog-20261007/manifest.json) : les anciens générateurs rough SABR et log-modulated rough Bergomi ont été rejoués, puis leurs JSON ont été comparés aux hashes et aux contrôles historiques. Les vérificateurs NUM-029, NUM-030 et rough Bergomi utilisent cette archive ; ils passent tous, ainsi que le gate de domaine. Le quadrature NUM-030 a reproduit exactement ses huit cas sur les anciennes entrées. Le contrôle NUM-029 retrouve zéro flag rough SABR et 26 flags SABR historiques. `NUM-028` et `NUM-030` restent ouverts ; la validité des autres payoffs rough, les queues rares et la portée de l'estimand QRH continu demandent des vérifications après repricing. Les releases et masques versionnés antérieurs restent des preuves historiques attachées à leurs entrées figées.

Depuis la racine `AI_factory`, après l'arrêt de la compilation lancée avant le déplacement, cette commande **compile et exécute en staging** les 655 recettes alignées. Elle utilise un nouveau build Ninja configuré sur l'arborescence finale et un nouveau répertoire de campagne. En cas d'interruption, relancer la même commande avec `--resume` :

```bash
python3 tools/datasets/generate_catalog.py \
  --build /tmp/ai_factory_aligned_post_layout_20261007 \
  --kind prices --construction aligned \
  --compile --compile-jobs 4 \
  --run-dir work/generation/aligned-price-refresh-post-layout-20261007 \
  --execute
```

Le périmètre de cette campagne est **uniquement aligné** : 655 recettes. Les recettes cartésiennes sont exclues. Le contrôleur gèle les sources et les entrées et refuse une publication qui écraserait les prix existants. La commande produit les résultats et reçus en staging pour revue ; elle ne modifie pas les prix canoniques.
