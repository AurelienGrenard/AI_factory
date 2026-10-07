# Domaines des modèles pour les prix equity alignés — 2026-10-07

> **Archive de la campagne candidate antérieure.** La release SABR v1 et les essais
> rough ci-dessous restent vérifiables avec leurs entrées figées. Le catalogue
> canonique modèles **et produits** a ensuite été régénéré ; voir la
> [reprise complète](parameter-catalog-refresh-2026-10-07.md). Les affirmations
> « produits inchangés » dans cette page décrivent exclusivement cette campagne v1.

Les 1 000 produits de chaque catalogue restent **bit à bit identiques**. Les
nouvelles plages portent seulement sur les générateurs de paramètres des modèles.
La publication des prix associés est versionnée : les anciens JSON et leurs
reçus restent lisibles comme historique, avec leurs qualifications d'audit.

## Décision sur les corrélations

Le domaine de référence equity courant impose `ρ<0` aux générateurs SABR
markovien, rough Bergomi et log-modulated rough Bergomi modifiés ici. Il s'agit
d'une politique de **référence MC alignée**, pas d'une restriction du moteur
ou d'une affirmation que tout modèle à `ρ>0` est invalide. Pour les deux
Bergomi exponentiels, le domaine positif pose aussi une question de martingale
continue démontrée dans [l'analyse NUM-030](num030-martingale-domain.md).
Pour SABR à `β<1`, la martingalité continue subsiste ; les anciens écarts
markoviens étaient numériques et concentrés dans des paramètres difficiles
([analyse SABR](sabr-martingale-domain.md)). Les autres modèles à corrélation
positive ne sont pas modifiés sans signal de qualité propre à leur famille.

## SABR markovien : release exécutée et vérifiée

Le générateur `sabr_01` produit désormais `β∈[0,30;0,95]`, `ρ<−0,20`, et
borne la volatilité initiale et sa volatilité stress. Le domaine peut décrire
un skew equity négatif ; il ne couvre plus les scénarios de skew inversé.
La [release SABR](../../work/catalog/releases/sabr-aligned-mc-domain-20261007-v1/manifest.json)
contient **un jeu de 1 000 modèles et les 29 jeux de 1 000 prix alignés**.
Chaque `generation.yaml` provient de l'exécution réelle du générateur du
worktree courant ; les binaires, sources, recettes, build et 23 entrées
modèle/produit ont été figés et hachés. La
[vérification de release](../../maintainer/tools/datasets/verify_aligned_model_release.py)
recalcule les empreintes des 30 paires, la parité européenne et les erreurs
relatives des 29 jeux. Résultat : **0 signal de parité** sur 1 000 paires,
`|z|` maximal 4,390 ; **0 prix >0,01** avec `SE/prix >25 %` parmi 29 000 prix.
La plage réalisée est `ρ∈[−0,9873;−0,2017]`. Les entrées produits sont
inchangées. Les anciennes 52 sorties SABR masquées par NUM-037 ne deviennent
pas certifiées par cette nouvelle release.

## Bergomi : domaines candidats validés sur les Européennes

Le rough Bergomi standard a désormais `ρ<−0,20` dans les lignes stress et
`η≤4` ; le Bergomi log modulé a `ρ<−0,20`, `η≤3,5` et borne également
`ξ₀`, `H`, l'échelle et la puissance de modulation stress. Le
[scan candidat](../../artifacts/audit/aligned-mc-domain-2026-10-07/bergomi_european_candidate_scan.json)
provient de **deux générateurs de modèles et quatre générateurs de prix**
réellement exécutés à 2²⁰ chemins par prix sur le build isolé courant, avec
les produits européens inchangés. Pour chacun des deux modèles : 1 000 paires,
0 signal de parité et 0 prix >0,01 avec `SE/prix >25 %`. Les maxima `|z|`
sont 3,236 et 3,421. Ces essais européens ne qualifient pas encore les 27
autres payoffs de chaque modèle ni les anciennes queues stress NUM-030.

## Quadratic rough Heston : domaine candidat de grille

Le générateur candidat réduit les coefficients de feedback quadratique. Le
maximum réalisé de `a` est 0,1771, de `λ` 1,1909 et de `η` 0,7878 ;
l'ancienne recette montait à 2, 6 et 4 sur les lignes stress. Les **1 000
paires européennes** de la [preuve candidate](../../artifacts/audit/aligned-mc-domain-2026-10-07/qrh_candidate_quality.json)
ont 0 signal de parité, `|z|` maximal 3,203 et 0 prix >0,01 avec
`SE/prix >25 %`. Sur la ligne 984, [18 replays](../../artifacts/audit/aligned-mc-domain-2026-10-07/qrh_grid_factor_replays.json)
à deux graines, 1/2/4 pas par jour et 2/3/7 facteurs donnent un moment
actualisé entre 99,958 % et 100,073 % du forward, avec `|z|≤1,224`.
Sur le lookback 770, [neuf replays](../../artifacts/audit/aligned-mc-domain-2026-10-07/qrh_lookback_tail_grid_replays.json)
à trois graines et trois grilles donnent une moyenne entre 0,2171 et 0,2202 ;
les cent plus grands payoffs portent au plus 0,196 % de la moyenne. Aucun
chemin de ces essais n'a un spot nul ou une sortie non finie.

Cette preuve concerne **l'estimand MC à grille finie et le domaine candidat**.
Elle ne démontre pas que le spot de la dynamique quadratique non bornée est
une vraie martingale à temps continu. Une [étude primaire récente de ce
modèle](https://www.researchgate.net/publication/397988507_Demystification_of_the_joint_SPXVIX_calibration_problem_with_the_quadratic_rough_Heston_model)
introduit une variance tronquée précisément pour assurer cette propriété ;
une restriction des paramètres ne fournit pas la même preuve. Nous ne
tronquons ni variance ni prix dans cette campagne. NUM-028 reste ouvert
jusqu'à qualification de tous les payoffs alignés, de l'estimand et des
anciennes sorties non fiables.

## Reproduction

Le contrôleur [generate_catalog.py](../../tools/datasets/generate_catalog.py)
accepte `--input-override LOGICAL_PATH=SOURCE_PATH` pour figer un nouveau jeu
modèle et les produits publiés sans réécrire les anciennes données. Le
[runner](../../maintainer/tools/datasets/run_aligned_model_campaign.py) exécute le
modèle puis les 29 prix ; le
[publisher](../../maintainer/tools/datasets/publish_aligned_model_release.py) refuse
une release qui conserve une parité signalée ou un prix significatif à forte
SE et conserve les entrées, sources et reçus. Le
[vérificateur](../../maintainer/tools/datasets/verify_aligned_model_release.py) contrôle
les fichiers publiés indépendamment du staging. Le build isolé est Release
CUDA 12.9.41, SM89. Les produits canoniques ne sont pas régénérés.
